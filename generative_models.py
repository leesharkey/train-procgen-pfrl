import torch
import torch.nn as nn
from policies import ResidualBlock
import layers as lyr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P


"""Proposal for generative model:

    - An encoder network
        - EncoderInputNetwork
            - see diagram on tablet
            - image -> residual (64)
            - init hidden state added to the channel dim (64+128)
            - Resblock down all (image, init hidden, and resid output) into 1x1 conv (3+64+128 -> 128) (32x32x128)
            - layer norm
            - non-local layer (with residual connection)
            - layer norm
            - ResOnebyOne (dense from all(?) previous)
            - Resblockdown 
            - layer norm
        - EncoderRNN
            - convGRU (8x8x256)
            - layer norm after every step
        - EncoderEmbedder (takes final convGRU output only)
            - Resblock down (8x8x256) -> (4x4x256)
            - layer norm
            - split
                - fc to mu (should have around 256 neurons)
                - fc to sigma
    - A decoder network
        - Initializer networks
            - InitHiddenStateNetwork = TwoLayerPerceptron
                - fc
                - layer norm
                - fc
            - PrevActionNetwork = TwoLayerPerceptron
            - ConvGRU initializer
                - This takes a guess at initializing the GRU so 
                  it doesn't start with 0-tensors. Informative initialization 
                  like this should work better.  
                - outputs something the size of the 'EnvStepper hidden state shape' for env
                  unrolling         
            - UnrollerNet
                - AssimilatorResidualBlock (takes EnvStepper hidden state shape block and also noise vector and outputs EnvStepper hidden state shape block) 
                - layer norm
                - ResidualConv
                - layer norm
        - Side decoders (take a EnvStepper hidden state shape block and produce predictions for obs and rew
            -reward decoder
                - Residual shrink block
                - layer norm
                - FC -> 1
            - obs decoder
                - deconv growth residual block (but reduce channels)
                - layer norm
                - deconv growth residual block (but reduce channels)
                - layer norm
                - deconv growth residual block (but reduce channels)
                - layer norm
                - conv (but reduce channels) (3)
            
    -  AssimilatorResidualBlock
        - has-a:
            - AssimilatorBlock (1x1 conv to 2d conv)
            - residual connection between non vec inputs to AssimilatorBlock and its outputs
"""

"""We'll use a convGRU unroller and have a non local layer in the obs decoder
At the input to the convGRU at every timestep we'll have an assimilator
that takes the noise, the action, and the hidden state and projects it
into something that's the same size as the hidden state, to which we'll add it.
This way, the assimilator is a res net and just updates the inputs to the
recurrent network.  
"""

"""
Classes we'll need:
- ResidualBlock for upsample and downsample and constant (from BigGAN)
- non-local layer (Attention)
- AssimilatorResidualBlock
- ResOnebyOne
- Encoder
    - EncoderInputNetwork
    - EncoderRNN
    - EncoderEmbedder
- Decoder
    - InitializerNets
        - IHANetwork
        - EnvStepperInitializer
    - EnvStepper
    - reward decoder
    - obs decoder
"""

"""
We can also explore whether or not it's worth learning the initialisation for
the encoder convGRU. For now, we'll just explore zero and noise 
initializations.
"""

class VAE(nn.Module):
    """The Variational Autoencoder that generates agent-environment sequences.

    Description

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    def __init__(self, agent, device, num_recon_obs, num_pred_steps):
        super(VAE, self).__init__()

        # Set settings
        self.num_recon_obs    = num_recon_obs
        self.num_pred_steps   = num_pred_steps
        self.num_unroll_steps = self.num_recon_obs + self.num_pred_steps
        self.device = device

        # Create networks
        self.encoder = EncoderNetwork(device).to(device)
        self.decoder = DecoderNetwork(device, agent,
                           num_unroll_steps=self.num_unroll_steps).to(device)

    def forward(self, obs, agent_h0):

        # Ensure the number of images in the input sequence is the same as the
        # number of observations that we're _reconstructing_.
        assert obs.shape[1] == self.num_recon_obs

        # Feed observation sequence into encoder, which returns the mean
        # and log(variance) for the latent sample (i.e. the encoded sequence)
        mu, logvar = self.encoder(obs, agent_h0)

        sigma = torch.exp(0.5 * logvar)  # log(var) -> standard deviation

        # Reparametrisation trick
        sample = (torch.randn(sigma.size()) * sigma) + mu

        # Decode
        pred_obs, pred_rews, pred_dones, pred_agent_hs, pred_agent_logprobs = \
            self.decoder(sample)

        preds = {'obs': pred_obs,
                 'reward': pred_rews,
                 'done': pred_dones,
                 'rec_h_state': pred_agent_hs,
                 'action_log_probs': pred_agent_logprobs}

        return mu, logvar, preds


class EncoderNetwork(nn.Module):
    """The Variational Autoencoder that generates agent-environment sequences.

    Description

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, device):
        super(EncoderNetwork, self).__init__()
        self.input_network = EncoderInputNetwork(device, agent_hidden_size=256).to(device)
        self.rnn = EncoderRNN(device)
        self.embedder = EncoderEmbedder(device)

    def forward(self, obs, agent_h0):

        # Reset encoder RNN's hidden state (note: not to be confused with the
        # agent's hidden state, agent_h0)
        h = None

        obs_seq_len = obs.shape[1] # (B, *T*, Ch, H, W)

        # Pass each image in the input input sequence into the encoder input
        # network to get a sequence of latent representations (one per image)
        inps = []
        for i in range(obs_seq_len):
            inps.append(self.input_network(obs[:,i,:], agent_h0))
        inps = torch.stack(inps, dim=1) # stack along time dimension

        # Pass sequence of latent representations
        h = self.rnn(inps, h)

        # Convert sequence embedding into VAE latent params
        mu, sigma = self.embedder(h)

        return mu, sigma




class EncoderInputNetwork(nn.Module):
    """Input Network for the encoder.

    Takes (a batch of) single images at a single timestep.

    Consists of a feedforward convolutional network. It has many residual
    connections, which sometimes skip several layers. In that sense, it is
    similar to a `dense` convolutional network, which has many such
    connections.

    It also `assimilates` the initial agent hidden state (1D vector) into the
    convolutional representations (3D tensor).

    Its output is passed to a recurrent network, which thus accumulates
    information about each image in the sequence.

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, device, agent_hidden_size=256):
        super(EncoderInputNetwork, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=128,
                               kernel_size=3, padding=1).to(device)
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.norm1 = nn.LayerNorm([128,32,32])
        self.resdown1 = lyr.ResBlockDown(128,128,downsample=self.pool)
        self.assimilateh0 = lyr.AssimilatorResidualBlock(128, agent_hidden_size)
        self.norm2 = nn.LayerNorm([128,32,32])
        self.resdown2 = lyr.ResBlockDown(128,128,downsample=self.pool)
        self.norm3 = nn.LayerNorm([128,16,16])
        self.attention = lyr.Attention(128)
        self.norm4 = nn.LayerNorm([128,16,16])
        self.res1x1   = lyr.ResOneByOne(128+128*3, 128)
        self.resdown3 = lyr.ResBlockDown(128,256,downsample=self.pool)
        self.norm5 = nn.LayerNorm([256,8,8])

    def forward(self, ob, h0):
        x  = ob
        z  = self.conv0(x)
        z  = self.resdown1(z)
        x1 = self.norm1(z)
        z  = self.assimilateh0(x1, h0)
        x2 = self.norm2(z)
        z  = self.resdown2(x2)
        x3 = self.norm3(z)
        z  = self.attention(x3)
        x4 = self.norm4(z)
        x123 = torch.cat([self.pool(x1), self.pool(x2), x3], dim=1)
        z  = self.res1x1(x4, x123)
        z  = self.resdown3(z)
        z  = self.norm5(z)
        return z

class EncoderRNN(nn.Module):
    """Recurrent network for input image sequences.

    The `EncoderRNN` takes as input the outputs of `EncoderInputNetwork`s for
    each input image. It thus takes as input a sequence of image
    representations (not raw images). It learns to encode the dynamics of the
    input image sequence in order that the decoder can reconstruct the input
    image sequence and also predict subsequent images that were not actually
    in the input sequence.

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, device):
        super(EncoderRNN, self).__init__()
        self.rnn = lyr.ConvGRU(input_size=[8, 8], # [H,W]
                               input_dim=256, # ch
                               hidden_dim=256,
                               kernel_size=(3,3),
                               num_layers=1).to(device)
        self.norm = nn.LayerNorm([256, 8, 8])

    def forward(self, inp, h=None):
        h = self.rnn(inp, h)
        h = self.norm(h[0][:,-1])  # only use final hidden state
        return h


class EncoderEmbedder(nn.Module):
    """Converts the output of the RNN to the VAE's latent sample params.

    The encoder embedder is the final layer of the VAE encoder.

    The output of the EncoderRNN is passed to the encoder embedders, which
    simply does some non-recurrent processing in order to generate the
    mean and log(variance) of the sample in the latent space of the VAE that
    is used, by the decoder, to reconstruct the input image sequence and
    predict future images.

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, device):
        super(EncoderEmbedder, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.resdown = lyr.ResBlockDown(in_channels=256,
                                        out_channels=256,
                                        downsample=self.pool).to(device)
        self.norm = nn.LayerNorm([256,4,4])
        self.fc_mu    = nn.Linear(4*4*256, 256).to(device)
        self.fc_sigma = nn.Linear(4*4*256, 256).to(device)

    def forward(self, inp):
        x = inp
        x = self.resdown(x)
        x = self.norm(x)
        mu = self.fc_mu(x.view(x.shape[0], -1))
        logvar = self.fc_sigma(x.view(x.shape[0], -1))
        return mu, logvar

class DecoderNetwork(nn.Module):
    """Reconstructs and predicts agent-environment sequences.

    Generates a whole agent-environment sequence from a latent sample from the
    VAE. It contains a copy of the agent. The VAE is trained such that the
    decoder produces:
        - a sequence of observations
        - a sequence of rewards
        - a sequence of hidden states of the agent
        - a sequence of actions from the agent (and their log probabilities)
    that match as closely as possible the sequences that were actually observed
    in real roll outs of the agent-environment system.

    It uses a recurrent architecture to generate sequences of arbitrary length,
    so sequences longer than the training sequences may be generated if wanted.
    The recurrent architecture consists of an environment and an agent part:
    the agent part is the agent itself, which takes as input an observation
    of the (simulated) environment and its own hidden state. The environment
    part consists of several networks:
        - An EnvStepper, which has 'environment hidden state' which unrolls
          through time
        - An ObservationDecoder, which takes the environment hidden state and
          converts it into an observation for that timestep.
        - A RewardDecoder, which does the same for the reward that agent
          received at that timestep.
        - # TODO a DoneDecoder, which predicts whether the episode is done at
          that timestep.

    Since both the agent and the EnvStepper are recurrent, they require inputs
    to get the ball rolling. There are several networks that convert the latent
    VAE sample into the inputs required. Those are:
        - inithidden_network, which generates the initial hidden state of the
          agent. We don't need to produce later hidden states, because the
          agent does that itself.
        - env_init_network, which generates the initial hidden state of the
          EnvStepper.
        - prev_act_network, which generates a vector with the same dimension
          as the action space. This is required for the env stepper, which
          can only unroll if it has a current env hidden state and the action
          generated by the agent at the previous timestep. The reason why it's
          the _previous_ action and not the current action is because that's
          just the way MDPs are formulated. Note that this action is not
          reconstructed like subsequent actions - it happened before t=0, so
          isn't part of the input sequence. So it's just some vector produced
          by the VAE to start the EnvStepper unrolling, and may not actually
          represent an action (which is a one-hot vector).

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, device, agent, num_unroll_steps):
        super(DecoderNetwork, self).__init__()
        self.action_dim = agent.model.n_actions

        # Initializers
        self.inithidden_network = TwoLayerPerceptron(
            insize=256,
            outsize=256)
        self.prev_act_network = TwoLayerPerceptron(
            insize=256,
            outsize=self.action_dim)
        self.env_init_network = EnvStepperInitializer(device=device)

        # Stepper (the one that gets unrolled)
        self.env_stepper      = EnvStepper(agent, env_h_ch=128, env_h_hw=8)

        # Decoders (used at every timestep
        self.reward_decoder = RewardDecoder(device)
        self.done_decoder   = DoneDecoder(device)
        self.obs_decoder    = ObservationDecoder(device)
        self.agent = agent
        self.num_unroll_steps = num_unroll_steps

    def forward(self, sample):

        # Get initial inputs to the agent and EnvStepper (both recurrent)
        env_h = self.env_init_network(sample)  # t=0
        prev_act = self.prev_act_network(sample)  # t=-1
        agent_h = self.inithidden_network(sample)  # t=0
        self.agent.train_recurrent_states = (agent_h.unsqueeze(0),)

        # Unroll the agent and EnvStepper and collect the generated data
        pred_obs = []
        pred_rews = []
        pred_dones = []
        pred_agent_hs = []
        pred_agent_logprobs = []

        for i in range(self.num_unroll_steps):

            # Observations
            obs = self.obs_decoder(env_h)
            pred_obs.append(obs)

            # Rewards
            rew = self.reward_decoder(env_h)
            pred_rews.append(rew)

            # Dones
            done = self.done_decoder(env_h)
            pred_dones.append(done)

            # Step environment forward using current state and *previous* action
            env_h = self.env_stepper(sample, prev_act, prev_h=env_h)

            # Store curr agent-hidden state, sample the action, and get next
            # agent-hidden state
            pred_agent_hs.append(self.agent.train_recurrent_states[0].squeeze())

            obs = obs.permute(0,2,3,1)
            act = self.agent.batch_act(obs)

            # Add obs, agent_h, actlogprobs, and rew to lists
            pred_agent_logprobs.append(self.agent.train_action_distrib)

            # Get ready for new step
            act = torch.tensor(act)
            act = torch.nn.functional.one_hot(act, num_classes=self.action_dim)
            prev_act = act
            self.agent.train_prev_recurrent_states = None

        return pred_obs, pred_rews, pred_dones, pred_agent_hs, pred_agent_logprobs


class TwoLayerPerceptron(nn.Module):
    """A two layer perceptron with layer norm and a linear output.

    It takes the VAE latent sample as input and outputs another vector.

    This class is used for multiple purposes:
      - In the prev_act_network, the output is a vector the size of the action
       space, which is used as one of the initial inputs to the EnvStepper.
      - In the inithidden_network, the output is a vector the size of the
        agent's hidden state and initializes the agent.

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, insize=256, outsize=256):
        super(TwoLayerPerceptron, self).__init__()
        self.net = \
            nn.Sequential(nn.Linear(insize, insize),
                          nn.ReLU(),
                          nn.LayerNorm(insize),
                          nn.Linear(insize, outsize))

    def forward(self, x):
        return self.net(x)


class EnvStepperInitializer(nn.Module):
    """Initializes the EnvStepper

    The EnvStepper is recurrent and therefore needs an initial hidden state.
    The EnvStepperInitializer generates an initial hidden state by taking the
    VAE latent sample as input and outputting something the size of the
    EnvStepper hidden state.

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, device, vae_latent_size=256, env_h_ch=128, env_h_hw=8):
        super(EnvStepperInitializer, self).__init__()

        self.env_h_ch = env_h_ch
        self.env_h_hw = env_h_hw
        self.fc = nn.Linear(vae_latent_size,
                            int((env_h_ch*env_h_hw*env_h_hw)/(2**2)))
        self.resblockup = lyr.ResBlockUp(in_channels=128,
                                         out_channels=128,
                                         hw=4)
        self.norm1 = nn.LayerNorm([128, 8, 8])
        self.norm2 = nn.LayerNorm([128, 8, 8])
        self.norm3 = nn.LayerNorm([128, 8, 8])
        self.norm4 = nn.LayerNorm([128, 8, 8])

        self.resblock1 = lyr.ResidualBlock(128)
        self.resblock2 = lyr.ResidualBlock(128)
        self.attention = lyr.Attention(128)

    def forward(self, x):
        x = self.fc(x)
        x = self.resblockup(x.view(x.shape[0],       self.env_h_ch,
                                   self.env_h_hw//2, self.env_h_hw//2)
                            )
        x = self.norm1(x)
        x = self.resblock1(x)
        x = self.norm2(x)
        x = self.attention(x)
        x = self.norm3(x)
        x = self.resblock2(x)
        x = self.norm4(x)
        return x

class EnvStepper(nn.Module):
    """A recurrent network that simulates the unrolling of the environment.

    The EnvStepper unrolls a latent representation of the environment through
    time. From its hidden state is decoded several things at each timestep:
      - The observation (by the ObservationDecoder), which is input to the
        agent.
      - The reward (by the RewardDecoder). It is trained to predict reward in
        the expectation that reward-salient aspects of the environment will be
        represented in the EnvStepper latent state.
      - #TODO DoneDecoder
      - #TODO ValueDecoder

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, agent, env_h_ch=128, env_h_hw=8, vae_latent_size=256):
        super(EnvStepper, self).__init__()
        action_dim = agent.model.n_actions
        self.assimilator = \
            lyr.AssimilatorResidualBlock(128,
                                         vec_size=(action_dim+vae_latent_size))
        # EnvStepper hidden state shape is 8x8x128
        self.attention = lyr.Attention(128)
        self.norm1 = nn.LayerNorm([128, 8, 8])
        self.norm2 = nn.LayerNorm([128, 8, 8])
        self.norm3 = nn.LayerNorm([128, 8, 8])
        self.resblock = ResidualBlock(128)

    def forward(self, vae_sample, prev_act, prev_h=None):
        vec = torch.cat([vae_sample,prev_act], dim=1)
        h = self.assimilator(prev_h, vec)
        h = self.norm1(h)
        h = self.attention(h)
        h = self.norm2(h)
        h = self.resblock(h)
        h = self.norm3(h)
        return h

class ObservationDecoder(nn.Module):
    """Decodes the observation from the EnvStepper latent state.

    At every timestep, the ObservationDecoder takes the EnvStepper latent state
    as input and outputs the observation (what the agent sees).

    It uses multiple upsampling residual blocks and a nonlocal layer
    (self-attention). It has a final tanh activation to ensure predicted pixel
    values have the same range as real pixels.

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, device, env_h_ch=128, env_h_hw=8):
        super(ObservationDecoder, self).__init__()
        ch0 = env_h_ch
        hw0 = env_h_hw
        self.resblockup1 = lyr.ResBlockUp(ch0,    ch0//2, hw=hw0)
        self.resblockup2 = lyr.ResBlockUp(ch0//2, ch0//4, hw=hw0*2)
        self.resblockup3 = lyr.ResBlockUp(ch0//4, 3,      hw=hw0*4)
        self.attention = lyr.Attention(64)
        # self.net = nn.Sequential(self.resblockup1,
        #                          self.attention,
        #                          self.resblockup2,
        #                          self.resblockup3)

    def forward(self, x):
        x = self.resblockup1(x)
        x = self.attention(x)
        x = self.resblockup2(x)
        x = self.resblockup3(x)
        x = ( torch.tanh(x) / 2.) + 0.5
        return x


class RewardDecoder(nn.Module):
    """Decodes the current reward from the EnvStepper latent state.

    At every timestep, the RewardDecoder takes the EnvStepper latent state
    as input and outputs the (predicted) reward for that timestep. The agent
    doesn't see this reward directly. It is just used to train the VAE
    in the hope that reward-salient aspects of the environment will be
    represented in the EnvStepper latent state.

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, device, env_h_ch=128, env_h_hw=8):
        super(RewardDecoder, self).__init__()
        ch0 = env_h_ch
        hw0 = env_h_hw
        self.resblockdown = lyr.ResBlockDown(ch0,ch0//4,
                               downsample=nn.AvgPool2d(kernel_size=2))
        self.fc = nn.Linear(4*4*(ch0//4), 1)

    def forward(self, x):
        x = self.resblockdown(x)
        x = self.fc(x.view(x.shape[0], -1))
        return x

class DoneDecoder(nn.Module):
    """Decodes the 'Done' status from the EnvStepper latent state.

    At every timestep, the DoneDecoder takes the EnvStepper latent state
    as input and outputs the (predicted) 'Done' status for that timestep. This
    indicates when the similator thinks the episode is over (e.g. if the agent
    dies or completes the level).

    It is identical to the RewardDecoder apart from the final sigmoid
    activation, since we want to return a prediction for a boolean here.

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, device, env_h_ch=128, env_h_hw=8):
        super(DoneDecoder, self).__init__()
        ch0 = env_h_ch
        hw0 = env_h_hw
        self.resblockdown = lyr.ResBlockDown(ch0,ch0//4,
                               downsample=nn.AvgPool2d(kernel_size=2))
        self.fc = nn.Linear(4*4*(ch0//4), 1)

    def forward(self, x):
        x = self.resblockdown(x)
        x = self.fc(x.view(x.shape[0], -1))
        return torch.sigmoid(x)


