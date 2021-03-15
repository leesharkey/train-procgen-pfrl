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
                - outputs something the size of the 'standard block' for env
                  unrolling         
            - UnrollerNet
                - AssimilatorResidualBlock (takes standard block and also noise vector and outputs standard block) 
                - layer norm
                - ResidualConv
                - layer norm
        - Side decoders (take a standard block and produce predictions for obs and rew
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
    def __init__(self, agent, device):
        super(VAE, self).__init__()
        self.encoder = EncoderNetwork(device).to(device)
        self.decoder = DecoderNetwork(device, agent, num_unroll_steps=10).to(device)
        self.device = device

    def forward(self, obs, agent_h0):
        mu, sigma = self.encoder(obs, agent_h0)
        sample = (torch.randn(mu.size()) + mu) * sigma
        return    self.decoder(sample)


class EncoderNetwork(nn.Module):

    def __init__(self, device):
        super(EncoderNetwork, self).__init__()
        self.input_network = EncoderInputNetwork(device, agent_hidden_size=256).to(device)
        self.rnn = EncoderRNN(device)
        self.embedder = EncoderEmbedder(device)

    def forward(self, obs, agent_h0):
        h = None
        obs_seq_len = obs.shape[1] # (B, *T*, Ch, H, W)
        inps = []
        for i in range(obs_seq_len):
            inps.append(self.input_network(obs[:,i,:], agent_h0)) #TODO why does this return 100 instead of 20 inps elements?
        inps = torch.stack(inps, dim=1) # stack along time dimension (batch is first)
        h = self.rnn(inps, h)
        mu, sigma = self.embedder(h)
        return mu, sigma




class EncoderInputNetwork(nn.Module):

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

    def __init__(self, device):
        super(EncoderRNN, self).__init__()
        self.rnn = lyr.ConvGRU(input_size=[8,8], # [H,W]
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
        sigma = self.fc_sigma(x.view(x.shape[0], -1))
        return mu, sigma

class DecoderNetwork(nn.Module):

    def __init__(self, device, agent, num_unroll_steps):
        super(DecoderNetwork, self).__init__()
        self.action_dim = agent.model.n_actions
        self.inithidden_network = TwoLayerPerceptron(
            insize=256,
            outsize=256)
        self.prev_act_network = TwoLayerPerceptron(
            insize=256,
            outsize=self.action_dim)
        self.env_init_network = EnvStepperInitializer(device=device)
        self.env_stepper = EnvStepper(agent)
        self.reward_decoder = RewardDecoder(device)
        self.obs_decoder = ObservationDecoder(device)
        self.agent = agent # TODO ensure the agent parameters are frozen during gen model training
        self.num_unroll_steps = num_unroll_steps

    def forward(self, sample):

        env_h = self.env_init_network(sample)
        prev_act = self.prev_act_network(sample)
        agent_h = self.inithidden_network(sample)

        pred_obs = []
        pred_rews = []
        pred_agent_hs = []
        pred_agent_logprobs = []

        # TODO: Stuff to fix:
        #  - log probs
        #  - what does the agent actually return?
        #  - storing obs, hs, etc in lists
        #  Then move on to loss functions.

        for i in range(self.num_unroll_steps):
            obs = self.obs_decoder(env_h)
            env_h = self.env_stepper(sample, prev_act, prev_h=env_h)
            agent_h, act = self.agent(agent_h, )

            # Add obs, agent_h, actlogprobs, and rew to lists


            # Get ready for new step
            prev_act = act

        return pred_obs, pred_rews, pred_agent_hs, pred_agent_logprobs


class TwoLayerPerceptron(nn.Module):

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

    def __init__(self, device, vae_latent_size=256, out_ch=128, out_hw=8):
        super(EnvStepperInitializer, self).__init__()

        self.out_ch = out_ch
        self.out_hw = out_hw
        self.fc = nn.Linear(vae_latent_size,
                            int((out_ch*out_hw*out_hw)/(2**2)))
        self.resblockup = lyr.ResBlockUp(in_channels=128,
                                         out_channels=128,
                                         upsample=nn.UpsamplingNearest2d)
        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(128)
        self.norm3 = nn.LayerNorm(128)
        self.norm4 = nn.LayerNorm(128)

        self.resblock1 = lyr.ResidualBlock(128)
        self.resblock2 = lyr.ResidualBlock(128)
        self.attention = lyr.Attention(128)


    def forward(self, x):
        x = self.fc(x)
        x = self.resblockup(x.view(-1,self.out_ch,self.out_hw,self.out_hw))
        x = self.norm1(x)
        x = self.resblock1(x)
        x = self.norm2(x)
        x = self.attention(x)
        x = self.norm3(x)
        x = self.resblock2(x)
        x = self.norm4(x)
        return x

class EnvStepper(nn.Module):

    def __init__(self, agent, vae_latent_size=256):
        super(EnvStepper, self).__init__()
        action_dim = agent.model.n_actions
        self.assimilator = \
            lyr.AssimilatorResidualBlock(128,
                                         vec_size=(action_dim+vae_latent_size))
        # standard block is 8x8x128
        self.attention = lyr.Attention(128)
        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(128)
        self.norm3 = nn.LayerNorm(128)
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

    def __init__(self, device, standard_actv_ch=128, standard_actv_hw=8):
        super(ObservationDecoder, self).__init__()
        ch0 = standard_actv_ch
        hw0 = standard_actv_hw
        self.resblockup1 = lyr.ResBlockUp(ch0,ch0//2)
        self.resblockup2 = lyr.ResBlockUp(ch0//2,ch0//4)
        self.resblockup3 = lyr.ResBlockUp(ch0//4,ch0//8)
        self.resblockup4 = lyr.ResBlockUp(ch0,3)
        self.attention = lyr.Attention(64)
        self.net = nn.Sequential(self.resblockup1,
                                 self.attention,
                                 self.resblockup2,
                                 self.resblockup3,
                                 self.resblockup4)
    def forward(self, x):
        return self.net(x)


class RewardDecoder(nn.Module):

    def __init__(self, device, standard_actv_ch=128, standard_actv_hw=8):
        super(RewardDecoder, self).__init__()
        ch0 = standard_actv_ch
        hw0 = standard_actv_hw
        self.resblockup1 = lyr.ResBlockDown(ch0,ch0//4)
        self.fc = nn.Linear(4*4*(ch0//4), 1)

    def forward(self, x):
        x = self.resblockup1(x)
        x = self.fc(x.view(x.shape[0], -1))
        return self.net(x)
