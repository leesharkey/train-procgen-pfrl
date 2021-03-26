import torch
import torch.nn as nn
import pfrl
from pfrl import experiments, utils
from pfrl.agents import PPO
from pfrl.policies import SoftmaxCategoricalHead

class ResidualBlock(nn.Module):

    def __init__(self, channels,
                 actv=torch.relu,
                 kernel_sizes=[3, 3],
                 paddings=[1, 1]):
        super(ResidualBlock, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_sizes[0],
                               padding=paddings[0])
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_sizes[1],
                               padding=paddings[0])
        self.actv = actv

    def forward(self, x):
        inputs = x
        x = self.actv(x)
        x = self.conv0(x)
        x = self.actv(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):

    def __init__(self, input_shape, out_channels):
        super(ConvSequence, self).__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0],
                              out_channels=self._out_channels,
                              kernel_size=3,
                              padding=1)
        self.max_pool2d = nn.MaxPool2d(kernel_size=3,
                                       stride=2,
                                       padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool2d(x)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return self._out_channels, (h + 1) // 2, (w + 1) // 2


class ImpalaCNN(nn.Module):
    """Network from IMPALA paper, to work with pfrl."""

    def __init__(self, obs_space, num_outputs):

        super(ImpalaCNN, self).__init__()

        h, w, c = obs_space.shape
        shape = (c, h, w)

        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2],
                                   out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)
        # Initialize weights of logits_fc
        nn.init.orthogonal_(self.logits_fc.weight, gain=0.01)
        nn.init.zeros_(self.logits_fc.bias)

    def forward(self, obs):
        assert obs.ndim == 4
        x = obs / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(x)
        x = self.hidden_fc(x)
        x = torch.relu(x)
        logits = self.logits_fc(x)
        dist = torch.distributions.Categorical(logits=logits)
        value = self.value_fc(x)
        return dist, value

    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path, device=None):
        self.load_state_dict(torch.load(model_path, map_location=device))


class CNNRecurrent(nn.Module):
    """."""

    def __init__(self, obs_space, act_space):

        super(CNNRecurrent, self).__init__()
        print("Observation space", obs_space)
        print("Action space", act_space)
        self.n_actions = act_space.n

        def lecun_init(layer, gain=1):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                pfrl.initializers.init_lecun_normal(layer.weight, gain)
                nn.init.zeros_(layer.bias)
            else:
                pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
                pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
                nn.init.zeros_(layer.bias_ih_l0)
                nn.init.zeros_(layer.bias_hh_l0)
            return layer

        self.feature_extractor = FeatureExtractorCNN(obs_space)

        # The pfrl.nn.RecurrentSequential class below has a FF part and an RNN
        # part. The class helps with passing observations to the FF part and
        # dealing with recursive hidden states.
        self.model = pfrl.nn.RecurrentSequential(
            self.feature_extractor,
            lecun_init(
                nn.GRU(num_layers=1, input_size=2048, hidden_size=256)),
            pfrl.nn.Branched(
                nn.Sequential(
                    lecun_init(nn.Linear(256, self.n_actions), 1e-2),
                    SoftmaxCategoricalHead(),
                ),
                lecun_init(nn.Linear(256, 1)),
            ),
        )

    def forward(self, obs, recurrent_state):
        return self.model(obs, recurrent_state)

    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path, device=None):
        self.load_state_dict(torch.load(model_path, map_location=device))


class FeatureExtractorCNN(nn.Module):
    """."""

    def __init__(self, obs_space):

        super(FeatureExtractorCNN, self).__init__()
        # obs_n_channels = obs_space.low.shape[2]
        h, w, c = obs_space.low.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)

    def forward(self, obs):
        x = (obs / 255.0)  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(x)
        return x

    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path, device=None):
        self.load_state_dict(torch.load(model_path, map_location=device))