import torch
import torch.nn as nn
from policies import ResidualBlock

"""The generative model consists of 
    - An encoder network
        - image -> residual (128)
        - init hidden state added to the channel dim (256)
        - both into 1x1 conv (-> 128)
        - residual block 256
        - layer norm
        - residual shrink block 256
        - layer norm
        - residual shrink block (or large kerns to ensure mixing) 256
        - layer norm
        - split
            - fc to mu (should have around 256 neurons)
            - fc to sigma
    - A decoder network
        - inithidden&andActionNetwork
            - residual block
            - layer norm
            - split
                - a)  
                    - residual shrink block
                    - FC to init hidden state
                - b)
                    - residual shrink block
                    - FC to prev action
        - standardisernet
            - fc to large block (4x4x256)
            - layernorm
            - deconv growth residual block 
            - layer norm
            - deconv growth residual block 
            - layer norm
            - conv
            - layer norm
            now at standard block, which should be roughly 8x8x128
        - UnrollerNet
            - AssimilatorResidualBlock
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
            
            

"""

"""
Classes we'll need:
DeconvResidualBlock
Deconv Growth Residual Block (a growth deconv in front of DeconvResidual)
- AssimilatorResidualBlock
"""


class AssimilatorResidualBlock(nn.Module):

    def __init__(self, channels,
                 actv=torch.relu,
                 kernel_sizes=[3, 3],
                 paddings=[1, 1],
                 action_dim=16):
        super(ResidualBlock, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=channels+action_dim,
                               out_channels=channels,
                               kernel_size=kernel_sizes[0],
                               padding=paddings[0])
        self.conv1 = nn.Conv2d(in_channels=channels+action_dim,
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