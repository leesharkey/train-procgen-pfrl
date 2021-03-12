import torch
import torch.nn as nn
from policies import ResidualBlock
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

from sync_batchnorm import SynchronizedBatchNorm2d as SyncBN2d

"""Proposal for generative model:

    - An encoder network
        - image -> residual (64)
        - init hidden state added to the channel dim (64+128)
        - all (image, init hidden, and resid output) into 1x1 conv (3+64+128 -> 128)
        - layer norm
        - Resblock down
        - non-local layer
        - layer norm
        - Resblock down 256 (with dense inputs from above)
        - layer norm
        - Resblock down (256)
        - layer norm
        - split
            - fc to mu (should have around 256 neurons)
            - fc to sigma
    - A decoder network
        - Initializer networks
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
            - ConvGRU initializer
                - This takes a guess at initializing the GRU so 
                  it doesn't start with 0-tensors. Informative initialization 
                  like this should work better.            
            
        - StandardizerNet
            - fc to large block (4x4x256)
            - layernorm
            - deconv growth residual block 
            - layer norm
            - deconv growth residual block 
            - layer norm
            - conv
            - layer norm
            now at standard block, which should be roughly 8x8x128
        - UnrollerNet (needs nonlocal)
            - AssimilatorResidualBlock (takes standard block and also noise vector and GRU hidden state and outputs standard block) 
            - ConvGRU
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
- AssimilatorBlock
- Encoder
- Decoder
    - InitializerNets
    - Standardizer
    - UnrollerNet
    - reward decoder
    - obs decoder
"""

"""
Later, if that isn't producing good results, we can try an unroller that's
just a resblock. The reason i didn't go for this for the main attempt is that
I don't see why we wouldn't get vanishing or exploding gradients, and I 
couldn't see any literature that did this. We can try something like:
- UnrollerNet
    - AssimilatorResidualBlock (takes standard block and also noise vector and outputs standard block) 
    - layer norm
    - ResidualConv
    - layer norm
"""

