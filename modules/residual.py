import torch
import torch.nn as nn
import torch.nn.functional as F
from .operations import *

__all__ = [
    'ResidualBlock',
]


class ResidualBlock(nn.Module):
    """
    https://github.com/YeeU/relightingNet/blob/1d6d18542d02c4da28fe464e630147c990339f80/modules/inverseRenderNet.py#L10
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 name='res'):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = int((self.kernel_size - 1) / 2)
        self.c1 = Conv2d(self.in_channels,
                         self.out_channels,
                         kernel_size=self.kernel_size,
                         stride=self.stride,
                         padding='VALID',
                         name=self.name + '_c1')
        self.c2 = Conv2d(self.out_channels,
                         self.out_channels,
                         kernel_size=self.kernel_size,
                         stride=self.stride,
                         padding='VALID',
                         name=self.name + '_c2')
        self.bn1 = InstanceNorm2d(self.out_channels, name=self.name + '_bn1')
        self.bn2 = InstanceNorm2d(self.out_channels, name=self.name + '_bn2')

    def forward(self, x):
        y = pad2d(x, self.padding, mode='reflect')
        y = F.relu(self.bn1(self.c1(y)))
        y = pad2d(y, self.padding, mode='reflect')
        y = self.bn2(self.c2(y))
        return y + x
