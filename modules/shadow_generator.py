import torch
import torch.nn as nn
import torch.nn.functional as F
from .operations import *

__all__ = [
    'InverseRenderNet',
]


class ShadowGenerator(nn.Module):
    """
    https://github.com/YeeU/relightingNet/blob/1d6d18542d02c4da28fe464e630147c990339f80/modules/sdNet.py#L7
    """
    def __init__(self, options, name='sd_generator'):
        super().__init__()
        self.name = name
        self.options = options
        # encoder
        self.e1_conv = Conv2d(30,
                              self.options.gf_dim * 8,
                              kernel_size=3,
                              stride=1,
                              name=self.name + '/g_e1_conv')
        self.bn_e1 = BatchNorm2d(self.options.gf_dim * 8,
                                 name=self.name + '/g_bn_e1')
        self.e2_conv = Conv2d(self.options.gf_dim * 8,
                              self.options.gf_dim * 16,
                              kernel_size=3,
                              stride=1,
                              name=self.name + '/g_e2_conv')
        self.bn_e2 = BatchNorm2d(self.options.gf_dim * 16,
                                 name=self.name + '/g_bn_e2')
        self.e3_conv = Conv2d(self.options.gf_dim * 16,
                              self.options.gf_dim * 32,
                              kernel_size=3,
                              stride=1,
                              name=self.name + '/g_e3_conv')
        self.bn_e3 = BatchNorm2d(self.options.gf_dim * 32,
                                 name=self.name + '/g_bn_e3')
        self.e4_conv = Conv2d(self.options.gf_dim * 32,
                              self.options.gf_dim * 64,
                              kernel_size=3,
                              stride=1,
                              name=self.name + '/g_e4_conv')
        self.bn_e4 = BatchNorm2d(self.options.gf_dim * 64,
                                 name=self.name + '/g_bn_e4')
        self.e5_conv = Conv2d(self.options.gf_dim * 64,
                              self.options.gf_dim * 64,
                              kernel_size=3,
                              stride=1,
                              name=self.name + '/g_e5_conv')
        self.bn_e5 = BatchNorm2d(self.options.gf_dim * 64,
                                 name=self.name + '/g_bn_e5')
        self.e6_conv = Conv2d(self.options.gf_dim * 64,
                              self.options.gf_dim * 64,
                              kernel_size=3,
                              stride=1,
                              name=self.name + '/g_e6_conv')
        self.bn_e6 = BatchNorm2d(self.options.gf_dim * 64,
                                 name=self.name + '/g_bn_e6')
        self.e7_conv = Conv2d(self.options.gf_dim * 64,
                              self.options.gf_dim * 64,
                              kernel_size=3,
                              stride=1,
                              name=self.name + '/g_e7_conv')
        self.bn_e7 = BatchNorm2d(self.options.gf_dim * 64,
                                 name=self.name + '/g_bn_e7')
        self.e8_conv = Conv2d(self.options.gf_dim * 64,
                              self.options.gf_dim * 64,
                              kernel_size=3,
                              stride=1,
                              name=self.name + '/g_e8_conv')
        # decoder
        self.bn_d1 = BatchNorm2d(self.options.gf_dim * 64,
                                 name=self.name + '/g_' + self.name + '_bn_d2')

    def forward(self, x):
        e1 = F.relu(self.e1_conv(x))
        e2 = max_pool2d(self.bn_e1(e1),
                        kernel_size=2,
                        stride=2,
                        padding='SAME')
        e2 = F.relu(self.e2_conv(e2))
        e3 = max_pool2d(self.bn_e2(e2),
                        kernel_size=2,
                        stride=2,
                        padding='SAME')
        e3 = F.relu(self.e3_conv(e3))
        e4 = max_pool2d(self.bn_e3(e3),
                        kernel_size=2,
                        stride=2,
                        padding='SAME')
        e4 = F.relu(self.e4_conv(e4))
        e5 = max_pool2d(self.bn_e4(e4),
                        kernel_size=2,
                        stride=2,
                        padding='SAME')
        e5 = F.relu(self.e5_conv(e5))
        e6 = max_pool2d(self.bn_e5(e5),
                        kernel_size=2,
                        stride=2,
                        padding='SAME')
        e6 = F.relu(self.e6_conv(e6))
        e7 = max_pool2d(self.bn_e6(e6),
                        kernel_size=2,
                        stride=2,
                        padding='SAME')
        e7 = F.relu(self.e7_conv(e7))
        e8 = max_pool2d(self.bn_e7(e7),
                        kernel_size=2,
                        stride=2,
                        padding='SAME')
        e8 = F.relu(self.e8_conv(e8))
        d1 = resize_bilinear(e8, e7.shape[2:4])
        d1 = self.bn_d1(d1)
        d1 = self.d1_dc(d1)
        d1 = torch.cat([F.relu(d1), e7], dim=1)
        d2 = resize_bilinear(d1, e6.shape[2:4])
        d2 = self.bn_d2(d2)
        d2 = self.d2_dc(d2)
        d2 = torch.cat([F.relu(d2), e6], dim=1)


def init_sd(nm_irn, lightings):
    lightings = lightings.view((-1, 27, 1, 1))
    lightings = torch.tile(lightings, (1, 1, nm_irn.size(1), nm_irn.size(2)))
    return torch.cat([nm_irn, lightings], axis=1)