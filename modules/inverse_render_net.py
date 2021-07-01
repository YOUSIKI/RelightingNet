import torch
import torch.nn as nn
import torch.nn.functional as F
from .lighting_model import *
from .operations import *
from .residual import *

__all__ = [
    'InverseRenderNet',
]


class InverseRenderNet(nn.Module):
    """
    https://github.com/YeeU/relightingNet/blob/1d6d18542d02c4da28fe464e630147c990339f80/modules/inverseRenderNet.py#L7
    """
    def __init__(self, options, name='inverserendernet'):
        super().__init__()
        self.name = name
        self.options = options
        # feature extractor
        self.e1_c = Conv2d(3,
                           self.options.gf_dim,
                           kernel_size=7,
                           stride=1,
                           padding='VALID',
                           name=self.name + '/g_e1_c')
        self.e2_c = Conv2d(self.options.gf_dim,
                           self.options.gf_dim * 2,
                           kernel_size=3,
                           stride=2,
                           name=self.name + '/g_e2_c')
        self.e3_c = Conv2d(self.options.gf_dim * 2,
                           self.options.gf_dim * 4,
                           kernel_size=3,
                           stride=2,
                           name=self.name + '/g_e3_c')
        self.e1_bn = InstanceNorm2d(self.options.gf_dim,
                                    name=self.name + '/g_e1_bn')
        self.e2_bn = InstanceNorm2d(self.options.gf_dim * 2,
                                    name=self.name + '/g_e2_bn')
        self.e3_bn = InstanceNorm2d(self.options.gf_dim * 4,
                                    name=self.name + '/g_e3_bn')
        # residual blocks
        self.r1 = ResidualBlock(self.options.gf_dim * 4,
                                self.options.gf_dim * 4,
                                name=self.name + '/g_r1')
        self.r2 = ResidualBlock(self.options.gf_dim * 4,
                                self.options.gf_dim * 4,
                                name=self.name + '/g_r2')
        self.r3 = ResidualBlock(self.options.gf_dim * 4,
                                self.options.gf_dim * 4,
                                name=self.name + '/g_r3')
        self.r4 = ResidualBlock(self.options.gf_dim * 4,
                                self.options.gf_dim * 4,
                                name=self.name + '/g_r4')
        self.r5 = ResidualBlock(self.options.gf_dim * 4,
                                self.options.gf_dim * 4,
                                name=self.name + '/g_r5')
        self.r6 = ResidualBlock(self.options.gf_dim * 4,
                                self.options.gf_dim * 4,
                                name=self.name + '/g_r6')
        self.r7 = ResidualBlock(self.options.gf_dim * 4,
                                self.options.gf_dim * 4,
                                name=self.name + '/g_r7')
        self.r8 = ResidualBlock(self.options.gf_dim * 4,
                                self.options.gf_dim * 4,
                                name=self.name + '/g_r8')
        self.r9 = ResidualBlock(self.options.gf_dim * 4,
                                self.options.gf_dim * 4,
                                name=self.name + '/g_r9')
        # albedo
        self.am1_dc = Conv2d(self.options.gf_dim * 4,
                             self.options.gf_dim * 2,
                             kernel_size=3,
                             stride=1,
                             name=self.name + '/g_am1_dc')
        self.am1_bn = InstanceNorm2d(self.options.gf_dim * 2,
                                     name=self.name + '/g_am1_bn')
        self.am2_dc = Conv2d(self.options.gf_dim * 2,
                             self.options.gf_dim,
                             kernel_size=3,
                             stride=1,
                             name=self.name + '/g_am2_dc')
        self.am2_bn = InstanceNorm2d(self.options.gf_dim,
                                     name=self.name + '/g_am2_bn')
        self.am_out_c = Conv2d(self.options.gf_dim,
                               self.options.am_out_c_dim,
                               kernel_size=7,
                               stride=1,
                               padding='VALID',
                               name=self.name + '/g_am_out_c')
        # normal
        self.nm1_dc = Conv2d(self.options.gf_dim * 4,
                             self.options.gf_dim * 2,
                             kernel_size=3,
                             stride=1,
                             name=self.name + '/g_nm1_dc')
        self.nm1_bn = InstanceNorm2d(self.options.gf_dim * 2,
                                     name=self.name + '/g_nm1_bn')
        self.nm2_dc = Conv2d(self.options.gf_dim * 2,
                             self.options.gf_dim,
                             kernel_size=3,
                             stride=1,
                             name=self.name + '/g_nm2_dc')
        self.nm2_bn = InstanceNorm2d(self.options.gf_dim,
                                     name=self.name + '/g_nm2_bn')
        self.nm_out_c = Conv2d(self.options.gf_dim,
                               self.options.nm_out_c_dim,
                               kernel_size=7,
                               stride=1,
                               padding='VALID',
                               name=self.name + '/g_nm_out_c')
        # mask
        self.mask1_dc = Conv2d(self.options.gf_dim * 4,
                               self.options.gf_dim * 2,
                               kernel_size=3,
                               stride=1,
                               name=self.name + '/g_mask1_dc')
        self.mask1_bn = InstanceNorm2d(self.options.gf_dim * 2,
                                       name=self.name + '/g_mask1_bn')
        self.mask2_dc = Conv2d(self.options.gf_dim * 2,
                               self.options.gf_dim,
                               kernel_size=3,
                               stride=1,
                               name=self.name + '/g_mask2_dc')
        self.mask2_bn = InstanceNorm2d(self.options.gf_dim,
                                       name=self.name + '/g_mask2_bn')
        self.mask_out_c = Conv2d(self.options.gf_dim,
                                 self.options.mask_out_c_dim,
                                 kernel_size=7,
                                 stride=1,
                                 padding='VALID',
                                 name=self.name + '/g_mask_out_c')
        # lighting model
        self.lighting_model = None

    def forward(self, images):
        # feature extractor
        c0 = pad2d(images, 3, mode='reflect')
        c1 = F.relu(self.e1_bn(self.e1_c(c0)))
        c2 = F.relu(self.e2_bn(self.e2_c(c1)))
        c3 = F.relu(self.e3_bn(self.e3_c(c2)))
        # residual blocks
        r1 = self.r1(c3)
        r2 = self.r2(r1)
        r3 = self.r3(r2)
        r4 = self.r4(r3)
        r5 = self.r5(r4)
        r6 = self.r6(r5)
        r7 = self.r7(r6)
        r8 = self.r8(r7)
        r9 = self.r9(r8)
        # albedo
        am1 = resize_bilinear(r9, c2.shape[2:4])
        am1 = self.am1_dc(am1)
        am1 = self.am1_bn(am1)
        am1 = F.relu(am1)
        am2 = resize_bilinear(am1, c1.shape[2:4])
        am2 = self.am2_dc(am2)
        am2 = self.am2_bn(am2)
        am2 = F.relu(am2)
        am2 = pad2d(am2, 3, mode='reflect')
        am_out = self.am_out_c(am2)
        # normal
        nm1 = resize_bilinear(r9, c2.shape[2:4])
        nm1 = self.nm1_dc(nm1)
        nm1 = self.nm1_bn(nm1)
        nm1 = F.relu(nm1)
        nm2 = resize_bilinear(nm1, c1.shape[2:4])
        nm2 = self.nm2_dc(nm2)
        nm2 = self.nm2_bn(nm2)
        nm2 = F.relu(nm2)
        nm2 = pad2d(nm2, 3, mode='reflect')
        nm_out = self.nm_out_c(nm2)
        # mask
        mask1 = resize_bilinear(r9, c2.shape[2:4])
        mask1 = self.mask1_dc(mask1)
        mask1 = self.mask1_bn(mask1)
        mask1 = F.relu(mask1)
        mask2 = resize_bilinear(mask1, c1.shape[2:4])
        mask2 = self.mask2_dc(mask2)
        mask2 = self.mask2_bn(mask2)
        mask2 = F.relu(mask2)
        mask2 = pad2d(mask2, 3, mode='reflect')
        mask_out = self.mask_out_c(mask2)

        return am_out, nm_out, mask_out

    def postprocess(self, images, masks, am_out, nm_out, mask_out):
        if masks is None:
            masks = torch.ones(
                (images.size(0), 1, images.size(2), images.size(3)))
        albedo = masks * torch.sigmoid(am_out)
        shadow = masks * torch.sigmoid(mask_out)
        nm_pred = nm_out
        nm_pred_sum = torch.sum(nm_pred**2, dim=1, keepdims=True)
        nm_pred_norm = torch.sqrt(1 + nm_pred_sum)
        nm_pred_xy = nm_pred / nm_pred_norm
        nm_pred_z = torch.ones_like(nm_pred_norm) / nm_pred_norm
        nm_pred = torch.cat([nm_pred_xy, nm_pred_z], axis=1)
        normal = masks * nm_pred
        if self.lighting_model is None and self.options.lighting_model is not None:
            self.lighting_model = LightingModel(self.options.lighting_model)
        if self.lighting_model is not None:
            lighting = self.lighting_model(
                images,
                masks,
                albedo,
                normal,
                shadow,
            )
        else:
            lighting = None
        return albedo, normal, shadow, lighting
