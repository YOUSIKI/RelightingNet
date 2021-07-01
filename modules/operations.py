import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'pad2d',
    'resize_bilinear',
    'max_pool2d',
    'InstanceNorm2d',
    'BatchNorm2d',
    'Conv2d',
]


def pad2d(x, pad, mode='zeros'):
    if mode == 'reflect':
        return nn.ReflectionPad2d(pad)(x)
    elif mode == 'zeros':
        return nn.ConstantPad2d(pad, 0)(x)
    else:
        raise NotImplementedError


def resize_bilinear(x, shape):
    return F.interpolate(x, shape, mode='bilinear', align_corners=False)


def same_padding(x, kernel_size, stride):
    B, C, N, M = x.size()
    outN = math.ceil(N / stride)
    outM = math.ceil(M / stride)
    padN = (outN - 1) * stride + kernel_size - N
    padM = (outM - 1) * stride + kernel_size - M
    padL = math.floor(padN / 2)
    padR = math.ceil(padN / 2)
    padT = math.floor(padM / 2)
    padB = math.ceil(padM / 2)
    x = pad2d(x, (padL, padR, padT, padB), mode='zeros')
    return x


def max_pool2d(x, kernel_size, stride, padding='VALID'):
    if padding == 'SAME':
        x = same_padding(x, kernel_size, stride)
    x = F.max_pool2d(x, kernel_size, stride)
    return x


class InstanceNorm2d(nn.InstanceNorm2d):
    """
    https://github.com/YeeU/relightingNet/blob/1d6d18542d02c4da28fe464e630147c990339f80/modules/ops.py#L13
    """
    def __init__(self, num_features, name='instance_norm'):
        super().__init__(num_features,
                         eps=1e-5,
                         affine=True,
                         track_running_stats=False)
        self.name = name

    def load_checkpoint(self, ckpt):
        scale = ckpt.get(self.name + '/scale')
        offset = ckpt.get(self.name + '/offset')
        assert scale is not None and offset is not None
        scale = torch.tensor(scale)
        offset = torch.tensor(offset)
        self.weight.data = scale.data
        self.bias.data = offset.data


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, name='batch_norm'):
        super().__init__(num_features,
                         eps=1e-5,
                         momentum=0.1,
                         affine=True,
                         track_running_stats=True)
        self.name = name

    def load_checkpoint(self, ckpt):
        beta = ckpt.get(self.name + '/beta')
        gamma = ckpt.get(self.name + '/gamma')
        moving_mean = ckpt.get(self.name + '/moving_mean')
        moving_variance = ckpt.get(self.name + '/moving_variance')
        assert beta and gamma and moving_mean and moving_variance
        beta = torch.tenosr(beta)
        gamma = torch.tensor(gamma)
        moving_mean = torch.tensor(moving_mean)
        moving_variance = torch.tensor(moving_variance)
        self.weight.data = gamma.data
        self.bias.data = beta.data
        self.running_mean.data = moving_mean.data
        self.running_var.data = moving_variance.data


class Conv2d(nn.Module):
    """
    https://github.com/YeeU/relightingNet/blob/1d6d18542d02c4da28fe464e630147c990339f80/modules/ops.py#L24
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding='SAME',
                 name=None):
        super().__init__()
        assert name is not None
        self.name = name + '/Conv'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = torch.nn.Parameter(
            torch.empty(
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            ))
        self.bias = None
        nn.init.xavier_normal_(self.weights)

    def forward(self, x):
        if self.padding == 'SAME':
            x = same_padding(x, self.kernel_size, self.stride)
        x = F.conv2d(x, self.weights, self.bias, self.stride)
        return x

    def load_checkpoint(self, ckpt):
        weights = ckpt.get(self.name + '/weights')
        assert weights is not None
        weights = torch.tensor(weights)
        weights = weights.permute(3, 2, 0, 1)
        self.weights.data = weights.data
