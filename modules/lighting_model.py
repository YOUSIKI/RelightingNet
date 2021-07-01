import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

__all__ = [
    'LightingModel',
]


class LightingModel(object):
    def __init__(self, root):
        self.root = root
        self.gamma = 2.2
        self.vectors = np.load(os.path.join(self.root, 'vectors.npy'))
        self.means = np.load(os.path.join(self.root, 'means.npy'))
        self.vectors = torch.FloatTensor(self.vectors)
        self.means = torch.FloatTensor(self.means)

    def __call__(self, images, masks, albedo, normal, shadow):
        lightings = self.decomposition(
            images,
            masks,
            albedo,
            normal,
            shadow,
            self.gamma,
        )
        lightings_pca = torch.matmul(lightings - self.means,
                                     LightingModel.pinv(self.vectors))
        lightings = torch.matmul(lightings_pca, self.vectors) + self.means
        lightings = lightings.view((-1, 9, 3))
        return lightings

    @classmethod
    def decomposition(cls, images, masks, albedo, normal, shadow, gamma):
        """
        https://github.com/YeeU/relightingNet/blob/1d6d18542d02c4da28fe464e630147c990339f80/modules/pred_illuDecomp_layer.py#L7
        """
        # compute shading by dividing input by albedo and shadow
        shading = torch.pow(images, gamma) / (albedo * shadow)
        # perform clamping on resulted shading to guarantee its numerical range
        shading = (torch.clamp(shading, 0, 1) + 1e-4) * masks
        # compute shading by linear equation regarding nm and L_SHcoeffs
        c1 = 0.429043
        c2 = 0.511664
        c3 = 0.743125
        c4 = 0.886227
        c5 = 0.247708

        outputs = list()

        for i in range(images.size(0)):  # batch_size
            si, ni = shading[i], normal[i]
            si = si.view((3, -1)).permute(1, 0)
            ni = ni.view((3, -1)).permute(1, 0)
            n0, n1, n2 = ni[:, 0], ni[:, 1], ni[:, 2]
            ones = torch.ones(ni.size(0), dtype=ni.dtype)
            a = [
                c4 * ones,
                2 * c2 * n1,
                2 * c2 * n2,
                2 * c2 * n0,
                2 * c1 * n0 * n1,
                2 * c1 * n1 * n2,
                c3 * n2 * n2 - c5,
                2 * c1 * n2 * n0,
                c1 * (n0**2 - n1**2),
            ]
            a = torch.stack(a, dim=-1)
            b = cls.pinv(a)
            c = torch.matmul(b, si)
            outputs.append(c)

        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.view((-1, 27))
        return outputs

    @classmethod
    def pinv(cls, a, reltol=1e-6):
        s, u, v = cls.svd(a)
        atol = torch.max(s) * reltol
        s = torch.where(s > atol, s, atol * torch.ones_like(s))
        s_inv = torch.diag(1.0 / s)
        return torch.matmul(v, torch.matmul(s_inv, u.transpose(0, 1)))

    @staticmethod
    def svd(a):
        u, s, v = torch.linalg.svd(a, full_matrices=False)
        v = torch.transpose(v, 0, 1)
        return s, u, v
