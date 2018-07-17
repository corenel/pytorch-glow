import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from network import module
from misc import ops


class FlowStep(nn.Module):
    flow_permutation = ['invconv', 'reverse', 'shuffle']
    flow_coupling = ['additive', 'affine']

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 permutation='invconv',
                 coupling='additive',
                 actnorm_scale=1.,
                 lu_decomposed=False):
        super().__init__()
        # permutation and coupling
        assert permutation in self.flow_permutation
        assert coupling in self.flow_coupling
        self.permutation = permutation
        self.coupling = coupling

        # activation normalization layer
        self.actnorm = module.ActNorm(num_channels=in_channels, scale=actnorm_scale)

        # flow permutation layer
        if permutation == 'invconv':
            self.invconv = module.Invertible1x1Conv(num_channels=in_channels,
                                                    lu_decomposed=lu_decomposed)
        elif permutation == 'reverse':
            self.reverse = module.Permutation2d(num_channels=in_channels, shuffle=False)
        else:
            self.shuffle = module.Permutation2d(num_channels=in_channels, shuffle=True)

        # flow coupling layer
        if coupling == 'additive':
            self.f = module.f(in_channels // 2, hidden_channels, in_channels // 2)
        else:
            self.f = module.f(in_channels // 2, hidden_channels, in_channels)

    def normal_flow(self, x, logdet=None):
        # activation normalization layer
        z, logdet = self.actnorm(x, logdet=logdet, reverse=False)

        # flow permutation layer
        if self.permutation == 'invconv':
            z, logdet = self.invconv(z, logdet, reverse=False)
        elif self.permutation == 'reverse':
            z = self.reverse(z, reverse=False)
        else:
            z = self.shuffle(z, reverse=False)

        # flow coupling layer
        nc = z.shape[1]
        z1 = z[:, :nc // 2, :, :]
        z2 = z[:, nc // 2:, :, :]
        if self.coupling == 'additive':
            z2 += self.f(z1)
        else:
            h = self.f(z1)
            shift = h[:, 0::2, :, :]
            scale = F.sigmoid(h[:, 1::2, :, :] + 2.)
            z2 += shift
            z2 *= scale
            logdet = ops.reduce_sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        return z, logdet

    def reverse_flow(self, x, logdet=None):
        # flow coupling layer
        nc = x.shape[1]
        z1 = x[:, :nc // 2, :, :]
        z2 = x[:, nc // 2:, :, :]
        if self.coupling == 'additive':
            z2 -= self.f(z1)
        else:
            h = self.f(z1)
            shift = h[:, 0::2, :, :]
            scale = F.sigmoid(h[:, 1::2, :, :] + 2.)
            z2 /= scale
            z2 -= shift
            logdet = -ops.reduce_sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        # flow permutation layer
        if self.permutation == 'invconv':
            z, logdet = self.invconv(z, logdet, reverse=True)
        elif self.permutation == 'reverse':
            z = self.reverse(z, reverse=True)
        else:
            z = self.shuffle(z, reverse=True)

        # activation normalization layer
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet

    def forward(self, x, logdet=None, reverse=False):
        assert x.shape[1] % 2 == 0
        if not reverse:
            return self.normal_flow(x, logdet)
        else:
            return self.reverse_flow(x, logdet)


class FlowModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class Glow(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
