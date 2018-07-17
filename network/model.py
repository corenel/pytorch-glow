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
                 lu_decomposition=False):
        """
        One step of flow described in paper

                      ▲
                      │
        ┌─────────────┼─────────────┐
        │  ┌──────────┴──────────┐  │
        │  │ flow coupling layer │  │
        │  └──────────▲──────────┘  │
        │             │             │
        │  ┌──────────┴──────────┐  │
        │  │  flow permutation   │  │
        │  │        layer        │  │
        │  └──────────▲──────────┘  │
        │             │             │
        │  ┌──────────┴──────────┐  │
        │  │     activation      │  │
        │  │ normalization layer │  │
        │  └──────────▲──────────┘  │
        └─────────────┼─────────────┘
                      │
                      │

        :param in_channels: number of input channels
        :type in_channels: int
        :param hidden_channels: number of hidden channels
        :type hidden_channels: int
        :param permutation: type of flow permutation
        :type permutation: str
        :param coupling: type of flow coupling
        :type coupling: str
        :param actnorm_scale: scale factor of actnorm layer
        :type actnorm_scale: float
        :param lu_decomposition: whether to use LU decomposition or not
        :type lu_decomposition: bool
        """
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
                                                    lu_decomposition=lu_decomposition)
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
        """
        Normal flow

        :param x: input tensor
        :type x: torch.Tensor
        :param logdet: log determinant
        :type logdet: torch.Tensor
        :return: output and logdet
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
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
        """
        Reverse flow

        :param x: input tensor
        :type x: torch.Tensor
        :param logdet: log determinant
        :type logdet: torch.Tensor
        :return: output and logdet
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
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
        """
        Forward oen step of flow

        :param x: input tensor
        :type x: torch.Tensor
        :param logdet: log determinant
        :type logdet: torch.Tensor
        :param reverse: whether to reverse flow
        :type reverse: bool
        :return: output and logdet
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        assert x.shape[1] % 2 == 0
        if not reverse:
            return self.normal_flow(x, logdet)
        else:
            return self.reverse_flow(x, logdet)


class FlowModel(nn.Module):
    def __init__(self,
                 in_shape,
                 hidden_channels,
                 K, L,
                 permutation='invconv',
                 coupling='additive',
                 actnorm_scale=1.,
                 lu_decomposition=False):
        """
        Flow model with multi-scale architecture

                         ┏━━━┓
                         ┃z_L┃
                         ┗━▲━┛
                           │
                ┌──────────┴──────────┐
                │    step of flow     │* K
                └──────────▲──────────┘
                ┌──────────┴──────────┐
                │       squeeze       │
                └──────────▲──────────┘
                           ├──────────────┐
        ┏━━━┓   ┌──────────┴──────────┐   │
        ┃z_i┃◀──┤        split        │   │
        ┗━━━┛   └──────────▲──────────┘   │
                ┌──────────┴──────────┐   │
                │    step of flow     │* K│ * (L-1)
                └──────────▲──────────┘   │
                ┌──────────┴──────────┐   │
                │       squeeze       │   │
                └──────────▲──────────┘   │
                           │◀─────────────┘
                         ┏━┻━┓
                         ┃ x ┃
                         ┗━━━┛

        :param in_shape: shape of input image in (H, W, C)
        :type in_shape: torch.Size or tuple(int) or list(int)
        :param hidden_channels: number of hidden channels
        :type hidden_channels: int
        :param K: depth of flow
        :type K: int
        :param L: number of levels
        :type L: int
        :param permutation: type of flow permutation
        :type permutation: str
        :param coupling: type of flow coupling
        :type coupling: str
        :param actnorm_scale: scale factor of actnorm layer
        :type actnorm_scale: float
        :param lu_decomposition: whether to use LU decomposition or not
        :type lu_decomposition: bool
        """
        super().__init__()
        self.K = K
        self.L = L

        # image shape
        assert len(in_shape) == 3
        assert in_shape[2] == 1 or in_shape[2] == 3
        nh, nw, nc = in_shape

        # initialize layers
        self.layers = nn.ModuleList()
        self.output_shapes = []
        for i in range(L):
            # squeeze
            self.layers.append(module.Squeeze2d(factor=2))
            nc, nh, nw = nc * 4, nh // 2, nw // 2
            self.output_shapes.append([-1, nc, nh, nw])
            # flow step * K
            for _ in range(K):
                self.layers.append(FlowStep(
                    in_channels=nc,
                    hidden_channels=hidden_channels,
                    permutation=permutation,
                    coupling=coupling,
                    actnorm_scale=actnorm_scale,
                    lu_decomposition=lu_decomposition))
                self.output_shapes.append([-1, nc, nh, nw])
            # split
            if i < L - 1:
                self.layers.append(module.Split2d(num_channels=nc))
                nc = nc // 2
                self.output_shapes.append([-1, nc, nh, nw])

    def encode(self, z, logdet=0.):
        """
        Encode input

        :param z: input tensor
        :type z: torch.Tensor
        :param logdet: log determinant
        :type logdet: torch.Tensor
        :return: encoded tensor
        :rtype: torch.Tensor
        """
        for layer in self.layers:
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, eps_std=None):
        """
        Decode input

        :param z: input tensor
        :type z: torch.Tensor
        :param eps_std: standard deviation of eps
        :type eps_std: float
        :return: decoded tensor
        :rtype: torch.Tensor
        """
        for layer in reversed(self.layers):
            if isinstance(layer, module.Split2d):
                z, logdet = layer(z, logdet=0., reverse=True, eps_std=eps_std)
            else:
                z, logdet = layer(z, logdet=0., reverse=True)
        return z, logdet

    def forward(self, z, logdet=0., reverse=False, eps_std=None):
        """
        Forward flow model

        :param z: input tensor
        :type z: torch.Tensor
        :param logdet: log determinant
        :type logdet: torch.Tensor
        :param reverse: whether to reverse flow
        :type reverse: bool
        :param eps_std: standard deviation of eps
        :type eps_std: float
        :return: output tensor
        :rtype: torch.Tensor
        """
        if not reverse:
            return self.encode(z, logdet)
        else:
            return self.decode(z, eps_std)


class Glow(nn.Module):
    def __init__(self, hps):
        """
        Glow network

        :param hps: hyper-parameters for this network
        """
        super().__init__()

        self.flow = FlowModel(
            in_shape=hps.model.image_size,
            hidden_channels=hps.model.hidden_channels,
            K=hps.model.K,
            L=hps.model.L,
            permutation=hps.ablation.flow_permutation,
            coupling=hps.ablation.flow_coupling,
            actnorm_scale=hps.model.actnorm_scale,
            lu_decomposition=hps.ablation.lu_decomposition)

    def forward(self, x):
        pass
