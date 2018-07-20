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
    bce_criterion = nn.BCEWithLogitsLoss()
    ce_criterion = nn.CrossEntropyLoss()

    def __init__(self, hps):
        """
        Glow network

        :param hps: hyper-parameters for this network
        """
        super().__init__()

        self.hps = hps
        self.flow = FlowModel(
            in_shape=hps.model.image_shape,
            hidden_channels=hps.model.hidden_channels,
            K=hps.model.K,
            L=hps.model.L,
            permutation=hps.ablation.flow_permutation,
            coupling=hps.ablation.flow_coupling,
            actnorm_scale=hps.model.actnorm_scale,
            lu_decomposition=hps.ablation.lu_decomposition)

        if hps.ablation.learn_top:
            nc = self.flow.output_shapes[-1][1]
            self.learn_top = module.Conv2dZeros(in_channels=2 * nc,
                                                out_channels=2 * nc)
        if hps.ablation.y_cond:
            nc = self.flow.output_shapes[-1][1]
            self.y_emb = module.LinearZeros(hps.dataset.n_classes, nc * 2)
            self.classifier = module.LinearZeros(nc, hps.dataset.n_classes)

        # TODO get number of deivce from config
        num_device = torch.cuda.device_count()
        assert hps.optim.n_batch_train % num_device == 0
        self.register_parameter('h_top',
                                nn.Parameter(torch.zeros([hps.optim.n_batch_train // num_device,
                                                          self.flow.output_shapes[-1][1] * 2,
                                                          self.flow.output_shapes[-1][2],
                                                          self.flow.output_shapes[-1][3]])))

    def prior(self, y_onehot=None):
        nc = self.h_top.shape[1]
        h = self.h_top.detach().clone()
        assert torch.sum(h) == 0.
        if self.hps.ablation.learn_top:
            h = self.learn_top(h)
        if self.hps.ablation.y_cond:
            assert y_onehot is not None
            h += self.y_emb(y_onehot).view(-1, nc, 1, 1)
        return ops.split_channel(h, 'simple')

    def preprocess(self, x):
        n_bins = 2 ** self.hps.model.n_bits_x
        if self.hps.model.n_bits_x < 8:
            x = torch.floor(x / 2 ** (8 - self.hps.model.n_bits_x))
        x = x / n_bins - .5
        return x

    def postprocess(self, x):
        n_bins = 2 ** self.hps.model.n_bits_x
        x = torch.clamp(torch.floor((x + .5) * n_bins) * (256. / n_bins), min=0, max=255)
        return x

    def normal_flow(self, x, y_onehot):
        # Pre-process for z
        n_bins = 2 ** self.hps.model.n_bits_x
        z = self.preprocess(x)
        z = z + torch.nn.init.uniform_(torch.empty(*z.shape, device=z.device), 0, 1. / n_bins)

        # Initialize logdet
        logdet_factor = z.shape[1] * z.shape[2]  # H * W
        objective = torch.zeros_like(x[:, 0, 0, 0])
        objective += float(-np.log(n_bins)) * logdet_factor

        # Encode
        z, objective = self.flow(z, logdet=objective, reverse=False)

        # Prior
        mean, logs = self.prior(y_onehot)
        objective += module.GaussianDiag.logp(mean, logs, z)

        # Prediction loss
        if self.hps.ablation.y_cond and self.hps.model.weight_y > 0:
            h_y = ops.reduce_mean(z, dim=[2, 3])
            y_logits = self.classifier(h_y)
        else:
            y_logits = None

        # Generative loss
        nobj = -objective
        # negative log-likelihood
        nll = nobj / float(np.log(2.) * x.shape[2] * x.shape[3])
        return z, nll, y_logits

    def reverse_flow(self, z, y_onehot, eps_std=None):
        with torch.no_grad():
            mean, logs = self.prior(y_onehot)
            if z is None:
                z = module.GaussianDiag.sample(mean, logs, eps_std)
            x, det = self.flow(z, eps_std=eps_std, reverse=True)
            x = self.postprocess(x)
            return x, det

    def forward(self,
                x=None, y_onehot=None,
                z=None, eps_std=None,
                reverse=False):
        if not reverse:
            return self.normal_flow(x, y_onehot)
        else:
            return self.reverse_flow(z, y_onehot, eps_std)

    @staticmethod
    def generative_loss(nll):
        return torch.mean(nll)

    @staticmethod
    def multi_class_loss(y_logits, y_onehot):
        """
        Classification loss for multiple target class problem

        :param y_logits: prediction in the shape of (N, Classes)
        :type y_logits: torch.Tensor
        :param y_onehot: one-hot targte vector in the shape of (N, Classes)
        :type y_onehot: torch.Tensor
        :return: classification loss
        :rtype: torch.Tensor
        """
        if y_logits is None:
            return 0
        return Glow.bce_criterion(y_logits, y_onehot.float())

    @staticmethod
    def single_class_loss(y_logits, y):
        """
        Classification loss for single target class problem

        :param y_logits: prediction in the shape of (N, Classes)
        :type y_logits: torch.Tensor
        :param y: target in the shape of (N)
        :type y: torch.Tensor
        :return: classification loss
        :rtype: torch.Tensor
        """
        if y_logits is None:
            return 0
        return Glow.ce_criterion(y_logits, y.long())

    def set_actnorm_inited(self, inited=True):
        """
        Set bias and logs of ActNorm layer initialized

        :param inited: initialization state
        :type inited: bool
        """
        for name, m in self.named_modules():
            if m.__class__.__name__.find("ActNorm") >= 0:
                m.inited = inited
