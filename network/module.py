import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActNorm(nn.Module):
    def __init__(self, num_channels, scale=1., logscale_factor=3., batch_variance=False):
        """
        Activation normalization layer

        :param num_channels: number of channels
        :type num_channels: int
        :param scale: scale
        :type scale: float
        :param logscale_factor: factor for logscale
        :type logscale_factor: float
        :param batch_variance: use batch variance
        :type batch_variance: bool
        """
        super().__init__()
        self.num_features = num_channels
        self.scale = scale
        self.logscale_factor = logscale_factor
        self.batch_variance = batch_variance
        self.logdet_factor = None

        self.initialized = False
        self.register_parameter('bias', nn.Parameter(torch.zeros(self.num_features)))
        self.register_parameter('logs', nn.Parameter(torch.zeros(self.num_features)))

    def actnorm_center(self, x, reverse=False):
        """
        center operation of activation normalization

        :param x: input
        :type x: torch.Tensor
        :param reverse: whether to reverse bias
        :type reverse: bool
        :return: centered input
        :rtype: torch.Tensor
        """
        if not reverse:
            x += self.bias
        else:
            x -= self.bias

        return x

    def actnorm_scale(self, x, logdet, reverse=False):
        """
        scale operation of activation normalization

        :param x: input
        :type x: torch.Tensor
        :param logdet:
        :type logdet:
        :param reverse: whether to reverse bias
        :type reverse: bool
        :return: centered input and logdet
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        logs = self.logs * self.logscale_factor

        if not reverse:
            x = x * torch.exp(logs)
        else:
            x = x * torch.exp(-logs)

        if logdet is not None:
            dlogdet = torch.sum(logs) * self.logdet_factor
            if reverse:
                dlogdet *= -1
            logdet += dlogdet

        return x, logdet

    def initialize_parameters(self, x):
        """
        Initialize bias and logs

        :param x: input
        :type x: torch.Tensor
        """
        with torch.no_grad():
            # Compute initial value
            # (N, C, H, W) -> (N, H, W, C) -> (N * H * W, C)
            x = x.permute([0, 2, 3, 1]).contiguous().view(-1, self.num_features)
            x_mean = -torch.mean(x, 0, keepdim=False)
            if self.batch_variance:
                raise NotImplementedError()
            else:
                x_var = torch.mean(x ** 2, 0, keepdim=False)
            logs = torch.log(self.scale / (torch.sqrt(x_var) + 1e-6)) / self.logscale_factor

            # Copy to parameters
            self.bias.data.copy_(x_mean.data)
            self.logs.data.copy_(logs.data)
            self.initialized = True

    def forward(self, x, logdet=None, reverse=False):
        """
        Forward activation normalization layer

        :param x: input
        :type x: torch.Tensor
        :param logdet:
        :type logdet:
        :param reverse: whether to reverse bias
        :type reverse: bool
        :return: normalized input and logdet
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        assert len(x.shape) == 4
        assert x.shape[1] == self.num_features, \
            'Input shape should be NxCxHxW, however channels are {} instead of {}'.format(x.shape[1], self.num_features)

        if not self.initialized:
            self.initialize_parameters(x)

        # TODO condition for non 4-dims input
        self.logdet_factor = int(x.shape[2]) * int(x.shape[3])

        # Transpose feature to last dim
        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute([0, 2, 3, 1]).contiguous()
        # record current shape
        x_shape = x.shape
        # (N, H, W, C) -> (N * H * W, C)
        x = x.view(-1, self.num_features)

        if not reverse:
            # center and scale
            x = self.actnorm_center(x, reverse)
            x, logdet = self.actnorm_scale(x, logdet, reverse)
        else:
            # scale and center
            x, logdet = self.actnorm_scale(x, logdet, reverse)
            x = self.actnorm_center(x, reverse)
        # reshape and transpose back
        # (N * H * W, C) -> (N, H, W, C) -> (N, C, H, W)
        x = x.view(*x_shape).permute([0, 3, 1, 2])
        return x, logdet


class LinearZeros(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, logscale_factor=3.):
        """
        Linear layer with zero initialization

        :param in_features: size of each input sample
        :type in_features: int
        :param out_features: size of each output sample
        :type out_features: int
        :param bias: whether to learn an additive bias.
        :type bias: bool
        :param logscale_factor: factor of logscale
        :type logscale_factor: float
        """
        super().__init__(in_features, out_features, bias)
        self.logscale_factor = logscale_factor
        # zero initialization
        self.weight.data.zero_()
        self.bias.data.zero_()
        # register parameter
        self.register_parameter('logs', nn.Parameter(torch.zeros(out_features)))

    def forward(self, x):
        """
        Forward linear zero layer

        :param x: input
        :type x: torch.Tensor
        :return: output
        :rtype: torch.Tensor
        """
        output = super().forward(x)
        output *= torch.exp(self.logs * self.logscale_factor)
        return output


class Conv2d(nn.Conv2d):
    @staticmethod
    def get_padding(padding_type, kernel_size, stride):
        """
        Get padding size.

        mentioned in https://github.com/pytorch/pytorch/issues/3867#issuecomment-361775080
        behaves as 'SAME' padding in TensorFlow
        independent on input size when stride is 1

        :param padding_type: type of padding in ['SAME', 'VALID']
        :type padding_type: str
        :param kernel_size: kernel size
        :type kernel_size: tuple(int) or int
        :param stride: stride
        :type stride: int
        :return: padding size
        :rtype: tuple(int)
        """
        assert padding_type in ['SAME', 'VALID'], "Unsupported padding type: {}".format(padding_type)
        if padding_type == 'SAME':
            assert stride == 1, "'SAME' padding only supports stride=1"
            return tuple((k - 1) // 2 for k in kernel_size)
        return tuple(0 for _ in kernel_size)

    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=1, padding_type='SAME',
                 do_weightnorm=False, do_actnorm=True,
                 dilation=1, groups=1):
        """
        Wrapper of nn.Conv2d with weight normalization and activation normalization

        :param padding_type: type of padding in ['SAME', 'VALID']
        :type padding_type: str
        :param do_weightnorm: whether to do weight normalization after convolution
        :type do_weightnorm: bool
        :param do_actnorm: whether to do activation normalization after convolution
        :type do_actnorm: bool
        """
        padding = self.get_padding(padding_type, kernel_size, stride)
        super().__init__(in_channels, out_channels,
                         kernel_size, stride, padding,
                         dilation, groups,
                         bias=(not do_actnorm))
        self.do_weight_norm = do_weightnorm
        self.do_actnorm = do_actnorm

        self.weight.data.normal_(mean=0.0, std=0.05)
        if self.do_actnorm:
            self.actnorm = ActNorm(out_channels)
        else:
            self.bias.data.zero_()

    def forward(self, x):
        """
        Forward wrapped Conv2d layer

        :param x: input
        :type x: torch.Tensor
        :return: output
        :rtype: torch.Tensor
        """
        x = super().forward(x)
        if self.do_weight_norm:
            # normalize N, H and W dims
            F.normalize(x, p=2, dim=0)
            F.normalize(x, p=2, dim=2)
            F.normalize(x, p=2, dim=3)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Conv2d):

    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=1, padding_type='SAME',
                 logscale_factor=3,
                 dilation=1, groups=1, bias=True):
        """
        Wrapper of nn.Conv2d with zero initialization and logs

        :param padding_type: type of padding in ['SAME', 'VALID']
        :type padding_type: str
        :param logscale_factor: factor for logscale
        :type logscale_factor: float
        """
        padding = Conv2d.get_padding(padding_type, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.logscale_factor = logscale_factor
        # initialize variables with zero
        self.bias.data.zero_()
        self.weight.data.zero_()
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))

    def forward(self, x):
        """
        Forward wrapped Conv2d layer

        :param x: input
        :type x: torch.Tensor
        :return: output
        :rtype: torch.Tensor
        """
        x = super().forward(x)
        x *= torch.exp(self.logs * self.logscale_factor)
        return x


class Invertible1x1Conv(nn.Module):

    def __init__(self, num_channels, lu_decomposed=False):
        """
        Invertible 1x1 convolution layer

        :param num_channels: number of channels
        :type num_channels: int
        :param lu_decomposed: whether to use LU decomposition
        :type lu_decomposed: bool
        """
        super().__init__()
        self.num_channels = num_channels
        self.lu_decomposed = lu_decomposed
        if self.lu_decomposed:
            raise NotImplementedError()
        else:
            self.w_shape = [num_channels, num_channels]
            self.logdet_factor = None
            # Sample a random orthogonal matrix
            w_init = np.linalg.qr(np.random.randn(*self.w_shape))[0].astype('float32')
            self.register_parameter('weight', nn.Parameter(torch.Tensor(w_init)))

    def forward(self, x, logdet=None, reverse=False):
        """

        :param x: input
        :type x: torch.Tensor
        :param logdet:
        :type logdet:
        :param reverse: whether to reverse bias
        :type reverse: bool
        :return: output and logdet
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        self.logdet_factor = x.shape[1] * x.shape[2]  # H * W
        dlogdet = torch.log(torch.abs(torch.det(self.weight))) * self.logdet_factor
        if not reverse:
            weight = self.weight.view(*self.w_shape, 1, 1)
            z = F.conv2d(x, weight)
            if logdet is not None:
                logdet += dlogdet
            return z, dlogdet
        else:
            weight = self.weight.inverse().view(*self.w_shape, 1, 1)
            z = F.conv2d(x, weight)
            if logdet is not None:
                logdet -= dlogdet
            return z, dlogdet


class GaussianDiag:
    @staticmethod
    def eps(mean):
        return torch.randn_like(mean)

    @staticmethod
    def flatten_sum(s):
        if len(s.shape) == 4:
            flatten = s.view(s.shape[0], -1)
            return torch.sum(flatten, dim=1)
        else:
            raise NotImplementedError()

    @staticmethod
    def logps(mean, logs, x):
        return -0.5 * (np.log(2 * np.pi) + 2. * logs + (x - mean) ** 2 / torch.exp(2. * logs))

    @staticmethod
    def logp(mean, logs, x):
        s = GaussianDiag.logps(mean, logs, x)
        return GaussianDiag.flatten_sum(s)

    @staticmethod
    def sample(mean, logs):
        eps = GaussianDiag.eps(mean)
        return mean + torch.exp(logs) * eps


class Split2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.conv2d_zeros = Conv2dZeros(num_channels // 2, num_channels)

    @staticmethod
    def unsequeeze2d(x, factor=2):
        assert factor >= 1
        if factor == 1:
            return x
        nc = x.shape[1]
        nh = x.shape[2]
        nw = x.shape[3]
        assert nc >= 4 and nc % 4 == 0
        x = x.view(-1, int(nc / factor ** 2), factor, factor, nh, nw)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(-1, int(nc / factor ** 2), int(nh * factor), int(nw * factor))
        return x

    @staticmethod
    def sequeeze2d(x, factor=2):
        assert factor >= 1
        if factor == 1:
            return x
        nc = x.shape[1]
        nh = x.shape[2]
        nw = x.shape[3]
        assert nh % factor == 0 and nw % factor == 0
        x = x.view(-1, nc, nh // factor, factor, nw // factor, factor)
        x = x.permute([0, 1, 3, 5, 2, 4]).contiguous()
        x = x.view(-1, nc * factor * factor, nh // factor, nw // factor)
        return x

    def prior(self, z):
        h = self.conv2d_zeros(z)
        mean = h[:, 0::2, :, :]
        logs = h[:, 1::2, :, :]
        return mean, logs

    def forward(self, x, logdet=None, reverse=False):
        pass
