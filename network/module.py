import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActNorm(nn.Module):
    def __init__(self, num_features, scale=1., logscale_factor=3., batch_variance=False):
        """
        Activation normalization layer

        :param num_features: number of channels
        :type num_features: int
        :param scale: scale
        :type scale: float
        :param logscale_factor: factor for logscale
        :type logscale_factor: float
        :param batch_variance: use batch variance
        :type batch_variance: bool
        """
        super().__init__()
        self.num_features = num_features
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
                raise NotImplementedError
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


class LinearZero(nn.Linear):
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


class Conv2D(nn.Conv2d):
    @staticmethod
    def get_edge_padding(pad, kernel_size, stride):
        pass

    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=1, pad='same',
                 do_weightnorm=False, weight_std=0.05, do_actnorm=True,
                 dilation=1, groups=1, bias=True):
        padding = None
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        return super().forward(x)


