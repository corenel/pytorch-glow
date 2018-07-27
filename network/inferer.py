import os
import time
import numpy as np
import torch

from torchvision import transforms
from torchvision.utils import make_grid

from misc import util, ops
from network.model import Glow


class Inferer:

    def __init__(self, hps, graph, devices, data_device):
        """
        Network inferer

        :param hps: hyper-parameters for this network
        :type hps: dict
        :param graph: model graph
        :type graph: torch.nn.Module
        :param devices: list of usable devices for model running
        :type devices: list
        :param data_device: list of usable devices for data loading
        :type data_device: str or int
        """
        # general
        self.hps = hps
        # state
        self.graph = graph
        self.devices = devices
        # data
        self.data_device = data_device
        self.num_classes = self.hps.dataset.num_classes
        # ablation
        self.y_condition = self.hps.ablation.y_condition

    def sample(self, z, y_onehot, eps_std=0.5):
        """
        Sample image

        :param z: latent feature vector
        :type z: torch.Tensor or None
        :param y_onehot: one-hot vector of label
        :type y_onehot: torch.Tensor or None
        :param eps_std: standard deviation of eps
        :type eps_std: float
        :return: generated image
        :rtype: np.ndarray
        """
        # generate sample from model
        img = self.graph(z=z, y_onehot=y_onehot, eps_std=eps_std, reverse=True)

        # create image grid
        grid = make_grid(img)

        # convert to numpy
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        grid_np = grid_np.astype(np.float32)
        grid_np = (grid_np * 255).astype(np.uint8)

        return grid_np
