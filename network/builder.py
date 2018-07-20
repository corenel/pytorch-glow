import os
import torch

from network import model
from misc import util, lr_scheduler


class Builder:
    optimizer_dict = {
        'adam': lambda params, **kwargs: torch.optim.Adam(params, **kwargs),
        'adamax': lambda params, **kwargs: torch.optim.Adamax(params, **kwargs)
    }
    lr_scheduler_dict = {
        'constant': lambda *args, **kwargs: lr_scheduler.constant(**kwargs)

    }

    def __init__(self, hps):
        """
        Network builder

        :param hps: hyper-parameters for this network
        :type hps: dict
        """
        super().__init__()
        self.hps = hps

    def build(self, training=True):
        """
        Build network

        :param training:
        :type training:
        :return:
        :rtype:
        """
        # initialize all variables
        graph, optimizer, scheduler, criterion_dict = None, None, None, None
        devices = None

        # build graph
        graph = model.Glow(self.hps)
        devices = util.get_devices(self.hps.device.graph)
        graph.to('cpu')

        # get optimizer and set lr schedule
        if training and graph is not None:
            optimizer_name = self.hps.optim.optimizer.lower()
            assert optimizer_name in self.optimizer_dict.keys(), \
                "Unsupported optimizer: {}".format(optimizer_name)
            optimizer = self.optimizer_dict[optimizer_name](graph.parameters(),
                                                            **self.hps.optim.optimizer_args)
            scheduler_name = self.hps.optim.lr_scheduler.lower()
