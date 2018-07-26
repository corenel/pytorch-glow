import os
import torch

from functools import partial
from network import model
from misc import util, lr_scheduler


class Builder:
    optimizer_dict = {
        'adam': lambda params, **kwargs: torch.optim.Adam(params, **kwargs),
        'adamax': lambda params, **kwargs: torch.optim.Adamax(params, **kwargs)
    }
    lr_scheduler_dict = {
        'constant': lambda **kwargs: lr_scheduler.constant(**kwargs),
        'noam': lambda **kwargs: lr_scheduler.noam_decay(**kwargs),
        'linear': lambda **kwargs: lr_scheduler.linear_anneal(**kwargs),
        'step': lambda **kwargs: lr_scheduler.step_anneal(**kwargs),
        'cyclic_cosine': lambda **kwargs: lr_scheduler.cyclic_cosine_anneal(**kwargs),
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
        step = 0
        state = None
        result_subdir = None
        graph, optimizer, scheduler, criterion_dict = None, None, None, None
        devices = util.get_devices(self.hps.device.graph)
        data_device = util.get_devices(self.hps.device.data)[0]

        # build graph
        graph = model.Glow(self.hps)
        graph.to('cpu')

        # load model
        if graph is not None:
            # locate or create result subdir
            if self.hps.general.warm_start and self.hps.general.resume_run_id != "":
                result_subdir = util.locate_result_subdir(self.hps.general.result_dir,
                                                          self.hps.general.resume_run_id)

            if training and result_subdir is None:
                result_subdir = util.create_result_subdir(self.hps.general.result_dir,
                                                          desc=self.hps.profile,
                                                          profile=self.hps)
            # load pre-trained model on first device
            if self.hps.general.warm_start:
                step_or_model_path = None
                if os.path.exists(self.hps.general.pre_trained):
                    step_or_model_path = self.hps.general.pre_trained
                elif self.hps.general.resume_step not in ['', 'best', 'latest']:
                    step_or_model_path = int(self.hps.general.resume_step)
                if step_or_model_path is not None:
                    state = util.load_model(result_subdir, step_or_model_path, graph,
                                            device=devices[0])
                if not training and state is None:
                    raise RuntimeError('No pre-trained model for inference')
            # move graph to devices
            if 'cpu' in devices:
                graph = graph.cpu()
                data_device = 'cpu'
            else:
                graph = graph.to(devices[0])
            print('Use {} for model running and {} for data loading'.format(devices[0], data_device))

        # setup optimizer and lr scheduler
        if training and graph is not None:
            # get optimizer
            optimizer_name = self.hps.optim.optimizer.lower()
            assert optimizer_name in self.optimizer_dict.keys(), \
                "Unsupported optimizer: {}".format(optimizer_name)
            # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
            optimizer = self.optimizer_dict[optimizer_name](
                graph.parameters(),
                **self.hps.optim.optimizer_args)
            if state is not None:
                optimizer.load_state_dict(state['optimizer'])
            # get lr scheduler
            scheduler_name = self.hps.optim.lr_scheduler.lower()
            scheduler_args = self.hps.optim.lr_scheduler_args
            assert scheduler_name in self.lr_scheduler_dict.keys(), \
                "Unsupported lr scheduler: {}".format(scheduler_name)
            if 'base_lr' not in scheduler_args:
                scheduler_args['base_lr'] = self.hps.optim.optimizer_args['lr']
            scheduler = partial(self.lr_scheduler_dict[scheduler_name], **scheduler_args)

        return {
            'step': step,
            'graph': graph,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'devices': devices,
            'data_device': data_device,
            'result_subdir': result_subdir
        }
