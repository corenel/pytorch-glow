import numpy as np
import torch

from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from misc import ops


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
        self.graph.eval()
        self.devices = devices
        self.use_cuda = 'cpu' not in self.devices
        # data
        self.data_device = data_device
        self.batch_size = self.graph.h_top.shape[0]
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
        with torch.no_grad():
            # generate sample from model
            img = self.graph(z=z, y_onehot=y_onehot, eps_std=eps_std, reverse=True)

            # create image grid
            grid = make_grid(img)

            # convert to numpy
            grid_np = grid.permute(1, 2, 0).cpu().numpy()
            grid_np = grid_np.astype(np.float32)
            grid_np = (grid_np * 255).astype(np.uint8)

            return grid_np

    def encode(self, img):
        """
        Encode input image to latent features

        :param img: input image
        :type img: torch.Tensor
        :return: latent features
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            x = img.unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
            if self.use_cuda:
                x = x.cuda()
            z, _, _ = self.graph(x)

    def compute_attribute_delta(self, dataset):
        """
        Compute feature vector delta of different attributes

        :param dataset: dataset for training model
        :type dataset: torch.utils.data.Dataset
        :return:
        :rtype:
        """
        print('[Inferer] Computing attribute delta')
        with torch.no_grad():
            # initialize variables
            attrs_z_pos = np.zeros([self.num_classes, *self.graph.flow.output_shapes[-1][1:]])
            attrs_z_neg = np.zeros([self.num_classes, *self.graph.flow.output_shapes[-1][1:]])
            num_z_pos = np.zeros(self.num_classes)
            num_z_neg = np.zeros(self.num_classes)
            delta = np.zeros([self.num_classes, *self.graph.flow.output_shapes[-1][1:]])

            data_loader = DataLoader(dataset, batch_size=self.batch_size,
                                     num_workers=self.hps.dataset.num_workers,
                                     shuffle=True,
                                     drop_last=True)

            progress = tqdm(data_loader)
            for idx, batch in enumerate(progress):
                # extract batch data
                assert 'y_onehot' in batch.keys(), 'Compute attribute delta needs "y_onehot" in batch data'
                for i in batch:
                    batch[i] = batch[i].to(self.data_device)
                x = batch['x']
                y_onehot = batch['y_onehot']

                # decode latent features
                z, _, _ = self.graph(x)

                # append to latent feature list by attributes
                for i in range(len(batch)):
                    for cls in range(self.num_classes):
                        if y_onehot[i, cls] > 0:
                            attrs_z_pos[cls] += z[i]
                            num_z_pos[cls] += 1
                        else:
                            attrs_z_neg[cls] += z[i]
                            num_z_neg[cls] += 1

            # compute delta
            num_z_pos = [max(1., float(num)) for num in num_z_pos]
            num_z_neg = [max(1., float(num)) for num in num_z_neg]
            for cls in range(self.num_classes):
                mean_z_pos = attrs_z_pos[cls] / num_z_pos[cls]
                mean_z_neg = attrs_z_neg[cls] / num_z_neg[cls]
                delta[cls] = mean_z_pos - mean_z_neg

            return delta
