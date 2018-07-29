import numpy as np
import torch

from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from misc import util


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

            return util.tensor_to_pil(grid)

    def encode(self, img):
        """
        Encode input image to latent features

        :param img: input image
        :type img: torch.Tensor or np.numpy or Image.Image
        :return: latent features
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            if not torch.is_tensor(img):
                img = util.image_to_tensor(
                    img,
                    shape=self.hps.model.image_shape)
                img = util.make_batch(img, self.batch_size)
            if self.use_cuda:
                img = img.cuda()
            z, _, _ = self.graph(img)
            return z[0, :, :, :]

    def decode(self, z):
        """

        :param z: input latent feature vector
        :type z: torch.Tensor
        :return: decoded image
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            if len(z.shape) == 3:
                z = util.make_batch(z, self.batch_size)
            if self.use_cuda:
                z = z.cuda()

            img = self.graph(z=z, y_onehot=None, reverse=True)[0, :, :, :]
            return img

    def compute_attribute_delta(self, dataset):
        """
        Compute feature vector deltaz of different attributes

        :param dataset: dataset for training model
        :type dataset: torch.utils.data.Dataset
        :return:
        :rtype:
        """
        print('[Inferer] Computing attribute deltaz')
        with torch.no_grad():
            # initialize variables
            attrs_z_pos = np.zeros([self.num_classes, *self.graph.flow.output_shapes[-1][1:]])
            attrs_z_neg = np.zeros([self.num_classes, *self.graph.flow.output_shapes[-1][1:]])
            num_z_pos = np.zeros(self.num_classes)
            num_z_neg = np.zeros(self.num_classes)
            deltaz = np.zeros([self.num_classes, *self.graph.flow.output_shapes[-1][1:]])

            data_loader = DataLoader(dataset, batch_size=self.batch_size,
                                     num_workers=self.hps.dataset.num_workers,
                                     shuffle=True,
                                     drop_last=True)

            progress = tqdm(data_loader)
            for idx, batch in enumerate(progress):
                # extract batch data
                assert 'y_onehot' in batch.keys(), 'Compute attribute deltaz needs "y_onehot" in batch data'
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

            # compute deltaz
            num_z_pos = [max(1., float(num)) for num in num_z_pos]
            num_z_neg = [max(1., float(num)) for num in num_z_neg]
            for cls in range(self.num_classes):
                mean_z_pos = attrs_z_pos[cls] / num_z_pos[cls]
                mean_z_neg = attrs_z_neg[cls] / num_z_neg[cls]
                deltaz[cls] = mean_z_pos - mean_z_neg

            return deltaz

    def apply_attribute_delta(self, img, deltaz, interpolation):
        """
        Apply attribute delta to image by given interpolation vector

        :param img: given image
        :type img: torch.Tensor or np.numpy or Image.Image
        :param deltaz: delta vector of attributes in latent space
        :type deltaz: np.ndarray
        :param interpolation: interpolation vector
        :type interpolation: torch.Tensor or np.ndarray or list[float]
        :return: processed image
        :rtype: torch.Tensor
        """
        print('[Inferer] Applying attribute deltaz')

        if isinstance(deltaz, np.ndarray):
            deltaz = torch.Tensor(deltaz)
        assert len(interpolation) == self.num_classes
        assert deltaz.shape == torch.Size([self.num_classes,
                                           *self.graph.flow.output_shapes[-1][1:]])

        # encode
        z = self.encode(img)

        # perform interpolation
        z_interpolated = z.clone()
        for i in range(len(interpolation)):
            z_delta = deltaz[i].mul(interpolation[i])
            if self.use_cuda:
                z_delta = z_delta.cuda()
            z_interpolated += z_delta

        # decode
        img = self.decode(z_interpolated)

        return img
