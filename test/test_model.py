import numpy as np
import cv2
import torch
import unittest

from network.model import FlowStep, FlowModel, Glow
from misc import ops, util


class TestModel(unittest.TestCase):
    def test_flow_step(self):
        flow_permutation = ['invconv', 'reverse', 'shuffle']
        flow_coupling = ['additive', 'affine']

        for permutation in flow_permutation:
            for coupling in flow_coupling:
                # initial variables
                x = torch.Tensor(np.random.rand(2, 16, 4, 4))
                flow_step = FlowStep(
                    in_channels=16,
                    hidden_channels=256,
                    permutation=permutation,
                    coupling=coupling,
                    actnorm_scale=1.,
                    lu_decomposition=False
                )
                # forward and reverse flow
                y, det = flow_step(x, 0, reverse=False)
                x_, det_ = flow_step(y, det, reverse=True)
                # assertion
                self.assertTrue(ops.tensor_equal(x, x_))

    def test_flow_model(self):
        flow_permutation = ['invconv', 'reverse', 'shuffle']
        flow_coupling = ['additive', 'affine']

        for permutation in flow_permutation:
            for coupling in flow_coupling:
                # initial variables
                x = torch.Tensor(np.random.rand(2, 3, 16, 16))
                flow_model = FlowModel(
                    in_shape=(16, 16, 3),
                    hidden_channels=256,
                    K=16, L=3,
                    permutation=permutation,
                    coupling=coupling,
                    actnorm_scale=1.,
                    lu_decomposition=False
                )
                # forward and reverse flow
                y, det = flow_model(x, 0, reverse=False)
                x_, _ = flow_model(y, det, reverse=True)
                # assertion
                self.assertEqual(x.shape, x_.shape)
                self.assertTupleEqual((2, 48, 2, 2), tuple(y.shape))

    def test_glow_model(self):
        # build model
        hps = util.load_profile('profile/test.json')
        glow_model = Glow(hps).cuda()
        image_shape = hps.model.image_shape
        # read image
        img = cv2.imread('misc/test.png')
        img = cv2.resize(img, (image_shape[0], image_shape[1]))
        img = (img / 255.0).astype(np.float32)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        x = torch.Tensor([img] * hps.optim.n_batch_train).cuda()
        y_onehot = torch.zeros((2, hps.dataset.n_classes)).cuda()
        # forward and reverse flow
        z, logdet, y_logits = glow_model(x=x, y_onehot=y_onehot, reverse=False)
        x_ = glow_model(z=z, y_onehot=y_onehot, reverse=True)
        # assertion
        # self.assertEqual(x.shape, x_.shape)
        # self.assertTupleEqual((2, 48, 2, 2), tuple(y.shape))


if __name__ == '__main__':
    unittest.main()
