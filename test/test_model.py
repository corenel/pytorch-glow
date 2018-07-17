import numpy as np
import torch
import unittest

from network.model import FlowStep, FlowModel
from misc import ops


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


if __name__ == '__main__':
    unittest.main()
