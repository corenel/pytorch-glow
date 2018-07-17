import numpy as np
import torch
import unittest

from network.model import FlowStep
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
                    lu_decomposed=False
                )
                # forward and reverse flow
                y, det = flow_step(x, 0)
                x_, det_ = flow_step(y, det, reverse=True)
                # assertion
                print('-------------')
                # self.assertTrue(ops.tensor_equal(x, x_))


if __name__ == '__main__':
    unittest.main()
