import numpy as np
import torch
import unittest

from network.module import Actnorm


class TestModule(unittest.TestCase):

    def test_actnorm(self):
        # initial variables
        x = torch.Tensor(np.random.rand(2, 16, 4, 4))
        actnorm = Actnorm(num_features=16)
        # forward and reverse flow
        y, _ = actnorm(x)
        x_, _ = actnorm(y, reverse=True)
        # assertion
        eps = 1e-6
        self.assertTrue(0 <= float(torch.max(torch.abs(x_ - x))) <= eps)


if __name__ == '__main__':
    unittest.main()
