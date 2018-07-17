import numpy as np
import torch
import unittest

from network.module import (ActNorm, LinearZeros, Conv2d, Conv2dZeros,
                            Invertible1x1Conv, Permutation2d, Split2d, Squeeze2d)
from misc import ops


class TestModule(unittest.TestCase):

    def test_actnorm(self):
        # initial variables
        x = torch.Tensor(np.random.rand(2, 16, 4, 4))
        actnorm = ActNorm(num_channels=16)
        # forward and reverse flow
        y, _ = actnorm(x)
        x_, _ = actnorm(y, reverse=True)
        # assertion
        self.assertTrue(ops.tensor_equal(x, x_))

    def test_linear_zeros(self):
        # initial variables
        x = torch.Tensor(np.random.rand(16))
        linear_zeros = LinearZeros(16, 16)
        # forward
        y = linear_zeros(x)
        # assertion
        self.assertTrue(torch.equal(y, torch.zeros(16)))

    def test_conv2d(self):
        # initial variables
        x = torch.Tensor(np.random.rand(2, 16, 4, 4))
        conv2d = Conv2d(in_channels=16, out_channels=5)
        # forward and reverse flow
        y = conv2d(x)
        # assertion
        self.assertTupleEqual((2, 5, 4, 4), tuple(y.shape))

    def test_conv2d_zeros(self):
        # initial variables
        x = torch.Tensor(np.random.rand(2, 16, 4, 4))
        conv2d_zeros = Conv2dZeros(in_channels=16, out_channels=5)
        # forward and reverse flow
        y = conv2d_zeros(x)
        # assertion
        self.assertTupleEqual((5, 16), tuple(conv2d_zeros.weight.shape[:2]))
        self.assertTupleEqual((2, 5, 4, 4), tuple(y.shape))

    def test_invertible_1x1_conv(self):
        # initial variables
        x = torch.Tensor(np.random.rand(2, 16, 4, 4))
        invertible_1x1_conv = Invertible1x1Conv(num_channels=16)
        # forward and reverse flow
        y, _ = invertible_1x1_conv(x)
        x_, _ = invertible_1x1_conv(y, reverse=True)
        # assertion
        self.assertTupleEqual((2, 16, 4, 4), tuple(y.shape))
        self.assertTrue(ops.tensor_equal(x, x_))

    def test_permutation2d(self):
        # initial variables
        x = torch.Tensor(np.random.rand(2, 16, 4, 4))
        reverse = Permutation2d(num_channels=16)
        shuffle = Permutation2d(num_channels=16, shuffle=True)
        # forward and reverse flow
        y_reverse = reverse(x)
        x_reverse = reverse(y_reverse, reverse=True)
        y_shuffle = shuffle(x)
        x_shuffle = shuffle(y_shuffle, reverse=True)
        # assertion
        self.assertTrue(ops.tensor_equal(x, x_reverse))
        self.assertTrue(ops.tensor_equal(x, x_shuffle))

    def test_squeeze2d(self):
        # initial variables
        x = torch.Tensor(np.random.rand(2, 16, 4, 4))
        squeeze = Squeeze2d(factor=2)
        # forward and reverse flow
        y, _ = squeeze(x)
        x_, _ = squeeze(y, reverse=True)
        # assertion
        self.assertTrue(ops.tensor_equal(x, x_))

    def test_split2d(self):
        # initial variables
        x = torch.Tensor(np.random.rand(2, 16, 4, 4))
        split2d = Split2d(num_channels=16)
        # forward and reverse flow
        y, _ = split2d(x, 0)
        x_, _ = split2d(y, 0, reverse=True)
        # assertion
        self.assertTrue(ops.tensor_equal(x, x_))


if __name__ == '__main__':
    unittest.main()
