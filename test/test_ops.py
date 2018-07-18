import numpy as np
import torch
import unittest

from misc import ops


class TestOps(unittest.TestCase):
    def test_tensor_equal(self):
        x = torch.Tensor(np.random.rand(2, 16, 4, 4))
        x_ = x + 1e-4
        self.assertTrue(ops.tensor_equal(x, x))
        self.assertFalse(ops.tensor_equal(x, x_))

    def test_reduce_mean(self):
        x = torch.ones(2, 3, 16, 16)
        mean = ops.reduce_mean(x, dim=[1, 2, 3])
        self.assertTrue(ops.tensor_equal(torch.ones(2), mean))

    def test_reduce_sum(self):
        x = torch.ones(2, 3, 16, 16)
        sum = ops.reduce_sum(x, dim=[1, 2, 3])
        sum_shape = float(x.shape[1] * x.shape[2] * x.shape[3])
        self.assertTrue(ops.tensor_equal(torch.Tensor([sum_shape, sum_shape]), sum))

    def test_split_channel(self):
        x = torch.ones(2, 4, 16, 16)
        nc = x.shape[1]
        # simple splitting
        x1, x2 = ops.split_channel(x, 'simple')
        for c in range(nc // 2):
            self.assertTrue(ops.tensor_equal(x1[:, c, :, :], x[:, c, :, :]))
            self.assertTrue(ops.tensor_equal(x2[:, c, :, :], x[:, nc // 2 + c, :, :]))
        # cross splitting
        x1, x2 = ops.split_channel(x, 'cross')
        for c in range(nc // 2):
            self.assertTrue(ops.tensor_equal(x1[:, c, :, :], x[:, 2 * c, :, :]))
            self.assertTrue(ops.tensor_equal(x2[:, c, :, :], x[:, 2 * c + 1, :, :]))

    def test_cat_channel(self):
        x = torch.ones(2, 4, 16, 16)
        x1, x2 = ops.split_channel(x, 'simple')
        self.assertTrue(ops.tensor_equal(ops.cat_channel(x1, x2), x))

    def test_count_pixels(self):
        x = torch.Tensor(np.random.rand(2, 16, 4, 4))
        nh, nw = x.shape[2], x.shape[3]
        self.assertEqual(nh * nw, ops.count_pixels(x))


if __name__ == '__main__':
    unittest.main()
