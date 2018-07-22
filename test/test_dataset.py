import unittest

from dataset import CelebA


class TestDataset(unittest.TestCase):
    def test_celeba(self):
        dataset = CelebA(root='/Data/CelebA')


if __name__ == '__main__':
    unittest.main()
