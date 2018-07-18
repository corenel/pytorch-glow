from misc import util
import unittest


class TestUtil(unittest.TestCase):
    def test_load_profile(self):
        hps = util.load_profile('profile/celebahq_256x256_5bit.json')
        self.assertIsInstance(hps, dict)


if __name__ == '__main__':
    unittest.main()
