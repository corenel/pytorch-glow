from misc import util
import torch
import unittest


class TestUtil(unittest.TestCase):
    def test_load_profile(self):
        hps = util.load_profile('profile/celebahq_256x256_5bit.json')
        self.assertIsInstance(hps, dict)

    def test_get_devices(self):
        # use cpu only
        devices = ['cpu']
        self.assertListEqual(devices, util.get_devices(devices))
        # use gpu
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
        if cuda_available and gpu_count >= 1:
            self.assertListEqual([0], util.get_devices(['cuda:0']))
            self.assertListEqual([0], util.get_devices(['cuda:0', 'cuda:1']))
            self.assertListEqual(['cpu'], util.get_devices(['cuda:1']))
            with self.assertRaises(AssertionError):
                util.get_devices(['cuda:1', 'cpu'])
        if cuda_available and gpu_count >= 2:
            devices = ['cuda:0', 'cuda:1']
            self.assertListEqual([0, 1], util.get_devices(devices))

    def test_result_subdir(self):
        result_dir = '/tmp'
        result_subdir = util.create_result_subdir(result_dir,
                                                  desc='test',
                                                  profile_path='profile/test.json')
        util.locate_result_subdir(result_dir, result_subdir)
        util.locate_result_subdir(result_dir, 0)
        util.locate_result_subdir(result_dir, '000')


if __name__ == '__main__':
    unittest.main()
