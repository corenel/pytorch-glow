import os
import re
import json
import torch
from easydict import EasyDict


def load_profile(filepath):
    """
    Load experiment profile as EasyDict

    :param filepath: path to profile
    :type filepath: str
    :return: hyper-parameters
    :rtype: EasyDict
    """
    if os.path.exists(filepath):
        with open(filepath) as f:
            return EasyDict(json.load(f))


def get_devices(devices):
    """
    Get devices for running model

    :param devices: list of devices from profile
    :type devices: list
    :return: list of usable devices according to desired and available hardware
    :rtype: list[str]
    """

    def parse_cuda_device(device):
        """
        Parse device into device id

        :param device: given device
        :type device: str or int
        :return: device id
        :rtype: int
        """
        origin = str(device)
        if isinstance(device, str) and re.search('cuda:([\d]+)', device):
            device = int(re.findall('cuda:([\d]+)', device)[0])
        if isinstance(device, int):
            if 0 <= device <= torch.cuda.device_count() - 1:
                return device
        print('Incorrect device "{}"'.format(origin))
        return

    use_cpu = any([d.find('cpu') >= 0 for d in devices])
    use_cuda = any([(d.find('cuda') >= 0 or isinstance(d, int)) for d in devices])
    assert not (use_cpu and use_cuda), 'CPU and GPU cannot be mixed.'

    if use_cuda:
        devices = [parse_cuda_device(d) for d in devices]
        devices = [d for d in devices if d is not None]
        if len(devices) == 0:
            print('No available GPU found, use CPU only')
            devices = ['cpu']

    return devices
