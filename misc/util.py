import os
import re
import sys
import glob
import json
import shutil
import numpy as np
import torch

from easydict import EasyDict


# Profile

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


# Device

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


# Logger

class OutputLogger(object):
    """Output logger"""

    def __init__(self):
        self.file = None
        self.buffer = ''

    def set_log_file(self, filename, mode='wt'):
        assert self.file is None
        self.file = open(filename, mode)
        if self.buffer is not None:
            self.file.write(self.buffer)
            self.buffer = None

    def write(self, data):
        if self.file is not None:
            self.file.write(data)
        if self.buffer is not None:
            self.buffer += data

    def flush(self):
        if self.file is not None:
            self.file.flush()


class TeeOutputStream(object):
    """Redirect output stream"""

    def __init__(self, child_streams, autoflush=False):
        self.child_streams = child_streams
        self.autoflush = autoflush

    def write(self, data):
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        for stream in self.child_streams:
            stream.write(data)
        if self.autoflush:
            self.flush()

    def flush(self):
        for stream in self.child_streams:
            stream.flush()


output_logger = None


def init_output_logging():
    """
    Initialize output logger
    """
    global output_logger
    if output_logger is None:
        output_logger = OutputLogger()
        sys.stdout = TeeOutputStream([sys.stdout, output_logger], autoflush=True)
        sys.stderr = TeeOutputStream([sys.stderr, output_logger], autoflush=True)


def set_output_log_file(filename, mode='wt'):
    """
    Set file name of output log

    :param filename: file name of log
    :type filename: str
    :param mode: the mode in which the file is opened
    :type mode: str
    """
    if output_logger is not None:
        output_logger.set_log_file(filename, mode)


# Result directory

def create_result_subdir(result_dir, desc, profile):
    """
    Create and initialize result sub-directory

    :param result_dir: path to root of result directory
    :type result_dir: str
    :param desc: description of current experiment
    :type desc: str
    :param profile: profile
    :type profile: dict
    :return: path to result sub-directory
    :rtype: str
    """
    # determine run id
    run_id = 0
    for fname in glob.glob(os.path.join(result_dir, '*')):
        fbase = os.path.basename(fname)
        finds = re.findall('^([\d]+)-', fbase)
        if len(finds) != 0:
            ford = int(finds[0])
            run_id = max(run_id, ford + 1)

    # create result sub-directory
    result_subdir = os.path.join(result_dir, '{:03d}-{:s}'.format(run_id, desc))
    if not os.path.exists(result_subdir):
        os.makedirs(result_subdir)
    set_output_log_file(os.path.join(result_subdir, 'log.txt'))
    print("Saving results to {}".format(result_subdir))

    # export profile
    with open(os.path.join(result_subdir, 'config.json'), 'w') as f:
        json.dump(profile, f)

    return result_subdir


def locate_result_subdir(result_dir, run_id_or_result_subdir):
    """
    Locate result subdir by given run id or path

    :param result_dir: path to root of result directory
    :type result_dir: str
    :param run_id_or_result_subdir: run id or subdir path
    :type run_id_or_result_subdir: int or str
    :return: located result subdir
    :rtype: str
    """
    if isinstance(run_id_or_result_subdir, str) and os.path.isdir(run_id_or_result_subdir):
        return run_id_or_result_subdir

    searchdirs = ['', 'results', 'networks']

    for searchdir in searchdirs:
        d = result_dir if searchdir == '' else os.path.join(result_dir, searchdir)
        # search directly by name
        d = os.path.join(d, str(run_id_or_result_subdir))
        if os.path.isdir(d):
            return d
        # search by prefix
        if isinstance(run_id_or_result_subdir, int):
            prefix = '{:03d}'.format(run_id_or_result_subdir)
        else:
            prefix = str(run_id_or_result_subdir)
        dirs = sorted(glob.glob(os.path.join(result_dir, searchdir, prefix + '-*')))
        dirs = [d for d in dirs if os.path.isdir(d)]
        if len(dirs) == 1:
            return dirs[0]
    print('Cannot locate result subdir for run: {}'.format(run_id_or_result_subdir))
    return None


def format_time(seconds):
    """
    Format seconds into desired format

    :param seconds: number of seconds
    :type seconds: float
    :return: formatted time
    :rtype: str
    """
    s = int(np.rint(seconds))
    if s < 60:
        return '{:d}s'.format(s)
    elif s < 60 * 60:
        return '{:d}m {:02d}s'.format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return '{:d}h {:02d}m {:02}ds'.format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return '{:d}d {:02d}h {:02d}m'.format(s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60)


# Model

def get_model_name(step):
    """
    Return filename of model snapshot by step

    :param step: global step of model
    :type step: int
    :return: model snapshot file name
    :rtype: str
    """
    return 'network-snapshot-{:06d}.pth'.format(step)


def get_best_model_name():
    """
    Return filename of best model snapshot by step

    :return: filename of best model snapshot
    :rtype: str
    """
    return 'network-snapshot-best.pth'


def get_last_model_name(result_subdir):
    """
    Return filename of best model snapshot by step

    :param result_subdir: path to result sub-directory
    :type result_subdir: str
    :return: filename of last model snapshot
    :rtype: str
    """
    latest = -1
    for f in os.listdir(result_subdir):
        if os.path.isfile(os.path.join(result_subdir, f)) and \
                re.search('network-snapshot-([\d]+).pth', f):
            f_step = int(re.findall('network-snapshot-([\d]+).pth', f)[0])
            if latest < f_step:
                latest = f_step

    return get_model_name(latest)


def save_model(result_subdir, step, graph, optimizer, seconds, is_best, criterion_dict=None):
    """
    Save model snapshot to result subdir

    :param result_subdir: path to result sub-directory
    :type result_subdir: str
    :param step: global step of model
    :type step: int
    :param graph: model graph
    :type graph: torch.nn.Module
    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :param seconds: seconds of running time
    :type seconds: float
    :param is_best: whether this model is best
    :type is_best: bool
    :param criterion_dict: dict of criterion
    :type criterion_dict: dict
    """
    # construct state
    state = {
        'step': step,
        # DataParallel wraps model in `module` attribute.
        'graph': graph.module.state_dict() if hasattr(graph, "module") else graph.state_dict(),
        'optimizer': optimizer.state_dict(),
        'criterion': {},
        'seconds': seconds
    }
    if criterion_dict is not None:
        state['criterion'] = {k: v.state_dict() for k, v in criterion_dict.items()}

    # save current state
    save_path = os.path.join(result_subdir, get_model_name(step))
    torch.save(state, save_path)

    # save best state
    if is_best:
        best_path = os.path.join(result_subdir, get_best_model_name())
        shutil.copy(save_path, best_path)


def load_model(result_subdir, step_or_model_path, graph, optimizer=None, criterion_dict=None, device=None):
    """
    lOad model snapshot from esult subdir

    :param result_subdir: path to result sub-directory
    :type result_subdir: str
    :param step_or_model_path: step or model path
    :type step_or_model_path: int or str
    :param graph: model graph
    :type graph: torch.nn.Module
    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :param criterion_dict: dict of criterion
    :type criterion_dict: dict
    :param device: device to run mode
    :type device: str
    :return: state
    :rtype: dict
    """
    # check existence of model file
    model_path = step_or_model_path
    if isinstance(step_or_model_path, int):
        model_path = get_model_name(step_or_model_path)
    if step_or_model_path == 'best':
        model_path = get_best_model_name()
    if step_or_model_path == 'latest':
        model_path = None
    if not os.path.exists(model_path):
        model_path = os.path.join(result_subdir, model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError('Failed to find model snapshot with {}'.format(step_or_model_path))

    # load model snapshot
    state = torch.load(model_path, map_location=device)
    step = state['step']
    graph.load_state_dict(state['graph'])
    graph.set_actnorm_inited()
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if criterion_dict is not None:
        for k in criterion_dict.keys():
            criterion_dict[k].load_state_dict(state['criterion'][k])
    print('Load model snapshot successfully from {}'.format(model_path))

    return state


# Dataset

def is_image(filepath):
    """
    Determine whether file is an image or not

    :param filepath: file path
    :type filepath: str
    :return: whether file is an image
    :rtype: bool
    """
    image_extensions = ['.png', '.jpg', '.jpeg']
    basename = os.path.basename(filepath)
    _, extension = os.path.splitext(basename)
    return extension.lower() in image_extensions


# Misc

def manual_seed(seed):
    """
    Set manual random seed

    :param seed: random seed
    :type seed: int
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
