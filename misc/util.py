import os
import re
import cv2
import sys
import glob
import json
import shutil
import numpy as np
import torch

from PIL import Image
from easydict import EasyDict
from torchvision.transforms import transforms


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

def get_devices(devices, verbose=True):
    """
    Get devices for running model

    :param devices: list of devices from profile
    :type devices: list
    :param verbose: print log
    :type verbose: bool
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
        _print('[Builder] Incorrect device "{}"'.format(origin), verbose=verbose)
        return

    use_cpu = any([d.find('cpu') >= 0 for d in devices])
    use_cuda = any([(d.find('cuda') >= 0 or isinstance(d, int)) for d in devices])
    assert not (use_cpu and use_cuda), 'CPU and GPU cannot be mixed.'

    if use_cuda:
        devices = [parse_cuda_device(d) for d in devices]
        devices = [d for d in devices if d is not None]
        if len(devices) == 0:
            _print('[Builder] No available GPU found, use CPU only', verbose=verbose)
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
    print("[Builder] Saving results to {}".format(result_subdir))

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
    print('[Builder] Cannot locate result subdir for run: {}'.format(run_id_or_result_subdir))
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
    if isinstance(device, int):
        device = 'cuda:{}'.format(device)
    state = torch.load(model_path, map_location=device)
    step = state['step']
    graph.load_state_dict(state['graph'])
    graph.set_actnorm_inited()
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if criterion_dict is not None:
        for k in criterion_dict.keys():
            criterion_dict[k].load_state_dict(state['criterion'][k])
    print('[Builder] Load model snapshot successfully from {}'.format(model_path))

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


def tensor_to_ndarray(tensor):
    """
    Convert float tensor into numpy image

    :param tensor: input tensor
    :type tensor: torch.Tensor
    :return: numpy image
    :rtype: np.ndarray
    """
    tensor_np = tensor.permute(1, 2, 0).cpu().numpy()
    tensor_np = tensor_np.astype(np.float32)
    tensor_np = (tensor_np * 255).astype(np.uint8)
    return tensor_np


def tensor_to_pil(tensor):
    """
    Convert float tensor into PIL image

    :param tensor: input tensor
    :type tensor: torch.Tensor
    :return: PIL image
    :rtype: Image.Image
    """
    transform = transforms.ToPILImage()
    tensor = tensor.cpu()
    return transform(tensor)


def ndarray_to_tensor(img, shape=(64, 64, 3), bgr2rgb=True):
    """
    Convert numpy image to float tensor

    :param img: numpy image
    :type img: np.ndarray
    :param shape: image shape in (H, W, C)
    :type shape: tuple or list
    :param bgr2rgb: convert color space from BGR to RGB
    :type bgr2rgb: bool
    :return: tensor
    :rtype: torch.Tensor
    """
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (shape[0], shape[1]))
    img = (img / 255.0).astype(np.float32)
    img = torch.Tensor(img).permute(2, 0, 1)
    return img


def pil_to_tensor(img, shape=(64, 64, 3), transform=None):
    """
    Convert PIL image to float tensor

    :param img: PIL image
    :type img: Image.Image
    :param shape: image shape in (H, W, C)
    :type shape: tuple or list
    :param transform: image transform
    :return: tensor
    :rtype: torch.Tensor
    """
    if transform is None:
        transform = transforms.Compose((
            transforms.Resize(shape[0]),
            transforms.ToTensor()
        ))
    return transform(img)


def image_to_tensor(img, shape=(64, 64, 3), bgr2rgb=True):
    """
    Convert image to torch tensor

    :param img: image
    :type img: Image.Image or np.ndarray
    :param shape: image shape in (H, W, C)
    :type shape: tuple or list
    :param bgr2rgb: convert color space from BGR to RGB
    :type bgr2rgb: bool
    :return: image tensor
    :rtype: torch.Tensor
    """
    if isinstance(img, Image.Image):
        return pil_to_tensor(img, shape)
    if isinstance(np.ndarray, img):
        return ndarray_to_tensor(img, shape, bgr2rgb)
    else:
        raise NotImplementedError('Unsupported image type: {}'.format(type(img)))


def save_deltaz(deltaz, save_dir):
    """
    Save deltaz as numpy

    :param deltaz: delta vector of attributes in latent space
    :type deltaz: np.ndarray
    :param save_dir: directory to save
    :type save_dir: str
    """
    check_path(save_dir)
    np.save(os.path.join(save_dir, 'deltaz.npy'), deltaz)


def load_deltaz(path):
    """
    Load deltaz as numpy

    :param path: path to numpy file
    :type path: str
    :return: delta vector of attributes in latent space
    :rtype: np.ndarray
    """
    if os.path.exists(path):
        return np.load(path)


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


def _print(*args, verbose=True, **kwargs):
    """
    Print with condition

    :param verbose: whether to verbose or not
    :type verbose: bool
    """
    if verbose:
        print(*args, **kwargs)


def check_path(path):
    """
    Check existence of directory path. If not, then create it.

    :param path: path to directory
    :type path: str
    """
    if not os.path.exists(path):
        os.makedirs(path)


def make_batch(tensor, batch_size):
    """
    Generate fake batch

    :param tensor: input tensor
    :type tensor: torch.Tensor
    :param batch_size: batch size
    :type batch_size: int
    :return: fake batch
    :rtype: torch.Tensor
    """
    assert len(tensor.shape) == 3, 'Assume 3D input tensor'
    return tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)


def make_interpolation_vector(num_classes, step=0.25,
                              minimum=-1., maximum=1.):
    """
    Generate interpolation vector

    :param num_classes: number of classes
    :type num_classes: int
    :param step: increasing step
    :type step: float
    :param minimum: minimum value
    :type minimum: float
    :param maximum: maximum value
    :type maximum: float
    :return: interpolation vector
    :rtype: np.ndarray
    """
    num_levels = int((maximum - minimum) / step) + 1
    levels = [-1. + step * i for i in range(num_levels)]

    interpolation_vector = np.zeros([num_classes, num_levels, num_classes])
    for cls in range(num_classes):
        for lv in range(num_levels):
            interpolation_vector[cls, lv, cls] = levels[lv]

    return interpolation_vector
