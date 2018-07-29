import cv2
import sys
import signal
import argparse
import torch

from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid

from misc import util
from network import Builder, Inferer
from dataset import CelebA


def parse_args():
    parser = argparse.ArgumentParser(
        'PyTorch implementation of "Glow: Generative Flow with Invertible 1x1 Convolutions"')
    parser.add_argument('--profile', '-p', type=str,
                        default='profile/celeba.json',
                        help='path to profile file')
    parser.add_argument('--weights', '-w', type=str,
                        default=None,
                        help='path to pre-trained weights')
    parser.add_argument('--delta', '-d', type=str,
                        default=None,
                        help='path to delta file')
    return parser.parse_args()


if __name__ == '__main__':
    # this enables a Ctrl-C without triggering errors
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    # parse arguments
    args = parse_args()

    # initialize logging
    util.init_output_logging()

    # load hyper-parameters
    hps = util.load_profile(args.profile)
    util.manual_seed(hps.ablation.seed)
    if args.weights is not None:
        hps.general.warm_start = True
        hps.general.pre_trained = args.weights

    # build graph
    builder = Builder(hps)
    state = builder.build(training=False)

    # load dataset
    dataset = CelebA(root=hps.dataset.root,
                     transform=transforms.Compose((
                         transforms.CenterCrop(160),
                         transforms.Resize(64),
                         transforms.ToTensor()
                     )))

    # start inference
    inferer = Inferer(
        hps=hps,
        graph=state['graph'],
        devices=state['devices'],
        data_device=state['data_device']
    )

    # 0. sample
    # img = inferer.sample(z=None, y_onehot=None, eps_std=0.5)
    # img = Image.fromarray(img, 'RGB')
    # img.save('sample.png')

    # 1. compute deltaz
    # deltaz = inferer.compute_attribute_delta(dataset)
    # util.save_deltaz(deltaz, '.')

    # 2. encode & decode
    # img = Image.open('misc/test_celeba.png').convert('RGB')
    # z = inferer.encode(img)
    # img = util.tensor_to_pil(inferer.decode(z))
    # img.save('reconstructed.png')

    # 3. apply delta
    # img = Image.open('misc/test_celeba.png').convert('RGB')
    # deltaz = util.load_deltaz('deltaz.npy')
    # interpolation = [0.] * hps.dataset.num_classes
    # interpolation[0] = 1.
    # img_interpolated = inferer.apply_attribute_delta(img, deltaz, interpolation)
    # img_interpolated = util.tensor_to_pil(img_interpolated)
    # img_interpolated.save('interpolated.png')

    # 4. batch apply
    interpolation_vector = util.make_interpolation_vector(hps.dataset.num_classes)
    img = Image.open('misc/test_celeba.png').convert('RGB')
    deltaz = util.load_deltaz('deltaz.npy')
    util.check_path('interpolation')
    for cls in range(interpolation_vector.shape[0]):
        imgs_interpolated = []
        for lv in range(interpolation_vector.shape[1]):
            img_interpolated = inferer.apply_attribute_delta(
                img, deltaz,
                interpolation_vector[cls, lv, :])
            imgs_interpolated.append(img_interpolated)
            # img_interpolated = util.tensor_to_pil(img_interpolated)
            # img_interpolated.save('interpolation/interpolated_{:s}_{:0.2f}.png'.format(
            #     dataset.attrs[cls],
            #     interpolation_vector[cls, lv, cls]))
        imgs_stacked = torch.stack(imgs_interpolated)
        imgs_grid = make_grid(imgs_stacked, nrow=interpolation_vector.shape[1])
        imgs = util.tensor_to_pil(imgs_grid)
        imgs.save('interpolation/interpolated_{:s}.png'.format(dataset.attrs[cls]))
