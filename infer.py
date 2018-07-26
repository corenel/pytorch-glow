import sys
import signal
import argparse

from PIL import Image

from misc import util
from network import Builder, Inferer


def parse_args():
    parser = argparse.ArgumentParser(
        'PyTorch implementation of "Glow: Generative Flow with Invertible 1x1 Convolutions"')
    parser.add_argument('--profile', '-p', type=str,
                        default='profile/celeba.json',
                        help='path to profile file')
    parser.add_argument('--weights', '-w', type=str,
                        default=None,
                        help='path to pre-trained weights')
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

    # start inference
    inferer = Inferer(
        hps=hps,
        graph=state['graph'],
        devices=state['devices'],
        data_device=state['data_device']
    )
    img = inferer.sample(z=None, y_onehot=None, eps_std=0.1)
    img.show()
