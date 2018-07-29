import os
import sys
import click
import signal
import torch

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import make_grid

from misc import util
from network import Builder, Inferer
from dataset import CelebA


@click.group(name='Inference for glow model')
@click.option('--profile', type=click.Path(exists=True))
@click.option('--snapshot', type=click.Path(exists=True))
@click.pass_context
def cli(ctx, profile, snapshot):
    # load hyper-parameters
    hps = util.load_profile(profile)
    util.manual_seed(hps.ablation.seed)
    if snapshot is not None:
        hps.general.warm_start = True
        hps.general.pre_trained = snapshot

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
    ctx.obj['hps'] = hps
    ctx.obj['dataset'] = dataset
    ctx.obj['inferer'] = inferer


@cli.command()
@click.pass_context
def sample(ctx):
    hps = ctx.obj['hps']
    inferer = ctx.obj['inferer']
    # smaple
    img = inferer.sample(z=None, y_onehot=None, eps_std=0.5)
    # save result
    result_subdir = util.create_result_subdir(hps.general.result_dir,
                                              desc='sample',
                                              profile=hps)
    util.tensor_to_pil(img).save(os.path.join(result_subdir, 'sample.png'))


@cli.command()
@click.pass_context
def compute_deltaz(ctx):
    hps = ctx.obj['hps']
    inferer = ctx.obj['inferer']
    dataset = ctx.obj['dataset']

    # compute delta
    deltaz = inferer.compute_attribute_delta(dataset)

    # save result
    result_subdir = util.create_result_subdir(hps.general.result_dir,
                                              desc='deltaz',
                                              profile=hps)
    util.save_deltaz(deltaz, result_subdir)


@cli.command()
@click.argument('image', type=click.Path(exists=True))
@click.pass_context
def reconstruct(ctx, image):
    hps = ctx.obj['hps']
    inferer = ctx.obj['inferer']

    # get image list
    img_list = []
    if os.path.isfile(image) and util.is_image(image):
        img_list = [image]
    elif os.path.isdir(image):
        img_list = [f for f in os.listdir(image)
                    if util.is_image(os.path.join(image, f))]

    # reconstruct images
    img_grid_list = []
    util.check_path('reconstructed')
    for img_path in img_list:
        img = Image.open(img_path).convert('RGB')
        x = util.pil_to_tensor(img)
        z = inferer.encode(img)
        x_ = inferer.decode(z)
        img_grid = torch.cat((x, x_.cpu()), dim=1)
        img_grid_list.append(img_grid)
        # util.tensor_to_pil(img_grid).save('reconstructed/{}'.format(os.path.basename(img_path)))

    # generate grid of reconstructed images
    imgs_grid = make_grid(torch.stack(img_grid_list))

    # save result
    result_subdir = util.create_result_subdir(hps.general.result_dir,
                                              desc='reconstruct',
                                              profile=hps)
    util.tensor_to_pil(imgs_grid).save(os.path.join(result_subdir, 'grid.png'))


@cli.command()
@click.argument('delta_file', type=click.Path(exists=True))
@click.argument('image_file', type=click.Path(exists=True, dir_okay=False))
@click.option('--batch', is_flag=True, default=True)
@click.pass_context
def interpolate(ctx, delta_file, image_file, batch):
    hps = ctx.obj['hps']
    inferer = ctx.obj['inferer']
    dataset = ctx.obj['dataset']

    img = Image.open(image_file).convert('RGB')
    deltaz = util.load_deltaz(delta_file)
    result_subdir = util.create_result_subdir(hps.general.result_dir,
                                              desc='interpolation',
                                              profile=hps)

    if batch:
        interpolation_vector = util.make_interpolation_vector(hps.dataset.num_classes)
        for cls in range(interpolation_vector.shape[0]):
            print('[Inferer] interpolating class "{}"'.format(dataset.attrs[cls]))
            imgs_interpolated = []
            progress = tqdm(range(interpolation_vector.shape[1]))
            for lv in progress:
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
            imgs.save(os.path.join(result_subdir,
                                   'interpolated_{:s}.png'.format(dataset.attrs[cls])))
    else:
        interpolation = [0.] * hps.dataset.num_classes
        interpolation[0] = 1.
        img_interpolated = inferer.apply_attribute_delta(img, deltaz, interpolation)
        img_interpolated = util.tensor_to_pil(img_interpolated)
        img_interpolated.save(os.path.join(result_subdir,
                                           'interpolated.png'))


if __name__ == '__main__':
    # this enables a Ctrl-C without triggering errors
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    # initialize logging
    util.init_output_logging()

    # command group
    cli(obj={})
