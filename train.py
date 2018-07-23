from torchvision import transforms

from misc import util
from network import Builder, Trainer
from dataset import CelebA

if __name__ == '__main__':
    # initialize logging
    util.init_output_logging()
    # load hyper-parameters
    hps = util.load_profile('profile/celeba.json')
    util.manual_seed(hps.ablation.seed)
    # build graph
    builder = Builder(hps)
    state = builder.build()
    # load dataset
    dataset = CelebA(root=hps.dataset.root,
                     transform=transforms.Compose((
                         transforms.CenterCrop(160),
                         transforms.Resize(64),
                         transforms.ToTensor()
                     )))
    # start training
    trainer = Trainer(hps=hps, dataset=dataset, **state)
    trainer.train()
