from misc import util
from network.builder import Builder
from network.trainer import Trainer

if __name__ == '__main__':
    hps = util.load_profile('profile/test.json')
    builder = Builder(hps)
    state = builder.build()
