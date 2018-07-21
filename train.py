from misc import util
from network.builder import Builder

if __name__ == '__main__':
    hps = util.load_profile('profile/test.json')
    builder = Builder(hps)
    state = builder.build()
