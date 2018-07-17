import os
import json

# General
verbose = False
root_dir = '.'
restore_dir = os.path.join(root_dir, 'snapshot')
log_dir = os.path.join(root_dir, 'log')
data_dir = os.path.join(root_dir, 'data')
profile_dir = os.path.join(root_dir, 'profile')

# Dataset hyper-parameters:
# problems = ['celeba', 'cifar10', 'mnist', 'imagenet', 'lsun']
problem = 'celeba'
# problem profile
profile = 'celebahq_256x256_5bit'
# data augmentation level ['none', 'standard', 'extra']
dal = 'standard'
# threads for parallel file reading
fmap = 1
# threads for parallel map
pmap = 1

# Optimization hyper-parameters:
# epoch size for training
n_train = 50000
# epoch size for validating
n_test = -1
# mini-batch size for training
n_batch_train = 64
# mini-batch size for validating
n_batch_test = 50
# mini-batch size for data-dependent initialization
n_batch_init = 256

# optimization method ['adam', 'adamax']
optimizer = "adamax"
# base learning rate
base_lr = 0.001
# adam beta1
beta1 = .9
# adam epsilon
eps = 1e-8
# number of averaging epochs for polyak and beta2
polyak_epochs = 1

# learning rate scheduler ['constant', 'noam', 'linear', 'step', 'cyclic' ]
lr_scheduler = 'constant'
lr_params = {
    'noam': {
        'warmup_steps': 4000
    },
    'linear': {
        'warmup_steps': 10
    },
    'step': {
        'anneal_rate': 0.98,
        'anneal_interval': 30000
    }
}
# weight decay. Switched off by default.
weight_decay = 1.,

# total number of training epochs
epochs = 1000000,
# epochs between valid
epochs_full_valid = 50
# whether to use memory saving gradients
gradient_checkpointing = True

# Model hyper-parameters:
# image size
image_size = -1
# anchor size for deciding batch size
anchor_size = 32
# width of hidden layers
width = 512
# depth of network
depth = 32
# weight of log p(y|x) in weighted loss
weight_y = 0.00
# number of bits of x
n_bits_x = 8
# number of levels
n_levels = 3

# Synthesis/Sampling hyper-parameters:
# mini-batch size for sampling
n_sample = 1
# epochs between full scale sample
epochs_full_sample = 50

# Ablation
# learn spatial prior
learn_top = False
# use y conditioning
y_cond = False
# LU decomposed
lu_decomposition = False
# random seed
seed = 0
# type of flow ['reverse', 'shuffle' , 'invconv']
flow_permutation = 'invconv'
# coupling type ['additive', 'affine']
flow_coupling = 'additive'
