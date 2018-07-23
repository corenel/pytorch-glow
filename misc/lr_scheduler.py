import numpy as np


def constant(base_lr, global_step):
    """
    Return constant learning rate

    :param base_lr: base learning rate
    :type base_lr: float
    :param global_step: global training steps
    :type global_step: int
    :return: scheduled learning rate
    :rtype: float
    """
    return base_lr


def noam_decay(base_lr, global_step, warmup_steps=4000, min_lr=1e-4):
    """
    Noam learning rate decay (from section 5.3 of Attention is all you need)

    :param base_lr: base learning rate
    :type base_lr: float
    :param global_step: global training steps
    :type global_step: int
    :param warmup_steps: number of steps for warming up
    :type warmup_steps: int
    :param min_lr: minimum learning rate
    :type min_lr: float
    :return: scheduled learning rate
    :rtype: float
    """
    step_num = global_step + 1.
    lr = base_lr * warmup_steps ** 0.5 * np.minimum(step_num ** -0.5, step_num * float(warmup_steps) ** -1.5)
    if global_step >= warmup_steps:
        lr = max(min_lr, lr)
    return lr


def linear_anneal(base_lr, global_step, num_train, warmup_steps=10):
    """
    Linearly annealed learning rate from 0 in the first warming up epochs.

    :param base_lr: base learning rate
    :type base_lr: float
    :param global_step: global training steps
    :type global_step: int
    :param num_train:
    :type num_train:
    :param warmup_steps: number of steps for warming up
    :type warmup_steps: int
    :return: scheduled learning rate
    :rtype: float
    """
    lr = base_lr * np.minimum(1., global_step / (num_train * warmup_steps))
    return lr


def step_anneal(base_lr, global_step,
                anneal_rate=0.98,
                anneal_interval=30000):
    """
    Annealing learning rate by steps

    :param base_lr: base learning rate
    :type base_lr: float
    :param global_step: global training steps
    :type global_step: int
    :param anneal_rate: rate of annealing
    :type anneal_rate: float
    :param anneal_interval: interval steps of annealing
    :type anneal_interval: int
    :return: scheduled learning rate
    :rtype: float
    """

    lr = base_lr * anneal_rate ** (global_step // anneal_interval)
    return lr


def cyclic_cosine_anneal(base_lr, global_step, t, m):
    """
    Cyclic cosine annealing (from section 3 of SNAPSHOT ENSEMBLES: TRAIN 1, GET M FOR FREE)

    :param base_lr: base learning rate
    :type base_lr: float
    :param global_step: global training steps
    :type global_step: int
    :param t: total number of epochs
    :type t: int
    :param m: number of ensembles we want
    :type m: int
    :return: scheduled learning rate
    :rtype: float
    """
    lr = (base_lr / 2.) * (np.cos(np.pi * ((global_step - 1) % (t // m)) / (t // m)) + 1.)
    return lr
