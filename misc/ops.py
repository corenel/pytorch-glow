import torch


def reduce_mean(tensor, dim=None, keepdim=False, out=None):
    """
    Returns the mean value of each row of the input tensor in the given dimension dim.

    Support multi-dim mean

    :param tensor: the input tensor
    :type tensor: torch.Tensor
    :param dim: the dimension to reduce
    :type dim: int or list[int]
    :param keepdim: whether the output tensor has dim retained or not
    :type keepdim: bool
    :param out: the output tensor
    :type out: torch.Tensor
    :return: mean result
    :rtype: torch.Tensor
    """
    # mean all dims
    if dim is None:
        return torch.mean(tensor)
    # prepare dim
    if isinstance(dim, int):
        dim = [dim]
    dim = sorted(dim)
    # get mean dim by dim
    for d in dim:
        tensor = tensor.mean(dim=d, keepdim=True)
    # squeeze reduced dimensions if not keeping dim
    if not keepdim:
        for cnt, d in enumerate(dim):
            tensor.squeeze_(d - cnt)
    if out is not None:
        out.copy_(tensor)
    return tensor


def reduce_sum(tensor, dim=None, keepdim=False, out=None):
    """
    Returns the sum of all elements in the input tensor.

    Support multi-dim sum

    :param tensor: the input tensor
    :type tensor: torch.Tensor
    :param dim: the dimension to reduce
    :type dim: int or list[int]
    :param keepdim: whether the output tensor has dim retained or not
    :type keepdim: bool
    :param out: the output tensor
    :type out: torch.Tensor
    :return: sum result
    :rtype: torch.Tensor
    """
    # summarize all dims
    if dim is None:
        return torch.sum(tensor)
    # prepare dim
    if isinstance(dim, int):
        dim = [dim]
    dim = sorted(dim)
    # get summary dim by dim
    for d in dim:
        tensor = tensor.sum(dim=d, keepdim=True)
    # squeeze reduced dimensions if not keeping dim
    if not keepdim:
        for cnt, d in enumerate(dim):
            tensor.squeeze_(d - cnt)
    if out is not None:
        out.copy_(tensor)
    return tensor


def tensor_equal(a, b, eps=1e-6):
    """
    Compare two tensors

    :param a: input tensor a
    :type a: torch.Tensor
    :param b: input tensor b
    :type b: torch.Tensor
    :param eps: epsilon
    :type eps: float
    :return: whether two tensors are equal
    :rtype: bool
    """
    if a.shape != b.shape:
        return False

    return 0 <= float(torch.max(torch.abs(a - b))) <= eps


def split_channel(tensor, split_type='simple'):
    """
    Split channels of tensor

    :param tensor: input tensor
    :type tensor: torch.Tensor
    :param split_type: type of splitting
    :type split_type: str
    :return: split tensor
    :rtype: tuple(torch.Tensor, torch.Tensor)
    """
    assert len(tensor.shape) == 4
    assert split_type in ['simple', 'cross']

    nc = tensor.shape[1]
    if split_type == 'simple':
        return tensor[:, :nc // 2, ...], tensor[:, nc // 2:, ...]
    elif split_type == 'cross':
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def cat_channel(a, b):
    """
    Concatenates channels of tensors

    :param a: input tensor a
    :type a: torch.Tensor
    :param b: input tensor b
    :type b: torch.Tensor
    :return: concatenated tensor
    :rtype: torch.Tensor
    """
    return torch.cat((a, b), dim=1)


def count_pixels(tensor):
    """
    Count number of pixels in given tensor

    :param tensor: input tensor
    :type tensor: torch.Tensor
    :return: number of pixels
    :rtype: int
    """
    assert len(tensor.shape) == 4
    return int(tensor.shape[2] * tensor.shape[3])


def onehot(y, num_classes):
    """
    Generate one-hot vector

    :param y: ground truth labels
    :type y: torch.Tensor
    :param num_classes: number os classes
    :type num_classes: int
    :return: one-hot vector generated from labels
    :rtype: torch.Tensor
    """
    assert len(y.shape) in [1, 2], "Label y should be 1D or 2D vector"
    y_onehot = torch.zeros(y.shape[0], num_classes, device=y.device)
    if len(y.shape) == 1:
        y_onehot = y_onehot.scatter_(1, y.unsqueeze(-1), 1)
    else:
        y_onehot = y_onehot.scatter_(1, y, 1)
    return y_onehot
