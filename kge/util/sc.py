"""
Assorted utilities for working with neural networks by Sanxing Chen (sc3hn@virginia.edu)
"""

import os
import torch
from torch import nn
from torch.nn import functional as F

import random
import numpy as np

from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``.

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_seed_from_env():
    seed_default = int(os.environ.get("TKGE_SEED", -1))
    seed_python = int(os.environ.get("TKGE_SEED_PYTHON", -1))
    seed_numpy = int(os.environ.get("TKGE_SEED_NUMPY", -1))
    seed_torch = int(os.environ.get("TKGE_SEED_TORCH", -1))
    seed_cuda = int(os.environ.get("TKGE_SEED_CUDA", -1))
    if seed_default < 0:
        seed_default = random.randint(0, 10000)
    if seed_python >= 0:
        random.seed(seed_python)
    else:
        random.seed(seed_default)
    if seed_numpy >= 0:
        np.random.seed(seed_numpy)
    else:
        np.random.seed(seed_default)
    if seed_torch >= 0:
        torch.manual_seed(seed_torch)
    else:
        torch.manual_seed(seed_default)
    if seed_cuda >= 0:
        torch.cuda.manual_seed(seed_cuda)
        torch.cuda.manual_seed_all(seed_cuda)
    else:
        torch.cuda.manual_seed(seed_default)
        torch.cuda.manual_seed_all(seed_default)


def pad_seq_of_seq(
    sequence: List[List],
    default_value: Callable[[], Any] = lambda: 0,
    padding_on_right: bool = True,
) -> Tuple[List[List], List]:
    lens = [len(i) for i in sequence]
    desired_length = max(lens)
    if padding_on_right:
        return [i + [default_value()] * (desired_length - len(i)) for i in sequence], lens
    else:
        return [[default_value()] * (desired_length - len(i)) + i for i in sequence], lens


def get_mask_from_sequence_lengths(sequence_lengths: torch.Tensor, max_length: int) -> torch.BoolTensor:
    """
    Given a variable of shape `(batch_size,)` that represents the sequence lengths of each batch
    element, this function returns a `(batch_size, max_length)` mask variable.  For example, if
    our input was `[2, 2, 3]`, with a `max_length` of 4, we'd return
    `[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]`.
    We require `max_length` here instead of just computing it from the input `sequence_lengths`
    because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
    that we can use it to construct a new tensor.
    Based on https://github.com/allenai/allennlp #5ad7a33
    """
    # (batch_size, max_length)
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return sequence_lengths.unsqueeze(1) >= range_tensor


def get_randperm_from_lengths(sequence_lengths: torch.Tensor, max_length: int):
    rand_vector = sequence_lengths.new_empty((len(sequence_lengths), max_length), dtype=torch.float).uniform_(0.1, 1)
    rand_vector.masked_fill_(~get_mask_from_sequence_lengths(sequence_lengths, max_length), 0)
    perm_vector = rand_vector.argsort(descending=True)
    return perm_vector


def get_bernoulli_mask(shape, prob, device):
    return torch.bernoulli(torch.full(shape, prob, device=device)).bool()
