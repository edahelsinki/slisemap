"""
This module used to contain the SLISEMAP loss functions.
"""

import torch

from slisemap.slisemap import make_loss, make_marginal_loss
from slisemap.utils import (
    _deprecated,
    _warn,
    softmax_column_kernel,
    softmax_row_kernel,
    squared_distance,
)


def softmax_kernel(D: torch.Tensor) -> torch.Tensor:
    _deprecated(softmax_kernel, softmax_row_kernel)
    return softmax_row_kernel(D)
