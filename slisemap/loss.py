"""Module that used to contain the SLISEMAP loss functions."""

import torch

from slisemap.slisemap import make_loss, make_marginal_loss  # noqa: F401
from slisemap.utils import (
    _deprecated,
    _warn,  # noqa: F401
    softmax_column_kernel,  # noqa: F401
    softmax_row_kernel,
    squared_distance,  # noqa: F401
)


def softmax_kernel(D: torch.Tensor) -> torch.Tensor:
    """Use `softmax_row_kernel` instead."""
    _deprecated(softmax_kernel, softmax_row_kernel)
    return softmax_row_kernel(D)
