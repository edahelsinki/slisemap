"""
This module contains the built-in alternatives for local white box models.
These can also be used as templates for implementing your own.
"""

from typing import Optional, Union

import numpy as np
import torch
from torch.nn.functional import softmax

from slisemap.utils import _assert


def linear_regression(X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Prediction function for linear regression.

    Args:
        X (torch.Tensor): Data matrix [n_x, m].
        B (torch.Tensor): Coefficient Matrix [n_b, m].

    Returns:
        torch.Tensor: Prediction tensor [n_b, n_x, 1]
    """
    return (B @ X.T)[:, :, None]


def multiple_linear_regression(X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Prediction function for multiple linear regression.

    Args:
        X (torch.Tensor): Data matrix [n_x, m].
        B (torch.Tensor): Coefficient Matrix [n_b, m*p].

    Returns:
        torch.Tensor: Prediction tensor [n_b, n_x, p]
    """
    n_x, m = X.shape
    n_b, o = B.shape
    p = torch.div(o, m, rounding_mode="trunc")
    a = torch.empty([n_b, n_x, p], device=B.device, dtype=B.dtype)
    for i in range(p):
        a[:, :, i] = B[:, (i * m) : ((i + 1) * m)] @ X.T
    return a


def linear_regression_loss(
    Ytilde: torch.Tensor, Y: torch.Tensor, B: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Least squares loss function for (multiple) linear regresson.

    Args:
        Ytilde (torch.Tensor): Predicted values [n_b, n_x, p].
        Y (torch.Tensor): Ground truth values [n_x, p].
        B (Optional[torch.Tensor], optional): Coefficient matrix (not used, the regularisation is part of Slisemap). Defaults to None.

    Returns:
        torch.Tensor: Loss values [n_b, n_x].
    """
    return ((Ytilde - Y.expand(Ytilde.shape)) ** 2).sum(dim=-1)


def linear_regression_coefficients(
    X: Union[torch.Tensor, np.ndarray],
    Y: Union[torch.Tensor, np.ndarray],
    intercept: bool = False,
) -> int:
    """Get the number of coefficients for a (multiple) linear regression.

    Args:
        X (Union[torch.Tensor, np.ndarray]): Data matrix.
        Y (Union[torch.Tensor, np.ndarray]): Target matrix.
        intercept (bool, optional): Add an (additional) intercept to X. Defaults to False.

    Returns:
        int: Number of coefficients (columns of B).
    """
    return (X.shape[1] + intercept) * (1 if len(Y.shape) < 2 else Y.shape[1])


def logistic_regression(X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Prediction function for (multinomial) logistic regression.
    Note that the number of coefficients is `m * (p-1)` due to the normalisation of softmax.

    Args:
        X (torch.Tensor): Data matrix [n_x, m].
        B (torch.Tensor): Coefficient Matrix [n_b, m*(p-1)].

    Returns:
        torch.Tensor: Prediction tensor [n_b, n_x, p]
    """
    n_x, m = X.shape
    n_b, o = B.shape
    p = 1 + torch.div(o, m, rounding_mode="trunc")
    a = torch.zeros([n_b, n_x, p], device=B.device, dtype=B.dtype)
    for i in range(p - 1):
        a[:, :, i] = B[:, (i * m) : ((i + 1) * m)] @ X.T
    return softmax(a, 2)


def logistic_regression_loss(
    Ytilde: torch.Tensor, Y: torch.Tensor, B: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Squared Hellinger distance function for (multinomial) logistic regression.

    Args:
        Ytilde (torch.Tensor): Predicted values [n_b, n_x, p].
        Y (torch.Tensor): Ground truth values [n_x, p].
        B (Optional[torch.Tensor], optional): Coefficient matrix (not used, the regularisation is part of Slisemap). Defaults to None.

    Returns:
        torch.Tensor: Loss values [n_b, n_x].
    """
    return ((Ytilde.sqrt() - Y.sqrt().expand(Ytilde.shape)) ** 2).sum(dim=-1) * 0.5


def logistic_regression_coefficients(
    X: Union[torch.Tensor, np.ndarray],
    Y: Union[torch.Tensor, np.ndarray],
    intercept: bool = False,
) -> int:
    """Get the number of coefficients for a (multinomial) logistic regression.

    Args:
        X (Union[torch.Tensor, np.ndarray]): Data matrix.
        Y (Union[torch.Tensor, np.ndarray]): Target matrix.
        intercept (bool, optional): Add an (additional) intercept to X. Defaults to False.

    Returns:
        int: Number of coefficients (columns of B).
    """
    _assert(
        len(Y.shape) > 1 and Y.shape[1] > 1,
        "Logistic regression requires Y:s with multiple classes",
    )
    return (X.shape[1] + intercept) * (Y.shape[1] - 1)


def logistic_regression_log(X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Prediction function for (multinomial) logistic regression that returns the **log of the prediction**.
    Note that the number of coefficients is `m * (p-1)` due to the normalisation of softmax.

    Args:
        X (torch.Tensor): Data matrix [n_x, m].
        B (torch.Tensor): Coefficient Matrix [n_b, m*(p-1)].

    Returns:
        torch.Tensor: Prediction tensor [n_b, n_x, p]
    """
    n_x, m = X.shape
    n_b, o = B.shape
    p = 1 + torch.div(o, m, rounding_mode="trunc")
    a = torch.zeros([n_b, n_x, p], device=B.device, dtype=B.dtype)
    for i in range(p - 1):
        a[:, :, i] = B[:, (i * m) : ((i + 1) * m)] @ X.T
    return a - torch.logsumexp(a, 2, True)


def logistic_regression_log_loss(
    Ytilde: torch.Tensor, Y: torch.Tensor, B: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Cross entropy loss function for (multinomial) logistic regression.
    Note that this loss function expects ``Ytilde`` to be the **log of the predicted probabilities**.

    Args:
        Ytilde (torch.Tensor): Predicted logits [n_b, n_x, p].
        Y (torch.Tensor): Ground truth values [n_x, p].
        B (Optional[torch.Tensor], optional): Coefficient matrix (not used, the regularisation is part of Slisemap). Defaults to None.

    Returns:
        torch.Tensor: Loss values [n_b, n_x].
    """
    return torch.sum(-Y * Ytilde - (1 - Y) * torch.log1p(-torch.exp(Ytilde)), -1)
