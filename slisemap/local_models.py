"""Module that contains the built-in alternatives for local white box models.

These functions can also be used as templates for implementing your own.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch.nn.functional import softmax

from slisemap.utils import CallableLike, _assert, _assert_no_trace, _deprecated, _warn


def local_predict(
    X: torch.Tensor,
    B: torch.Tensor,
    local_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Get individual predictions when every data item has a separate model.

    Args:
        X: Data matrix [n, m].
        B: Coefficient matrix [n, q].
        local_model: Prediction function: [1, m], [1, q] -> [1, 1, o].

    Returns:
        Matrix of local predictions [n, o].
    """
    n = X.shape[0]
    _assert(n == B.shape[0], "X and B must have the same number of rows", local_predict)
    y = local_model(X[:1, :], B[:1, :])[0, 0, ...]
    Y = torch.empty((n, *y.shape), dtype=y.dtype, device=y.device)
    Y[0, ...] = y
    for i in range(1, n):
        Y[i, ...] = local_model(X[i : i + 1, :], B[i : i + 1, :])[0, 0, ...]
    return Y


class ALocalModel(ABC):
    """Abstract class for gathering all the functions needed for a local model (predict, loss, coefficients)."""

    @staticmethod
    @abstractmethod
    def predict(X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Prediction function.

        Args:
            X: Data matrix.
            B: Coefficient matrix.

        Returns:
            Y: Prediction matrix.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def loss(Ytilde: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Loss function.

        Args:
            Ytilde: Prediction matrix.
            Y: Target matrix

        Returns:
            L: Loss matrix.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def coefficients(
        X: Union[torch.Tensor, np.ndarray],
        Y: Union[torch.Tensor, np.ndarray],
        intercept: bool,
    ) -> int:
        """Get for the number of columns of B.

        Args:
            X: Data matrix.
            Y: Target matrix.
            intercept: Add intercept.

        Returns:
            Number of columns.
        """
        raise NotImplementedError

    @staticmethod
    def regularisation(
        X: torch.Tensor,
        Y: torch.Tensor,
        B: torch.Tensor,
        Z: torch.Tensor,
        Ytilde: torch.Tensor,
    ) -> torch.Tensor:
        """Regularisation function.

        Args:
            X: Data matrix.
            Y: Target matrix.
            B: Coefficient matrix.
            Z: Embedding matrix.
            Ytilde: Prediction matrix.

        Returns:
            Additional loss term.
        """
        return 0.0


def linear_regression(X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Prediction function for (multiple) linear regression.

    Args:
        X: Data matrix [n_x, m].
        B: Coefficient Matrix [n_b, m * p].

    Returns:
        Prediction tensor [n_b, n_x, p]
    """
    # return (B @ X.T)[:, :, None] # Only for single linear regression
    return (B.view(B.shape[0], -1, X.shape[1]) @ X.T).transpose(1, 2)


def multiple_linear_regression(X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Prediction function for multiple linear regression. **DEPRECATED**.

    Args:
        X: Data matrix [n_x, m].
        B: Coefficient Matrix [n_b, m*p].

    Returns:
        Prediction tensor [n_b, n_x, p]

    Deprecated:
        1.4: In favour of a combined `linear_regression`
    """
    _deprecated(multiple_linear_regression, linear_regression)
    return linear_regression(X, B)


def linear_regression_loss(
    Ytilde: torch.Tensor, Y: torch.Tensor, B: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Least squares loss function for (multiple) linear regresson.

    Args:
        Ytilde: Predicted values [n_b, n_x, p].
        Y: Ground truth values [n_x, p].
        B: Coefficient matrix. **Deprecated**. Defaults to None.

    Returns:
        Loss values [n_b, n_x].

    Deprecated:
        1.6: B
    """
    return ((Ytilde - Y.expand(Ytilde.shape)) ** 2).sum(dim=-1)


def linear_regression_coefficients(
    X: Union[torch.Tensor, np.ndarray],
    Y: Union[torch.Tensor, np.ndarray],
    intercept: bool = False,
) -> int:
    """Get the number of coefficients for a (multiple) linear regression.

    Args:
        X: Data matrix.
        Y: Target matrix.
        intercept: Add an (additional) intercept to X. Defaults to False.

    Returns:
        Number of coefficients (columns of B).
    """
    return (X.shape[1] + intercept) * (1 if len(Y.shape) < 2 else Y.shape[1])


class LinearRegression(ALocalModel):
    """A class that contains all the functions needed for linear regression."""

    predict = linear_regression
    loss = linear_regression_loss
    coefficients = linear_regression_coefficients


def absolute_error(
    Ytilde: torch.Tensor, Y: torch.Tensor, B: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Absolute error function for (multiple) linear regresson.

    Args:
        Ytilde: Predicted values [n_b, n_x, p].
        Y: Ground truth values [n_x, p].
        B: Coefficient matrix. **Deprecated**. Defaults to None.

    Returns:
        Loss values [n_b, n_x].

    Deprecated:
        1.6: B
    """
    return torch.abs(Ytilde - Y.expand(Ytilde.shape)).sum(dim=-1)


class LinearAbsoluteRegression(ALocalModel):
    """A class that contains all the functions needed for linear regression with absolute errors."""

    predict = linear_regression
    loss = absolute_error
    coefficients = linear_regression_coefficients


def logistic_regression(X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Prediction function for (multinomial) logistic regression.

    Note that the number of coefficients is `m * (p-1)` due to the normalisation of softmax.

    Args:
        X: Data matrix [n_x, m].
        B: Coefficient Matrix [n_b, m*(p-1)].

    Returns:
        Prediction tensor [n_b, n_x, p]
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
        Ytilde: Predicted values [n_b, n_x, p].
        Y: Ground truth values [n_x, p].
        B: Coefficient matrix. **Deprecated**. Defaults to None.

    Returns:
        Loss values [n_b, n_x].

    Deprecated:
        1.6: B
    """
    _assert_no_trace(
        lambda: (
            Ytilde.shape[-1] <= Y.shape[-1],
            f"Too few columns in Y: {Y.shape[-1]} < {Ytilde.shape[-1]}",
        ),
        logistic_regression_loss,
    )
    return ((Ytilde.sqrt() - Y.sqrt().expand(Ytilde.shape)) ** 2).sum(dim=-1) * 0.5


def logistic_regression_coefficients(
    X: Union[torch.Tensor, np.ndarray],
    Y: Union[torch.Tensor, np.ndarray],
    intercept: bool = False,
) -> int:
    """Get the number of coefficients for a (multinomial) logistic regression.

    Args:
        X: Data matrix.
        Y: Target matrix.
        intercept: Add an (additional) intercept to X. Defaults to False.

    Returns:
        Number of coefficients (columns of B).
    """
    return (X.shape[1] + intercept) * max(1, Y.shape[1] - 1)


class LogisticRegression(ALocalModel):
    """A class that contains all the functions needed for logistic regression."""

    predict = logistic_regression
    loss = logistic_regression_loss
    coefficients = logistic_regression_coefficients


def logistic_regression_log(X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Prediction function for (multinomial) logistic regression that returns the **log of the prediction**.

    Note that the number of coefficients is `m * (p-1)` due to the normalisation of softmax.

    Args:
        X: Data matrix [n_x, m].
        B: Coefficient Matrix [n_b, m*(p-1)].

    Returns:
        Prediction tensor [n_b, n_x, p]
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

    Note that this loss function expects `Ytilde` to be the **log of the predicted probabilities**.

    Args:
        Ytilde: Predicted logits [n_b, n_x, p].
        Y: Ground truth values [n_x, p].
        B: Coefficient matrix. **Deprecated**. Defaults to None.

    Returns:
        Loss values [n_b, n_x].

    Deprecated:
        1.6: B
    """
    _assert_no_trace(
        lambda: (
            Ytilde.shape[-1] <= Y.shape[-1],
            f"Too few columns in Y: {Y.shape[-1]} < {Ytilde.shape[-1]}",
        ),
        logistic_regression_loss,
    )
    return torch.sum(-Y * Ytilde - (1 - Y) * torch.log1p(-torch.exp(Ytilde)), -1)


class LogisticLogRegression(ALocalModel):
    """A class that contains all the functions needed for logistic regression.

    The predictions are in log-space rather than probabilities for numerical stability.
    """

    predict = logistic_regression_log
    loss = logistic_regression_log_loss
    coefficients = logistic_regression_coefficients


LocalModelCollection = Union[
    ALocalModel,
    Type[ALocalModel],
    Tuple[
        CallableLike[ALocalModel.predict],
        CallableLike[ALocalModel.loss],
        Union[None, int, CallableLike[ALocalModel.coefficients]],
        Union[None, CallableLike[ALocalModel.regularisation]],
    ],
    linear_regression,
    logistic_regression,
    logistic_regression_log,
]


def identify_local_model(
    local_model: Union[LocalModelCollection, CallableLike[ALocalModel.predict]],
    local_loss: Optional[CallableLike[ALocalModel.loss]] = None,
    coefficients: Union[None, int, CallableLike[ALocalModel.coefficients]] = None,
    regularisation: Union[None, CallableLike[ALocalModel.regularisation]] = None,
) -> Tuple[Callable, Callable, Callable, Callable]:
    """Identify the "predict", "loss", and "coefficients" functions for a local model.

    Args:
        local_model: A instance/subclass of `ALocalModel`, a predict function, or a tuple of functions.
        local_loss: A loss function or None if it is part of `local_model`. Defaults to None.
        coefficients: The number of coefficients, or a function giving that number, or None if it is part of `local_model`. Defaults to None.
        regularisation: Additional regularisation function. Defaults to None.

    Returns:
        predict: "prediction" function (takes X and B and returns predicted Y for every X and B combination).
        loss: "loss" function (takes predicted Y and real Y and returns the loss).
        coefficients: "coefficients" function (takes X and Y and returns the number of coefficients for B).
        regularisation: "regularisation" function (takes X, Y, B, Z, and, Ytilde and returns an additional loss scalar).
    """
    pred_fn = None
    loss_fn = None
    coef_fn = linear_regression_coefficients
    regu_fn = ALocalModel.regularisation
    if isinstance(local_model, ALocalModel) or (
        isinstance(local_model, type) and issubclass(local_model, ALocalModel)
    ):
        pred_fn = local_model.predict
        loss_fn = local_model.loss
        coef_fn = local_model.coefficients
        regu_fn = local_model.regularisation
    elif callable(local_model):
        pred_fn = local_model
        if local_model in (linear_regression, multiple_linear_regression):
            loss_fn = linear_regression_loss
            coef_fn = linear_regression_coefficients
        elif local_model == logistic_regression:
            loss_fn = logistic_regression_loss
            coef_fn = logistic_regression_coefficients
        elif local_model == logistic_regression_log:
            loss_fn = logistic_regression_log_loss
            coef_fn = logistic_regression_coefficients
        else:
            loss_fn = local_loss
            coef_fn = coefficients
    elif isinstance(local_model, Sequence):
        pred_fn = local_model[0]
        loss_fn = local_model[1] if len(local_model) > 1 else loss_fn
        coef_fn = local_model[2] if len(local_model) > 2 else coef_fn
        regu_fn = local_model[3] if len(local_model) > 3 else regu_fn
    else:
        _warn("Could not identity the local model", identify_local_model)
        pred_fn = local_model
    if local_loss is not None:
        loss_fn = local_loss
    if coefficients is not None:
        coef_fn = coefficients
    if regularisation is not None:
        regu_fn = regularisation
    if isinstance(coef_fn, int):
        i_coef = coef_fn
        coef_fn = lambda X, Y: i_coef  # noqa: E731
    _assert(pred_fn is not None, "`local_model` function missing")
    _assert(loss_fn is not None, "`local_loss` function missing")
    return pred_fn, loss_fn, coef_fn, regu_fn
