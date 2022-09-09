"""
This module contains various useful functions.
"""

import warnings
from timeit import default_timer as timer
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
import torch


class SlisemapException(Exception):
    # Custom Exception type (for filtering)
    pass


class SlisemapWarning(Warning):
    # Custom Warning type (for filtering)
    pass


def _assert(condition: bool, message: str, method: Optional[Callable] = None):
    if not condition:
        if method is None:
            raise SlisemapException(f"AssertionError: {message}")
        else:
            raise SlisemapException(f"AssertionError, {method.__qualname__}: {message}")


def _assert_no_trace(
    condition: Callable[[], bool], message: str, method: Optional[Callable] = None
):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        _assert(condition(), message, method)


def _deprecated(old: Callable, new: Optional[Callable] = None):
    if new is None:
        warnings.warn(
            f"{old.__qualname__} is deprecated and may be removed in a future version",
            DeprecationWarning,
        )
    else:
        warnings.warn(
            f"{old.__qualname__} is deprecated in favour of {new.__qualname__} and may be removed in a future version",
            DeprecationWarning,
        )


def _warn(warning: str, method: Optional[Callable] = None):
    if method is None:
        warnings.warn(warning, SlisemapWarning, 2)
    else:
        warnings.warn(f"{method.__qualname__}: {warning}", SlisemapWarning, 2)


def tonp(x: torch.Tensor) -> np.ndarray:
    """Convert a torch.Tensor to a numpy.ndarray.

    Args:
        x (torch.Tensor): Input torch.Tensor.

    Returns:
        (np.ndarray): Output numpy.ndarray.
    """
    return x.cpu().detach().numpy()


_tonp = tonp


class CheckConvergence:
    """An object that tries to estimate when an optimisation has converged.
    Use it for, e.g., escape+optimisation cycles in Slisemap.

    Args:
        patience (float, optional): How long should the optimisation continue without improvement. Defaults to 3.
        max_iter (int, optional): The maximum number of iterations. Defaults to `2**20`.
    """

    __slots__ = {
        "current": "Current loss value.",
        "best": "Best loss value, so far.",
        "counter": "Number of steps since the best loss value.",
        "patience": "Number of steps allowed without improvement.",
        "optimal": "Cache for storing the state that produced the best loss value.",
        "max_iter": "The maximum number of iterations.",
        "iter": "The current number of iterations.",
    }

    def __init__(self, patience: float = 3, max_iter=1 << 20):
        self.current = np.inf
        self.best = np.asarray(np.inf)
        self.counter = 0
        self.patience = patience
        self.optimal = None
        self.max_iter = max_iter
        self.iter = 0

    def has_converged(
        self,
        loss: Union[float, Sequence[float]],
        store: Optional[Callable[[], Any]] = None,
    ) -> bool:
        """Check if the optimisation has converged.

        If more than one loss value is provided, then only the first one is checked when storing the `optimal_state`.
        The other losses are only used for checking convergence.

        Args:
            loss (Union[float, Sequence[float]]): The latest loss value(s).
            store (Optional[Callable[[], Any]], optional): Function that returns the current state for storing in `self.optimal_state`. Defaults to None.

        Returns:
            (bool): True if the optimisation has converged.
        """
        self.iter += 1
        loss = np.asarray(loss)
        if np.any(np.isnan(loss)):
            _warn("Loss is `nan`", CheckConvergence.has_converged)
            return True
        if np.any(loss < self.best):
            self.counter = 0  # Reset the counter if a new best
            if store is not None and loss.item(0) < self.best.item(0):
                self.optimal = store()
            self.best = np.minimum(loss, self.best)
        else:
            # Increase the counter if no improvement
            self.counter += np.mean(self.current <= loss)
        self.current = loss
        return self.counter >= self.patience or self.iter >= self.max_iter


def LBFGS(
    loss_fn: Callable[[], torch.Tensor],
    variables: List[torch.Tensor],
    max_iter: int = 500,
    max_eval: Optional[int] = None,
    line_search_fn: Optional[str] = "strong_wolfe",
    time_limit: Optional[float] = None,
    **kwargs,
) -> torch.optim.LBFGS:
    """Optimise a function using LBFGS.

    Args:
        loss_fn (Callable[[], torch.Tensor]): Function that returns a value to be minimised.
        variables (List[torch.Tensor]): List of variables to optimise (must have `requires_grad=True`).
        max_iter (int, optional): Maximum number of LBFGS iterations. Defaults to 500.
        max_eval (Optional[int], optional): Maximum number of function evaluations. Defaults to `1.25 * max_iter`.
        line_search_fn (Optional[str], optional): Line search method (None or "strong_wolfe"). Defaults to "strong_wolfe".
        time_limit (Optional[float], optional): Optional time limit for the optimisation (in seconds). Defaults to None.
        **kwargs (optional): Argumemts passed to `torch.optim.LBFGS`.

    Returns:
        (torch.optim.LBFGS): The LBFGS optimiser.
    """
    optimiser = torch.optim.LBFGS(
        variables,
        max_iter=max_iter if time_limit is None else 20,
        max_eval=max_eval,
        line_search_fn=line_search_fn,
        **kwargs,
    )

    def closure():
        optimiser.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    if time_limit is None:
        optimiser.step(closure)
    else:
        start = timer()
        prev_evals = 0
        for _ in range((max_iter - 1) // 20 + 1):
            optimiser.step(closure)
            if timer() - start > time_limit:
                break  # Time limit exceeded
            tot_evals = optimiser.state_dict()["state"][0]["func_evals"]
            if prev_evals + 1 == tot_evals:
                break  # LBFGS has converged if it returns after one evaluation
            prev_evals = tot_evals
            if max_eval is not None:
                if tot_evals >= max_eval:
                    break  # Number of evaluations exceeded max_eval
                optimiser.param_groups[0]["max_eval"] -= tot_evals
            # The number of steps is limited by ceiling(max_iter/20) with 20 iterations per step

    return optimiser


def PCA_rotation(
    X: torch.Tensor, components: int = -1, full: bool = True, niter: int = 10
) -> torch.Tensor:
    """Calculate the rotation matrix from PCA.

    If the PCA fails (e.g. if original matrix is not full rank) then this shows a warning instead of throwing an error (returns a dummy rotation).

    Args:
        X (torch.Tensor): The original matrix.
        components (int, optional): The maximum number of components in the embedding. Defaults to `min(*X.shape)`.
        full (bool, optional): Use a full SVD for the PCA (slower). Defaults to True.
        niter (int, optional): The number of iterations when a randomised approach is used. Defaults to 10.

    Returns:
        (torch.Tensor): Rotation matrix that turns the original matrix into the embedded space.
    """
    try:
        components = min(*X.shape, components) if components > 0 else min(*X.shape)
        if full:
            return torch.linalg.svd(X, full_matrices=False)[2].T[:, :components]
        else:
            return torch.pca_lowrank(X, components, center=False, niter=niter)[2]
    except:
        _warn("Could not perform PCA", PCA_rotation)
        z = torch.zeros((X.shape[1], components), dtype=X.dtype, device=X.device)
        z.fill_diagonal_(1.0, True)
        return z


def global_model(
    X: torch.Tensor,
    Y: torch.Tensor,
    local_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    local_loss: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    coefficients: Optional[int] = None,
    lasso: float = 0.0,
    ridge: float = 0.0,
) -> torch.Tensor:
    """Find coefficients for a global model.

    Args:
        X (torch.Tensor): Data matrix.
        Y (torch.Tensor): Target matrix.
        local_model (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Prediction function for the model.
        local_loss (Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]): Loss function for the model.
        coefficients (Optional[int], optional): Number of coefficients. Defaults to X.shape[1].
        lasso (float, optional): Lasso-regularisation coefficient for B ($\\lambda_{lasso} * ||B||_1$). Defaults to 0.0.
        ridge (float, optional): Ridge-regularisation coefficient for B ($\\lambda_{ridge} * ||B||_2$). Defaults to 0.0.

    Returns:
        (torch.Tensor): Global model coefficients.
    """
    shape = (1, X.shape[1] * Y.shape[1] if coefficients is None else coefficients)
    B = torch.zeros(shape, dtype=X.dtype, device=X.device).requires_grad_(True)

    def loss():
        l = local_loss(local_model(X, B), Y, B).mean()
        if lasso > 0:
            l += lasso * torch.sum(B.abs())
        if ridge > 0:
            l += ridge * torch.sum(B**2)
        return l

    LBFGS(loss, [B])
    return B.detach()


def dict_array(dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Turn a dictionary of various values to a dictionary of numpy arrays with equal length inplace.

    Args:
        dict (Dict[str, Any]): Dictionary.

    Returns:
        (Dict[str, np.ndarray]): The same dictionary where the values are numpy arrays with equal length.
    """
    n = 1
    for k, v in dict.items():
        v = np.asarray(v).ravel()
        dict[k] = v
        n = max(n, len(v))
    for k, v in dict.items():
        if len(v) == 1:
            dict[k] = np.repeat(v, n)
        elif len(v) != n:
            _warn(f"Uneven lengths in dictionary ({k}: {len(v)} != {n})", dict_array)
    return dict


def dict_append(df: Dict[str, np.ndarray], d: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Append a dictionary of values to a dictionary of numpy arrays (see `dict_array`) inplace.

    Args:
        df (Dict[str, np.ndarray]): Dictionary of numpy arrays.
        d (Dict[str, Any]): Dictionary to append.

    Returns:
        (Dict[str, np.ndarray]): The same dictionary as `df` with the values from `d` appended.
    """
    d = dict_array(d)
    for k in df.keys():
        df[k] = np.concatenate((df[k], d[k]), 0)
    return df


def dict_concat(
    dicts: Union[Sequence[Dict[str, Any]], Iterator[Dict[str, Any]]]
) -> Dict[str, np.ndarray]:
    """Combine multiple dictionaries into one by concatenating the values.
    Calls `dict_array` to pre-process the dictionaries.

    Args:
        dicts (Union[Sequence[Dict[str, Any]], Iterator[Dict[str, Any]]]): Sequence or Generator with dictionaries (all must have the same keys).

    Returns:
        (Dict[str, np.ndarray]): Combined dictionary.
    """
    if isinstance(dicts, Sequence):
        dicts = (d for d in dicts)
    df = dict_array(next(dicts))
    for d in dicts:
        dict_append(df, d)
    return df


def _expand_variable_names(
    variables: Sequence[str],
    intercept: bool,
    columns: int,
    targets: Union[None, str, Sequence[str]],
    coefficients: int,
) -> List[str]:
    if intercept and len(variables) == columns - 1:
        variables = list(variables) + ["Intercept"]
    if targets and not isinstance(targets, str) and len(targets) > 0:
        if coefficients % len(variables) == 0 and coefficients % len(targets) == 0:
            variables = [f"{t}: {v}" for t in targets for v in variables]
            variables = variables[:coefficients]
    _assert(
        len(variables) == coefficients,
        f"The number of variable names ({len(variables)}) must match the number of coefficients ({coefficients})",
        _expand_variable_names,
    )
    return variables
