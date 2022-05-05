"""
This module contains various useful functions.
"""

from timeit import default_timer as timer
from typing import Any, Callable, List, Optional, Sequence, Union
from warnings import warn

import numpy as np
import torch


class SlisemapException(Exception):
    # Custom Exception type (for filtering)
    pass


class SlisemapWarning(Warning):
    # Custom Warning type (for filtering)
    pass


def _assert(condition: bool, message: str):
    if not condition:
        raise SlisemapException(message)


def _warn(warning: str, method: Optional[Callable] = None):
    if method is None:
        warn(warning, SlisemapWarning, 2)
    else:
        warn(f"{method.__qualname__}: {warning}", SlisemapWarning, 2)


def _tonp(x: torch.Tensor) -> np.ndarray:
    """Convert a torch.Tensor to a numpy.ndarray.

    Args:
        x (torch.Tensor): Input torch.Tensor.

    Returns:
        np.ndarray: Output numpy.ndarray.
    """
    return x.cpu().detach().numpy()


class CheckConvergence:
    """An object that tries to estimate when an optimisation has converged.
    Use it for, e.g., escape+optimisation cycles in Slisemap.
    """

    def __init__(self, patience: int = 3):
        """Create a CheckConvergence object.

        Args:
            patience (int, optional): How long should the optimisation continue without improvement. Defaults to 3.
        """
        self.current = np.inf
        self.best = np.inf
        self.counter = 0
        self.patience = max(1e-8, patience)
        self.optimal_state = None

    def has_converged(
        self,
        loss: Union[float, Sequence[float]],
        store: Optional[Callable[[], Any]] = None,
    ) -> bool:
        """Check if the optimisation has converged.

        Args:
            loss (Union[float, Sequence[float]]): The latest loss value(s).
            store (Optional[Callable[[], Any]], optional): Function that stores the optimal state in self.optimal_state. Defaults to None.


        Returns:
            bool: True if the optimisation has converged.
        """
        loss = np.asarray(loss)
        if np.any(np.isnan(loss)):
            _warn("Loss is `nan`", CheckConvergence.has_converged)
            return True
        if np.any(loss < self.best):
            self.best = np.minimum(loss, self.best)
            self.current = loss
            self.counter = 0  # Reset the counter if a new best
            if store is not None and loss.item(0) < self.best.item(0):
                self.optimal_state = store()
            return False
        # Increase the counter if no improvement
        self.counter += np.mean(self.current <= loss)
        self.current = loss
        return self.counter >= self.patience  # Has the patience run out


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
        torch.optim.LBFGS: The LBFGS optimiser.
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
    X: torch.Tensor, components: int, full: bool = True, niter: int = 10
) -> torch.Tensor:
    """Calculate the rotation matrix from PCA.

    If the PCA fails (e.g. if original matrix is not full rank) then this shows a warning instead of throwing an error (returns a dummy rotation).

    Args:
        X (torch.Tensor): The original matrix.
        components (int): The number of components in the embedding.
        full (bool, optional): Use a full SVD for the PCA (slower). Defaults to True.
        niter (int, optional): The number of iterations when a randomised approach is used. Defaults to 10.

    Returns:
        torch.Tensor: Rotation matrix that turns the original matrix into the embedded space.
    """
    try:
        components = min(*X.shape, components)
        if full:
            return torch.linalg.svd(X, full_matrices=False)[2][:, :components]
        else:
            return torch.pca_lowrank(X, components, center=False, niter=niter)[2]
    except:
        _warn("Could not perform PCA", PCA_rotation)
        z = torch.zeros((X.shape[1], components), dtype=X.dtype, device=X.device)
        z.fill_diagonal_(1.0, True)
        return z


def varimax(
    X: torch.Tensor, gamma: float = 1.0, max_iter: float = 100, tolerance: float = 1e-8
) -> torch.Tensor:
    """Rotate a matrix using varimax (so that the first dimension has the largest variance).

    Code adapted from: http://en.wikipedia.org/wiki/Talk:Varimax_rotation

    Args:
        X (torch.Tensor): Matrix
        gamma (float, optional): Learning rate. Defaults to 1.0.
        max_iter (float, optional): Maximum number of iterations. Defaults to 100.
        tolerance (float, optional): Early stopping tolerance (relative). Defaults to 1e-8.

    Returns:
        torch.Tensor: Rotated matrix.
    """
    gamma /= X.shape[0]
    eye = torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
    R = eye.clone()
    d = 0.0
    for _ in range(max_iter):
        XR = X @ R
        u, s, vh = torch.linalg.svd(
            X.T @ (XR**3 - gamma * (XR @ (eye * (XR.T @ XR)))), full_matrices=False
        )
        R = u @ vh
        d_old = d
        d = torch.sum(s)
        if d_old != 0.0 and d / d_old < 1.0 + tolerance:
            break
    return X @ R


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
        torch.Tensor: Global model coefficients.
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
