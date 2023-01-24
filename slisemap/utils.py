"""
This module contains various useful functions.
"""

import warnings
from timeit import default_timer as timer
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

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


def _deprecated(
    old: Union[Callable, str],
    new: Union[None, Callable, str] = None,
):
    try:
        old = f"'{old.__qualname__}'"
    except AttributeError:
        old = str(old)
    if new is None:
        warnings.warn(
            f"{old} is deprecated and may be removed in a future version",
            DeprecationWarning,
            2,
        )
    else:
        try:
            new = f"'{new.__qualname__}'"
        except AttributeError:
            new = str(new)
        warnings.warn(
            f"{old} is deprecated in favour of {new} and may be removed in a future version",
            DeprecationWarning,
            2,
        )


def _warn(warning: str, method: Optional[Callable] = None):
    if method is None:
        warnings.warn(warning, SlisemapWarning, 2)
    else:
        warnings.warn(f"{method.__qualname__}: {warning}", SlisemapWarning, 2)


def tonp(x: Union[torch.Tensor, Any]) -> np.ndarray:
    """Convert a `torch.Tensor` to a `numpy.ndarray`.
    If `x` is not a `torch.Tensor` then `np.asarray` is used instead.

    Args:
        x: Input `torch.Tensor`.

    Returns:
        Output `numpy.ndarray`.
    """
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    else:
        return np.asarray(x)


_tonp = tonp


class CheckConvergence:
    """An object that tries to estimate when an optimisation has converged.
    Use it for, e.g., escape+optimisation cycles in Slisemap.

    Args:
        patience: How long should the optimisation continue without improvement. Defaults to 3.
        max_iter: The maximum number of iterations. Defaults to `2**20`.
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
        self.counter = 0.0
        self.patience = patience
        self.optimal = None
        self.max_iter = max_iter
        self.iter = 0

    def has_converged(
        self,
        loss: Union[float, Sequence[float]],
        store: Optional[Callable[[], Any]] = None,
        verbose: bool = False,
    ) -> bool:
        """Check if the optimisation has converged.

        If more than one loss value is provided, then only the first one is checked when storing the `optimal_state`.
        The other losses are only used for checking convergence.

        Args:
            loss: The latest loss value(s).
            store: Function that returns the current state for storing in `self.optimal_state`. Defaults to None.
            verbose: Pring debug messages. Defaults to False.

        Returns:
            True if the optimisation has converged.
        """
        self.iter += 1
        loss = np.asarray(loss)
        if np.any(np.isnan(loss)):
            _warn("Loss is `nan`", CheckConvergence.has_converged)
            return True
        if np.any(loss < self.best):
            self.counter = 0.0  # Reset the counter if a new best
            if store is not None and loss.item(0) < self.best.item(0):
                self.optimal = store()
            self.best = np.minimum(loss, self.best)
        else:
            # Increase the counter if no improvement
            self.counter += np.mean(self.current <= loss)
        self.current = loss
        if verbose:
            print(
                f"CheckConvergence: patience={self.patience-self.counter:g}/{self.patience:g}   iter={self.iter}/{self.max_iter}"
            )
        return self.counter >= self.patience or self.iter >= self.max_iter


def LBFGS(
    loss_fn: Callable[[], torch.Tensor],
    variables: List[torch.Tensor],
    max_iter: int = 500,
    max_eval: Optional[int] = None,
    line_search_fn: Optional[str] = "strong_wolfe",
    time_limit: Optional[float] = None,
    increase_tolerance: bool = False,
    verbose: bool = False,
    **kwargs,
) -> torch.optim.LBFGS:
    """Optimise a function using LBFGS.

    Args:
        loss_fn: Function that returns a value to be minimised.
        variables: List of variables to optimise (must have `requires_grad=True`).
        max_iter: Maximum number of LBFGS iterations. Defaults to 500.
        max_eval: Maximum number of function evaluations. Defaults to `1.25 * max_iter`.
        line_search_fn: Line search method (None or "strong_wolfe"). Defaults to "strong_wolfe".
        time_limit: Optional time limit for the optimisation (in seconds). Defaults to None.
        increase_tolerance: Increase the tolerances for convergence checking. Defaults to False.
        verbose: Print status messages. Defaults to False.
        **kwargs: Argumemts passed to `torch.optim.LBFGS`.

    Returns:
        The LBFGS optimiser.
    """
    if increase_tolerance:
        kwargs["tolerance_grad"] = 100 * kwargs.get("tolerance_grad", 1e-7)
        kwargs["tolerance_change"] = 100 * kwargs.get("tolerance_change", 1e-9)
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
        loss = optimiser.step(closure)
    else:
        start = timer()
        prev_evals = 0
        for _ in range((max_iter - 1) // 20 + 1):
            loss = optimiser.step(closure)
            if not torch.all(torch.isfinite(loss)).cpu().detach().item():
                break
            if timer() - start > time_limit:
                if verbose:
                    print("LBFGS: Time limit exceeded!")
                break
            tot_evals = optimiser.state_dict()["state"][0]["func_evals"]
            if prev_evals + 1 == tot_evals:
                break  # LBFGS has converged if it returns after one evaluation
            prev_evals = tot_evals
            if max_eval is not None:
                if tot_evals >= max_eval:
                    break  # Number of evaluations exceeded max_eval
                optimiser.param_groups[0]["max_eval"] -= tot_evals
            # The number of steps is limited by ceiling(max_iter/20) with 20 iterations per step

    if verbose:
        iters = optimiser.state_dict()["state"][0]["n_iter"]
        evals = optimiser.state_dict()["state"][0]["func_evals"]
        loss = loss.mean().cpu().detach().item()
        if not np.isfinite(loss):
            print("LBFGS: Loss is not finite {}!")
        elif iters >= max_iter:
            print("LBFGS: Maximum number of iterations exceeded!")
        elif max_eval is not None and evals >= max_eval:
            print("LBFGS: Maximum number of evaluations exceeded!")
        else:
            print(f"LBFGS: Converged in {iters} iterations")

    return optimiser


def PCA_rotation(
    X: torch.Tensor,
    components: int = -1,
    center: bool = True,
    full: bool = True,
    niter: int = 10,
) -> torch.Tensor:
    """Calculate the rotation matrix from PCA.

    If the PCA fails (e.g. if original matrix is not full rank) then this shows a warning instead of throwing an error (returns a dummy rotation).

    Args:
        X: The original matrix.
        components: The maximum number of components in the embedding. Defaults to `min(*X.shape)`.
        center: Center the matrix before calculating the PCA.
        full: Use a full SVD for the PCA (slower). Defaults to True.
        niter: The number of iterations when a randomised approach is used. Defaults to 10.

    Returns:
        Rotation matrix that turns the original matrix into the embedded space.
    """
    try:
        components = min(*X.shape, components) if components > 0 else min(*X.shape)
        if full:
            if center:
                X = X - X.mean(dim=(-2,), keepdim=True)
            return torch.linalg.svd(X, full_matrices=False)[2].T[:, :components]
        else:
            return torch.pca_lowrank(X, components, center=center, niter=niter)[2]
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
        X: Data matrix.
        Y: Target matrix.
        local_model: Prediction function for the model.
        local_loss: Loss function for the model.
        coefficients: Number of coefficients. Defaults to X.shape[1].
        lasso: Lasso-regularisation coefficient for B ($\\lambda_{lasso} * ||B||_1$). Defaults to 0.0.
        ridge: Ridge-regularisation coefficient for B ($\\lambda_{ridge} * ||B||_2$). Defaults to 0.0.

    Returns:
        Global model coefficients.
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
        dict: Dictionary.

    Returns:
        The same dictionary where the values are numpy arrays with equal length.
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
        df: Dictionary of numpy arrays.
        d: Dictionary to append.

    Returns:
        The same dictionary as `df` with the values from `d` appended.
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
        dicts: Sequence or Generator with dictionaries (all must have the same keys).

    Returns:
        Combined dictionary.
    """
    if isinstance(dicts, Sequence):
        dicts = (d for d in dicts)
    df = dict_array(next(dicts))
    for d in dicts:
        dict_append(df, d)
    return df


def to_tensor(
    input: Union[np.ndarray, torch.Tensor, Dict[str, Any], Sequence[Any], Any],
    **tensorargs,
) -> Tuple[torch.Tensor, Optional[Sequence[Any]], Optional[Sequence[Any]]]:
    """Convert the input into a `torch.Tensor` (via `numpy.ndarray` if necessary).
    This function wrapps `torch.as_tensor` (and `numpy.asarray`) and tries to extract row and column names.
    This function can handle arbitrary objects (such as `pandas.DataFrame`) if they implement `.to_numpy()` and, optionally, `.index` and `.columns`.

    Args:
        data: input data
        **tensorargs: additional arguments to `torch.as_tensor`

    Returns:
        output: output tensor
        rows: row names or `None`
        columns: column names or `None`
    """
    if isinstance(input, dict):
        output = torch.as_tensor(np.asarray(tuple(input.values())).T, **tensorargs)
        return output, None, list(input.keys())
    elif isinstance(input, np.ndarray):
        return (torch.as_tensor(input, **tensorargs), None, None)
    elif isinstance(input, torch.Tensor):
        return (torch.as_tensor(input, **tensorargs), None, None)
    else:
        # Check if X is similar to a Pandas DataFrame
        try:
            output = torch.as_tensor(input.to_numpy(), **tensorargs)
        except (AttributeError, TypeError):
            try:
                output = torch.as_tensor(input.numpy(), **tensorargs)
            except (AttributeError, TypeError):
                output = torch.as_tensor(np.asarray(input), **tensorargs)
        try:
            columns = input.columns if len(input.columns) == output.shape[1] else None
        except (AttributeError, TypeError):
            columns = None
        try:
            rows = input.index if len(input.index) == output.shape[0] else None
        except (AttributeError, TypeError):
            rows = None
        return output, rows, columns


class Metadata(dict):
    """Metadata for Slisemap objects.
    Primarily row names, column names, and scaling information about the matrices.
    These are mostly used when plotting.
    """

    def __init__(self, root: "Slisemap"):
        super().__init__()
        self.root = root

    def set_rows(self, *rows: Optional[Sequence[Any]]):
        """Set the row names with checks.

        Args:
            *rows: row names
        """
        for row in rows:
            if row is not None:
                _assert(
                    len(rows) == self.root.n,
                    f"Wrong number of row names {len(rows)} != {self.root.n}",
                    Metadata.set_rows,
                )
                self["rows"] = rows
                break

    def set_variables(
        self,
        variables: Optional[Sequence[Any]] = None,
        add_intercept: Optional[bool] = None,
    ):
        """Set the variable names with checks.

        Args:
            variables: variable names
            add_intercept: add "Intercept" to the variable names. Defaults to `self.root.intercept`,
        """
        if add_intercept is None:
            add_intercept = self.root.intercept
        if variables is not None:
            if add_intercept:
                variables = list(variables) + ["Intercept"]
            _assert(
                len(variables) == self.root.m,
                f"Wrong number of variables {len(variables)} != {self.root.m}",
                Metadata.set_variables,
            )
            self["variables"] = variables

    def set_targets(self, targets: Union[None, str, Sequence[Any]] = None):
        """Set the target names with checks.

        Args:
            targets: target names
        """
        if targets is not None:
            if isinstance(targets, str):
                targets = [targets]
            _assert(
                len(targets) == self.root.o,
                f"Wrong number of targets {len(targets)} != {self.root.o}",
                Metadata.set_targets,
            )
            self["targets"] = targets

    def set_coefficients(self, coefficients: Optional[Sequence[Any]] = None):
        """Set the coefficient names with checks.

        Args:
            coefficients: coefficient names
        """
        if coefficients is not None:
            _assert(
                len(coefficients) == self.root.q,
                f"Wrong number of targets {len(coefficients)} != {self.root.q}",
                Metadata.set_coefficients,
            )
            self["coefficients"] = coefficients

    def set_dimensions(self, dimensions: Optional[Sequence[Any]] = None):
        """Set the dimension names with checks.

        Args:
            dimensions: dimension names
        """
        if dimensions is not None:
            _assert(
                len(dimensions) == self.root.d,
                f"Wrong number of targets {len(dimensions)} != {self.root.d}",
                Metadata.set_dimensions,
            )
            self["dimensions"] = dimensions

    def get_coefficients(self) -> List[str]:
        """Get a list of coefficient names (with a fallback if not assigned)

        Returns:
            list of coefficient names
        """
        if "coefficients" in self:
            return self["coefficients"]
        if "variables" in self:
            if self.root.m == self.root.q:
                return self["variables"]
            if "targets" in self and self.root.m * self.root.o >= self.root.q:
                return [
                    f"{t}: {v}" for t in self["targets"] for v in self["variables"]
                ][: self.root.q]
        return [f"B_{i}" for i in range(self.root.q)]

    def get_targets(self) -> List[str]:
        """Get a list of target names (with a fallback if not assigned)

        Returns:
            list of target names
        """
        if "targets" in self:
            return self["targets"]
        else:
            return [f"Y_{i}" for i in range(self.root.o)] if self.root.o > 1 else ["Y"]

    def get_variables(self, intercept: bool = True) -> List[str]:
        """Get a list of variable names (with a fallback if not assigned)

        Args:
            intercept: include the intercept in the list. Defaults to True.

        Returns:
            list of variable names
        """
        if "variables" in self:
            if self.root.intercept and not intercept:
                return self["variables"][:-1]
            else:
                return self["variables"]
        if self.root.intercept:
            if not intercept:
                return [f"X_{i}" for i in range(self.root.m - 1)]
            else:
                return [f"X_{i}" for i in range(self.root.m - 1)] + ["X_Intercept"]
        return [f"X_{i}" for i in range(self.root.m)]

    def get_dimensions(self, long: bool = False) -> List[str]:
        """Get a list of dimension names (with a fallback if not assigned)

        Args:
            long: Use "SLISEMAP 1",... as fallback instead of "Z_0",...

        Returns:
            list of dimension names
        """
        if "dimensions" in self:
            return self["dimensions"]
        elif long:
            return [f"SLISEMAP {i+1}" for i in range(self.root.d)]
        else:
            return [f"Z_{i}" for i in range(self.root.d)]

    def set_scale_X(
        self,
        X_center: Union[None, torch.Tensor, np.ndarray, Sequence[float]] = None,
        X_scale: Union[None, torch.Tensor, np.ndarray, Sequence[float]] = None,
    ):
        """Set scaling information with checks.
        Use if `X` has been scaled before being input to Slisemap.
        Assuming the scaling can be converted to the form `X = (X_unscaled - X_center) / X_scale`.
        This allows some plots to (temporarily) revert the scaling (for more intuitive units).

        Args:
            X_center: The constant offset of `X`. Defaults to None.
            X_scale: The scaling factor of `X`. Defaults to None.
        """
        if X_center is not None:
            X_center = tonp(X_center).ravel()
            assert X_center.size == self.root.m - self.root.intercept
            self["X_center"] = X_center
        if X_scale is not None:
            X_scale = tonp(X_scale).ravel()
            assert X_scale.size == self.root.m - self.root.intercept
            self["X_scale"] = X_scale

    def set_scale_Y(
        self,
        Y_center: Union[None, torch.Tensor, np.ndarray, Sequence[float]] = None,
        Y_scale: Union[None, torch.Tensor, np.ndarray, Sequence[float]] = None,
    ):
        """Set scaling information with checks.
        Use if `Y` has been scaled before being input to Slisemap.
        Assuming the scaling can be converted to the form `Y = (Y_unscaled - Y_center) / Y_scale`.
        This allows some plots to (temporarily) revert the scaling (for more intuitive units).

        Args:
            Y_center: The constant offset of `Y`. Defaults to None.
            Y_scale: The scaling factor of `Y`. Defaults to None.
        """
        if Y_center is not None:
            Y_center = tonp(Y_center).ravel()
            assert Y_center.size == self.root.o
            self["Y_center"] = Y_center
        if Y_scale is not None:
            Y_scale = tonp(Y_scale).ravel()
            assert Y_scale.size == self.root.o
            self["Y_scale"] = Y_scale

    def unscale_X(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Unscale X if the scaling information has been given (see `set_scale_X`).

        Args:
            X: The data matrix X (or `self.root.get_X(intercept=False)` if None).

        Returns:
            Possibly scaled X.
        """
        if X is None:
            X = self.root.get_X(intercept=False)
        if "X_scale" in self:
            X = X * self["X_scale"][None, :]
        if "X_center" in self:
            X = X + self["X_center"][None, :]
        return X

    def unscale_Y(self, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Unscale Y if the scaling information has been given (see `set_scale_Y`).

        Args:
            Y: The response matrix Y (or `self.root.get_Y()` if None).

        Returns:
            Possibly scaled Y.
        """
        if Y is None:
            Y = self.root.get_Y()
        if "Y_scale" in self:
            Y = Y * self["Y_scale"][None, :]
        if "Y_center" in self:
            Y = Y + self["Y_center"][None, :]
        return Y
