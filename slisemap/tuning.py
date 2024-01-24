"""
Find optimal hyper-parameters for Slisemap during the optimisation.
This should be faster than the usual "outer-loop" hyperperameter optimisation.
But the local search dynamics might be less exhaustive.
"""

import gc
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union
from functools import cache

import numpy as np
import torch

from slisemap.metrics import accuracy
from slisemap.slisemap import Slisemap
from slisemap.slipmap import Slipmap
from slisemap.utils import CheckConvergence, ToTensor, _assert, _warn, tonp


def hyperparameter_tune(
    method: Union[Type[Slisemap], Type[Slipmap]],
    X: ToTensor,
    y: ToTensor,
    X_test: ToTensor,
    y_test: ToTensor,
    lasso: Union[float, Tuple[float, float]] = (0.001, 10.0),
    ridge: Union[float, Tuple[float, float]] = (0.0001, 1.0),
    radius: Union[float, Tuple[float, float]] = (1.5, 4.0),
    *args,
    model: bool = True,
    n_calls: int = 15,
    verbose: bool = False,
    random_state: int = 42,
    predict_kws: Dict[str, object] = {},
    optim_kws: Dict[str, object] = {},
    gp_kws: Dict[str, object] = {},
    **kwargs,
) -> Union[Slisemap, Slipmap, Dict[str, float]]:
    """Tune the `lasso`, `ridge`, and `radius` hyperparameters using Bayesian optimisation.
    This function requires "scikit-optimize" to be installed.

    The search space is configured through the `lasso`/`ridge`/`radius` arguments as follows:
        - float: Skip the tuning of that hyperparameter.
        - tuple: tune the parameters limited to the space of `(lowerbound, upperbound)`.

    This function selects a candidate set of hyperparameters using `skopt.gp_minimize`.
    For a given set of hyperparameters, a Slisemap/Slipmap model is trained on `X` and `y`.
    Then the solution is evaluated using `X_test` and `y_test`.
    This procedure is repeated for `n_calls` iterations before the best result is returned.

    Args:
        method: Method to tune, either `Slisemap` or `Slipmap`.
        X: Data matrix.
        y: target matrix.
        X_test: New data for evaluation.
        y_test: New data for evaluation.
        lasso: Limits for the `lasso` parameter. Defaults to (0.001, 10.0).
        ridge: Limits for the `ridge` parameter. Defaults to (0.0001, 1.0).
        radius: Limits for the `radius` parameter. Defaults to (1.5, 4.0).
        *args: Arguments forwarded to `method`.
    Keyword Args:
        model: Return a trained model instead of a dictionary with tuned parameters. Defaults to True.
        n_calls: Number of parameter evaluations. Defaults to 15.
        verbose: Print status messages. Defaults to False.
        random_state: Random seed. Defaults to 42.
        predict_kws: Keyword arguments forwarded to `sm.predict`.
        optim_kws: Keyword arguments forwarded to `sm.optimise`.
        gp_kws: Keyword arguments forwarded to `skopt.gp_minimize`.
        **kwargs: Keyword arguments forwarded to `method`.

    Raises:
        ImportError: If `scikit-optimize` is not installed.

    Returns:
        Dictionary with hyperparameter values or a Slisemap/Slipmap model trained on those (see the `model` argument).
    """
    try:
        import skopt
    except ImportError:
        raise ImportError(
            "Hyperparameter tuning requires `scikit-optimize` to be installed"
        )

    space = []
    params = {}

    def make_space(grid, name, prior):
        if isinstance(grid, (float, int)):
            params[name] = grid
        else:
            _assert(
                len(grid) == 2,
                f"Wrong size `len({name}) = {len(grid)} != 2`",
                hyperparameter_tune,
            )
            space.append(skopt.space.Real(*grid, prior=prior, name=name))
            params[name] = (grid[0] * grid[1]) ** 0.5

    make_space(lasso, "lasso", "log-uniform")
    make_space(ridge, "ridge", "log-uniform")
    make_space(radius, "radius", "uniform")
    if len(space) == 0:
        _warn("No hyperparameters to tune", hyperparameter_tune)
        if model:
            sm = method(X, y, radius=radius, lasso=lasso, ridge=ridge, *args, **kwargs)
            sm.optimise(**optim_kws)
            return sm
        else:
            return params

    if model:
        best_loss = np.inf
        best_sm = None

    @skopt.utils.use_named_args(space)
    @cache
    def objective(
        lasso=params["lasso"], ridge=params["ridge"], radius=params["radius"]
    ):
        sm = method(X, y, radius=radius, lasso=lasso, ridge=ridge, *args, **kwargs)
        sm.optimise(**optim_kws)
        Xt = sm._as_new_X(X_test)
        Yt = sm._as_new_Y(y_test, Xt.shape[0])
        P = sm.predict(Xt, **predict_kws, numpy=False)
        l = sm.local_loss(Yt, P).mean().cpu().item()
        if verbose:
            print(f"Loss with {dict(lasso=lasso, ridge=ridge, radius=radius)}: {l}")
        if model:
            nonlocal best_loss, best_sm
            if l < best_loss:
                best_sm = sm
                best_loss = l
        del sm
        return l

    res = skopt.gp_minimize(
        objective,
        space,
        n_initial_points=min(10, max(3, (n_calls - 1) // 3 + 1)),
        n_calls=n_calls,
        random_state=random_state,
        **gp_kws,
    )
    for s, v in zip(space, res.x):
        params[s.name] = v
    if verbose:
        print(f"Final parameter values:", params)

    if model:
        return best_sm
    else:
        return params


def _hyper_grid(
    sm: Slisemap,
    lasso_grid: float = 3.0,
    ridge_grid: float = 3.0,
    search_size: int = 6,
    **kwargs: Any,
):
    # Do a local random search over hyperparameters
    sm.lbfgs(only_B=True, **kwargs)
    B0 = sm._B.detach().clone()
    rnd = torch.empty(search_size + 1, **sm.tensorargs)
    rnd.uniform_(-1, 1, generator=sm._random_state) * np.log(max(1.0, lasso_grid))
    lasso_grid = tonp(torch.exp(rnd) * sm.lasso)
    lasso_grid[0] = sm.lasso  # Make sure "no change" is included
    rnd.uniform_(-1, 1, generator=sm._random_state) * np.log(max(1.0, ridge_grid))
    ridge_grid = tonp(torch.exp(rnd) * sm.ridge)
    ridge_grid[0] = sm.ridge  # Make sure "no change is included"
    for lasso, ridge in zip(lasso_grid, ridge_grid):
        if np.allclose(lasso, sm.lasso) and np.allclose(ridge, sm.ridge):
            yield sm
        else:
            sm._B[...] = B0
            sm.lasso = lasso
            sm.ridge = ridge
            sm.lbfgs(only_B=True, **kwargs)
            yield sm


def _hyper_select(
    sm: Slisemap,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    test: Callable[[Slisemap, torch.Tensor, torch.Tensor], float] = accuracy,
    verbose: bool = False,
    **kwargs: Any,
) -> Tuple[Slisemap, float]:
    # Select new hyperparameters
    ev = np.inf
    for sm2 in _hyper_grid(sm, **kwargs):
        ev2 = test(sm2, X_test, y_test)
        if verbose:
            _hyper_verbose2(_hyper_select, sm2, ev2)
        if ev2 < ev:
            ev = ev2
            sm = sm2.copy()
    del sm2, ev2
    gc.collect(0)
    return sm, ev


def _hyper_init(
    method: Callable,
    sm: Slisemap,
    kwargs: Dict[str, Any],
    lasso_grid: float,
    ridge_grid: float,
    search_size: int,
    verbose: int,
) -> Tuple[Slisemap, Dict[str, Any], Optional[Dict[str, Any]]]:
    # Initialise kwargs and check the search space
    hs_kws = kwargs.copy()
    hs_kws.setdefault("lasso_grid", max(1, lasso_grid))
    hs_kws.setdefault("ridge_grid", max(1, ridge_grid))
    hs_kws.setdefault("search_size", max(0, search_size))
    hs_kws.setdefault("verbose", verbose > 1)
    hs_kws.setdefault("increase_tolerance", True)

    if search_size < 1:
        _warn(
            "`search_size` is less than one. No hyperparameter optimisation will be performed.",
            method,
        )
        hs_kws = None
    if lasso_grid <= 1.0 and ridge_grid <= 1.0:
        _warn(
            "Both `lasso_grid` and `ridge_grid` are less than one. No hyperparameter optimisation will be performed.",
            method,
        )
        hs_kws = None
    if sm.lasso <= 0.0 and lasso_grid > 1.0:
        _warn(
            "Slisemap `lasso` is zero with a `lasso_grid` greater than one. Setting the starting `lasso` to 0.001.",
            method,
        )
        sm.lasso = 0.001
    if sm.ridge <= 0.0 and ridge_grid > 1.0:
        _warn(
            "Slisemap `ridge` is zero with a `ridge_grid` greater than one. Setting the starting `ridge` to 0.001.",
            method,
        )
        sm.ridge = 0.001
    return sm, kwargs, hs_kws


def _hyper_noreg_loss(sm: Slisemap) -> float:
    # Slisemap loss without the regularisation terms
    return (
        sm.value()
        - sm.lasso * torch.sum(torch.abs(sm._B)).cpu().item()
        - sm.ridge * torch.sum(sm._B**2).cpu().item()
    )


def _hyper_verbose(method, sm: Slisemap, iter: int, test: float):
    # Print debug messages for the hyperparameter optimisation
    print(
        f"{method.__qualname__} {iter:>3}: lasso = {sm.lasso:5g}, ridge = {sm.ridge:5g}, loss = {_hyper_noreg_loss(sm):5g}, test = {test:5g}"
    )


def _hyper_verbose2(method, sm: Slisemap, test: float):
    # Print debug messages for the hyperparameter optimisation
    print(
        f"  {method.__qualname__}: lasso = {sm.lasso:5g}, ridge = {sm.ridge:5g}, loss = {_hyper_noreg_loss(sm):5g}, test = {test:5g}"
    )


def _hyper_verbose3(method, sms: List[Slisemap], iter: int, test: List[float]):
    # Print debug messages for the hyperparameter optimisation
    lasso = [sm.lasso for sm in sms]
    ridge = [sm.ridge for sm in sms]
    loss = [_hyper_noreg_loss(sm) for sm in sms]
    print(
        f"{method.__qualname__} {iter:>3}:",
        f"lasso = {np.mean(lasso):5g}±{np.std(lasso):5g},",
        f"ridge = {np.mean(ridge):5g}±{np.std(ridge):5g},",
        f"loss = {np.mean(loss):5g}±{np.std(loss):5g},",
        f"test = {np.mean(test):5g}±{np.std(test):5g}",
    )


def _hyper_verbose4(method, sm: Slisemap):
    # Print debug messages for the hyperparameter optimisation
    print(
        f"{method.__qualname__} Final: lasso = {sm.lasso:5g}, ridge = {sm.ridge:5g}, loss = {_hyper_noreg_loss(sm):5g}"
    )


def optimise_with_test_set(
    sm: Slisemap,
    X_test: Union[np.ndarray, torch.Tensor],
    y_test: Union[np.ndarray, torch.Tensor],
    lasso_grid: float = 3.0,
    ridge_grid: float = 3.0,
    search_size: int = 6,
    test: Callable[[Slisemap, torch.Tensor, torch.Tensor], float] = accuracy,
    patience: int = 2,
    max_iterations: int = 100,
    verbose: Literal[0, 1, 2, 3] = 0,
    escape_kws: Dict[str, Any] = {},
    **kwargs: Any,
) -> Slisemap:
    """Optimise a Slisemap object using test data to tune the regularisation.

    How this works:
        - The procedure is very similar to [Slisemap.optimise][slisemap.slisemap.Slisemap.optimise], which alternates between [LBFGS][slisemap.slisemap.Slisemap.lbfgs] optimisation and an ["escape" heuristic][slisemap.slisemap.Slisemap.escape] until convergence.
        - The hyperoptimisation tuning adds an additional step after each call to [LBFGS][slisemap.slisemap.Slisemap.lbfgs] where a small local search is performed to tune the hyperparameters.
        - The convergence criteria is also changed to use the test data (see the `test` parameter).
        - This should be faster than the usual "outer-loop" hyperperameter optimisation, but the local search dynamics might be less exhaustive.

    Args:
        sm: Slisemap object.
        X_test: Data matrix for the test set.
        y_test: Target matrix/vector for the test set.
        lasso_grid: The extent of the local search for the lasso parameter `(lasso/lasso_grid, lasso*lasso_grid)`. Set to zero to disable the hyperparameter search. Defaults to 3.0.
        ridge_grid: The extent of the local search for the ridge parameter `(ridge/ridge_grid, ridge*ridge_grid)`. Set to zero to disable the hyperparameter search. Defaults to 3.0.
        search_size: The number of evaluations in the local random search. Defaults to 6.
        test: Test to measure the performance of different hyperparameter values. Defaults to [accuracy][slisemap.metrics.accuracy].
        patience: Number of optimisation rounds without improvement before stopping. Defaults to 2.
        max_iterations: Maximum numbers optimisation rounds. Defaults to 100.
        verbose: Print status messages. Defaults to 0.
        escape_kws: Keyword arguments forwarded to [sm.escape][slisemap.slisemap.Slisemap.escape]. Defaults to {}.
    Keyword Args:
        **kwargs: Optional keyword arguments to [sm.lbfgs][slisemap.slisemap.Slisemap.lbfgs].

    Returns:
        Optimised Slisemap object. This is not the same object as the input!
    """
    sm, kwargs, hs_kws = _hyper_init(
        optimise_with_test_set,
        sm=sm,
        kwargs=kwargs,
        lasso_grid=lasso_grid,
        ridge_grid=ridge_grid,
        search_size=search_size,
        verbose=verbose,
    )
    if hs_kws is None:
        sm.optimise(verbose=verbose, **escape_kws, **kwargs)
        return sm

    X_test = sm._as_new_X(X_test)
    y_test = sm._as_new_Y(y_test, X_test.shape[0])
    if verbose:
        _hyper_verbose(optimise_with_test_set, sm, 0, test(sm, X_test, y_test))

    # Initial optimisation with: _hyper_select -> escape -> lbfgs
    sm, loss = _hyper_select(sm, X_test, y_test, test, **hs_kws)
    cc = CheckConvergence(patience, max_iterations)
    while not cc.has_converged(loss, sm.copy, verbose=verbose > 1):
        sm.escape(**escape_kws)
        sm.lbfgs(increase_tolerance=True, verbose=verbose > 2, **kwargs)
        sm, loss = _hyper_select(sm, X_test, y_test, test, **hs_kws)
        if verbose:
            _hyper_verbose(optimise_with_test_set, sm, cc.iter, loss)

    # Secondary optimisation with: lbfgs -> _hyper_select
    sm, loss = cc.optimal, cc.best
    hs_kws["increase_tolerance"] = False
    cc.patience = min(patience, 1)
    cc.counter = 0.0
    while not cc.has_converged(loss, sm.copy, verbose=verbose > 1):
        sm.lbfgs(increase_tolerance=False, verbose=verbose > 2, **kwargs)
        sm, loss = _hyper_select(sm, X_test, y_test, test, **hs_kws)
        if verbose:
            _hyper_verbose(optimise_with_test_set, sm, cc.iter, loss)

    return cc.optimal


optimize_with_test_set = optimise_with_test_set


def optimise_with_cross_validation(
    sm: Slisemap,
    k: int = 5,
    lasso_grid: float = 3.0,
    ridge_grid: float = 3.0,
    search_size: int = 6,
    lerp: float = 0.5,
    test: Callable[[Slisemap, torch.Tensor, torch.Tensor], float] = accuracy,
    patience: int = 1,
    max_iterations: int = 100,
    verbose: Literal[0, 1, 2, 3] = 0,
    escape_kws: Dict[str, Any] = {},
    **kwargs: Any,
) -> Slisemap:
    """Optimise a Slisemap object using cross validation to tune the regularisation.

    How this works:
        - The data is split into k folds for cross validation.
        - Then a procedure like [optimise_with_test_set][slisemap.tuning.optimise_with_test_set] is used.
        - After every hyperparameter tuning the regularisation coefficients are smoothed across the folds (see the `lerp` parameter).
        - Finally, when the cross validation has converged the solution is transferred to the complete data for one final optimisation.
        - Note that this is significantly slower than just training on Slisemap solution.
        - However, this should be faster than the usual "outer-loop" hyperperameter optimisation (but the local search dynamics might be less exhaustive).

    Args:
        sm: Slisemap object.
        k: Number of folds for the cross validation. Defaults to 5.
        lasso_grid: The extent of the local search for the lasso parameter `(lasso/lasso_grid, lasso*lasso_grid)`. Set to zero to disable the hyperparameter search. Defaults to 3.0.
        ridge_grid: The extent of the local search for the ridge parameter `(ridge/ridge_grid, ridge*ridge_grid)`. Set to zero to disable the hyperparameter search. Defaults to 3.0.
        search_size: The number of evaluations in the local random search. Defaults to 6.
        lerp: Smooth regularisation coefficients across folds (linearly interpolating towards the mean coefficients). Defaults to 0.5.
        test: Test to measure the performance of different hyperparameter values. Defaults to [accuracy][slisemap.metrics.accuracy].
        patience: Number of optimisation rounds without improvement before stopping. Defaults to 1.
        max_iterations: Maximum numbers optimisation rounds. Defaults to 100.
        verbose: Print status messages. Defaults to 0.
        escape_kws: Keyword arguments forwarded to [sm.escape][slisemap.slisemap.Slisemap.escape]. Defaults to {}.
    Keyword Args:
        **kwargs: Optional keyword arguments to [sm.lbfgs][slisemap.slisemap.Slisemap.lbfgs].

    Returns:
        Optimised Slisemap object.
    """
    sm, kwargs, hs_kws = _hyper_init(
        optimise_with_cross_validation,
        sm=sm,
        kwargs=kwargs,
        lasso_grid=lasso_grid,
        ridge_grid=ridge_grid,
        search_size=search_size,
        verbose=verbose,
    )
    if hs_kws is None:
        sm.optimise(verbose=verbose, **escape_kws, **kwargs)
        return sm

    # Create k folds
    fold_size = (sm.n - 1) // k + 1
    folds = torch.tile(torch.arange(k, **sm.tensorargs), (fold_size,))
    folds = folds[torch.randperm(sm.n, generator=sm._random_state)]
    sms = []
    tests = []
    for i in range(k):
        X_test = sm._X[folds == i, ...]
        y_test = sm._Y[folds == i, ...]
        tests.append((X_test, y_test))
        sm2 = sm.copy()
        sm2._X = sm._X[folds != i, ...]
        sm2._Y = sm._Y[folds != i, ...]
        sm2._B = sm._B[folds != i, ...]
        sm2._Z = sm._Z[folds != i, ...]
        sms.append(sm2)

    # Helper functions
    def hyper():
        loss = []
        for i, (X_test, y_test) in enumerate(tests):
            sms[i], l = _hyper_select(sms[i], X_test, y_test, test, **hs_kws)
            loss.append(l)
        return [np.mean(loss)] + loss

    def optim():
        lasso = np.mean([sm2.lasso for sm2 in sms])
        ridge = np.mean([sm2.ridge for sm2 in sms])
        for sm2 in sms:
            sm2.lasso = sm2.lasso * (1 - lerp) + lasso * lerp
            sm2.ridge = sm2.ridge * (1 - lerp) + ridge * lerp
            sm2.escape(**escape_kws)
            sm2.lbfgs(increase_tolerance=True, verbose=verbose > 2, **kwargs)
        return hyper()

    # Optimise the cross validation folds with hyperparameter tuning
    cc = CheckConvergence(patience, max_iterations)
    loss = hyper()
    if verbose:
        _hyper_verbose3(optimise_with_cross_validation, sms, 0, loss[1:])
    while not cc.has_converged(loss, lambda: [sm2.copy() for sm2 in sms], verbose > 1):
        loss = optim()
        if verbose:
            _hyper_verbose3(optimise_with_cross_validation, sms, cc.iter, loss[1:])

    # Apply the tuned parameters on the complete model
    loss = [test(sm2, sm._X, sm._Y) for sm2 in cc.optimal]
    sm3 = cc.optimal[np.argmin(loss)]
    B, Z = sm3.fit_new(sm._X, sm._Y, optimise=False, numpy=False)
    sm._B = B
    sm._Z = Z
    sm.lasso = (1 - lerp) * sm3.lasso + lerp * np.mean([sm2.lasso for sm2 in sms])
    sm.ridge = (1 - lerp) * sm3.ridge + lerp * np.mean([sm2.ridge for sm2 in sms])

    # Optimise the complete model
    kwargs.setdefault("max_iter", 500)
    kwargs["max_iter"] *= 2
    sm.lbfgs(increase_tolerance=False, verbose=verbose > 2, **kwargs)
    if verbose:
        _hyper_verbose4(optimise_with_cross_validation, sm)

    return sm


optimize_with_cross_validation = optimise_with_cross_validation


def optimise(
    sm: Slisemap,
    X_test: Union[None, np.ndarray, torch.Tensor] = None,
    y_test: Union[None, np.ndarray, torch.Tensor] = None,
    **kwargs: Any,
) -> Slisemap:
    """Optimise a Slisemap object with hyperparameter tuning.
    This can either be done using a [test set][slisemap.tuning.optimise_with_test_set] or [cross validation][slisemap.tuning.optimise_with_cross_validation].
    The choice of method is based on whether `X_test` and `y_test` is given.

    Args:
        sm: Slisemap object.
        X_test: Data matrix for the test set. Defaults to None.
        y_test: Target matrix/vector for the test set. Defaults to None.
    Keyword Args:
        **kwargs: Optional keyword arguments to [slisemap.tuning.optimise_with_test_set][] or [slisemap.tuning.optimise_with_cross_validation][].

    Returns:
        Optimised Slisemap object. This is not the same object as the input!
    """
    if X_test is None or y_test is None:
        return optimise_with_cross_validation(sm, **kwargs)
    else:
        return optimise_with_test_set(sm, X_test, y_test, **kwargs)


optimize = optimise
