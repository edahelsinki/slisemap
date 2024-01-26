"""
Find optimal hyper-parameters for Slisemap and Slipmap.
"""

from functools import cache
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch

try:
    import skopt
except ImportError:
    raise ImportError(
        "Hyperparameter tuning requires `scikit-optimize` to be installed"
    )

from slisemap.metrics import accuracy
from slisemap.slipmap import Slipmap
from slisemap.slisemap import Slisemap
from slisemap.utils import CheckConvergence, ToTensor, _assert, _deprecated, _warn, tonp


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


def _hyper_init(
    method: Callable,
    sm: Slisemap,
    kwargs: Dict[str, Any],
    lasso_grid: float,
    ridge_grid: float,
    radius_grid: float,
    search_size: int,
) -> Tuple[Slisemap, Dict[str, Any], Optional[Dict[str, Any]]]:
    # Check the search space
    _assert(search_size > 1, "No hyperparameter optimisation", method)
    if sm.lasso <= 0.0 and lasso_grid > 1.0:
        _warn("`lasso_grid >= 1.0` with `sm.lasso == 0`", method)
        lasso_grid = 0.0
    if sm.ridge <= 0.0 and ridge_grid > 1.0:
        _warn("`ridge_grid >= 1.0` with `sm.ridge == 0`", method)
        ridge_grid = 0.0
    if sm.radius <= 0.0 and radius_grid > 1.0:
        _warn("`radius_grid >= 1.0` with `sm.radius == 0`", method)
        radius_grid = 0.0
    if lasso_grid <= 1.0 and ridge_grid <= 1.0 and radius_grid <= 1.0:
        _warn("Empty hyperparameter search space.", method)
        hs_kws = None
    # Initialise kwargs
    hs_kws = {
        "lasso_grid": max(1, lasso_grid),
        "ridge_grid": max(1, ridge_grid),
        "radius_grid": max(1, radius_grid),
        "n_calls": max(0, search_size),
    }
    kwargs.setdefault("increase_tolerance", True)
    return kwargs, hs_kws


def _hyper_tune(
    sm: Union[Slisemap, Slipmap],
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    test: Callable[[Slisemap, torch.Tensor, torch.Tensor], float],
    lasso_grid: float,
    ridge_grid: float,
    radius_grid: float,
    n_calls: int = 6,
    random_state: int = 42,
    gp_kws: Dict[str, object] = {},
    **kwargs: Any,
):
    space = []
    make_space = lambda value, grid, name: space.append(
        skopt.space.Real(value / grid, value * grid, "log-uniform", name=name)
    )
    if ridge_grid > 1.0:
        make_space(sm.ridge, ridge_grid, "ridge")
    if lasso_grid > 1.0:
        make_space(sm.lasso, lasso_grid, "lasso")
    if radius_grid > 1.0:
        make_space(sm.radius, radius_grid, "radius")
    best_ev = test(sm, X_test, y_test)
    best_sm = sm

    @skopt.utils.use_named_args(space)
    @cache
    def objective(lasso=sm.lasso, ridge=sm.ridge, radius=sm.radius):
        sm2 = sm.copy()
        sm2.lasso = lasso
        sm2.ridge = ridge
        sm2.radius = radius
        sm2.lbfgs(only_B=True, **kwargs)
        ev2 = test(sm2, X_test, y_test)
        nonlocal best_ev, best_sm
        if ev2 < best_ev:
            best_sm = sm2
            best_ev = ev2
        else:
            del sm2
        return ev2

    skopt.gp_minimize(
        objective,
        space,
        n_initial_points=min(10, max(3, (n_calls - 1) // 3 + 1)),
        n_calls=n_calls,
        random_state=random_state,
        **gp_kws,
    )
    return best_sm, best_ev


def _hyper_verbose(
    method,
    sm: Union[Slisemap, List[Slisemap]],
    iter: int,
    test: Union[float, List[float], None],
):
    # Print debug messages for the hyperparameter optimisation
    if isinstance(sm, list):
        lasso = [sm2.lasso for sm2 in sm]
        ridge = [sm2.ridge for sm2 in sm]
        radius = [sm2.radius for sm2 in sm]
        print(
            f"{method.__qualname__} {iter:>3}:",
            f"lasso = {np.mean(lasso):5g}±{np.std(lasso):5g},",
            f"ridge = {np.mean(ridge):5g}±{np.std(ridge):5g},",
            f"radius = {np.mean(radius):5g}±{np.std(radius):5g},",
            f"test = {np.mean(test):5g}±{np.std(test):5g}",
        )
    else:
        print(
            f"{method.__qualname__} {iter:>3}:",
            f"lasso = {sm.lasso:5g},",
            f"ridge = {sm.ridge:5g},",
            f"radius = {sm.radius:5g},",
            "" if test is None else f"test = {test:5g}",
        )


def optimise_with_test(
    sm: Union[Slisemap, Slipmap],
    X_test: Union[np.ndarray, torch.Tensor],
    y_test: Union[np.ndarray, torch.Tensor],
    lasso_grid: float = 3.0,
    ridge_grid: float = 3.0,
    radius_grid: float = 1.1,
    search_size: int = 6,
    test: Callable[[Slisemap, torch.Tensor, torch.Tensor], float] = accuracy,
    patience: int = 2,
    max_escapes: int = 100,
    verbose: Literal[0, 1, 2, 3] = 0,
    escape_kws: Dict[str, Any] = {},
    *,
    max_iterations=None,
    **kwargs: Any,
) -> Union[Slisemap, Slipmap]:
    """Optimise a Slisemap or Slipmap object using test data to tune the regularisation.

    How this works:
        - The procedure is very similar to [Slisemap.optimise][slisemap.slisemap.Slisemap.optimise], which alternates between [LBFGS][slisemap.slisemap.Slisemap.lbfgs] optimisation and an ["escape" heuristic][slisemap.slisemap.Slisemap.escape] until convergence.
        - The hyperoptimisation tuning adds an additional step after each call to [LBFGS][slisemap.slisemap.Slisemap.lbfgs] where a small local search is performed to tune the hyperparameters.
        - The convergence criteria is also changed to use the test data (see the `test` parameter).
        - This should be faster than the usual "outer-loop" hyperperameter optimisation, but the local search dynamics might be less exhaustive.

    Args:
        sm: Slisemap or Slipmap object.
        X_test: Data matrix for the test set.
        y_test: Target matrix/vector for the test set.
        lasso_grid: The extent of the local search for the lasso parameter `(lasso/lasso_grid, lasso*lasso_grid)`. Set to zero to disable the hyperparameter search. Defaults to 3.0.
        ridge_grid: The extent of the local search for the ridge parameter `(ridge/ridge_grid, ridge*ridge_grid)`. Set to zero to disable the hyperparameter search. Defaults to 3.0.
        radius_grid: The extent of the local search for the radius parameter `(radius/radius_grid, radius*radius_grid)`. Set to zero to disable the hyperparameter search. Defaults to 1.5.
        search_size: The number of evaluations in the local random search. Defaults to 6.
        test: Test to measure the performance of different hyperparameter values. Defaults to [accuracy][slisemap.metrics.accuracy].
        patience: Number of optimisation rounds without improvement before stopping. Defaults to 2.
        max_escapes: Maximum numbers optimisation rounds. Defaults to 100.
        verbose: Print status messages. Defaults to 0.
        escape_kws: Keyword arguments forwarded to [sm.escape][slisemap.slisemap.Slisemap.escape]. Defaults to {}.
    Keyword Args:
        **kwargs: Optional keyword arguments to [sm.lbfgs][slisemap.slisemap.Slisemap.lbfgs].

    Returns:
        Optimised Slisemap or Slipmap object. This is not the same object as the input!

    Deprecated:
        1.6: `max_iterations` renamed to `max_escapes`
    """
    if max_iterations is not None:
        _deprecated(
            optimise_with_cv.max_iterations,
            optimise_with_cv.max_escapes,
        )
        max_escapes = max_iterations
    kwargs, hs_kws = _hyper_init(
        optimise_with_test,
        sm=sm,
        kwargs=kwargs,
        lasso_grid=lasso_grid,
        ridge_grid=ridge_grid,
        radius_grid=radius_grid,
        search_size=search_size,
    )
    if hs_kws is None:
        kwargs["increase_tolerance"] = False
        sm.optimise(verbose=verbose, **escape_kws, **kwargs)
        return sm

    X_test = sm._as_new_X(X_test)
    y_test = sm._as_new_Y(y_test, X_test.shape[0])
    if verbose:
        _hyper_verbose(optimise_with_test, sm, 0, test(sm, X_test, y_test))

    # Initial optimisation with: _hyper_select -> escape -> lbfgs
    sm.lbfgs(only_B=True, verbose=verbose > 2, **kwargs)
    sm, ev = _hyper_tune(sm, X_test, y_test, test, **hs_kws, **kwargs)
    cc = CheckConvergence(patience, max_escapes)
    while not cc.has_converged(ev, sm.copy, verbose=verbose > 1):
        sm.escape(**escape_kws)
        sm.lbfgs(verbose=verbose > 2, **kwargs)
        sm, ev = _hyper_tune(sm, X_test, y_test, test, **hs_kws, **kwargs)
        if verbose:
            _hyper_verbose(optimise_with_test, sm, cc.iter, ev)

    # Secondary optimisation with: lbfgs -> _hyper_select
    sm, ev = cc.optimal, cc.best
    kwargs["increase_tolerance"] = False
    cc.patience = min(patience, 1)
    cc.counter = 0.0
    while not cc.has_converged(ev, sm.copy, verbose=verbose > 1):
        sm.lbfgs(verbose=verbose > 2, **kwargs)
        sm, ev = _hyper_tune(sm, X_test, y_test, test, **hs_kws, **kwargs)
        if verbose:
            _hyper_verbose(optimise_with_test, sm, cc.iter, ev)

    return cc.optimal


optimize_with_test = optimise_with_test
optimize_with_test_set = optimise_with_test
optimise_with_test_set = optimise_with_test


def optimise_with_cv(
    sm: Union[Slisemap, Slipmap],
    k: int = 5,
    lasso_grid: float = 3.0,
    ridge_grid: float = 3.0,
    radius_grid: float = 1.1,
    search_size: int = 6,
    lerp: float = 0.3,
    test: Callable[[Slisemap, torch.Tensor, torch.Tensor], float] = accuracy,
    patience: int = 2,
    max_escapes: int = 100,
    verbose: Literal[0, 1, 2, 3] = 0,
    escape_kws: Dict[str, Any] = {},
    *,
    max_iterations=None,
    **kwargs: Any,
) -> Union[Slisemap, Slipmap]:
    """Optimise a Slisemap or Slipmap object using cross validation to tune the regularisation.

    How this works:
        - The data is split into k folds for cross validation.
        - Then a procedure like [optimise_with_test][slisemap.tuning.optimise_with_test] is used.
        - After every hyperparameter tuning the regularisation coefficients are smoothed across the folds (see the `lerp` parameter).
        - Finally, when the cross validation has converged the solution is transferred to the complete data for one final optimisation.
        - Note that this is significantly slower than just training on Slisemap solution.
        - However, this should be faster than the usual "outer-loop" hyperperameter optimisation (but the local search dynamics might be less exhaustive).

    Args:
        sm: Slisemap or Slipmap object.
        k: Number of folds for the cross validation. Defaults to 5.
        lasso_grid: The extent of the local search for the lasso parameter `(lasso/lasso_grid, lasso*lasso_grid)`. Set to zero to disable the hyperparameter search. Defaults to 3.0.
        ridge_grid: The extent of the local search for the ridge parameter `(ridge/ridge_grid, ridge*ridge_grid)`. Set to zero to disable the hyperparameter search. Defaults to 3.0.
        radius_grid: The extent of the local search for the radius parameter `(radius/radius_grid, radius*radius_grid)`. Set to zero to disable the hyperparameter search. Defaults to 1.5.
        search_size: The number of evaluations in the local random search. Defaults to 6.
        lerp: Smooth regularisation coefficients across folds (linearly interpolating towards the mean coefficients). Defaults to 0.3.
        test: Test to measure the performance of different hyperparameter values. Defaults to [accuracy][slisemap.metrics.accuracy].
        patience: Number of optimisation rounds without improvement before stopping. Defaults to 1.
        max_escapes: Maximum numbers optimisation rounds. Defaults to 100.
        verbose: Print status messages. Defaults to 0.
        escape_kws: Keyword arguments forwarded to [sm.escape][slisemap.slisemap.Slisemap.escape]. Defaults to {}.
    Keyword Args:
        **kwargs: Optional keyword arguments to [sm.lbfgs][slisemap.slisemap.Slisemap.lbfgs].

    Returns:
        Optimised Slisemap or Slipmap object.

    Deprecated:
        1.6: `max_iterations` renamed to `max_escapes`
    """
    if max_iterations is not None:
        _deprecated(
            optimise_with_cv.max_iterations,
            optimise_with_cv.max_escapes,
        )
        max_escapes = max_iterations
    kwargs, hs_kws = _hyper_init(
        optimise_with_cv,
        sm=sm,
        kwargs=kwargs,
        lasso_grid=lasso_grid,
        ridge_grid=ridge_grid,
        radius_grid=radius_grid,
        search_size=search_size,
    )
    if hs_kws is None:
        kwargs["increase_tolerance"] = False
        sm.optimise(verbose=verbose, **escape_kws, **kwargs)
        return sm

    # Create k folds
    fold_size = (sm.n - 1) // k + 1
    folds = torch.tile(torch.arange(k, **sm.tensorargs), (fold_size,))[: sm.n]
    sms = []
    tests = []
    for i in range(k):
        X_test = sm._X[folds == i, ...]
        y_test = sm._Y[folds == i, ...]
        tests.append((X_test, y_test))
        sm2 = sm.copy()
        sm2._X = sm._X[folds != i, ...].clone()
        sm2._Y = sm._Y[folds != i, ...].clone()
        if isinstance(sm, Slisemap):
            sm2._B = sm._B[folds != i, ...].clone()
        sm2._Z = sm._Z[folds != i, ...].clone()
        sm2.lbfgs(only_B=True, verbose=verbose > 2, **kwargs)
        sms.append(sm2)

    # Helper functions
    def hyper():
        nonlocal sms
        loss = []
        for i, (X_test, y_test) in enumerate(tests):
            sms[i], l = _hyper_tune(sms[i], X_test, y_test, test, **hs_kws, **kwargs)
            loss.append(l)
        return [np.mean(loss)] + loss

    def optim():
        lasso = np.mean([sm2.lasso for sm2 in sms])
        ridge = np.mean([sm2.ridge for sm2 in sms])
        radius = np.mean([sm2.radius for sm2 in sms])
        for sm2 in sms:
            sm2.lasso = sm2.lasso * (1 - lerp) + lasso * lerp
            sm2.ridge = sm2.ridge * (1 - lerp) + ridge * lerp
            sm2.radius = sm2.radius * (1 - lerp) + radius * lerp
            sm2.escape(**escape_kws)
            sm2.lbfgs(verbose=verbose > 2, **kwargs)
        return hyper()

    # Optimise the cross validation folds with hyperparameter tuning
    cc = CheckConvergence(patience, max_escapes)
    loss = hyper()
    if verbose:
        _hyper_verbose(optimise_with_cv, sms, 0, loss[1:])
    while not cc.has_converged(loss, lambda: [sm2.copy() for sm2 in sms], verbose > 1):
        loss = optim()
        if verbose:
            _hyper_verbose(optimise_with_cv, sms, cc.iter, loss[1:])

    # Apply the tuned parameters on the complete model
    loss = [test(sm2, sm._X, sm._Y) for sm2 in cc.optimal]
    opt = np.argmin(loss)
    sm._Z[folds != opt, ...] = cc.optimal[opt]._Z
    if isinstance(sm, Slisemap):
        sm._B[folds != opt, ...] = cc.optimal[opt]._B
    sm.lasso = np.mean([sm2.lasso for sm2 in cc.optimal])
    sm.ridge = np.mean([sm2.ridge for sm2 in cc.optimal])
    sm.radius = np.mean([sm2.radius for sm2 in cc.optimal])

    # Optimise the complete model
    kwargs["increase_tolerance"] = False
    sm.lbfgs(verbose=verbose > 2, **kwargs)
    if verbose:
        _hyper_verbose(optimise_with_cv, sm, "Final", None)

    return sm


optimize_with_cv = optimise_with_cv
optimize_with_cross_validation = optimise_with_cv
optimise_with_cross_validation = optimise_with_cv


def optimise(
    sm: Union[Slisemap, Slipmap],
    X_test: Union[None, np.ndarray, torch.Tensor] = None,
    y_test: Union[None, np.ndarray, torch.Tensor] = None,
    **kwargs: Any,
) -> Union[Slisemap, Slipmap]:
    """Optimise a Slisemap or Slipmap object with hyperparameter tuning.
    This can either be done using a [test set][slisemap.tuning.optimise_with_test] or [cross validation][slisemap.tuning.optimise_with_cv].
    The choice of method is based on whether `X_test` and `y_test` is given.

    Args:
        sm: Slisemap or Slipmap object.
        X_test: Data matrix for the test set. Defaults to None.
        y_test: Target matrix/vector for the test set. Defaults to None.
    Keyword Args:
        **kwargs: Optional keyword arguments to [slisemap.tuning.optimise_with_test][] or [slisemap.tuning.optimise_with_cv][].

    Returns:
        Optimised Slisemap or Slipmap object. This is not the same object as the input!

    Deprecated:
        1.6: Use the uncerlying function directly instead
    """
    if X_test is None or y_test is None:
        _deprecated(optimise, optimise_with_cv)
        return optimise_with_cv(sm, **kwargs)
    else:
        _deprecated(optimise, optimise_with_test)
        return optimise_with_test(sm, X_test, y_test, **kwargs)


optimize = optimise
