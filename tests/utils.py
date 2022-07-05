import random
from itertools import combinations
from typing import Tuple, Union

import numpy as np
import torch
from slisemap.slisemap import Slisemap
from slisemap.local_models import *


def set_seed(seed):
    np.random.seed(seed)
    random.seed(42)
    torch.manual_seed(42)


def get_slisemap(
    n=20, m=5, classes=0, seed=None, randomB=False, lasso=1e-4, **kwargs
) -> Slisemap:
    if seed is None:
        npr = np.random
    else:
        npr = np.random.RandomState(seed)
    X = npr.normal(size=(n, m))
    if classes > 0:
        y = npr.uniform(size=(n, classes))
        sm = Slisemap(
            X,
            y,
            local_model=logistic_regression,
            local_loss=logistic_regression_loss,
            coefficients=(m + 1) * (classes - 1),
            lasso=lasso * 10,
            **kwargs,
        )
    else:
        y = npr.normal(size=n)
        sm = Slisemap(X, y, lasso=lasso, **kwargs)
    if randomB:
        sm._B = torch.normal(0, 1, sm.B.shape, **sm.tensorargs)
    return sm


def get_slisemap2(
    n=20,
    m=5,
    s=0.3,
    classes=False,
    seed=None,
    randomB=False,
    cheat=False,
    lasso=1e-4,
    **kwargs,
) -> Tuple[Slisemap, np.ndarray]:
    cl, X, y, B = get_rsynth(n, m, s=s, seed=seed)
    if classes:
        y = 1 / (1 + np.exp(-y))
        sm = Slisemap(
            X,
            y,
            local_model=logistic_regression,
            local_loss=logistic_regression_loss,
            lasso=lasso * 10,
            random_state=seed,
            **kwargs,
        )
    else:
        sm = Slisemap(X, y, lasso=lasso, random_state=seed, **kwargs)
    if randomB:
        sm._B = torch.normal(
            0, 1, sm.B.shape, **sm.tensorargs, generator=sm._random_state
        )
    if cheat:
        angles = np.pi * cl / 3  # Assume k=3, d=2
        Z = np.stack((np.sin(angles), np.cos(angles)), 1)
        sm._Z = torch.as_tensor(Z, **sm.tensorargs)
    return (sm, cl)


def assert_allclose(x, y, label="", *args, **kwargs):
    assert np.allclose(x, y, *args, **kwargs), f"{label}: {x} != {y}"


def all_finite(x: Union[float, np.ndarray], *args: Union[float, np.ndarray]) -> bool:
    if len(args) > 0:
        return np.all(np.all(np.isfinite(y)) for y in [x, *args])
    return np.all(np.isfinite(x))


def assert_approx_ge(x, y, label=None, tolerance=0.05):
    tolerance *= (abs(x) + abs(y)) * 0.5
    if label:
        assert np.all(x > y - tolerance), f"{label}: {x} !>= {y}"
    else:
        assert np.all(x > y - tolerance), f"{x} !>= {y}"


def get_rsynth(
    N: int = 100,
    M: int = 11,
    k: int = 3,
    s: float = 0.25,
    se: float = 0.075,
    seed: Union[None, int, np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic regression data

    Note: X is already preprocessed (normalised and intercept added)

    Args:
        N (int, optional): Number of rows in X. Defaults to 100.
        M (int, optional): Number of columns in X. Defaults to 11.
        k (int, optional): Number of clusters (with their own true model). Defaults to 3.
        s (float, optional): Scale for the randomisation of the cluster centers. Defaults to 0.25.
        se (float, optional): Scale for the noise of y. Defaults to 0.075.
        seed (Union[None, int, np.random.RandomState], optional): Local random seed. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: cluster_ids[N], X[N,M+1], y[N], B[k,M+1].
    """
    if seed is None:
        npr = np.random
    elif isinstance(seed, np.random.RandomState):
        npr = seed
    else:
        npr = np.random.RandomState(seed)

    B = npr.normal(size=[k, M + 1])  # k x (M+1)
    while not _are_models_different(B):
        B = npr.normal(size=[k, M + 1])
    c = npr.normal(scale=s, size=[k, M])  # k X M
    j = np.repeat(np.arange(k), N // k + 1)[:N]  # N
    e = npr.normal(scale=se, size=N)  # N
    X = npr.normal(loc=c[j, :])  # N x M
    X = (X - np.mean(X, 0, keepdims=True)) / np.std(X, 0, keepdims=True)
    yhat = np.sum(B[j, :-1] * X, 1) + B[j, -1] + e
    return j, X, yhat, B


def _are_models_different(B: np.ndarray, treshold: float = 0.5) -> bool:
    for i, j in combinations(range(B.shape[0]), 2):
        cosine_similarity = B[i] @ B[j] / (B[i] @ B[i] * B[j] @ B[j])
        if cosine_similarity > treshold:
            return False
    return True
