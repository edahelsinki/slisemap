import random
from itertools import combinations
from typing import Tuple, Union

import numpy as np
import torch

from slisemap.local_models import LogisticLogRegression, LogisticRegression
from slisemap.slisemap import Slisemap


def set_seed(seed):
    np.random.seed(seed)
    random.seed(42)
    torch.manual_seed(42)


def get_slisemap(
    n=20, m=5, classes=0, seed=None, randomB=False, lasso=1e-4, **kwargs
) -> Slisemap:
    npr = np.random if seed is None else np.random.RandomState(seed)
    X = npr.normal(size=(n, m))
    if classes > 0:
        y = npr.uniform(size=(n, classes))
        sm = Slisemap(
            X,
            y,
            local_model=LogisticRegression,
            coefficients=(m + 1) * (classes - 1),
            lasso=lasso * 10,
            **kwargs,
        )
    else:
        y = npr.normal(size=n)
        sm = Slisemap(X, y, lasso=lasso, **kwargs)
    if randomB:
        sm._B = torch.normal(0, 1, sm._B.shape, **sm.tensorargs)
        sm._B0 = sm._B.clone()
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
    k=3,
    **kwargs,
) -> Tuple[Slisemap, np.ndarray]:
    cl, X, y, B = get_rsynth(n, m, s=s, k=k, seed=seed)
    if classes:
        y = 1 / (1 + np.exp(-y))
        sm = Slisemap(
            X,
            np.stack((y, 1 - y), -1),
            local_model=LogisticLogRegression,
            lasso=lasso * 10,
            **kwargs,
        )
    else:
        sm = Slisemap(X, y, lasso=lasso, **kwargs)
    if randomB:
        if seed is not None:
            generator = torch.Generator(sm._B.device).manual_seed(seed)
            sm._B = torch.normal(sm._B * 0.0, 1.0, generator=generator)
        else:
            sm._B = torch.normal(sm._B * 0.0, 1.0)
    if cheat:
        angles = 2 * np.pi * cl / 3  # Assume k=3, d=2
        Z = np.stack((np.sin(angles), np.cos(angles)), 1)
        end = [sm.radius * 0.99, sm.radius * 1.01]
        Z = Z * np.linspace(end, end[::-1], len(cl))
        sm._Z = torch.as_tensor(Z, **sm.tensorargs)
        sm._B = torch.as_tensor(B[cl], **sm.tensorargs)
        sm._normalise()
    return (sm, cl)


def assert_allclose(x, y, label="", *args, **kwargs):
    if isinstance(x, torch.Tensor):
        allclose = torch.allclose(x, y, *args, **kwargs)
        maxdiff = torch.max(torch.abs(x - y)).cpu().detach().item()
    else:
        allclose = np.allclose(x, y, *args, **kwargs)
        maxdiff = np.max(np.abs(x - y))
    assert allclose, f"{label}: {x} != {y}\nmax abs diff: {maxdiff}"


def all_finite(x: Union[float, np.ndarray], *args: Union[float, np.ndarray]) -> bool:
    if len(args) > 0:
        return np.all(np.all(np.isfinite(y)) for y in [x, *args])
    return np.all(np.isfinite(x))


def assert_approx_ge(x, y, label=None, tolerance=0.05):
    tolerance *= (np.abs(x) + np.abs(y)) * 0.5
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
    seed: Union[None, int, np.random.RandomState, np.random.Generator] = None,
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
    elif isinstance(seed, (np.random.RandomState, np.random.Generator)):
        npr = seed
    else:
        npr = np.random.Generator(np.random.PCG64(seed))

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
