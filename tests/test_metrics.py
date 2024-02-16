from functools import partial

import numpy as np
import pytest

from slisemap.metrics import (
    accuracy,
    cluster_purity,
    coherence,
    coverage,
    entropy,
    euclidean_nearest_neighbours,
    fidelity,
    kernel_neighbours,
    kernel_purity,
    kmeans_matching,
    median_loss,
    precision,
    radius_neighbours,
    recall,
    relevance,
    slisemap_loss,
    stability,
)
from slisemap.slipmap import Slipmap
from slisemap.slisemap import Slisemap

from .utils import all_finite, assert_approx_ge, get_slisemap, get_slisemap2


@pytest.fixture(scope="session")
def metrics_data():
    sm1, c = get_slisemap2(30, 4, randomB=True, seed=357938)
    sm2 = sm1.copy()
    sm2.lbfgs()
    sm3 = get_slisemap(30, 4, classes=3, randomB=True, seed=53498)
    sm4 = sm3.copy()
    sm4.lbfgs()
    sm5 = Slipmap.convert(sm1)
    sm6 = sm5.copy()
    sm6.lbfgs(only_B=True)
    sm6.lbfgs()
    return (sm1, sm2, sm3, sm4, sm5, sm6, c)


def test_loss(metrics_data):
    for sm1, sm2 in zip(metrics_data[::2], metrics_data[1::2]):
        loss1 = slisemap_loss(sm1)
        loss2 = slisemap_loss(sm2)
        ent1 = entropy(sm1)
        ent2 = entropy(sm2)
        assert all_finite(loss1, loss2, ent1, ent2)
        assert loss2 <= loss1


def test_fidelity(metrics_data):
    for sm1, sm2 in zip(metrics_data[::2], metrics_data[1::2]):
        for neighbours in (
            None,
            partial(euclidean_nearest_neighbours, k=10),
            kernel_neighbours,
            radius_neighbours,
            metrics_data[-1],
        ):
            val1 = fidelity(sm1, neighbours)
            val2 = fidelity(sm2, neighbours)
            assert all_finite(val1, val2)
            assert val1 >= val2


def test_coverage(metrics_data):
    for sm1, sm2 in zip(metrics_data[::2], metrics_data[1::2]):
        for neighbours in (
            None,
            partial(euclidean_nearest_neighbours, k=10),
            kernel_neighbours,
            radius_neighbours,
            metrics_data[-1],
        ):
            val1 = coverage(sm1, 0.5, neighbours)
            val2 = coverage(sm2, 0.5, neighbours)
            assert all_finite(val1, val2)
            assert val1 <= val2


def test_median_loss(metrics_data):
    for sm1, sm2 in zip(metrics_data[::2], metrics_data[1::2]):
        for neighbours in (
            None,
            partial(euclidean_nearest_neighbours, k=10),
            kernel_neighbours,
            radius_neighbours,
            metrics_data[-1],
        ):
            val1 = median_loss(sm1, neighbours)
            val2 = median_loss(sm2, neighbours)
            assert all_finite(val1, val2)
            assert val1 >= val2


def test_stability(metrics_data):
    for sm1, sm2 in zip(metrics_data[::2], metrics_data[1::2]):
        if isinstance(sm1, Slipmap):
            continue  # Slipmap.get_B is not smooth
        for neighbours in (
            None,
            partial(euclidean_nearest_neighbours, k=10),
            kernel_neighbours,
            radius_neighbours,
            metrics_data[-1],
        ):
            val1 = stability(sm1, neighbours)
            val2 = stability(sm2, neighbours)
            assert all_finite(val1, val2)
            assert val1 >= val2, f"{val1} < {val2}, n={neighbours}, sm={sm1}"


def test_coherence(metrics_data):
    for sm1, sm2 in zip(metrics_data[::2], metrics_data[1::2]):
        if isinstance(sm1, Slipmap):
            continue  # Slipmap.get_B is not smooth
        for neighbours in (
            None,
            partial(euclidean_nearest_neighbours, k=10),
            kernel_neighbours,
            radius_neighbours,
            metrics_data[-1],
        ):
            val1 = coherence(sm1, neighbours)
            val2 = coherence(sm2, neighbours)
            assert all_finite(val1, val2)
            assert val1 > val2 * 0.99, f"{val1} < {val2}, n={neighbours}, sm={sm1}"


def test_relevance(metrics_data):
    for sm1, sm2 in zip(metrics_data[::2], metrics_data[1::2]):
        if sm1.o < 2:
            pred = lambda x: np.random.normal()  # noqa: E731
            assert all_finite(relevance(sm1, pred, 0.5))
            assert all_finite(relevance(sm2, pred, 0.5))
    # Does not (currently) work for num_classes > 2
    sm = get_slisemap(classes=2)
    pred = lambda x: np.random.uniform()  # noqa: E731
    assert all_finite(relevance(sm, pred, 0.5))


def test_purity(metrics_data):
    c = metrics_data[-1]
    for sm1, sm2 in zip(metrics_data[::2], metrics_data[1::2]):
        cp1 = cluster_purity(sm1, c)
        dp1 = kernel_purity(sm1, c, 1.0)
        lp1 = kernel_purity(sm1, c, 1.0, True)
        cp2 = cluster_purity(sm2, c)
        dp2 = kernel_purity(sm2, c, 1.0)
        lp2 = kernel_purity(sm2, c, 1.0, True)
        assert all_finite(cp1, cp2, dp1, dp2, lp1, lp2)
        if isinstance(sm1, Slipmap):
            assert_approx_ge(cp2, cp1, f"cluster purity ({type(sm1).__name__})")
        if isinstance(sm1, Slisemap):
            assert_approx_ge(dp2, dp1, f"embedding purity ({type(sm1).__name__})")
        assert_approx_ge(lp2, lp1, "loss purity")


def test_prec_rec(metrics_data):
    for sm1, sm2 in zip(metrics_data[::2], metrics_data[1::2]):
        p1 = precision(sm1, 1.0, 1.0)
        r1 = recall(sm1, 1.0, 1.0)
        p2 = precision(sm2, 1.0, 1.0)
        r2 = recall(sm2, 1.0, 1.0)
        assert all_finite(r1, r2, p1, p2)
        assert_approx_ge(p2, p1, "precision")
        assert_approx_ge(r2, r1, "recall")


def test_accuracy(metrics_data):
    for sm1, sm2 in zip(metrics_data[::2], metrics_data[1::2]):
        val1 = accuracy(sm1)
        val2 = accuracy(sm2)
        assert all_finite(val1, val2)
        assert val1 > val2
        X_test = np.random.normal(size=(2, sm1.m))
        a = accuracy(sm2, X_test, np.random.uniform(size=(2, sm1.o)))
        assert all_finite(a)


def test_kmeans_matching(metrics_data):
    for sm1, sm2 in zip(metrics_data[::2], metrics_data[1::2]):
        kmm1 = kmeans_matching(sm1)
        assert all_finite(kmm1)
        kmm2 = kmeans_matching(sm2)
        assert all_finite(kmm2)
        assert np.all(kmm1 <= kmm2)
