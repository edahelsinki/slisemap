import numpy as np
import pytest

from slisemap.metrics import *

from .utils import *


@pytest.fixture(scope="session")
def metrics_data():
    sm1, c = get_slisemap2(30, 4, randomB=True, seed=357938)
    sm2 = sm1.copy()
    sm2.lbfgs()
    sm3 = get_slisemap(30, 4, classes=3, randomB=True, seed=53498)
    sm4 = sm3.copy()
    sm4.lbfgs()
    return (sm1, sm2, sm3, sm4, c)


def test_loss(metrics_data):
    sm1, sm2, sm3, sm4, c = metrics_data
    loss1 = slisemap_loss(sm1)
    loss2 = slisemap_loss(sm2)
    ent1 = entropy(sm1)
    ent2 = entropy(sm2)
    assert all_finite(loss1, loss2, ent1, ent2)
    assert loss2 <= loss1
    loss3 = slisemap_loss(sm3)
    loss4 = slisemap_loss(sm4)
    ent3 = entropy(sm3)
    ent4 = entropy(sm4)
    assert all_finite(loss3, loss4, ent3, ent4)
    assert loss4 <= loss3


def test_fidelity(metrics_data):
    sm1, sm2, sm3, sm4, c = metrics_data
    fid1 = [
        fidelity(sm1),
        fidelity(sm1, euclidean_nearest_neighbours, k=10),
        fidelity(sm1, kernel_neighbours),
        fidelity(sm1, radius_neighbours),
        fidelity(sm1, c),
    ]
    assert all_finite(fid1)
    fid2 = [
        fidelity(sm2),
        fidelity(sm2, euclidean_nearest_neighbours, k=10),
        fidelity(sm2, kernel_neighbours),
        fidelity(sm2, radius_neighbours),
        fidelity(sm2, c),
    ]
    assert all_finite(fid2)
    assert np.all(fid1 >= fid2)
    fid3 = [
        fidelity(sm3),
        fidelity(sm3, euclidean_nearest_neighbours, k=10),
        fidelity(sm3, kernel_neighbours),
        fidelity(sm3, radius_neighbours),
        fidelity(sm3, c),
    ]
    assert all_finite(fid3)
    fid4 = [
        fidelity(sm4),
        fidelity(sm4, euclidean_nearest_neighbours, k=10),
        fidelity(sm4, kernel_neighbours),
        fidelity(sm4, radius_neighbours),
        fidelity(sm4, c),
    ]
    assert all_finite(fid4)
    assert np.all(fid3 >= fid4)


def test_coverage(metrics_data):
    sm1, sm2, sm3, sm4, c = metrics_data
    cov1 = [
        coverage(sm1, 0.5),
        coverage(sm1, 0.5, euclidean_nearest_neighbours, k=10),
        coverage(sm1, 0.5, kernel_neighbours),
        coverage(sm1, 0.5, radius_neighbours),
        coverage(sm1, 0.5, c),
    ]
    assert all_finite(cov1)
    cov2 = [
        coverage(sm2, 0.5),
        coverage(sm2, 0.5, euclidean_nearest_neighbours, k=10),
        coverage(sm2, 0.5, kernel_neighbours),
        coverage(sm2, 0.5, radius_neighbours),
        coverage(sm2, 0.5, c),
    ]
    assert all_finite(cov2)
    assert np.all(cov1 <= cov2)
    cov3 = [
        coverage(sm3, 0.5),
        coverage(sm3, 0.5, euclidean_nearest_neighbours, k=10),
        coverage(sm3, 0.5, kernel_neighbours),
        coverage(sm3, 0.5, radius_neighbours),
        coverage(sm3, 0.5, c),
    ]
    assert all_finite(cov3)


def test_median_loss(metrics_data):
    sm1, sm2, sm3, sm4, c = metrics_data
    med1 = [
        median_loss(sm1),
        median_loss(sm1, euclidean_nearest_neighbours, k=10),
        median_loss(sm1, kernel_neighbours),
        median_loss(sm1, radius_neighbours),
        median_loss(sm1, c),
    ]
    assert all_finite(med1)
    med2 = [
        median_loss(sm2),
        median_loss(sm2, euclidean_nearest_neighbours, k=10),
        median_loss(sm2, kernel_neighbours),
        median_loss(sm2, radius_neighbours),
        median_loss(sm2, c),
    ]
    assert all_finite(med2)
    assert np.all(med1 >= med2)
    med3 = [
        median_loss(sm3),
        median_loss(sm3, euclidean_nearest_neighbours, k=10),
        median_loss(sm3, kernel_neighbours),
        median_loss(sm3, radius_neighbours),
        median_loss(sm3, c),
    ]
    assert all_finite(med3)


def test_stability(metrics_data):
    sm1, sm2, sm3, sm4, c = metrics_data
    stab1 = [
        stability(sm1),
        stability(sm1, euclidean_nearest_neighbours, k=10),
        stability(sm1, kernel_neighbours),
        stability(sm1, radius_neighbours),
        stability(sm1, c),
    ]
    assert all_finite(stab1)
    stab2 = [
        stability(sm2),
        stability(sm2, euclidean_nearest_neighbours, k=10),
        stability(sm2, kernel_neighbours),
        stability(sm2, radius_neighbours),
        stability(sm2, c),
    ]
    assert all_finite(stab2)
    assert np.all(stab1 >= stab2)
    stab3 = [
        stability(sm3),
        stability(sm3, euclidean_nearest_neighbours, k=10),
        stability(sm3, kernel_neighbours),
        stability(sm3, radius_neighbours),
        stability(sm3, c),
    ]
    assert all_finite(stab3)


def test_coherence(metrics_data):
    sm1, sm2, sm3, sm4, c = metrics_data
    coh1 = [
        coherence(sm1),
        coherence(sm1, euclidean_nearest_neighbours, k=10),
        coherence(sm1, kernel_neighbours),
        coherence(sm1, radius_neighbours),
        coherence(sm1, c),
    ]
    assert all_finite(coh1)
    coh2 = [
        coherence(sm2),
        coherence(sm2, euclidean_nearest_neighbours, k=10),
        coherence(sm2, kernel_neighbours),
        coherence(sm2, radius_neighbours),
        coherence(sm2, c),
    ]
    assert all_finite(coh2)
    assert np.all(coh1[1:] >= coh2[1:])
    coh3 = [
        coherence(sm3),
        coherence(sm3, euclidean_nearest_neighbours, k=10),
        coherence(sm3, kernel_neighbours),
        coherence(sm3, radius_neighbours),
        coherence(sm3, c),
    ]
    assert all_finite(coh3)


def test_relevance(metrics_data):
    sm1, sm2, sm3, sm4, c = metrics_data
    pred = lambda x: np.random.normal()
    assert all_finite(relevance(sm1, pred, 0.5))
    assert all_finite(relevance(sm2, pred, 0.5))
    # Does not (currently) work for num_classes > 2
    sm = get_slisemap(classes=2)
    pred = lambda x: np.random.uniform()
    assert all_finite(relevance(sm, pred, 0.5))


def test_purity(metrics_data):
    sm1, sm2, sm3, sm4, c = metrics_data
    cp1 = cluster_purity(sm1, c)
    dp1 = kernel_purity(sm1, c, 1.0)
    lp1 = kernel_purity(sm1, c, 1.0, True)
    cp2 = cluster_purity(sm2, c)
    dp2 = kernel_purity(sm2, c, 1.0)
    lp2 = kernel_purity(sm2, c, 1.0, True)
    assert all_finite(cp1, cp2, dp1, dp2, lp1, lp2)
    # assert_approx_ge(cp2, cp1, "cluster purity")
    # Sometimes the cluster purity gets worse!
    assert_approx_ge(dp2, dp1, "embedding purity")
    assert_approx_ge(lp2, lp1, "loss purity")


def test_prec_rec(metrics_data):
    sm1, sm2, sm3, sm4, c = metrics_data
    p1 = precision(sm1, 1.0, 1.0)
    r1 = recall(sm1, 1.0, 1.0)
    p2 = precision(sm2, 1.0, 1.0)
    r2 = recall(sm2, 1.0, 1.0)
    assert all_finite(r1, r2, p1, p2)
    assert_approx_ge(p2, p1, "precision")
    assert_approx_ge(r2, r1, "recall")


def test_accuracy(metrics_data):
    sm1, sm2, sm3, sm4, c = metrics_data
    for f in [False, True]:
        ab1 = accuracy(sm1, between=True, fidelity=f)
        an1 = accuracy(sm1, between=True, fidelity=f, optimise=False)
        af1 = accuracy(sm1, between=False, fidelity=f)
        assert all_finite(ab1, an1, af1)
        ab2 = accuracy(sm2, between=True, fidelity=f)
        an2 = accuracy(sm2, between=True, fidelity=f, optimise=False)
        af2 = accuracy(sm2, between=False, fidelity=f)
        assert all_finite(ab2, an2, af2)
        assert ab1 > ab2
        assert an1 > an2
        assert af1 > af2
        assert an2 > ab2 * 0.95
        X_test = np.random.normal(size=(2, sm1.m))
        a = accuracy(sm1, X_test, np.random.uniform(size=2), fidelity=f)
        assert all_finite(a)
        a = accuracy(sm3, X_test, np.random.uniform(size=(2, sm3.o)), fidelity=f)
        assert all_finite(a)


def test_kmeans_matching(metrics_data):
    sm1, sm2, sm3, sm4, c = metrics_data
    kmm1 = kmeans_matching(sm1)
    assert all_finite(kmm1)
    kmm2 = kmeans_matching(sm2)
    assert all_finite(kmm2)
    assert np.all(kmm1 <= kmm2)
    kmm3 = kmeans_matching(sm3)
    assert all_finite(kmm3)
    kmm4 = kmeans_matching(sm4)
    assert all_finite(kmm4)
    assert np.all(kmm3 <= kmm4)
