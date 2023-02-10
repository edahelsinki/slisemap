import numpy as np
from slisemap.metrics import *
from .utils import *


def test_loss():
    sm = get_slisemap()
    loss0 = slisemap_loss(sm)
    sm.lbfgs()
    loss1 = slisemap_loss(sm)
    ent = slisemap_entropy(sm)
    assert all_finite(loss0, loss1, ent)
    assert loss1 <= loss0
    sm = get_slisemap(classes=3)
    loss0 = slisemap_loss(sm)
    sm.restore()
    sm.lbfgs()
    loss1 = slisemap_loss(sm)
    ent = slisemap_entropy(sm)
    assert all_finite(loss0, loss1, ent)
    assert loss1 <= loss0


def test_fidelity():
    sm, c = get_slisemap2(30, 4)
    fid1 = [
        fidelity(sm),
        fidelity(sm, euclidean_nearest_neighbours, k=10),
        fidelity(sm, kernel_neighbours),
        fidelity(sm, radius_neighbours),
        fidelity(sm, c),
    ]
    assert all_finite(fid1)
    sm.lbfgs()
    fid2 = [
        fidelity(sm),
        fidelity(sm, euclidean_nearest_neighbours, k=10),
        fidelity(sm, kernel_neighbours),
        fidelity(sm, radius_neighbours),
        fidelity(sm, c),
    ]
    assert all_finite(fid2)
    assert np.all(fid1 >= fid2)
    sm = get_slisemap(30, 4, classes=3)
    fid3 = [
        fidelity(sm),
        fidelity(sm, euclidean_nearest_neighbours, k=10),
        fidelity(sm, kernel_neighbours),
        fidelity(sm, radius_neighbours),
        fidelity(sm, c),
    ]
    assert all_finite(fid3)


def test_coverage():
    sm, c = get_slisemap2(30, 4)
    cov1 = [
        coverage(sm, 0.5),
        coverage(sm, 0.5, euclidean_nearest_neighbours, k=10),
        coverage(sm, 0.5, kernel_neighbours),
        coverage(sm, 0.5, radius_neighbours),
        coverage(sm, 0.5, c),
    ]
    assert all_finite(cov1)
    sm.lbfgs()
    cov2 = [
        coverage(sm, 0.5),
        coverage(sm, 0.5, euclidean_nearest_neighbours, k=10),
        coverage(sm, 0.5, kernel_neighbours),
        coverage(sm, 0.5, radius_neighbours),
        coverage(sm, 0.5, c),
    ]
    assert all_finite(cov2)
    assert np.all(cov1 <= cov2)
    sm = get_slisemap(30, 4, classes=3)
    cov3 = [
        coverage(sm, 0.5),
        coverage(sm, 0.5, euclidean_nearest_neighbours, k=10),
        coverage(sm, 0.5, kernel_neighbours),
        coverage(sm, 0.5, radius_neighbours),
        coverage(sm, 0.5, c),
    ]
    assert all_finite(cov3)


def test_median_loss():
    set_seed(12945784)
    sm, c = get_slisemap2(30, 4)
    med1 = [
        median_loss(sm),
        median_loss(sm, euclidean_nearest_neighbours, k=10),
        median_loss(sm, kernel_neighbours),
        median_loss(sm, radius_neighbours),
        median_loss(sm, c),
    ]
    assert all_finite(med1)
    sm.lbfgs()
    med2 = [
        median_loss(sm),
        median_loss(sm, euclidean_nearest_neighbours, k=10),
        median_loss(sm, kernel_neighbours),
        median_loss(sm, radius_neighbours),
        median_loss(sm, c),
    ]
    assert all_finite(med2)
    assert np.all(med1 >= med2)
    sm = get_slisemap(30, 4, classes=3)
    med3 = [
        median_loss(sm),
        median_loss(sm, euclidean_nearest_neighbours, k=10),
        median_loss(sm, kernel_neighbours),
        median_loss(sm, radius_neighbours),
        median_loss(sm, c),
    ]
    assert all_finite(med3)


def test_stability():
    sm, c = get_slisemap2(30, 4, randomB=True)
    stab1 = [
        stability(sm),
        stability(sm, euclidean_nearest_neighbours, k=10),
        stability(sm, kernel_neighbours),
        stability(sm, radius_neighbours),
        stability(sm, c),
    ]
    assert all_finite(stab1)
    sm.lbfgs()
    stab2 = [
        stability(sm),
        stability(sm, euclidean_nearest_neighbours, k=10),
        stability(sm, kernel_neighbours),
        stability(sm, radius_neighbours),
        stability(sm, c),
    ]
    assert all_finite(stab2)
    assert np.all(stab1 >= stab2)
    sm = get_slisemap(30, 4, classes=3, randomB=True)
    stab3 = [
        stability(sm),
        stability(sm, euclidean_nearest_neighbours, k=10),
        stability(sm, kernel_neighbours),
        stability(sm, radius_neighbours),
        stability(sm, c),
    ]
    assert all_finite(stab3)


def test_coherence():
    sm, c = get_slisemap2(30, 4, randomB=True)
    coh1 = [
        coherence(sm),
        coherence(sm, euclidean_nearest_neighbours, k=10),
        coherence(sm, kernel_neighbours),
        coherence(sm, radius_neighbours),
        coherence(sm, c),
    ]
    assert all_finite(coh1)
    sm.lbfgs()
    coh2 = [
        coherence(sm),
        coherence(sm, euclidean_nearest_neighbours, k=10),
        coherence(sm, kernel_neighbours),
        coherence(sm, radius_neighbours),
        coherence(sm, c),
    ]
    assert all_finite(coh2)
    assert np.all(coh1[1:] >= coh2[1:])
    sm = get_slisemap(30, 4, classes=3, randomB=True)
    coh3 = [
        coherence(sm),
        coherence(sm, euclidean_nearest_neighbours, k=10),
        coherence(sm, kernel_neighbours),
        coherence(sm, radius_neighbours),
        coherence(sm, c),
    ]
    assert all_finite(coh3)


def test_relevance():
    sm = get_slisemap()
    pred = lambda x: np.random.normal()
    assert all_finite(relevance(sm, pred, 0.5))
    sm.lbfgs()
    assert all_finite(relevance(sm, pred, 0.5))
    # Does not (currently) work for num_classes > 2
    sm = get_slisemap(classes=2)
    pred = lambda x: np.random.uniform()
    assert all_finite(relevance(sm, pred, 0.5))


def test_purity():
    sm, clusters = get_slisemap2(50, 5)
    cp1 = cluster_purity(sm, clusters)
    dp1 = kernel_purity(sm, clusters, 1.0)
    lp1 = kernel_purity(sm, clusters, 1.0, True)
    sm.lbfgs()
    cp2 = cluster_purity(sm, clusters)
    dp2 = kernel_purity(sm, clusters, 1.0)
    lp2 = kernel_purity(sm, clusters, 1.0, True)
    # print(cp1, cp2, dp1, dp2, lp1, lp2)
    assert all_finite(cp1, cp2, dp1, dp2, lp1, lp2)
    # assert_approx_ge(cp2, cp1, "cluster purity")
    # Sometimes the cluster purity gets worse!
    assert_approx_ge(dp2, dp1, "embedding purity")
    assert_approx_ge(lp2, lp1, "loss purity")


def test_prec_rec():
    sm, _ = get_slisemap2(50, 5)
    p1 = precision(sm, 1.0, 1.0)
    r1 = recall(sm, 1.0, 1.0)
    sm.lbfgs()
    p2 = precision(sm, 1.0, 1.0)
    r2 = recall(sm, 1.0, 1.0)
    # print(r1, r2, p1, p2)
    assert all_finite(r1, r2, p1, p2)
    assert_approx_ge(p2, p1, "precision")
    assert_approx_ge(r2, r1, "recall")


def test_accuracy():
    set_seed(73409409)
    for f in [False, True]:
        sm, _ = get_slisemap2(30, 4, randomB=True)
        ab1 = accuracy(sm, between=True, fidelity=f)
        an1 = accuracy(sm, between=True, fidelity=f, optimise=False)
        af1 = accuracy(sm, between=False, fidelity=f)
        assert all_finite(ab1, an1, af1)
        sm.lbfgs()
        ab2 = accuracy(sm, between=True, fidelity=f)
        an2 = accuracy(sm, between=True, fidelity=f, optimise=False)
        af2 = accuracy(sm, between=False, fidelity=f)
        assert all_finite(ab2, an2, af2)
        assert ab1 > ab2
        assert an1 > an2
        assert af1 > af2
        assert an2 > ab2 * 0.95
        X_test = np.random.normal(size=(2, 4))
        a = accuracy(sm, X_test, np.random.uniform(size=2), fidelity=f)
        assert all_finite(a)
        sm = get_slisemap(30, 4, classes=3, intercept=True)
        a = accuracy(sm, X_test, np.random.uniform(size=(2, 3)), fidelity=f)
        assert all_finite(a)


def test_kmeans_matching():
    sm, c = get_slisemap2(30, 4, randomB=True)
    kmm1 = kmeans_matching(sm)
    assert all_finite(kmm1)
    sm.lbfgs()
    kmm2 = kmeans_matching(sm)
    assert all_finite(kmm2)
    assert np.all(kmm1 <= kmm2)
    sm = get_slisemap(30, 4, classes=3, randomB=True)
    assert all_finite(kmeans_matching(sm))
