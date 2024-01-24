from timeit import default_timer as timer

import pytest
from slisemap.utils import *

from .utils import *


def test_LBFGS():
    sm1 = get_slisemap()
    prev = np.inf
    for _ in range(5):
        v = sm1.lbfgs(max_iter=100)
        assert v <= prev
        prev = v


def test_LBFGS_timelimit():
    sm1 = get_slisemap(seed=42)
    sm2 = sm1.copy()
    t0 = timer()
    sm1.lbfgs(max_iter=100, max_eval=125, time_limit=100)
    t1 = timer()
    sm2.lbfgs(max_iter=100, max_eval=125, time_limit=0.01)
    t2 = timer()
    assert (t1 - t0) > (t2 - t1)


def test_convergence():
    cc = CheckConvergence(0)
    assert cc.has_converged(1)
    cc = CheckConvergence(0.01)
    assert not cc.has_converged(1)
    assert cc.has_converged(1)
    cc = CheckConvergence(1)
    assert not cc.has_converged([2, 3], lambda: 1)
    assert not cc.has_converged((1, 1), lambda: 2)
    assert cc.has_converged((1, 1), lambda: 3)
    assert cc.optimal == 2
    cc = CheckConvergence(100, 2)
    assert not cc.has_converged([1, 3])
    assert cc.has_converged((1, 1))


def test_global_model():
    sm, _ = get_slisemap2(40, 5)
    global_model(sm._X, sm._Y, sm.local_model, sm.local_loss, sm.q, 0.01, 0.01)
    sm, _ = get_slisemap2(40, 5, classes=True)
    global_model(sm._X, sm._Y, sm.local_model, sm.local_loss, sm.q, 0.01, 0.01)


def test_PCA():
    set_seed(453789359)
    x = torch.normal(0, 1, (5, 3))
    assert PCA_rotation(x, 2, full=True).shape == (3, 2)
    assert PCA_rotation(x, 2, full=False).shape == (3, 2)
    assert PCA_rotation(x, 5, full=True).shape == (3, 3)
    assert PCA_rotation(x, 5, full=False).shape == (3, 3)
    with pytest.warns(SlisemapWarning, match="PCA"):
        assert PCA_rotation(x * np.nan, 5, full=True).shape == (3, 3)
    with pytest.warns(SlisemapWarning, match="PCA"):
        assert PCA_rotation(x * np.nan, 5, full=False).shape == (3, 3)
    assert_allclose(
        np.abs(PCA_rotation(x, 3, center=False).numpy()),
        np.abs(np.linalg.svd(x.numpy())[2].T),
    )
    x = torch.normal(0, 1, (3, 5))
    assert PCA_rotation(x, 2, full=False).shape == (5, 2)
    assert PCA_rotation(x, 2, full=True).shape == (5, 2)
    assert_allclose(
        np.abs(PCA_rotation(x, 2, center=False).numpy()),
        np.abs(np.linalg.svd(x.numpy())[2][:2].T),
        "rotation",
    )


def test_PCA_for_rotation():
    for i in range(1, 6):
        X = torch.normal(0, 1, size=(10 + i * 5, i))
        XR = X @ PCA_rotation(X, center=False)
        assert X.shape == XR.shape
        assert torch.allclose(torch.cdist(X, X), torch.cdist(XR, XR), atol=2e-3)
        Xvar = torch.sqrt(torch.sum(X**2) / X.shape[0])
        XRvar = torch.sqrt(torch.sum(XR**2) / X.shape[0])
        assert torch.allclose(Xvar, XRvar, atol=1e-6)
        Xmd = torch.sqrt(torch.sum(torch.mean(X, 0) ** 2))
        XRmd = torch.sqrt(torch.sum(torch.mean(XR, 0) ** 2))
        assert torch.allclose(Xmd, XRmd, atol=1e-6)


def test_dict():
    dict_array(dict())
    dict_array(dict(a=2, b="asd", c=[1, 2, 3], d=np.arange(3), e=0.2, f=None))
    dict_concat(dict() for _ in range(3))
    dict_concat(
        dict(a=2, b="asd", c=list(range(i)), d=np.arange(i), e=0.2, f=None)
        for i in range(2, 3)
    )


def test_to_tensor():
    X = np.random.normal(0, 1, (5, 5))
    X2 = torch.as_tensor(X)
    X3 = [c for c in X]
    X4 = {k: v for k, v in enumerate(X.T)}

    def test(X):
        Xn, r, c = to_tensor(X)
        assert torch.allclose(X2, Xn)
        assert r is None or X2.shape[0] == len(r)
        assert c is None or X2.shape[1] == len(c)

    test(X)
    test(X2)
    with pytest.warns(UserWarning, match="extremely slow"):
        test(X3)
    test(X4)

    try:
        import pandas

        test(pandas.DataFrame(X))
        test(pandas.DataFrame(X4))
        rows = [1, 4, 2, 0, 3]
        assert_allclose(rows, to_tensor(pandas.DataFrame(X4).iloc[rows])[1])
    except ImportError:
        pass

    Slisemap(X=X4, y=X3[1], lasso=0.1)


def test_metadata():
    sm, _ = get_slisemap2(40, 5)
    assert len(sm.metadata.get_coefficients()) == 6
    assert len(sm.metadata.get_variables(False)) == 5
    assert len(sm.metadata.get_variables(True)) == 6
    sm.metadata.set_variables(range(6), add_intercept=False)
    sm.metadata.set_variables(range(5), add_intercept=True)
    assert len(sm.metadata.get_variables(False)) == 5
    assert len(sm.metadata.get_variables(True)) == 6
    assert len(sm.metadata.get_coefficients()) == 6
    assert len(sm.metadata.get_targets()) == 1
    sm.metadata.set_targets("asd")
    assert len(sm.metadata.get_targets()) == 1
    sm.metadata.set_targets(["asd"])
    assert len(sm.metadata.get_targets()) == 1
    assert len(sm.metadata.get_coefficients()) == 6
    sm.metadata.set_coefficients(range(6))
    assert len(sm.metadata.get_coefficients()) == 6
    assert_allclose(sm.metadata.unscale_X(), sm.get_X(intercept=False))
    assert_allclose(sm.metadata.unscale_Y(), sm.get_Y())
    sm.metadata.set_scale_X(np.zeros(5), np.ones(5))
    sm.metadata.set_scale_Y(0, 1)
    assert_allclose(sm.metadata.unscale_X(), sm.get_X(intercept=False))
    assert_allclose(sm.metadata.unscale_Y(), sm.get_Y())
    assert_allclose(np.arange(sm.n), sm.metadata.get_rows(True))
    sm.metadata.set_rows(np.arange(2, 2 + sm.n), None, None, None)
    assert_allclose(np.arange(2, 2 + sm.n), sm.metadata.get_rows())
    sm.metadata.set_rows(np.arange(1, 1 + sm.n))
    assert_allclose(np.arange(1, 1 + sm.n), sm.metadata.get_rows())
