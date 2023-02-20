import pytest
from slisemap.slisemap import *
from slisemap.local_models import *
from slisemap.escape import escape_neighbourhood, escape_marginal
from slisemap.utils import SlisemapWarning

from .utils import *


@pytest.fixture
def sm_data():
    sm = get_slisemap(
        40, 5, intercept=True, lasso=0, ridge=0, randomB=True, seed=459872
    )
    sm0 = sm.copy()
    sm.lbfgs(100)
    sm.z_norm = 0
    sm._normalise()
    return (sm0, sm)


def test_parameters():
    # Just a naïve check to see that some parameters are working
    sm4 = get_slisemap(30, 4, z_norm=1, radius=4)
    sm5 = get_slisemap(30, 4, intercept=False)
    sm6 = get_slisemap(30, 4, classes=2, radius=3)
    with pytest.warns(SlisemapWarning, match="dimensions"):
        sm7 = get_slisemap(30, 4, d=10)
    sm7 = get_slisemap(10, 15)
    assert all_finite(
        sm4.lbfgs(10),
        sm5.lbfgs(10),
        sm6.lbfgs(10),
        sm7.lbfgs(10),
    )


def test_lbfgs():
    set_seed(10983422)
    sm = get_slisemap(30, 4)
    l1 = sm.value()
    l2 = sm.lbfgs()
    assert all_finite(l1, l2)
    assert l1 >= l2
    assert_allclose(l2, sm.value(), rtol=1e-3)
    sm = get_slisemap(30, 4, classes=3)
    l1 = sm.value()
    l2 = sm.lbfgs()
    assert all_finite(l1, l2)
    assert l1 >= l2
    assert_allclose(l2, sm.value(), rtol=1e-3)


def test_only_B(sm_data):
    sm = sm_data[1].copy()
    sm2 = Slisemap(
        X=sm.get_X(intercept=False, numpy=False),
        y=sm.get_Y(numpy=False),
        radius=sm.radius,
        lasso=sm.lasso,
        intercept=sm.intercept,
        Z0=sm.get_Z(numpy=False),
        random_state=123009857,
    )
    sm2.optimise(1, 3, only_B=True)
    assert_allclose(sm.get_Z(), sm2.get_Z())
    assert_approx_ge(sm.value(), sm2.value())
    sm.lbfgs(only_B=True)


def test_fit_new():
    sm, _ = get_slisemap2(60, 5, cheat=True, seed=239177)
    sm.lbfgs(100)
    X = sm.get_X(intercept=False)
    y = sm.get_Y()
    x1 = X[4, :]
    y1 = float(y[4])
    x2 = X[:2, :]
    y2 = y[:2, :]
    y2b = y[:2, 0]
    _, _, l1a = sm.fit_new(x1, y1, False, loss=True)
    _, _, l1b = sm.fit_new(x1, y1, True, loss=True)
    assert l1b[0] <= l1a[0] * 1.1
    _, _, l2a = sm.fit_new(
        x2, y2, False, between=True, escape_fn=escape_neighbourhood, loss=True
    )
    B2, Z2, l2b = sm.fit_new(
        x2, y2, True, between=False, escape_fn=escape_neighbourhood, loss=True
    )
    _, _, l2c = sm.fit_new(
        x2, y2, True, between=True, escape_fn=escape_neighbourhood, loss=True
    )
    _, _, l2d = sm.fit_new(
        x2, y2, optimise=True, between=True, escape_fn=escape_marginal, loss=True
    )
    _, _, l2e = sm.fit_new(
        x2, y2, optimise=False, between=True, escape_fn=escape_marginal, loss=True
    )
    assert np.sum(np.abs(l2c - l2b)) < 0.01
    assert_approx_ge(l2a, l2c)
    assert_approx_ge(l2e, l2d)
    assert_approx_ge(l2e, l2a)
    B2d, Z2d, l2d = sm.fit_new(x2, y2b, False, loss=True)
    B2e, Z2e, l2e = sm.fit_new(x2, y2b, True, False, loss=True)
    B2f, Z2f, l2f = sm.fit_new(x2, y2b, True, True, loss=True)
    assert np.sum(np.abs(l2d - l2a)) < 1e-4
    assert np.sum(np.abs(l2e - l2b)) < 1e-4
    assert np.sum(np.abs(l2f - l2c)) < 1e-4
    lf = sm._get_loss_fn(individual=True)
    l2b_ = lf(
        X=sm._as_new_X(np.concatenate((X, x2), 0)),
        Y=sm._as_new_Y(np.concatenate((y, y2), 0)),
        B=torch.cat((sm._B, torch.as_tensor(B2, **sm.tensorargs)), 0),
        Z=torch.cat((sm._Z, torch.as_tensor(Z2 / sm.radius, **sm.tensorargs)), 0),
    )[sm.n :]
    assert np.sum(np.abs(tonp(l2b_) - l2b)) < 0.01
    sm.fit_new(x1, y1, optimise=False, between=True, loss=True, numpy=False)
    sm.fit_new(x2, y2, optimise=False, between=False, loss=True, numpy=False)
    sm.fit_new(x1, y1, optimise=False, between=True, loss=True, numpy=False)
    sm.fit_new(x2, y2, optimise=False, between=False, loss=True, numpy=False)


def test_loss():
    sm = get_slisemap()
    assert all_finite(sm.value())
    assert all_finite(sm.value(True))
    sm = Slisemap(
        np.random.normal(size=(10, 3)),
        np.random.normal(size=(10, 1)),
        lasso=1e-4,
        local_model=linear_regression,
        local_loss=linear_regression_loss,
    )
    assert sm.q == linear_regression_coefficients(sm._X, sm._Y)
    assert all_finite(sm.value())
    sm = Slisemap(
        np.random.normal(size=(10, 3)),
        np.random.normal(size=(10, 3)),
        lasso=1e-4,
        local_model=linear_regression,
        local_loss=linear_regression_loss,
    )
    assert sm.q == linear_regression_coefficients(sm._X, sm._Y)
    assert all_finite(sm.value())
    sm = Slisemap(
        np.random.normal(size=(10, 3)),
        np.random.uniform(size=(10, 2)),
        lasso=1e-3,
        coefficients=4,
        local_model=logistic_regression,
        local_loss=logistic_regression_loss,
    )
    assert sm.q == logistic_regression_coefficients(sm._X, sm._Y)
    assert all_finite(sm.value())
    sm = Slisemap(
        np.random.normal(size=(10, 3)),
        np.random.uniform(size=(10, 3)),
        lasso=1e-3,
        coefficients=8,
        local_model=logistic_regression,
        local_loss=logistic_regression_loss,
    )
    assert sm.q == logistic_regression_coefficients(sm._X, sm._Y)
    assert all_finite(sm.value())


def test_predict(sm_data):
    sm = sm_data[1]
    y1 = sm.predict(X=np.random.normal(size=(10, 5)), Z=np.random.normal(size=(10, 2)))
    y2 = sm.predict(X=np.random.normal(size=5), Z=np.random.normal(size=2))
    assert y1.shape == (10, 1)
    assert y2.shape == (1, 1)
    assert all_finite(y1, y2)
    y1 = sm.predict(X=np.random.normal(size=(10, 5)), B=np.random.normal(size=(10, 6)))
    y2 = sm.predict(X=np.random.normal(size=5), B=np.random.normal(size=6))
    assert y1.shape == (10, 1)
    assert y2.shape == (1, 1)
    assert all_finite(y1, y2)


def test_get(sm_data):
    sm = sm_data[1]
    assert sm.get_Y(False, True).shape == (40,)
    assert sm.get_Y(False, False).shape == (40, 1)
    assert sm.get_X(intercept=False).shape[1] == sm.m - 1
    assert sm.get_X(False, False).shape == (40, 5)
    assert sm.get_X(False, True).shape == (40, 6)
    assert sm.get_B(False).shape == (40, 6)
    assert_allclose(sm._Z, sm.get_Z(False, False, False), "Z")
    assert_allclose(torch.sqrt(torch.sum(sm._Z**2) / sm.n), torch.ones(1), "scale")
    Z = sm.get_Z(numpy=False)
    assert_allclose(torch.sqrt(torch.sum(Z**2) / sm.n) / sm.radius, torch.ones(1))
    assert_allclose(sm.get_D(numpy=False), torch.cdist(Z, Z), "D", 1e-4, 1e-6)
    W = sm.get_W(numpy=False)
    assert_allclose(W, sm.kernel(torch.cdist(Z, Z)), "W")
    L = sm.get_L(numpy=False)
    assert_allclose(sm.value(), torch.sum(W * L).cpu().item(), "loss", 1e-4, 1e-6)
    assert_allclose(sm.value(True, False), torch.sum(W * L, 1), "ind_loss", 1e-4, 1e-4)
    assert_allclose(
        sm.get_L(numpy=False),
        sm.get_L(X=sm._X[:, :-1], Y=sm._Y, numpy=False),
        "L",
    )


def test_set(sm_data):
    sm = sm_data[1]
    sm.d = 5
    sm.jit = False
    sm.random_state = None
    assert np.isfinite(sm.value())
    sm.lasso = 1
    sm.ridge = 1
    sm.z_norm = 0
    sm.radius = 2
    sm.d = 2
    sm.jit = True
    sm.random_state = 42
    assert np.isfinite(sm.value())


def test_restore(sm_data):
    sm0, sm1 = sm_data
    B1 = sm0.get_B()
    Z1 = sm0.get_Z()
    v1 = sm0.value()
    sm2 = sm1.copy()
    sm2.restore()
    assert_allclose(B1, sm2.get_B(), "restore B")
    assert_allclose(Z1, sm2.get_Z(), "restore Z")
    assert_allclose(v1, sm2.value(), "restore value")


def test_cluster(sm_data):
    sm = sm_data[1]
    id, cm = sm.get_model_clusters(5)
    assert id.max() == 4
    assert id.min() == 0
    D = torch.cdist(sm._B.cpu(), torch.as_tensor(cm))
    id2 = torch.argmin(D, 1).numpy()
    assert np.all(id == id2)


def test_cuda(sm_data):
    sm = sm_data[1].copy()
    if torch.cuda.is_available():
        sm.cuda()
        sm.optimise(max_escapes=2, max_iter=10)
        sm.cpu()
        sm.optimise(max_escapes=2, max_iter=10)
