import pytest
from slisemap.slisemap import *
from slisemap.local_models import *
from slisemap.escape import escape_neighbourhood, escape_marginal
from slisemap.utils import SlisemapWarning

from .utils import *


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
    sm = get_slisemap(30, 4)
    l1 = sm.value()
    l2 = sm.lbfgs()
    assert all_finite(l1, l2)
    assert l1 >= l2
    sm = get_slisemap(30, 4, classes=3)
    l1 = sm.value()
    l2 = sm.lbfgs()
    assert all_finite(l1, l2)
    assert l1 >= l2


def test_fit_new():
    set_seed(2391)
    sm, _ = get_slisemap2(60, 5)
    sm.lbfgs()
    X = sm.get_X(intercept=False)
    y = sm.get_Y()
    x1 = X[4, :]
    y1 = float(y[4])
    x2 = X[:2, :]
    y2 = y[:2, :]
    y2b = y[:2, 0]
    losses = sm.value(True)
    B1a, Z1a, l1a = sm.fit_new(x1, y1, False, loss=True)
    B1b, Z1b, l1b = sm.fit_new(x1, y1, True, loss=True)
    assert l1b[0] < l1a[0] * 1.1
    assert l1b[0] < losses[4] * 1.1
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
    assert_approx_ge(l2a, l2b)
    assert_approx_ge(l2e, l2d)
    assert_approx_ge(l2e, l2a)
    B2d, Z2d, l2d = sm.fit_new(x2, y2b, False, loss=True)
    B2e, Z2e, l2e = sm.fit_new(x2, y2b, True, False, loss=True)
    B2f, Z2f, l2f = sm.fit_new(x2, y2b, True, True, loss=True)
    assert np.sum(np.abs(l2d - l2a)) < 1e-4
    assert np.sum(np.abs(l2e - l2b)) < 1e-4
    assert np.sum(np.abs(l2f - l2c)) < 1e-4
    lf = sm.get_loss_fn(individual=True)
    l2b_ = lf(
        X=sm._as_new_X(np.concatenate((X, x2), 0)),
        Y=sm._as_new_Y(np.concatenate((y, y2), 0)),
        B=torch.cat((sm.B, torch.as_tensor(B2, **sm.tensorargs)), 0),
        Z=torch.cat((sm.Z, torch.as_tensor(Z2 / sm.radius, **sm.tensorargs)), 0),
    )[sm.n :]
    assert_approx_ge(l2b, l2b_.cpu().numpy())


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
    assert sm.coefficients == linear_regression_coefficients(sm.X, sm.Y)
    assert all_finite(sm.value())
    sm = Slisemap(
        np.random.normal(size=(10, 3)),
        np.random.normal(size=(10, 3)),
        lasso=1e-4,
        local_model=multiple_linear_regression,
        local_loss=linear_regression_loss,
    )
    assert sm.coefficients == linear_regression_coefficients(sm.X, sm.Y)
    assert all_finite(sm.value())
    sm = Slisemap(
        np.random.normal(size=(10, 3)),
        np.random.uniform(size=(10, 2)),
        lasso=1e-3,
        coefficients=4,
        local_model=logistic_regression,
        local_loss=logistic_regression_loss,
    )
    assert sm.coefficients == logistic_regression_coefficients(sm.X, sm.Y)
    assert all_finite(sm.value())
    sm = Slisemap(
        np.random.normal(size=(10, 3)),
        np.random.uniform(size=(10, 3)),
        lasso=1e-3,
        coefficients=8,
        local_model=logistic_regression,
        local_loss=logistic_regression_loss,
    )
    assert sm.coefficients == logistic_regression_coefficients(sm.X, sm.Y)
    assert all_finite(sm.value())


def test_predict():
    sm = get_slisemap(40, 5)
    y1 = sm.predict(np.random.normal(size=(10, 5)), np.random.normal(size=(10, 2)))
    y2 = sm.predict(np.random.normal(size=5), np.random.normal(size=2))
    assert y1.shape == (10, 1)
    assert y2.shape == (1, 1)
    assert all_finite(y1, y2)


def test_get():
    sm = get_slisemap(40, 5, intercept=True, lasso=0, ridge=0)
    assert torch.allclose(sm.Z, sm.get_Z(False, False, False))
    assert torch.allclose(torch.sqrt(torch.sum(sm.Z**2) / sm.n), torch.ones(1))
    Z = sm.get_Z(numpy=False)
    assert torch.allclose(
        torch.sqrt(torch.sum(Z**2) / sm.n) / sm.radius, torch.ones(1)
    )
    assert torch.allclose(sm.get_D(numpy=False), torch.cdist(Z, Z))
    assert torch.allclose(sm.get_W(numpy=False), sm.kernel(torch.cdist(Z, Z)))
    assert sm.get_X(intercept=False).shape[1] == sm.m - 1
    assert np.allclose(sm.value(True), np.sum(sm.get_L() * sm.get_W(), 1))
    sm.get_Y(False, True)
    sm.get_Y(False, False)
    sm.get_X(False, False)
    sm.get_X(False, True)
    sm.get_B(False)
    assert torch.allclose(
        sm.get_L(numpy=False), sm.get_L(X=sm._X[:, :-1], Y=sm._Y, numpy=False)
    )


def test_set():
    sm = get_slisemap(40, 5, intercept=True, lasso=0, ridge=0, random_state=42)
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


def test_restore():
    sm = get_slisemap(40, 3, lasso=0.01, random_state=42)
    B1 = sm.get_B()
    Z1 = sm.get_Z()
    v1 = sm.value()
    sm.optimise(max_escapes=2, max_iter=10)
    B2 = sm.get_B()
    Z2 = sm.get_Z()
    v2 = sm.value()
    sm.restore()
    assert np.allclose(B1, sm.get_B())
    assert np.allclose(Z1, sm.get_Z())
    assert np.allclose(v1, sm.value())
    sm.optimise(max_escapes=2, max_iter=10)
    assert np.allclose(B2, sm.get_B())
    assert np.allclose(Z2, sm.get_Z())
    assert np.allclose(v2, sm.value())


def test_cluster():
    sm = get_slisemap(40, 5, randomB=True)
    id, cm = sm.get_model_clusters(5)
    assert id.max() == 4
    assert id.min() == 0
    D = torch.cdist(sm._B.cpu(), torch.as_tensor(cm))
    id2 = torch.argmin(D, 1).numpy()
    assert np.all(id == id2)


def test_cuda():
    if torch.cuda.is_available():
        sm = get_slisemap(40, 3, randomB=True, cuda=False, random_state=42)
        sm.optimise(max_escapes=2, max_iter=10)
        sm.cuda()
        sm.optimise(max_escapes=2, max_iter=10)
        sm.cpu()
        sm.optimise(max_escapes=2, max_iter=10)
