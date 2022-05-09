from timeit import default_timer as timer

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
    assert not cc.has_converged(1)
    assert cc.has_converged(1)
    cc = CheckConvergence(1)
    assert not cc.has_converged([1, 3])
    assert not cc.has_converged((1, 1))
    assert cc.has_converged((1, 1))


def test_global_model():
    sm, _ = get_slisemap2(40, 5)
    global_model(sm.X, sm.Y, sm.local_model, sm.local_loss, sm.coefficients, 0.01, 0.01)
    sm, _ = get_slisemap2(40, 5, classes=True)
    global_model(sm.X, sm.Y, sm.local_model, sm.local_loss, sm.coefficients, 0.01, 0.01)


def varimax_numpy(Phi, gamma=1.0, q=20, tol=1e-6):
    # From: http://en.wikipedia.org/wiki/Talk:Varimax_rotation
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, vh = np.linalg.svd(
            np.dot(
                Phi.T,
                np.asarray(Lambda) ** 3
                - (gamma / p)
                * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda)))),
            )
        )
        R = np.dot(u, vh)
        d = sum(s)
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return np.dot(Phi, R)


def test_varimax():
    X = np.random.normal(size=(20, 2))
    XR = varimax_numpy(X, 1, 20, 1e-6)
    XRb = varimax_numpy(X[:, :1], 1, 20, 1e-6)
    X = torch.as_tensor(X)
    XR = torch.as_tensor(XR)
    XRb = torch.as_tensor(XRb)
    assert torch.allclose(torch.cdist(X, X), torch.cdist(XR, XR))
    assert torch.allclose(torch.cdist(X[:, :1], X[:, :1]), torch.cdist(XRb, XRb))
    XR2 = varimax(X, 1, 20, 1e-6)
    assert torch.allclose(XR, XR2)
    assert torch.allclose(torch.cdist(X, X), torch.cdist(XR2, XR2))
    XR2b = varimax(X[:, :1], 1, 20, 1e-6)
    assert torch.allclose(XRb, XR2b)
    assert torch.allclose(torch.cdist(X[:, :1], X[:, :1]), torch.cdist(XR2b, XR2b))


def test_PCA():
    x = torch.normal(0, 1, (5, 3))
    assert PCA_rotation(x, 2, full=True).shape == (3, 2)
    assert PCA_rotation(x, 2, full=False).shape == (3, 2)
    assert PCA_rotation(x, 5, full=True).shape == (3, 3)
    assert PCA_rotation(x, 5, full=False).shape == (3, 3)
    assert PCA_rotation(x * np.nan, 5, full=True).shape == (3, 3)
    assert PCA_rotation(x * np.nan, 5, full=False).shape == (3, 3)
