from slisemap.escape import *

from .utils import *


def escape_one(x, y, B, Z, local_model, local_loss, kernel, radius=3.5, avoid=-1):
    # Escape one data item like in the paper
    Zss = torch.sum(Z**2)
    Zs = Z * (radius / (torch.sqrt(Zss / Z.shape[0]) + 1e-8))
    D = torch.cdist(Zs, Zs)
    W = kernel(D)
    L = local_loss(local_model(x, B), y).ravel()
    K = torch.sum(W * L[None, :], 1)
    if avoid > 0:
        K[avoid] = np.inf
    index = torch.argmin(K)
    return B[index], Z[index]


def test_escape_internal():
    # Test that the vectorised version is as good as the slow implementation of escape
    set_seed(3451653)
    sm, _ = get_slisemap2(40, 3, cheat=True)
    sm.lbfgs()
    B2, Z2 = escape_neighbourhood(
        X=sm._X[:4],
        Y=sm._Y[:4],
        B=sm._B,
        Z=sm._Z,
        local_model=sm.local_model,
        local_loss=sm.local_loss,
        distance=sm.distance,
        kernel=sm.kernel,
        radius=sm.radius,
        force_move=False,
    )
    assert B2.shape[0] == Z2.shape[0] == 4
    B2, Z2 = escape_neighbourhood(
        X=sm._X,
        Y=sm._Y,
        B=sm._B,
        Z=sm._Z,
        local_model=sm.local_model,
        local_loss=sm.local_loss,
        distance=sm.distance,
        kernel=sm.kernel,
        radius=sm.radius,
        force_move=True,
    )
    B4 = B2.clone()
    Z4 = Z2.clone()
    for i in range(sm.n):
        x = sm._X[i : i + 1]
        y = sm._Y[i : i + 1]
        B4[i], Z4[i] = escape_one(
            x, y, sm._B, sm._Z, sm.local_model, sm.local_loss, sm.kernel, sm.radius, i
        )
    # Sometimes the vectorised version chooses a different point to jump to.
    # This asserts that that happens less than 20% of the time:
    assert torch.sum(torch.sum(torch.abs(Z2 - Z4), 1) < 1e-4) >= 0.8 * sm.n
    assert torch.sum(torch.sum(torch.abs(B2 - B4), 1) < 1e-4) >= 0.8 * sm.n
    loss = sm._get_loss_fn()
    l1 = loss(sm._X, sm._Y, sm._B, sm._Z)
    l2 = loss(sm._X, sm._Y, B2, Z2)
    l4 = loss(sm._X, sm._Y, B4, Z4)
    assert l2 <= l4 * 1.01
    sm._B = B2
    sm._Z = Z2
    sm.lbfgs()
    l5 = loss(sm._X, sm._Y, sm._B, sm._Z)
    assert l5 <= l1


def test_escape():
    sm, _ = get_slisemap2(40, 3, seed=883313)
    l1 = sm.value()
    l2 = sm.lbfgs(only_B=True)
    for fn in [escape_greedy, escape_marginal, escape_combined, escape_neighbourhood]:
        sm2 = sm.copy()
        sm2.escape(force_move=False, escape_fn=fn)
        sm2 = sm.copy()
        sm2.fit_new(
            sm2.get_X(intercept=False)[:2],
            sm2.get_Y()[:2],
            optimise=False,
            escape_fn=fn,
        )
        sm2.escape(force_move=True, escape_fn=fn)
    assert torch.all(sm._B.max(0)[0] >= sm2._B.max(0)[0])
    assert torch.all(sm._B.min(0)[0] <= sm2._B.min(0)[0])
    assert torch.allclose(torch.sqrt(torch.sum(sm2._Z**2) / sm2.n), torch.ones(1))
    l3 = sm2.lbfgs()
    assert torch.allclose(torch.sqrt(torch.sum(sm2._Z**2) / sm2.n), torch.ones(1))
    assert all_finite(l1, l2, l3)
    assert l3 <= l2 * 1.02
    assert l2 <= l1 * 1.02
    sm, _ = get_slisemap2(40, 3, classes=True, randomB=True, seed=853313)
    l1 = sm.value()
    l2 = sm.lbfgs()
    sm2 = sm.copy()
    sm2.escape(False)
    assert torch.all(sm._B.max(0)[0] >= sm2._B.max(0)[0])
    assert torch.all(sm._B.min(0)[0] <= sm2._B.min(0)[0])
    assert torch.allclose(torch.sqrt(torch.sum(sm2._Z**2) / sm2.n), torch.ones(1))
    sm2 = sm.copy()
    sm2.escape(True)
    l3 = sm2.lbfgs()
    assert torch.allclose(torch.sqrt(torch.sum(sm2._Z**2) / sm2.n), torch.ones(1))
    assert all_finite(l1, l2, l3)
    assert l3 <= l2 * 1.05
    assert l2 <= l1 * 1.02
