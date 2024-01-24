import pytest
from slisemap.slipmap import *
from slisemap.utils import SlisemapWarning, tonp

from .utils import *


def test_grid():
    for d in [1, 2, 3, 4]:
        for i in [20, 40, 80, 160]:
            for hex in [True, False] if d == 2 else [False]:
                if 4**d * 3 / 4 < i:
                    grid = make_grid(i, d, hex)
                    assert i * 0.8 <= grid.shape[0] <= i * 1.4
                    assert 0.99 <= grid.max() <= 1.1
                    sum = grid.sum(0)
                    for e in range(d):
                        grid[:, e] *= -1
                        assert_allclose(sum, grid.sum(0), f"{i} {d} {e} {grid.shape}")


def test_parameters():
    # Just a naïve check to see that some parameters are working
    with pytest.warns(SlisemapWarning, match="regularisation"):
        sm3 = Slipmap(np.random.normal(size=(10, 5)), np.random.normal(size=10))
        sm3.into()
        sm3.radius = 2.5
        sm3.jit = False
        sm3.cpu()
        assert all_finite(sm3.value())
    sm4 = Slipmap.convert(get_slisemap(30, 4, lasso=0, radius=4))
    sm5 = Slipmap.convert(get_slisemap(30, 4, intercept=False), True)
    sm6 = Slipmap.convert(get_slisemap(30, 4, classes=2, radius=3))
    with pytest.warns(SlisemapWarning, match="dimensions"):
        sm7 = get_slisemap(30, 4, d=10)
    with pytest.warns(SlisemapWarning, match="dimensions"):
        sm7 = Slipmap.convert(sm7)
    sm8 = Slipmap.convert(get_slisemap(10, 15, seed=42))
    sms = [sm4, sm5, sm6, sm7, sm8]
    losses = [sm.value() for sm in sms]
    assert all_finite(losses)
    [sm.lbfgs() for sm in sms]
    losses2 = [sm.value() for sm in sms]
    assert all_finite(losses2)
    assert_approx_ge(losses, losses2)


def test_save(tmp_path):
    sm1 = Slipmap.convert(get_slisemap(30, 4))
    sm1.save(tmp_path / "tmp.sp")
    sm2 = Slipmap.load(tmp_path / "tmp.sp")
    assert_allclose(tonp(sm1._Z), tonp(sm2._Z))
    assert_allclose(tonp(sm1._Bp), tonp(sm2._Bp))


def test_getters():
    sm = Slipmap.convert(get_slisemap(30, 4, intercept=True))
    assert sm.intercept
    p = sm.p
    assert sm.get_Z(numpy=True).shape == (30, 2)
    assert sm.get_B().shape == (30, 5)
    assert sm.get_Bp().shape == (p, 5)
    assert sm.get_D(proto_rows=True, proto_cols=True, numpy=True).shape == (p, p)
    assert sm.get_D(proto_rows=True, proto_cols=False, numpy=True).shape == (p, 30)
    assert sm.get_D(proto_rows=False, proto_cols=False, numpy=False).shape == (30, 30)
    assert sm.get_L().shape == (p, 30)
    assert sm.get_W().shape == (p, 30)
    assert sm.get_X().shape == (30, 5)
    assert sm.get_X(intercept=False).shape == (30, 4)
    assert sm.get_Y().shape == (30, 1)
    assert sm.get_Zp().shape == (p, 2)
    assert sm.n == 30
    assert sm.o == 1
    assert sm.q == 5


def test_optim():
    set_seed(213879243)
    sm = Slipmap.convert(get_slisemap(30, 4))
    l1 = sm.value()
    l2 = sm.lbfgs(only_B=True)
    assert_approx_ge(l1, l2)
    assert_allclose(l2, sm.value())
    l3 = sm.lbfgs(only_Z=True)
    assert_approx_ge(l2, l3)
    assert_allclose(l3, sm.value(), rtol=1e-3)
    l4 = sm.lbfgs()
    assert_approx_ge(l2, l4)
    assert_allclose(l4, sm.value(), rtol=1e-3)
    l5 = sm.optimise(verbose=3)
    assert_approx_ge(l4, l5)
    assert_allclose(l5, sm.value())
    sm.optimise(verbose=3, max_iter=10, max_escapes=1)


def test_plot():
    sp = Slipmap.convert(get_slisemap2(60, 5, cheat=True, randomB=True)[0], radius=2.0)
    try:
        sp.plot("title", jitter=0.1, show=False)
        sp.plot(bars=False, clusters=3, show=False)
        sp.plot(clusters=3, bars=True, show=False)
        sp.plot(bars=3, clusters=3, show=False)
        sp.plot_dist(scatter=True, show=False)
        sp.plot_dist(scatter=False, show=False)
        sp.plot_dist(clusters=3, scatter=True, show=False)
        sp.plot_dist(clusters=3, scatter=False, show=False)
        sp.plot_position(index=1, show=False)
        sp.plot_position(index=[1, 2], show=False)
        sp.plot_position(
            X=sp.get_X(True, False)[:5, :], Y=sp.get_Y()[:5, :], show=False
        )
    finally:
        plt.close("all")
