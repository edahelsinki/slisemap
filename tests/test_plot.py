from matplotlib import pyplot as plt

from .utils import *


def test_plot():
    try:
        sm, cl = get_slisemap2(30, 4, randomB=True)
        sm.plot(title="ASD", clusters=4, show=False)
        sm.plot(title="ASD", clusters=4, bars=True, show=False)
        sm.plot(title="ASD", clusters=4, bars=-1, show=False)
        sm.plot(title="ASD", clusters=0, show=False)
        sm.plot(clusters=cl, bars=False, show=False)
        sm.plot(clusters=cl, bars=True, show=False)
        cl2 = np.asarray([f"A{9-i}" for i in np.unique(cl)])[cl]
        sm.plot(clusters=cl2, bars=False, show=False)
        sm.plot(clusters=cl2, bars=True, show=False)
        sm.plot(jitter=1, figsize=(2, 2), show=False)
        sm.metadata.set_variables(range(4), add_intercept=True)
        sm.plot(title="ASD", clusters=4, bars=3, show=False)
        sm.plot(title="ASD", clusters=4, bars=[1, 0, 3], show=False)
        sm.plot(jitter=1, show=False)
        sm = Slisemap(
            np.random.normal(size=(10, 3)), np.random.normal(size=(10, 3)), lasso=0
        )
        sm.plot(show=False)
        sm.metadata.set_variables(range(3), add_intercept=True)
        sm.metadata.set_targets(range(3))
        sm.plot(show=False)
    finally:
        plt.close("all")


def test_plot_position():
    try:
        sm, cl = get_slisemap2(30, 4, randomB=True)
        sm.plot_position(index=1, title="ASD", selection=False, jitter=1, show=False)
        sm.plot_position(
            index=[1, 2, 3], legend_inside=True, col_wrap=2, selection=False, show=False
        )
        sm.plot_position(
            index=range(3), title="ASD", legend_inside=False, col_wrap=2, show=False
        )
        sm.plot_position(
            X=sm.get_X(intercept=False)[:3], Y=sm.get_Y()[:3], col_wrap=2, show=False
        )
        sm.plot_position(
            X=sm._X[:3, :-1], Y=sm._Y[:3], col_wrap=5, legend_inside=True, show=False
        )
    finally:
        plt.close("all")


def test_plot_dist():
    try:
        sm, cl = get_slisemap2(30, 4, randomB=True)
        sm.plot_dist(title="ASD", clusters=4, show=False)
        sm.plot_dist(clusters=cl, show=False)
        sm.plot_dist(legend_inside=True, col_wrap=3, show=False)
        sm.plot_dist(legend_inside=False, col_wrap=3, show=False)
        sm.plot_dist(title="ASD", scatter=True, show=False)
        sm.plot_dist(scatter=True, legend_inside=True, col_wrap=3, show=False)
        sm.plot_dist(scatter=True, legend_inside=False, col_wrap=3, show=False)
        sm.metadata.set_variables(range(4), add_intercept=True)
        sm.metadata.set_targets(["asd"])
        sm.metadata.set_scale_X(np.zeros(4), np.ones(4))
        sm.metadata.set_scale_Y(0, 1)
        sm.plot_dist(jitter=1, show=False)
        sm.plot_dist(scatter=True, show=False)
    finally:
        plt.close("all")


def test_get_model_clusters():
    sm, cl = get_slisemap2(30, 4, randomB=True)
    B = sm.get_B()
    Z = sm.get_Z(rotate=True)
    cl1, cm1 = sm.get_model_clusters(2, B, Z)
    cl2, cm2 = sm.get_model_clusters(2, B, -Z)
    assert_allclose(cl1, 1 - cl2)
    assert cl1.mean() > 0.1
    cl3, cm3 = sm.get_model_clusters(3, B, Z)
    d0 = np.sum((B - cm3[0:1, :]) ** 2, 1)
    d1 = np.sum((B - cm3[1:2, :]) ** 2, 1)
    d2 = np.sum((B - cm3[2:3, :]) ** 2, 1)
    assert np.all(d0[cl3 == 0] <= d1[cl3 == 0])
    assert np.all(d0[cl3 == 0] <= d2[cl3 == 0])
    assert np.all(d1[cl3 == 1] <= d0[cl3 == 1])
    assert np.all(d1[cl3 == 1] <= d2[cl3 == 1])
    assert np.all(d2[cl3 == 2] <= d0[cl3 == 2])
    assert np.all(d2[cl3 == 2] <= d1[cl3 == 2])
