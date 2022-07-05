from matplotlib import pyplot as plt
import numpy as np

from slisemap.diagnostics import *

from .utils import *


def test_print_plot():
    sm = get_slisemap(30, 4)
    diags = diagnose(sm, conservative=True)
    print_diagnostics(diags, False)
    print_diagnostics(diags, True)
    try:
        plot_diagnostics(sm, diags, False, show=False)
        plot_diagnostics(sm, diags, True, show=False)
        plot_diagnostics(sm, {k: v for k, v in [diags.popitem()]}, True, show=False)
    finally:
        plt.close("all")


def test_diagnose():
    sm, _ = get_slisemap2(60, 4, cheat=True, seed=4590849)
    sm.lbfgs()
    for name, mask in diagnose(sm, conservative=False).items():
        assert isinstance(name, str)
        assert isinstance(mask, np.ndarray)
        assert mask.shape[0] == sm.n
        assert np.mean(mask) <= 0.25, f"Too many data items flagged in {name}."


def test_underfit():
    sm, _ = get_slisemap2(60, 4, radius=0.5, seed=359784)
    sm.optimise(max_escapes=2)
    assert np.mean(lightweight_diagnostic(sm)) > 0.1
    assert np.mean(weight_neighbourhood_diagnostic(sm)) > 0.1
    assert np.mean(loss_neighbourhood_diagnostic(sm)) > 0.08
    assert np.mean(global_loss_diagnostic(sm)) > 0.08
    assert np.mean(quantile_loss_diagnostic(sm)) > 0.1


def test_overfit():
    sm, _ = get_slisemap2(60, 4, radius=10, seed=75308)
    sm.lbfgs()
    assert np.mean(distant_diagnostic(sm)) > 0.1
    assert np.mean(heavyweight_diagnostic(sm)) > 0.1
    assert np.mean(weight_neighbourhood_diagnostic(sm)) > 0.1
    assert np.mean(loss_neighbourhood_diagnostic(sm)) > 0.02
    assert np.mean(optics_diagnostic(sm)) > 0.05


def test_loss_neigh():
    sm, _ = get_slisemap2(60, 4)
    sm._Y += 10
    sm.lbfgs()
    sm.lasso = 10000
    assert np.mean(loss_neighbourhood_diagnostic(sm)) == 0
