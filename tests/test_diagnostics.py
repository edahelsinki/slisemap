from matplotlib import pyplot as plt
import numpy as np

from slisemap.diagnostics import *

from .utils import *


def test_print_plot():
    sm = get_slisemap(30, 4)
    diags = diagnose(sm, conservative=True)
    print_diagnostics(diags, False)
    print_diagnostics(diags, True)
    plot_diagnostics(sm, diags, False)
    plot_diagnostics(sm, diags, True)
    plot_diagnostics(sm, {k: v for k, v in [diags.popitem()]}, True)
    plt.close("all")


def test_diagnose():
    sm = get_slisemap(30, 4)
    for name, mask in diagnose(sm, conservative=False).items():
        assert isinstance(name, str)
        assert isinstance(mask, np.ndarray)
        assert mask.shape[0] == sm.n
    for name, mask in diagnose(sm, conservative=True).items():
        assert isinstance(name, str)
        assert isinstance(mask, np.ndarray)
        assert mask.shape[0] == sm.n


def test_underfit():
    sm = get_slisemap(60, 4, radius=0.5)
    sm.optimise()
    assert np.mean(lightweight_diagnostic(sm)) > 0.1
    assert np.mean(weight_neighbourhood_diagnostic(sm)) > 0.1
    assert np.mean(loss_neighbourhood_diagnostic(sm)) > 0.1
    assert np.mean(global_loss_diagnostic(sm)) > 0.1


def test_overfit():
    set_seed(1821)
    sm, _ = get_slisemap2(60, 4, radius=10)
    sm.optimise()
    assert np.mean(distant_diagnostic(sm)) > 0.1
    assert np.mean(heavyweight_diagnostic(sm)) > 0.1
    assert np.mean(weight_neighbourhood_diagnostic(sm)) > 0.1
    assert np.mean(loss_neighbourhood_diagnostic(sm)) > 0.02
    assert np.mean(optics_diagnostic(sm)) > 0.1


def test_loss_neigh():
    sm, _ = get_slisemap2(60, 4)
    sm._Y += 10
    sm.optimise()
    sm.lasso = 10000
    assert np.mean(loss_neighbourhood_diagnostic(sm)) == 0
