import warnings
from slisemap.tuning import *
from .utils import get_slisemap2
import pytest


@pytest.fixture(scope="session")
def tune_data():
    sm, _ = get_slisemap2(200, 5, lasso=1, ridge=1, seed=245230984)
    X_test = sm.get_X(intercept=False)[:50, ...]
    y_test = sm.get_Y()[:50, ...]
    X_vali = sm.get_X(intercept=False)[50:100, ...]
    y_vali = sm.get_Y()[50:100, 0]
    sm._X = sm._X[100:, ...]
    sm._Y = sm._Y[100:, ...]
    sm._B = sm._B[100:, ...]
    sm._Z = sm._Z[100:, ...]
    sm._Z0 = sm._Z0[100:, ...]
    sm._B0 = sm._B0[100:, ...]
    sm.optimise(patience=1)
    sp = Slipmap.convert(sm, radius=2.0)
    sp.optimise(patience=1)
    # Ignore some skopt warnings
    warnings.filterwarnings(
        "ignore", message="The objective has been evaluated at this point before."
    )
    return (sm, sp, X_test, y_test, X_vali, y_vali)


@pytest.hookimpl(trylast=True)
def test_with_set(tune_data):
    sm0, sp0, X_test, y_test, X_vali, y_vali = tune_data
    for sm in (sm0, sp0):
        sm2 = sm.copy()
        sm2 = optimise_with_test(
            sm2, X_test, y_test, patience=2, max_escapes=5, max_iter=20
        )
        assert accuracy(sm2, X_vali, y_vali) < accuracy(sm, X_vali, y_vali) * 1.05, sm2
        assert sm2.lasso != sm.lasso or sm2.ridge != sm.ridge or sm2.radius != sm.radius


@pytest.hookimpl(trylast=True)
def test_with_cv(tune_data):
    sm0, sp0, X_test, y_test, X_vali, y_vali = tune_data
    for sm in (sm0, sp0):
        sm2 = sm.copy()
        sm2 = optimise_with_cv(sm2, patience=1, max_escapes=5, max_iter=20, k=3)
        assert accuracy(sm2, X_vali, y_vali) < accuracy(sm, X_vali, y_vali) * 1.05
        assert sm2.lasso != sm.lasso or sm2.ridge != sm.ridge or sm2.radius != sm.radius


@pytest.hookimpl(trylast=True)
def test_tune(tune_data):
    sm0, sp0, X_test, y_test, X_vali, y_vali = tune_data
    for sm in (sm0, sp0):
        sm2 = hyperparameter_tune(
            sm.__class__,
            sm0.get_X(intercept=False),
            sm0.get_Y(),
            X_test,
            y_test,
            radius=sm0.radius,
            optim_kws=dict(patience=1, max_escapes=5, max_iter=20),
            n_calls=10,
        )
        assert accuracy(sm2, X_vali, y_vali) < accuracy(sm, X_vali, y_vali)
        assert sm2.lasso != sm.lasso or sm2.ridge != sm.ridge or sm2.radius != sm.radius
