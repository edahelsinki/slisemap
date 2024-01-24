from slisemap.tuning import *
from .utils import get_slisemap2
import pytest


@pytest.fixture(scope="session")
def tune_data():
    sm, _ = get_slisemap2(100, 5, lasso=1, ridge=1, seed=245230984)
    X_test = sm.get_X(intercept=False)[:30, ...]
    y_test = sm.get_Y()[:30, 0]
    X_vali = sm.get_X(intercept=False)[30:50, ...]
    y_vali = sm.get_Y()[30:50, 0]
    sm._X = sm._X[50:, ...]
    sm._Y = sm._Y[50:, ...]
    sm._B = sm._B[50:, ...]
    sm._Z = sm._Z[50:, ...]
    sm._Z0 = sm._Z0[50:, ...]
    sm._B0 = sm._B0[50:, ...]
    sm2 = sm.copy()
    sm2.optimise(patience=1)
    return (sm, sm2, X_test, y_test, X_vali, y_vali)


def test_with_set(tune_data):
    sm0, sm2, X_test, y_test, X_vali, y_vali = tune_data
    sm = sm0.copy()
    sm = optimise_with_test_set(sm, X_test, y_test, patience=1, max_iterations=20)
    assert accuracy(sm, X_vali, y_vali) < accuracy(sm2, X_vali, y_vali)
    assert sm.lasso != sm2.lasso
    assert sm.ridge != sm2.ridge


def test_cv(tune_data):
    sm0, sm2, X_test, y_test, X_vali, y_vali = tune_data
    sm = sm0.copy()
    sm = optimise_with_cross_validation(sm, patience=1, k=2, max_iterations=20)
    assert accuracy(sm, X_vali, y_vali) < accuracy(sm2, X_vali, y_vali)
    assert sm.lasso != sm2.lasso
    assert sm.ridge != sm2.ridge


def test_tune(tune_data):
    sm0, sm2, X_test, y_test, X_vali, y_vali = tune_data
    sm = hyperparameter_tune(
        Slisemap,
        sm0.get_X(intercept=False),
        sm0.get_Y(),
        X_test,
        y_test,
        radius=sm0.radius,
        optim_kws=dict(patience=1, max_escapes=5, max_iter=100),
        n_calls=10,
    )
    sm.optimise(patience=1)
    assert accuracy(sm, X_vali, y_vali) < accuracy(sm2, X_vali, y_vali)
    assert sm.lasso != sm2.lasso
    assert sm.ridge != sm2.ridge
