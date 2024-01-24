import pytest
import torch

from slisemap.local_models import *
from slisemap.utils import SlisemapException

from .utils import *


def test_linear_model():
    X = torch.as_tensor(np.random.normal(size=(10, 4)))
    B = torch.as_tensor(np.random.normal(size=(10, 4)))
    Y = torch.as_tensor(np.random.normal(size=(10, 1)))
    assert B.shape[1] == linear_regression_coefficients(X, Y)
    P = linear_regression(X, B)
    L = linear_regression_loss(P, Y)
    assert P.shape == (10, 10, 1)
    assert L.shape == (10, 10)
    L = linear_regression_loss(Y, Y)
    assert L.shape == (10,)
    assert linear_regression(X, B[:1, :]).shape == (1, 10, 1)
    assert linear_regression(X[:1, :], B).shape == (10, 1, 1)
    torch.jit.trace(linear_regression, (X, B))
    torch.jit.trace(linear_regression_loss, (P, Y))
    B = torch.as_tensor(np.random.normal(size=(10, 8)))
    Y = torch.as_tensor(np.random.normal(size=(10, 2)))
    assert B.shape[1] == linear_regression_coefficients(X, Y)
    P = linear_regression(X, B)
    L = linear_regression_loss(P, Y)
    assert P.shape == (10, 10, 2)
    assert L.shape == (10, 10)
    L = linear_regression_loss(Y, Y)
    assert L.shape == (10,)
    assert linear_regression(X, B[:1, :]).shape == (1, 10, 2)
    assert linear_regression(X[:1, :], B).shape == (10, 1, 2)
    torch.jit.trace(linear_regression, (X, B))
    torch.jit.trace(linear_regression_loss, (P, Y))


def test_logistic_model():
    X = torch.as_tensor(np.random.normal(size=(10, 4)))
    B = torch.as_tensor(np.random.normal(size=(10, 4)))
    Y = torch.as_tensor(np.eye(2)[np.random.randint(0, 2, 10)])
    assert B.shape[1] == logistic_regression_coefficients(X, Y)
    P = logistic_regression(X, B)
    L = logistic_regression_loss(P, Y)
    assert P.shape == (10, 10, 2)
    assert L.shape == (10, 10)
    L = logistic_regression_loss(Y, Y)
    assert L.shape == (10,)
    assert logistic_regression(X, B[:1, :]).shape == (1, 10, 2)
    assert logistic_regression(X[:1, :], B).shape == (10, 1, 2)
    torch.jit.trace(logistic_regression, (X, B))
    torch.jit.trace(logistic_regression_loss, (P, Y))
    with pytest.raises(SlisemapException, match="AssertionError"):
        logistic_regression_loss(
            logistic_regression(X, B),
            torch.as_tensor(np.random.uniform(0, 1.0, size=(10, 1))),
        )
    B = torch.as_tensor(np.random.normal(size=(10, 8)))
    Y = torch.as_tensor(np.eye(3)[np.random.randint(0, 3, 10)])
    assert B.shape[1] == logistic_regression_coefficients(X, Y)
    P = logistic_regression(X, B)
    L = logistic_regression_loss(P, Y)
    assert P.shape == (10, 10, 3)
    assert L.shape == (10, 10)
    L = logistic_regression_loss(Y, Y)
    assert L.shape == (10,)
    assert logistic_regression(X, B[:1, :]).shape == (1, 10, 3)
    assert logistic_regression(X[:1, :], B).shape == (10, 1, 3)
    torch.jit.trace(logistic_regression, (X, B))
    torch.jit.trace(logistic_regression_loss, (P, Y))


def test_logistic_log_model():
    X = torch.as_tensor(np.random.normal(size=(10, 4)))
    B = torch.as_tensor(np.random.normal(size=(10, 4)))
    Y = torch.as_tensor(np.random.normal(size=(10, 2)))
    assert B.shape[1] == logistic_regression_coefficients(X, Y)
    P = logistic_regression_log(X, B)
    print(P.shape, Y.shape)
    L = logistic_regression_log_loss(P, Y)
    assert P.shape == (10, 10, 2)
    assert L.shape == (10, 10)
    L = logistic_regression_log_loss(Y, Y)
    assert L.shape == (10,)
    assert logistic_regression_log(X, B[:1, :]).shape == (1, 10, 2)
    assert logistic_regression_log(X[:1, :], B).shape == (10, 1, 2)
    torch.jit.trace(logistic_regression_log, (X, B))
    torch.jit.trace(logistic_regression_log_loss, (P, Y))
    B = torch.as_tensor(np.random.normal(size=(10, 8)))
    Y = torch.as_tensor(np.random.normal(size=(10, 3)))
    assert B.shape[1] == logistic_regression_coefficients(X, Y)
    P = logistic_regression_log(X, B)
    L = logistic_regression_log_loss(P, Y)
    assert P.shape == (10, 10, 3)
    assert L.shape == (10, 10)
    L = logistic_regression_log_loss(Y, Y)
    assert L.shape == (10,)
    assert logistic_regression_log(X, B[:1, :]).shape == (1, 10, 3)
    assert logistic_regression_log(X[:1, :], B).shape == (10, 1, 3)
    torch.jit.trace(logistic_regression_log, (X, B))
    torch.jit.trace(logistic_regression_log_loss, (P, Y))


def test_identify():
    p, l, c, r = identify_local_model(LinearRegression)
    assert p == linear_regression
    assert l == linear_regression_loss
    assert c == linear_regression_coefficients
    assert r == ALocalModel.regularisation
    p, l, c, r = identify_local_model(LinearRegression, logistic_regression_loss, 3)
    assert p == linear_regression
    assert l == logistic_regression_loss
    assert c(None, None) == 3
    assert r == ALocalModel.regularisation


def test_local_predict():
    X = torch.normal(0.0, 1.0, (10, 5))
    B = torch.normal(0.0, 1.0, (10, 5))
    Y = local_predict(X, B, linear_regression)
    assert Y.shape == (10, 1)
    B = torch.normal(0.0, 1.0, (10, 10))
    Y = local_predict(X, B, linear_regression)
    assert Y.shape == (10, 2)
    Y = local_predict(X, B, logistic_regression)
    assert Y.shape == (10, 3)
