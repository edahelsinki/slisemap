from slisemap.local_models import *

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
    B = torch.as_tensor(np.random.normal(size=(10, 8)))
    Y = torch.as_tensor(np.random.normal(size=(10, 2)))
    assert B.shape[1] == linear_regression_coefficients(X, Y)
    P = multiple_linear_regression(X, B)
    L = linear_regression_loss(P, Y)
    assert P.shape == (10, 10, 2)
    assert L.shape == (10, 10)
    L = linear_regression_loss(Y, Y)
    assert L.shape == (10,)


def test_logistic_model():
    X = torch.as_tensor(np.random.normal(size=(10, 4)))
    B = torch.as_tensor(np.random.normal(size=(10, 4)))
    Y = torch.as_tensor(np.eye(2)[np.random.randint(0, 2, 10)])
    assert B.shape[1] == logistic_regression_coefficients(X, Y)
    P = logistic_regression(X, B)
    print(P.shape, Y.shape)
    L = logistic_regression_loss(P, Y)
    assert P.shape == (10, 10, 2)
    assert L.shape == (10, 10)
    L = linear_regression_loss(Y, Y)
    assert L.shape == (10,)
    B = torch.as_tensor(np.random.normal(size=(10, 8)))
    Y = torch.as_tensor(np.eye(3)[np.random.randint(0, 3, 10)])
    assert B.shape[1] == logistic_regression_coefficients(X, Y)
    P = logistic_regression(X, B)
    L = logistic_regression_loss(P, Y)
    assert P.shape == (10, 10, 3)
    assert L.shape == (10, 10)
    L = linear_regression_loss(Y, Y)
    assert L.shape == (10,)
