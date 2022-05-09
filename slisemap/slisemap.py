"""
This module contains the Slisemap class.
"""

from copy import copy
from os import PathLike
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from slisemap.escape import escape_neighbourhood
from slisemap.local_models import linear_regression, linear_regression_loss
from slisemap.loss import make_loss, make_marginal_loss, softmax_kernel
from slisemap.utils import (
    LBFGS,
    CheckConvergence,
    PCA_rotation,
    _assert,
    _tonp,
    _warn,
    global_model,
    varimax,
)


class Slisemap:
    """Slisemap: combine local explanations with dimensionality reduction.

    __Example Usage__
    X = np.array([[0.1,0.5,0.7], [0.8,0.9,1], [0.8,0.5,0.3], [0.1,0.2,0.3], [1,2,5], [2,3,4], [2,0,1]])
    y = np.array([1, 2, 3, 4, 1.5, 1.8, 1.7])
    sm = Slisemap(X, y, radius=3.5, lasso=1e-4, ridge=2e-4)
    sm.optimise()
    sm.plot()
    """

    # Make Python faster and safer by not creating a Slisemap.__dict__
    __slots__ = (
        "_X",
        "_Y",
        "_Z",
        "_Z0",
        "_B",
        "_B0",
        "_radius",
        "_lasso",
        "_ridge",
        "_z_norm",
        "_intercept",
        "_local_model",
        "_local_loss",
        "_loss",
        "_distance",
        "_kernel",
        "_jit",
    )

    def __init__(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        radius: float = 3.5,
        d: int = 2,
        lasso: Optional[float] = None,
        ridge: Optional[float] = None,
        z_norm: float = 0.01,
        intercept: bool = True,
        local_model: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = linear_regression,
        local_loss: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ] = linear_regression_loss,
        coefficients: Union[
            None, int, Callable[[torch.Tensor, torch.Tensor], int]
        ] = None,
        distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.cdist,
        kernel: Callable[[torch.Tensor], torch.Tensor] = softmax_kernel,
        B0: Union[None, np.ndarray, torch.Tensor] = None,
        Z0: Union[None, np.ndarray, torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
        jit: bool = True,
        cuda: Optional[bool] = None,
    ):
        """Create a new Slisemap object.

        Args:
            X (Union[np.ndarray, torch.Tensor]): Data matrix. Note that the data is assumed to be normalised.
            y (Union[np.ndarray, torch.Tensor]): Target vector or matrix.
            radius (float, optional): The radius of the embedding Z. Defaults to 3.5.
            d (int, optional): The number of embedding dimensions. Defaults to 2.
            lasso (Optional[float], optional): Lasso regularisation coefficient. Defaults to 0.0.
            ridge (Optional[float], optional): Ridge regularisation coefficient. Defaults to 0.0.
            z_norm (float, optional): Z normalisation regularisation coefficient. Defaults to 0.01.
            intercept (bool, optional): Should an intercept term be added to self.X. Defaults to True.
            local_model (Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional): Local model prediction function. Defaults to linear_regression.
            local_loss (Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor], optional): Local model loss function. Defaults to linear_regression_loss.
            coefficients (Union[None, int, Callable[[torch.Tensor, torch.Tensor], int]], optional): The number of local model coefficients or a function: `(X,Y)->coefficients`. Defaults to self.X.shape[1] * self.Y.shape[1].
            distance (Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional): Distance function. Defaults to torch.cdist (Euclidean distance).
            kernel (Callable[[torch.Tensor], torch.Tensor], optional): Kernel function. Defaults to softmax_kernel.
            B0 (Union[None, np.ndarray, torch.Tensor], optional): Initial value for B (random if None). Defaults to None.
            Z0 (Union[None, np.ndarray, torch.Tensor], optional): Initial value for Z (PCA if None). Defaults to None.
            dtype (torch.dtype, optional): Floating type. Defaults to torch.float32.
            jit (bool, optional): Just-In-Time compile the loss function for increased performance (see `torch.jit.trace` for caveats). Defaults to True.
            cuda (Optional[bool], optional): Use cuda if available (defaults to true if the data is large enough). Defaults to None.
        """
        if lasso is None and ridge is None:
            _warn(
                "Consider using regularisation (lasso/l1 and ridge/l2 regularisation is built-in, via the parameters `lasso` and `ridge`). "
                + "Regularisation is important for handling small neighbourhoods, and also makes the local models more local. "
                + "Set `lasso=0` to disable this warning (if no regularisation is really desired).",
                Slisemap,
            )
        self.lasso = 0.0 if lasso is None else lasso
        self.ridge = 0.0 if ridge is None else ridge
        self.kernel = kernel
        self.distance = distance
        self.local_model = local_model
        self.local_loss = local_loss
        self.z_norm = z_norm
        self.radius = radius
        self._intercept = intercept
        self._jit = jit

        if cuda is None:
            scale = (
                X.shape[0] ** 2 * X.shape[1] * (1 if len(y.shape) < 2 else y.shape[1])
            )
            cuda = scale > 1_000_000 and torch.cuda.is_available()
        tensorargs = {
            "device": torch.device("cuda" if cuda else "cpu"),
            "dtype": dtype,
        }

        self._X = torch.as_tensor(X, **tensorargs)
        if intercept:
            self._X = torch.cat((self._X, torch.ones_like(self._X[:, :1])), 1)
        n, m = self._X.shape

        self._Y = torch.as_tensor(y, **tensorargs)
        _assert(
            self._Y.shape[0] == n,
            f"The length of y must match X: {self._Y.shape[0]} != {n}",
        )
        if len(self._Y.shape) == 1:
            self._Y = self._Y[:, None]

        if Z0 is None:
            self._Z0 = self._X @ PCA_rotation(self._X, min(m, d))
            if d > m:
                _warn(
                    "The number of embedding dimensions is larger than the number of data dimensions",
                    Slisemap,
                )
                Z0fill = torch.normal(mean=0, std=0.05, size=[n, d - m])
                self._Z0 = torch.cat((self._Z0, Z0fill), 1)
        else:
            self._Z0 = torch.as_tensor(Z0, **tensorargs)
            _assert(
                self._Z0.shape == (n, d),
                f"Z0 has the wrong shape: {self._Z0.shape} != ({n}, {d})",
            )
        if radius > 0:
            norm = 1 / (torch.sqrt(torch.sum(self._Z0**2) / self._Z0.shape[0]) + 1e-8)
            self._Z0 = self._Z0 * norm
        self._Z = self._Z0.detach().clone()

        if callable(coefficients):
            coefficients = coefficients(self._X, self._Y)
        if B0 is None:
            if coefficients is None:
                coefficients = m * self.p
            B0 = global_model(
                X=self._X,
                Y=self._Y,
                local_model=self.local_model,
                local_loss=self.local_loss,
                coefficients=coefficients,
                lasso=self.lasso,
                ridge=self.ridge,
            )
            if torch.all(torch.isfinite(B0)):
                self._B0 = B0.expand((n, coefficients))
            else:
                _warn(
                    "Optimising a global model as initialisation resulted in non-finite values. Consider using stronger regularisation (increase `lasso` or `ridge`).",
                    Slisemap,
                )
                self._B0 = torch.zeros((n, coefficients), **tensorargs)
        else:
            self._B0 = torch.as_tensor(B0, **tensorargs)
            if coefficients is None:
                _assert(len(B0.shape) > 1, "B0 must have more than one dimension")
                coefficients = B0.shape[1]
            _assert(
                self._B0.shape == (n, coefficients),
                f"B0 has the wrong shape: {self._B0.shape} != ({n}, {coefficients})",
            )
        self._B = self._B0.detach().clone()

    def restore(self):
        """Reset B and Z to their initial values (B0 and Z0)."""
        self._Z = self._Z0.clone().detach()
        self._B = self._B0.clone().detach()

    @property
    def n(self) -> int:
        """The number of data items."""
        return self._X.shape[0]

    @property
    def m(self) -> int:
        """The number of variables (including potential intercept)."""
        return self._X.shape[1]

    @property
    def p(self) -> int:
        """The number of target variables (i.e. the number of classes)."""
        return self._Y.shape[-1]

    @property
    def d(self) -> int:
        """The number of embedding dimensions."""
        return self._Z.shape[1]

    @d.setter
    def d(self, value: int):
        _assert(value > 0, "The number of embedding dimensions must be positive")
        if self.d != value:
            self._loss = None  # invalidate cached loss function
            self._Z = self._Z.detach()
            if self.d > value:
                self._Z = self._Z @ PCA_rotation(self._Z, value)
            else:
                zn = [self._Z, torch.zeros((self.n, value - self.d), **self.tensorargs)]
                self._Z = torch.concat(zn, 1)

    @property
    def coefficients(self) -> int:
        """The number of local model coefficients."""
        return self._B.shape[1]

    @property
    def intercept(self) -> bool:
        """Is an intercept column added to the data?"""
        return self._intercept

    @property
    def radius(self) -> float:
        """The radius of the embedding."""
        return self._radius

    @radius.setter
    def radius(self, value: float):
        _assert(value >= 0, "radius must not be negative")
        self._radius = value
        self._loss = None  # invalidate cached loss function

    @property
    def lasso(self) -> float:
        """Lasso regularisation strength."""
        return self._lasso

    @lasso.setter
    def lasso(self, value: float):
        _assert(value >= 0, "lasso must not be negative")
        self._lasso = value
        self._loss = None  # invalidate cached loss function

    @property
    def ridge(self) -> float:
        """Ridge regularisation strength."""
        return self._ridge

    @ridge.setter
    def ridge(self, value: float):
        _assert(value >= 0, "ridge must not be negative")
        self._ridge = value
        self._loss = None  # invalidate cached loss function

    @property
    def z_norm(self) -> float:
        """Z normalisation regularisation strength."""
        return self._z_norm

    @z_norm.setter
    def z_norm(self, value: float):
        _assert(value >= 0, "z_norm must not be negative")
        self._z_norm = value
        self._loss = None  # invalidate cached loss function

    @property
    def local_model(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Local model prediction function. Takes in X[n, m] and B[n, coefficients], and returns Ytilde[n, n, p]."""
        return self._local_model

    @local_model.setter
    def local_model(self, value: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        _assert(callable(value), "local_model must be callable")
        self._local_model = value
        self._loss = None  # invalidate cached loss function

    @property
    def local_loss(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Local model loss function. Takes in Ytilde[n, n, p], Y[n, p], and B[n, coefficients], and returns L[n, n]"""
        return self._local_loss

    @local_loss.setter
    def local_loss(
        self, value: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    ):
        _assert(callable(value), "local_loss must be callable")
        self._local_loss = value
        self._loss = None  # invalidate cached loss function

    @property
    def distance(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Distance function. Takes in Z[n1, d] and Z[n2, d], and returns D[n1, n2]"""
        return self._distance

    @distance.setter
    def distance(self, value: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        _assert(callable(value), "distance must be callable")
        self._distance = value
        self._loss = None  # invalidate cached loss function

    @property
    def kernel(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Kernel function. Takes in D[n, n] and returns W[n, n]"""
        return self._kernel

    @kernel.setter
    def kernel(self, value: Callable[[torch.Tensor], torch.Tensor]):
        _assert(callable(value), "kernel must be callable")
        self._kernel = value
        self._loss = None  # invalidate cached loss function

    @property
    def jit(self) -> bool:
        """Just-In-Time compile the loss function."""
        return self._jit

    @jit.setter
    def jit(self, value: bool):
        self._jit = value
        self._loss = None  # invalidate cached loss function

    @property
    def X(self) -> torch.Tensor:
        """Data matrix as a `torch.Tensor` (use `Slisemap.get_X()` for options)."""
        return self._X

    @property
    def Y(self) -> torch.Tensor:
        """Target matrix as a `torch.Tensor` (use `Slisemap.get_Y()` for options)."""
        return self._Y

    @property
    def Z(self) -> torch.Tensor:
        """Normalised embedding matrix as a `torch.Tensor` (use `Slisemap.get_Z()` for options)."""
        return self._Z

    @property
    def B(self) -> torch.Tensor:
        """Coefficient matrix for the local models as a `torch.Tensor` (use `Slisemap.get_B()` for options)."""
        return self._B

    @property
    def tensorargs(self) -> Dict[str, Any]:
        """When creating a new `torch.Tensor` add these keyword arguments to match the `dtype` and `device` of this Slisemap object."""
        return dict(device=self._X.device, dtype=self._X.dtype)

    def cuda(self, **kwargs):
        """Move the tensors to CUDA memory (and run the calculations there).

        Args:
            **kwargs: Optional arguments to `torch.Tensor.cuda`
        """
        X = self._X.cuda(**kwargs)
        self._X = X
        self._Y = self._Y.cuda(**kwargs)
        self._Z = self._Z.detach().cuda(**kwargs)
        self._B = self._B.detach().cuda(**kwargs)
        self._loss = None  # invalidate cached loss function

    def cpu(self, **kwargs):
        """Move the tensors to CPU memory (and run the calculations there).

        Args:
            **kwargs: Optional arguments to `torch.Tensor.cpu`
        """
        X = self._X.cpu(**kwargs)
        self._X = X
        self._Y = self._Y.cpu(**kwargs)
        self._Z = self._Z.detach().cpu(**kwargs)
        self._B = self._B.detach().cpu(**kwargs)
        self._loss = None  # invalidate cached loss function

    def get_loss_fn(
        self, individual: bool = False
    ) -> Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ]:
        """Returns the slisemap loss function.
        This function JITs and caches the loss function for efficiency.

        Args:
            individual (bool, optional): Make a loss function for individual losses. Defaults to False.

        Returns:
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]: Loss function: (X, Y, B, Z) -> loss.
        """
        if individual:
            return make_loss(
                local_model=self.local_model,
                local_loss=self.local_loss,
                distance=self.distance,
                kernel=self.kernel,
                radius=self.radius,
                lasso=self.lasso,
                ridge=self.ridge,
                z_norm=0.0,
                individual=True,
            )
        if self._loss is None:
            # Caching the loss function
            self._loss = make_loss(
                local_model=self.local_model,
                local_loss=self.local_loss,
                distance=self.distance,
                kernel=self.kernel,
                radius=self.radius,
                lasso=self.lasso,
                ridge=self.ridge,
                z_norm=self.z_norm,
            )
            # JITting the loss function improves the performance
            if self._jit:
                self._loss = torch.jit.trace(
                    self._loss, (self._X[:1], self._Y[:1], self._B[:1], self._Z[:1])
                )
        return self._loss

    def _as_new_X(
        self, X: Union[None, np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        if X is None:
            return self._X
        tensorargs = self.tensorargs
        X = torch.atleast_2d(torch.as_tensor(X, **tensorargs))
        if self._intercept:
            X = torch.cat([X, torch.ones((X.shape[0], 1), **tensorargs)], 1)
        _assert(
            X.shape[1] == self.m,
            f"X has the wrong shape {X.shape[1]} != {self.m}",
        )
        return X

    def _as_new_Y(
        self, Y: Union[None, float, np.ndarray, torch.Tensor] = None, n: int = -1
    ) -> torch.Tensor:
        if Y is None:
            return self._Y
        Y = torch.as_tensor(Y, **self.tensorargs)
        if len(Y.shape) < 2:
            Y = torch.reshape(Y, (n, self.p))
        n = Y.shape[0]
        _assert(
            Y.shape == (n, self.p),
            f"Y has the wrong shape {Y.shape} != {(n, self.p)}",
        )
        return Y

    def get_Z(
        self, scale: bool = True, rotate: bool = False, numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """Get the Z matrix

        Args:
            scaled (bool, optional): Scale the returned Z to match self.radius. Defaults to True.
            rotate (bool, optional): Rotate the returned Z so that the first dimension is the major axis. Defaults to False.
            numpy (bool, optional): Return the matrix as a numpy (True) or pytorch (False) matrix. Defaults to True.

        Returns:
            Union[np.ndarray, torch.Tensor]: The Z matrix
        """
        if scale:
            if self.radius > 0:
                Zss = torch.sum(self._Z**2)
                Z = self._Z * (
                    self.radius / (torch.sqrt(Zss / self._Z.shape[0]) + 1e-8)
                )
            else:
                Z = self._Z - self._Z.mean(dim=0, keepdim=True)
        else:
            Z = self._Z
        if rotate:
            Z = varimax(Z)
        if numpy:
            return Z.cpu().detach().numpy()
        else:
            return Z.detach().clone()

    def get_B(self, numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Get the B matrix

        Args:
            numpy (bool, optional): Return the matrix as a numpy (True) or pytorch (False) matrix. Defaults to True.

        Returns:
            Union[np.ndarray, torch.Tensor]: The B matrix
        """
        return _tonp(self._B) if numpy else self._B.detach().clone()

    def get_D(self, numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Get the embedding distance matrix

        Args:
            numpy (bool, optional): Return the matrix as a numpy (True) or pytorch (False) matrix. Defaults to True.

        Returns:
            Union[np.ndarray, torch.Tensor]: The D matrix
        """
        Z = self.get_Z(rotate=False, numpy=False)
        D = self._distance(Z, Z)
        return _tonp(D) if numpy else D.detach()

    def get_L(
        self,
        X: Union[None, np.ndarray, torch.Tensor] = None,
        Y: Union[None, float, np.ndarray, torch.Tensor] = None,
        numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Get the loss matrix: [B.shape[0], X.shape[0]].

        Args:
            X (Union[None, np.ndarray, torch.Tensor], optional): Optional replacement for the training X. Defaults to None.
            Y (Union[None, float, np.ndarray, torch.Tensor], optional): Optional replacement for the training Y. Defaults to None.
            numpy (bool, optional): Return the matrix as a numpy (True) or pytorch (False) matrix. Defaults to True.

        Returns:
            Union[np.ndarray, torch.Tensor]: The L matrix
        """
        X = self._as_new_X(X)
        Y = self._as_new_Y(Y, X.shape[0])
        L = self.local_loss(self.local_model(X, self._B), Y, self._B)
        return _tonp(L) if numpy else L.detach()

    def get_W(self, numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Get the weight matrix

        Args:
            numpy (bool, optional): Return the matrix as a numpy.ndarray instead of a torch.Tensor. Defaults to True.

        Returns:
            Union[np.ndarray, torch.Tensor]: The W matrix
        """
        W = self.kernel(self.get_D(numpy=False))
        return _tonp(W) if numpy else W.detach()

    def get_X(
        self, numpy: bool = True, intercept: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """Get the data matrix

        Args:
            numpy (bool, optional): Return the matrix as a numpy.ndarray instead of a torch.Tensor. Defaults to True.
            intercept (bool, optional): Include the intercept column (if `self.intercept == True`). Defaults to True.

        Returns:
            Union[np.ndarray, torch.Tensor]: The X matrix
        """
        X = self._X if intercept or not self._intercept else self._X[:, :-1]
        return _tonp(X) if numpy else X.detach().clone()

    def get_Y(
        self, numpy: bool = True, ravel: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """Get the target matrix

        Args:
            numpy (bool, optional): Return the matrix as a numpy.ndarray instead of a torch.Tensor. Defaults to True.
            ravel (bool, optional): Remove the second dimension if it is singular (i.e. turn it into a vector). Defaults to False.

        Returns:
            Union[np.ndarray, torch.Tensor]: The Y matrix
        """
        Y = self._Y.ravel() if ravel else self._Y
        return _tonp(Y) if numpy else Y.detach().clone()

    def value(
        self, individual: bool = False, numpy: bool = True
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """Calculate the loss value.

        Args:
            individual (bool, optional): Give loss individual loss values for the data points. Defaults to False.

        Returns:
            Union[float, np.ndarray]: loss value(s).
        """
        loss = self.get_loss_fn(individual)
        loss = loss(X=self._X, Y=self._Y, B=self._B, Z=self._Z)
        if individual:
            return _tonp(loss) if numpy else loss.detach()
        else:
            return loss.cpu().detach().item() if numpy else loss.detach()

    def entropy(
        self, aggregate: bool = True, numpy: bool = True
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """Compute row-wise entropy of the W matrix induced by Z.

        Args:
            aggregate (bool, optional): Aggregate the row-wise entropies into one scalar. Defaults to True.
            numpy (bool, optional): return as a numpy.ndarray or float instead of torch.Tensor. Defaults to True.

        Returns:
            Union[float, np.ndarray, torch.Tensor]: entropy
        """
        W = self.get_W(False)
        entropy = -(W * W.log()).sum(dim=1)
        if aggregate:
            entropy = (entropy.mean().exp() / self.n).detach()
            return entropy.cpu().item() if numpy else entropy
        else:
            return _tonp(entropy) if numpy else entropy.detach()

    def lbfgs(self, max_iter: int = 500, *, only_B: bool = False, **kwargs) -> float:

        """Optimise Slisemap using LBFGS.

        Args:
            max_iter (int, optional): Maximum number of LBFGS iterations. Defaults to 500.
            only_B (bool, optional): Only optimise B. Defaults to False.
            **kwargs: Optional keyword arguments to LBFGS.

        Returns:
            float: The loss value.
        """
        Z = self._Z.detach().clone().requires_grad_(True)
        B = self._B.detach().clone().requires_grad_(True)

        loss_ = self.get_loss_fn()
        loss_fn = lambda: loss_(self._X, self._Y, B, Z)
        LBFGS(loss_fn, [B] if only_B else [Z, B], max_iter=max_iter, **kwargs)

        if np.isnan(loss_fn().cpu().detach().item()):
            _warn(
                "An LBFGS optimisation resulted in `nan` (try strengthening the regularisation or reducing the radius)",
                Slisemap.lbfgs,
            )
            # Some datasets (with logistic local models) are initially numerically unstable.
            # Just running LBFGS for one iteration seems to avoid those issues.
            Z = self._Z.detach().requires_grad_(True)
            B = self._B.detach().requires_grad_(True)
            LBFGS(loss_fn, [B] if only_B else [Z, B], max_iter=1, **kwargs)

        self._Z = Z.detach()
        self._B = B.detach()
        self._normalise()

        return self.value()

    def escape(
        self,
        force_move: bool = True,
        escape_fn: Callable = escape_neighbourhood,
    ):
        """Try to escape a local optimum by moving the items (embedding and local model) to the neighbourhoods best suited for them.
        This is done by finding another item (in the optimal neighbourhood) and copying its values for Z and B.

        Args:
            force_move (bool, optional): Do not allow the items to pair with themselves. Defaults to True.
            escape_fn (Callable, optional): Escape function (escape_neighbourhood/escape_greedy/escape_marginal). Defaults to escape_neighbourhood.
        """
        self._B, self._Z = escape_fn(
            X=self._X,
            Y=self._Y,
            B=self._B,
            Z=self._Z,
            local_model=self.local_model,
            local_loss=self.local_loss,
            distance=self.distance,
            kernel=self.kernel,
            radius=self.radius,
            force_move=force_move,
            jit=self.jit,
        )
        self._normalise()

    def _normalise(self):
        """Normalise Z."""
        if self.radius > 0:
            self._Z *= 1 / (torch.sqrt(torch.sum(self._Z**2) / self.n) + 1e-8)

    def optimise(
        self,
        patience: int = 3,
        max_escapes: int = 100,
        max_iter: int = 500,
        escape_fn: Callable = escape_neighbourhood,
        verbose: bool = False,
        noise_scale: float = 1e-4,
        **kwargs,
    ) -> float:
        """Optimise Slisemap by alternating between Slisemap.lbfgs and Slisemap.escape until convergence.

        Args:
            patience (int, optional): Number of escapes without improvement before stopping. Defaults to 3.
            max_escapes (int, optional): Maximum number of escapes. Defaults to 100.
            max_iter (int, optional): Maximum number of LBFGS iterations per round. Defaults to 500.
            escape_fn (Callable, optional): Escape function (escape_neighbourhood/escape_greedy/escape_marginal). Defaults to escape_neighbourhood.
            verbose (bool, optional): Print status messages. Defaults to False.
            noise_scale (float, optional): Scale of the noise used to avoid loosing dimensions in the embedding after an escape (when using a gradient based optimiser). Defaults to 1e-4.
            **kwargs: Optional keyword arguments to Slisemap.lbfgs.

        Returns:
            float: The loss value.
        """
        loss = np.repeat(np.inf, 2)
        loss[0] = self.lbfgs(max_iter=max_iter, only_B=True, **kwargs)
        if verbose:
            print(f"LBFGS   0: {loss[0]:.2f}")
        cc = CheckConvergence(patience)
        for i in range(max_escapes):
            self.escape(escape_fn=escape_fn)
            loss[1] = self.value()
            if verbose:
                print(f"Escape {i:2d}: {loss[1]:.2f}")
            if noise_scale > 0.0:
                self._Z = torch.normal(self.Z, noise_scale)
            loss[0] = self.lbfgs(max_iter=max_iter, **kwargs)
            if verbose:
                print(f"LBFGS  {i+1:2d}: {loss[0]:.2f}")
            if cc.has_converged(loss, self.copy):
                break
        if cc.optimal_state is not None:
            self._Z = cc.optimal_state._Z
            self._B = cc.optimal_state._B
        return self.lbfgs(max_iter=max_iter * 2, **kwargs)

    def fit_new(
        self,
        Xnew: Union[np.ndarray, torch.Tensor],
        ynew: Union[float, np.ndarray, torch.Tensor],
        optimise: bool = True,
        between: bool = True,
        escape_fn: Callable = escape_neighbourhood,
        loss: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Generate embedding(s) and model(s) for new data item(s).

        This works as follows:
            1. Find good initial embedding(s) and local model(s) using the escape_fn.
            2. Optionally finetune the embedding(s) and model(s) using LBFG.

        Args:
            Xnew (Union[np.ndarray, torch.Tensor]): New data point(s).
            ynew (Union[float, np.ndarray, torch.Tensor]): New target(s).
            optimise (bool, optional): Should the embedding and model be optimised (after finding the neighbourhood). Defaults to True.
            between (bool, optional): If `optimise=True` should the new points affect each other. Defaults to True.
            escape_fn (Callable, optional): Escape function to use as initialisation (escape_neighbourhood/escape_greedy/escape_marginal). Defaults to escape_neighbourhood.
            loss (bool, optional): Return a vector of individual losses for the new items. Defaults to False.
            **kwargs: Optional keyword arguments to LBFGS.

        Returns:
            Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]: Bnew, Znew, (loss).
        """
        Xnew = self._as_new_X(Xnew)
        n = Xnew.shape[0]
        ynew = self._as_new_Y(ynew, n)
        Bnew, Znew = escape_fn(
            X=Xnew,
            Y=ynew,
            B=self.B,
            Z=self.Z,
            local_model=self.local_model,
            local_loss=self.local_loss,
            distance=self.distance,
            kernel=self.kernel,
            radius=self.radius,
            force_move=False,
            Xold=self.X,
            Yold=self.Y,
            jit=self.jit,
        )

        if optimise:
            lf, set_new = make_marginal_loss(
                X=self.X,
                Y=self.Y,
                B=self.B,
                Z=self.Z,
                Xnew=Xnew if between else Xnew[:1],
                Ynew=ynew if between else ynew[:1],
                local_model=self.local_model,
                local_loss=self.local_loss,
                distance=self.distance,
                kernel=self.kernel,
                radius=self.radius,
                lasso=self.lasso,
                ridge=self.ridge,
                jit=self.jit,
            )
            if between:
                Bnew = Bnew.detach().requires_grad_(True)
                Znew = Znew.detach().requires_grad_(True)
                LBFGS(lambda: lf(Bnew, Znew), [Bnew, Znew], **kwargs)
            else:
                for j in range(n):
                    set_new(Xnew[None, j], ynew[None, j])
                    Bi = Bnew[None, j].detach().clone().requires_grad_(True)
                    Zi = Znew[None, j].detach().clone().requires_grad_(True)
                    LBFGS(lambda: lf(Bi, Zi), [Bi, Zi], **kwargs)
                    Bnew[j] = Bi.detach()
                    Znew[j] = Zi.detach()
        if self.radius > 0:
            Zout = Znew * (self.radius / (torch.sum(self._Z**2) + 1e-8))
        else:
            Zout = Znew
        if loss:
            lf = self.get_loss_fn(individual=True)
            if between:
                loss = lf(
                    X=torch.cat((self.X, Xnew), 0),
                    Y=torch.cat((self.Y, ynew), 0),
                    B=torch.cat((self.B, Bnew), 0),
                    Z=torch.cat((self.Z, Znew), 0),
                )[self.n :]
                loss = _tonp(loss)
            else:
                if self._jit:
                    lf = torch.jit.trace(lf, (Xnew[:1], ynew[:1], Bnew[:1], Znew[:1]))
                loss = np.zeros(n)
                for j in range(n):
                    l = lf(
                        X=torch.cat((self.X, Xnew[None, j]), 0),
                        Y=torch.cat((self.Y, ynew[None, j]), 0),
                        B=torch.cat((self.B, Bnew[None, j]), 0),
                        Z=torch.cat((self.Z, Znew[None, j]), 0),
                    )[-1]
                    loss[j] = l.detach().cpu().item()
            return _tonp(Bnew), _tonp(Zout), loss
        else:
            return _tonp(Bnew), _tonp(Zout)

    def predict(
        self,
        Xnew: Union[np.ndarray, torch.Tensor],
        Znew: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ) -> np.ndarray:
        """Predict new outcomes when the data and embedding is known.

        Args:
            Xnew (Union[np.ndarray, torch.Tensor]): Data matrix.
            Znew (Union[np.ndarray, torch.Tensor]): Embedding matrix.
            **kwargs: Optional keyword arguments to LBFGS.

        Returns:
            np.ndarray: Prediction matrix.
        """
        Xnew = torch.as_tensor(Xnew, **self.tensorargs)
        Xnew = torch.atleast_2d(Xnew)
        N = Xnew.shape[0]
        if self._intercept:
            Xnew = torch.cat([Xnew, torch.ones([N, 1], **self.tensorargs)], 1)
        _assert(
            Xnew.shape == (N, self.m),
            f"Xnew has the wrong shape {Xnew.shape} != {(N, self.m)}",
        )
        Znew = torch.as_tensor(Znew, **self.tensorargs)
        Znew = torch.atleast_2d(Znew)
        _assert(
            Znew.shape == (N, self.d),
            f"Znew has the wrong shape {Znew.shape} != {(N, self.d)}",
        )
        D = self._distance(Znew, self._Z)
        W = self.kernel(D)
        Bnew = self._B[torch.argmin(D, 1)].clone().requires_grad_(True)
        loss = lambda: (
            torch.sum(
                W * self.local_loss(self.local_model(self._X, Bnew), self._Y, Bnew)
            )
            + self.lasso * torch.sum(torch.abs(Bnew))
            + self.ridge * torch.sum(Bnew**2)
        )
        LBFGS(loss, [Bnew], **kwargs)
        return _tonp(torch.diagonal(self.local_model(Xnew, Bnew), dim1=0, dim2=1).T)

    def copy(self) -> "Slisemap":
        """Make a copy of this Slisemap that references as much of the same torch-data as possible.

        Returns:
            Slisemap: An almost shallow copy of this Slisemap object.
        """
        other = copy(self)  # Shallow copy!
        # Deep copy these:
        other._B = other._B.clone().detach()
        other._Z = other._Z.clone().detach()
        return other

    def save(self, f: Union[str, PathLike, BinaryIO], **kwargs):
        """Save the Slisemap object to a file.
        This method uses `torch.save`, which uses `pickle` for the non-pytorch properties.
        This comes with the normal caveats like lambda-functions not being supported.
        However, the default pickle module can be overridden (see `torch.save`).
        The default ending for `torch.save` is ".pt" and by default the file is compressed.

        Args:
            f (Union[str, PathLike, BinaryIO]): Either a Path-like object or a (writable) File-like object.
            **kwargs: Parameters forwarded to `torch.save`.
        """
        loss = self._loss
        try:
            self._B = self._B.detach()
            self._Z = self._Z.detach()
            self._loss = None
            torch.save(self, f, **kwargs)
        finally:
            self._loss = loss

    @classmethod
    def load(
        cls,
        f: Union[str, PathLike, BinaryIO],
        device: Union[None, str, torch.device] = None,
        **kwargs,
    ) -> "Slisemap":
        """Load a Slisemap object from a file.
        This method uses `torch.load`, which uses `pickle` for the non-pytorch properties.
        However, the default pickle module can be overridden (see `torch.load`).

        Args:
            f (Union[str, PathLike, BinaryIO]): Either a Path-like object or a (readable) File-like object.
            device (Union[None, str, torch.device], optional): Device to load the tensors to (or the original if None). Defaults to None.
            **kwargs: Parameters forwarded to `torch.load`.

        Returns:
            Slisemap: The loaded Slisemap object.
        """
        sm = torch.load(f, map_location=device, **kwargs)
        return sm

    def _cluster_models(
        self, clusters: int, B: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get cluster labels from k-means on the local model coefficients.

        Args:
            clusters (int): Number of clusters.
            B (Optional[np.ndarray], optional): B matrix. Defaults to self.get_B().

        Returns:
            Tuple[np.ndarray, np.ndarray]: vector of cluster labels an matrix of cluster centers.
        """
        from sklearn.cluster import KMeans

        km = KMeans(clusters).fit(B if B is not None else self.get_B())
        influence = np.abs(km.cluster_centers_)
        influence = influence.max(0) + influence.mean(0)
        col = np.argmax(influence)
        ord = np.argsort(km.cluster_centers_[:, col])
        return np.argsort(ord)[km.labels_], km.cluster_centers_[ord]

    def plot(
        self,
        title: str = "",
        variables: Optional[Sequence[str]] = None,
        targets: Union[None, str, Sequence[str]] = None,
        clusters: Union[None, int, np.ndarray] = None,
        bars: Union[bool, int] = False,
        jitter: float = 0.0,
        B: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
        show: bool = True,
        **kwargs,
    ) -> Optional[Figure]:
        """Plot the Slisemap solution using seaborn.

        Args:
            title (str, optional): Title of the plot. Defaults to "".
            variables (Optional[Sequence[str]], optional): List of variable names. Defaults to None.
            targets (Union[None, str, Sequence[str]], optional): Target name(s). Defaults to None.
            clusters (Union[None, int, np.ndarray], optional): Can be None (plot individual losses), an int (plot k-means clusters of B), or an array of known cluster id:s. Defaults to None.
            bars (Union[bool, int], optional): If the clusters are from k-means, plot the local models in a bar plot. If `bar` is an int then only plot the most influential variables. Defaults to False.
            jitter (float, optional): Add random (normal) noise to the scatterplot. Defaults to 0.0.
            B (Optional[np.ndarray], optional): Override self.get_B() in the plot. Defaults to None.
            Z (Optional[np.ndarray], optional): Override self.get_Z() in the plot. Defaults to None.
            show (bool, optional): Show the plot. Defaults to True.
            **kwargs: Additional arguments to `plt.subplots`.

        Returns:
            Optional[Figure]: Matplotlib figure if show=False.
        """
        Z = self.get_Z(rotate=True) if Z is None else Z
        B = self.get_B() if B is None else B
        if Z.shape[1] == 1:
            Z = np.concatenate((Z, np.zeros_like(Z)), 1)
        elif Z.shape[1] > 2:
            _warn(
                "Only the first two dimensions in the embedding are plotted",
                Slisemap.plot,
            )
        if jitter > 0:
            Z = np.random.normal(Z, jitter)
        kwargs.setdefault("figsize", (12, 6))
        fig, (ax1, ax2) = plt.subplots(1, 2, **kwargs)
        if variables is not None:
            if self._intercept and len(variables) == self.m - 1:
                variables = list(variables) + ["Intercept"]
            if targets and not isinstance(targets, str) and len(targets) > 0:
                if B.shape[1] % len(variables) == 0 and B.shape[1] % len(targets) == 0:
                    variables = [f"{t}: {v}" for t in targets for v in variables]
                    variables = variables[: B.shape[1]]
            _assert(
                len(variables) == B.shape[1],
                f"The number of variable names ({len(variables)}) must match the number of coefficients ({B.shape[1]})",
            )
        if isinstance(clusters, int):
            clusters, centers = self._cluster_models(clusters, B)
            if bars:
                y = np.arange(B.shape[1]) if variables is None else variables
                if not isinstance(bars, bool):
                    influence = np.abs(centers)
                    influence = influence.max(0) + influence.mean(0)
                    mask = np.argsort(-influence)[:bars]
                    y = np.asarray(y)[mask]
                    B = B[:, mask]
                g = sns.barplot(
                    y=np.tile(y, B.shape[0]),
                    x=B.ravel(),
                    hue=np.repeat(clusters, B.shape[1]),
                    ax=ax2,
                    palette="bright",
                    orient="h",
                )
                g.legend().remove()
                lim = np.max(np.abs(g.get_xlim()))
                g.set(xlabel=None, ylabel=None, xlim=(-lim, lim))
            else:
                B = centers
        else:
            B = B[np.argsort(Z[:, 0])]
            _assert(not bars, "`bar!=False` requires that `clusters` is an integer")
        if clusters is None:
            L = self.value(individual=True)
            sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=L, palette="crest", ax=ax1)
            sns.heatmap(B, ax=ax2, center=0, cmap="RdBu", robust=True)
            ax1.legend(title="Fidelity")
            if variables is not None:
                ax2.set_xticks(np.arange(len(variables)) + 0.5)
                ax2.set_xticklabels(variables, rotation=30)
        else:
            sns.scatterplot(
                x=Z[:, 0],
                y=Z[:, 1],
                hue=clusters,
                style=clusters,
                palette="bright",
                ax=ax1,
            )
            ax1.legend(title="Cluster")
            if not bars:
                sns.heatmap(B, ax=ax2, center=0, cmap="RdBu", robust=True)
                if variables is not None:
                    ax2.set_xticks(np.arange(len(variables)) + 0.5)
                    ax2.set_xticklabels(variables, rotation=30)
        ax1.set_xlabel("SLISEMAP 1")
        ax1.set_ylabel("SLISEMAP 2")
        ax1.axis("equal")
        ax2.set_xlabel("Coefficients")
        ax1.set_title("Embedding")
        ax2.set_title("Local Models")
        sns.despine(fig)
        plt.suptitle(title)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            return fig

    def plot_position(
        self,
        X: Union[None, np.ndarray, torch.Tensor] = None,
        Y: Union[None, float, np.ndarray, torch.Tensor] = None,
        index: Union[None, int, Sequence[int]] = None,
        title: str = "",
        jitter: float = 0.0,
        col_wrap: int = 4,
        selection: bool = True,
        legend_inside: bool = True,
        show: bool = True,
        **kwargs,
    ) -> Optional[sns.FacetGrid]:
        """Plot fidelities for alternative locations for the selected item(s).
        Indicate the selected item(s) either via X&Y or via index.

        Args:
            X (Union[None, np.ndarray, torch.Tensor], optional): Data matrix for the selected data item(s). Defaults to None.
            Y (Union[None, float, np.ndarray, torch.Tensor], optional): Response matrix for the selected data item(s). Defaults to None.
            index (Union[None, int, Sequence[int]], optional): Index/indices of the selected data item(s). Defaults to None.
            title (str, optional): Title of the plot. Defaults to "".
            jitter (float, optional): Add random (normal) noise to the embedding. Defaults to 0.0.
            col_wrap (int, optional): Maximum number of columns. Defaults to 4.
            selection (bool, optional): Mark the selected data item(s), if index is given. Defaults to True.
            legend_inside (bool, optional): Move the legend inside the grid (if there is an empty cell). Defaults to True.
            show (bool, optional): Show the plot. Defaults to True.
            **kwargs: Additional arguments to seaborn.relplot.

        Returns:
            Optional[sns.FacetGrid]: Seaborn FacetGrid if show=False.
        """
        import pandas as pd

        Z = self.get_Z(rotate=True)
        if Z.shape[1] == 1:
            Z = np.concatenate((Z, np.zeros_like(Z)), 1)
        elif Z.shape[1] > 2:
            _warn(
                "Only the first two dimensions in the embedding are plotted",
                Slisemap.plot_position,
            )
        if jitter > 0:
            Z = np.random.normal(Z, jitter)
        if index is None:
            _assert(
                X is not None and Y is not None, "Either index or X and Y must be given"
            )
            L = self.get_L(X, Y)
        else:
            if isinstance(index, int):
                index = [index]
            L = self.get_L()[:, index]
        df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "SLISEMAP 1": Z[:, 0],
                        "SLISEMAP 2": Z[:, 1],
                        "Fidelity": loss,
                        "i": i,
                    }
                )
                for i, loss in enumerate(L.T)
            ],
            ignore_index=True,
        )
        kwargs.setdefault("palette", "crest")
        kwargs.setdefault("kind", "scatter")
        g: sns.FacetGrid = sns.relplot(
            data=df,
            x="SLISEMAP 1",
            y="SLISEMAP 2",
            hue="Fidelity",
            hue_norm=tuple(np.quantile(df["Fidelity"], (0.0, 0.9))),
            col="i",
            col_wrap=min(col_wrap, L.shape[1]),
            legend=False,
            **kwargs,
        )
        cmap = plt.get_cmap(kwargs["palette"])
        legend = {
            f"{l:.{2}f}": Patch(color=cmap(n))
            for l, n in zip(
                np.linspace(*np.quantile(df["Fidelity"], (0.0, 0.9)), 6).round(2),
                np.linspace(0.0, 1.0, 6),
            )
        }
        inside = legend_inside and col_wrap < L.shape[1] and L.shape[1] % col_wrap != 0
        w = 1 / col_wrap
        h = 1 / ((L.shape[1] - 1) // col_wrap + 1)
        if selection and index is not None:
            for i, ax in zip(index, g.axes.ravel()):
                ax.scatter(Z[i, 0], Z[i, 1], 100, "#9b52b4", "X")
            g.add_legend(
                legend,
                "Fidelity",
                loc="lower center" if inside else "upper right",
                bbox_to_anchor=(1 - w, h * 0.4, w * 0.9, h * 0.5) if inside else None,
            )
            g.add_legend(
                {"": Line2D([], [], None, "None", "#9b52b4", "X", 8)},
                "Selected",
                loc="upper center" if inside else "lower right",
                bbox_to_anchor=(1 - w, 0.0, w * 0.9, h * 0.4) if inside else None,
            )
        else:
            g.add_legend(
                legend,
                "Fidelity",
                loc="center" if inside else "center right",
                bbox_to_anchor=(1 - w, 0, w * 0.9, h) if inside else None,
            )
        g.set_titles("")
        plt.suptitle(title)
        if inside:
            plt.tight_layout()
        else:
            g.tight_layout()
        if show:
            plt.show()
        else:
            return g

    def plot_dist(
        self,
        title: str = "",
        X: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
        variables: Optional[List[str]] = None,
        targets: Union[None, str, Sequence[str]] = None,
        clusters: Union[None, int, np.ndarray] = None,
        scatter: bool = False,
        jitter: float = 0.0,
        col_wrap: int = 4,
        legend_inside: bool = True,
        show: bool = True,
        **kwargs,
    ) -> Optional[sns.FacetGrid]:
        """Plot the distribution of the variables, either as density plots (with clusters) or as scatterplots.

        Args:
            title (str, optional): Title of the plot. Defaults to "".
            X (Optional[np.ndarray], optional): Replacement data matrix (e.g. without normalisation). Defaults to None.
            Y (Optional[np.ndarray], optional): Replacement target matrix (e.g. without normalisation). Defaults to None.
            variables (Optional[List[str]], optional): List of variable names. Defaults to None.
            targets (Union[None, str, Sequence[str]], optional): Target name(s). Defaults to None.
            clusters (Union[None, int, np.ndarray], optional): Number of cluster or vector of cluster labels. Defaults to None.
            scatter (bool, optional): Use scatterplots instead of density plots (clusters are ignored). Defaults to False.
            jitter (float, optional): Add jitter to the scatterplots. Defaults to 0.0.
            col_wrap (int, optional): Maximum number of columns. Defaults to 4.
            legend_inside (bool, optional): Move the legend inside the grid (if there is an empty cell). Defaults to True.
            show (bool, optional): Show the plot. Defaults to True.
            **kwargs: Additional arguments to seaborn.relplot.

        Returns:
            Optional[sns.FacetGrid]: Seaborn FacetGrid if show=False.
        """
        import pandas as pd

        if X is None:
            X = self.get_X(intercept=False)
        if Y is None:
            Y = self.get_Y()
        else:
            Y = np.reshape(Y, (X.shape[0], -1))
        if variables is None:
            variables = [f"Variable {i}" for i in range(self.m - self._intercept)]
        if targets is None:
            targets = (
                [f"Target {i}" for i in range(self.p)] if self.p > 1 else ["Target"]
            )
        elif isinstance(targets, str):
            targets = [targets]

        if not scatter:
            if isinstance(clusters, int):
                clusters, _ = self._cluster_models(clusters)
            elif clusters is None:
                legend_inside = False
            df = pd.concat(
                [
                    pd.DataFrame(dict(var=n, Value=XY[:, i], Cluster=clusters))
                    for v, XY in [(targets, Y), (variables, X)]
                    for i, n in enumerate(v)
                ],
                ignore_index=True,
            )
            if kwargs.setdefault("kind", "kde") == "kde":
                kwargs.setdefault("bw_adjust", 0.75)
                kwargs.setdefault("common_norm", False)
            kwargs.setdefault("palette", "bright")
            kwargs.setdefault("facet_kws", dict(sharex=False, sharey=False))
            g = sns.displot(
                data=df,
                x="Value",
                hue=None if clusters is None else "Cluster",
                col="var",
                col_wrap=col_wrap,
                **kwargs,
            )
            g.set_titles("")
            for ax, n in zip(g.axes.flat, g.col_names):
                ax.set_xlabel(n)
        else:
            Z = self.get_Z(rotate=True)
            if Z.shape[1] == 1:
                Z = np.concatenate((Z, np.zeros_like(Z)), 1)
            elif Z.shape[1] > 2:
                _warn(
                    "Only the first two dimensions in the embedding are plotted",
                    Slisemap.plot_dist,
                )
            if jitter > 0:
                Z = np.random.normal(Z, jitter)
            df = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "var": n,
                            "Value": XY[:, i],
                            "SLISEMAP 1": Z[:, 0],
                            "SLISEMAP 2": Z[:, 1],
                        }
                    )
                    for v, XY in [(targets, Y), (variables, X)]
                    for i, n in enumerate(v)
                ],
                ignore_index=True,
            )
            kwargs.setdefault("palette", "rocket")
            kwargs.setdefault("kind", "scatter")
            g = sns.relplot(
                data=df,
                x="SLISEMAP 1",
                y="SLISEMAP 2",
                hue="Value",
                col="var",
                col_wrap=col_wrap,
                **kwargs,
            )
            g.set_titles("{col_name}")
        plt.suptitle(title)
        cells = len(targets) + len(variables)
        if legend_inside and col_wrap < cells and cells % col_wrap != 0:
            w = 1 / col_wrap
            h = 1 / ((cells - 1) // col_wrap + 1)
            sns.move_legend(
                g,
                "center",
                bbox_to_anchor=(1 - w, h * 0.1, w * 0.9, h * 0.9),
                frameon=False,
            )
            plt.tight_layout()
        else:
            g.tight_layout()
        if show:
            plt.show()
        else:
            return g
