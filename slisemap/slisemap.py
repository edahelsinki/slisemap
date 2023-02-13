"""
This module contains the `Slisemap` class.
"""

from copy import copy
from os import PathLike
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.figure import Figure
from sklearn.cluster import KMeans

from slisemap.escape import escape_neighbourhood
from slisemap.local_models import (
    identify_local_model,
    LinearRegression,
    ALocalModel,
    local_predict,
)
from slisemap.loss import make_loss, make_marginal_loss, softmax_row_kernel
from slisemap.plot import (
    _expand_variable_names,
    legend_inside_facet,
    plot_barmodels,
    plot_density_facet,
    plot_embedding,
    plot_embedding_facet,
    plot_matrix,
    plot_position_legend,
)
from slisemap.utils import (
    LBFGS,
    CheckConvergence,
    Metadata,
    PCA_rotation,
    _assert,
    _deprecated,
    _warn,
    global_model,
    to_tensor,
    tonp,
)


class Slisemap:
    """__Slisemap__: Combine local explanations with dimensionality reduction.

    This class contains the data and the parameters needed for finding a Slisemap solution.
    It also contains the solution (remember to [optimise()][slisemap.slisemap.Slisemap.optimise] first) in the form of an embedding matrix, see [get_Z()][slisemap.slisemap.Slisemap.get_Z], and a matrix of coefficients for the local model, see [get_B()][slisemap.slisemap.Slisemap.get_B].
    Other methods of note are the various plotting methods, the [save()][slisemap.slisemap.Slisemap.save] method, and the [fit_new()][slisemap.slisemap.Slisemap.fit_new] method.

    The use of some regularisation is highly recommended. Slisemap comes with built-in lasso/L1 and ridge/L2 regularisation (if these are used it is also a good idea to normalise the data in advance).

    Attributes:
        n: The number of data items (`X.shape[0]`).
        m: The number of variables (`X.shape[1]`).
        o: The number of targets (`Y.shape[1]`).
        d: The number of embedding dimensions (`Z.shape[1]`).
        q: The number of coefficients (`B.shape[1]`).
        intercept: Has an intercept term been added to `X`.
        radius: The radius of the embedding.
        lasso: Lasso regularisation coefficient.
        ridge: Ridge regularisation coefficient.
        z_norm: Z normalisation regularisation coefficient.
        local_model: Local model prediction function (see [slisemap.local_models][]).
        local_loss: Local model loss function (see [slisemap.local_models][]).
        coefficients: The number of local model coefficients.
        distance: Distance function.
        kernel: Kernel function.
        metadata: Dictionary of arbitrary metadata such as variable names.
        jit: Just-In-Time compile the loss function for increased performance (see `torch.jit.trace` for caveats).
        random_state: Set an explicit seed for the random number generator (i.e. `torch.manual_seed`).
        metadata: A dictionary for storing variable names and other metadata (see [slisemap.utils.Metadata][]).
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
        "_random_state",
        "_rs0",
        "metadata",
    )

    def __init__(
        self,
        X: Union[np.ndarray, torch.Tensor, Sequence[Any]],
        y: Union[np.ndarray, torch.Tensor, Sequence[Any]],
        radius: float = 3.5,
        d: int = 2,
        lasso: Optional[float] = None,
        ridge: Optional[float] = None,
        z_norm: float = 0.01,
        intercept: bool = True,
        local_model: Union[
            ALocalModel, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = LinearRegression,
        local_loss: Optional[
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        coefficients: Union[
            None, int, Callable[[torch.Tensor, torch.Tensor], int]
        ] = None,
        distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.cdist,
        kernel: Callable[[torch.Tensor], torch.Tensor] = softmax_row_kernel,
        B0: Union[None, np.ndarray, torch.Tensor, Sequence[Any]] = None,
        Z0: Union[None, np.ndarray, torch.Tensor, Sequence[Any]] = None,
        jit: bool = True,
        random_state: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        cuda: Optional[bool] = None,
    ):
        """Create a Slisemap object.

        Args:
            X: Data matrix.
            y: Target vector or matrix.
            radius: The radius of the embedding Z. Defaults to 3.5.
            d: The number of embedding dimensions. Defaults to 2.
            lasso: Lasso regularisation coefficient. Defaults to 0.0.
            ridge: Ridge regularisation coefficient. Defaults to 0.0.
            z_norm: Z normalisation regularisation coefficient. Defaults to 0.01.
            intercept: Should an intercept term be added to `X`. Defaults to True.
            local_model: Local model prediction function (see [slisemap.local_models.identify_local_model][]). Defaults to [LinearRegression][slisemap.local_models.LinearRegression].
            local_loss: Local model loss function (see [slisemap.local_models.identify_local_model][]). Defaults to None.
            coefficients: The number of local model coefficients (see [slisemap.local_models.identify_local_model][]). Defaults to None.
            distance: Distance function. Defaults to `torch.cdist` (Euclidean distance).
            kernel: Kernel function. Defaults to [softmax_row_kernel][slisemap.loss.softmax_row_kernel].
            B0: Initial value for B (random if None). Defaults to None.
            Z0: Initial value for Z (PCA if None). Defaults to None.
            jit: Just-In-Time compile the loss function for increased performance (see `torch.jit.trace` for caveats). Defaults to True.
            random_state: Set an explicit seed for the random number generator (i.e. `torch.manual_seed`). Defaults to None.
            dtype: Floating type. Defaults to `torch.float32`.
            device: Torch device (see `cuda` if None). Defaults to None.
            cuda: Use cuda if available. Defaults to True, if the data is large enough.
        """
        for s in Slisemap.__slots__:
            # Initialise all attributes (to avoid attribute errors)
            setattr(self, s, None)
        if lasso is None and ridge is None:
            _warn(
                "Consider using regularisation!\n"
                + "Regularisation is important for handling small neighbourhoods, and also makes the local models more local. "
                + "Lasso (l1) and ridge (l2) regularisation is built-in, via the parameters `lasso` and `ridge`. "
                + "Set `lasso=0` to disable this warning (if no regularisation is really desired).",
                Slisemap,
            )
        local_model, local_loss, coefficients = identify_local_model(
            local_model, local_loss, coefficients
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
        self._rs0 = random_state
        self.metadata = Metadata(self)

        if device is None:
            if cuda is None and isinstance(X, torch.Tensor):
                device = X.device
            elif cuda == True:
                device = torch.device("cuda")
        tensorargs = {"device": device, "dtype": dtype}

        self._X, X_rows, X_columns = to_tensor(X, **tensorargs)
        if intercept:
            self._X = torch.cat((self._X, torch.ones_like(self._X[:, :1])), 1)
        n, m = self._X.shape
        self.metadata.set_variables(X_columns, intercept)

        self._Y, Y_rows, Y_columns = to_tensor(y, **tensorargs)
        self.metadata.set_targets(Y_columns)
        _assert(
            self._Y.shape[0] == n,
            f"The length of y must match X: {self._Y.shape[0]} != {n}",
            Slisemap,
        )
        if len(self._Y.shape) == 1:
            self._Y = self._Y[:, None]

        if device is None and cuda is None:
            if self.n**2 * self.m * self.o > 1_000_000 and torch.cuda.is_available():
                tensorargs["device"] = torch.device("cuda")
                self._X = self._X.cuda()
                self._Y = self._Y.cuda()

        self.random_state = random_state

        if Z0 is None:
            self._Z0 = self._X @ PCA_rotation(self._X, d)
            if self._Z0.shape[1] < d:
                _warn(
                    "The number of embedding dimensions is larger than the number of data dimensions",
                    Slisemap,
                )
                Z0fill = torch.zeros(size=[n, d - self._Z0.shape[1]], **tensorargs)
                self._Z0 = torch.cat((self._Z0, Z0fill), 1)
            Z_rows = None
        else:
            self._Z0, Z_rows, Z_columns = to_tensor(Z0, **tensorargs)
            self.metadata.set_dimensions(Z_columns)
            _assert(
                self._Z0.shape == (n, d),
                f"Z0 has the wrong shape: {self._Z0.shape} != ({n}, {d})",
                Slisemap,
            )
        if radius > 0:
            norm = 1 / (torch.sqrt(torch.sum(self._Z0**2) / self._Z0.shape[0]) + 1e-8)
            self._Z0 = self._Z0 * norm
        self._Z = self._Z0.detach().clone()

        if B0 is None:
            B0 = global_model(
                X=self._X,
                Y=self._Y,
                local_model=self.local_model,
                local_loss=self.local_loss,
                coefficients=coefficients(self._X, self._Y),
                lasso=self.lasso,
                ridge=self.ridge,
            )
            if not torch.all(torch.isfinite(B0)):
                _warn(
                    "Optimising a global model as initialisation resulted in non-finite values. Consider using stronger regularisation (increase `lasso` or `ridge`).",
                    Slisemap,
                )
                B0 = torch.zeros_like(B0)
            self._B0 = B0.expand((n, B0.shape[1]))
            B_rows = None
        else:
            self._B0, B_rows, B_columns = to_tensor(B0, **tensorargs)
            self.metadata.set_coefficients(B_columns)
            _assert(
                self._B0.shape == (n, B0.shape[1]),
                f"B0 has the wrong shape: {self._B0.shape} != ({n}, {coefficients(self._X, self._Y)})",
                Slisemap,
            )
        self._B = self._B0.detach().clone()

        self.metadata.set_rows(X_rows, Y_rows, B_rows, Z_rows)

    @property
    def n(self) -> int:
        # The number of data items
        return self._X.shape[0]

    @property
    def m(self) -> int:
        # The number of variables (including potential intercept)
        return self._X.shape[1]

    @property
    def p(self) -> int:
        # The number of target variables (i.e. the number of classes)
        # Deprecated (1.2), use `o` instead!
        _deprecated(Slisemap.p, Slisemap.o)
        return self._Y.shape[-1]

    @property
    def o(self) -> int:
        # The number of target variables (i.e. the number of classes)
        return self._Y.shape[-1]

    @property
    def d(self) -> int:
        # The number of embedding dimensions
        return self._Z.shape[1]

    @d.setter
    def d(self, value: int):
        _assert(
            value > 0, "The number of embedding dimensions must be positive", Slisemap.d
        )
        if self.d != value:
            self._loss = None  # invalidate cached loss function
            self._Z = self._Z.detach()
            if self.d > value:
                self._Z = self._Z @ PCA_rotation(self._Z, value)
            else:
                zn = [self._Z, torch.zeros((self.n, value - self.d), **self.tensorargs)]
                self._Z = torch.concat(zn, 1)

    @property
    def q(self) -> int:
        # The number of local model coefficients
        return self._B.shape[1]

    @property
    def coefficients(self) -> int:
        # The number of local model coefficients
        # Deprecated (1.2), use `q` instead!
        _deprecated(Slisemap.coefficients, Slisemap.q)
        return self.q

    @property
    def intercept(self) -> bool:
        # Is an intercept column added to the data?
        return self._intercept

    @property
    def radius(self) -> float:
        # The radius of the embedding
        return self._radius

    @radius.setter
    def radius(self, value: float):
        if self._radius != value:
            _assert(value >= 0, "radius must not be negative", Slisemap.radius)
            self._radius = value
            self._loss = None  # invalidate cached loss function

    @property
    def lasso(self) -> float:
        # Lasso regularisation strength
        return self._lasso

    @lasso.setter
    def lasso(self, value: float):
        if self._lasso != value:
            _assert(value >= 0, "lasso must not be negative", Slisemap.lasso)
            self._lasso = value
            self._loss = None  # invalidate cached loss function

    @property
    def ridge(self) -> float:
        # Ridge regularisation strength
        return self._ridge

    @ridge.setter
    def ridge(self, value: float):
        if self._ridge != value:
            _assert(value >= 0, "ridge must not be negative", Slisemap.ridge)
            self._ridge = value
            self._loss = None  # invalidate cached loss function

    @property
    def z_norm(self) -> float:
        # Z normalisation regularisation strength
        return self._z_norm

    @z_norm.setter
    def z_norm(self, value: float):
        if self._z_norm != value:
            _assert(value >= 0, "z_norm must not be negative", Slisemap.z_norm)
            self._z_norm = value
            self._loss = None  # invalidate cached loss function

    @property
    def local_model(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        # Local model prediction function. Takes in X[n, m] and B[n, q], and returns Ytilde[n, n, o]
        return self._local_model

    @local_model.setter
    def local_model(
        self,
        value: Union[ALocalModel, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ):
        value, _, _ = identify_local_model(value)
        if self._local_model != value:
            self._local_model = value
            self._loss = None  # invalidate cached loss function

    @property
    def local_loss(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        # Local model loss function. Takes in Ytilde[n, n, o], Y[n, o], and B[n, q], and returns L[n, n]
        return self._local_loss

    @local_loss.setter
    def local_loss(
        self,
        value: Callable[
            [torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor
        ],
    ):
        if self._local_loss != value:
            _assert(callable(value), "local_loss must be callable", Slisemap.local_loss)
            self._local_loss = value
            self._loss = None  # invalidate cached loss function

    @property
    def distance(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        # Distance function. Takes in Z[n1, d] and Z[n2, d], and returns D[n1, n2]
        return self._distance

    @distance.setter
    def distance(self, value: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        if self._distance != value:
            _assert(callable(value), "distance must be callable", Slisemap.distance)
            self._distance = value
            self._loss = None  # invalidate cached loss function

    @property
    def kernel(self) -> Callable[[torch.Tensor], torch.Tensor]:
        # Kernel function. Takes in D[n, n] and returns W[n, n]
        return self._kernel

    @kernel.setter
    def kernel(self, value: Callable[[torch.Tensor], torch.Tensor]):
        if self._kernel != value:
            _assert(callable(value), "kernel must be callable", Slisemap.kernel)
            self._kernel = value
            self._loss = None  # invalidate cached loss function

    @property
    def jit(self) -> bool:
        # Just-In-Time compile the loss function?
        return self._jit

    @jit.setter
    def jit(self, value: bool):
        if self._jit != value:
            self._jit = value
            self._loss = None  # invalidate cached loss function

    def random_state(self, value: Optional[int]):
        # Set the seed for the random number generator specific for this object (None reverts to the global `torch` PRNG)
        if value is None:
            self._random_state = None
        else:
            if self._X.device.type == "cpu":
                self._random_state = torch.random.manual_seed(value)
            elif self._X.device.type == "cuda":
                gen = torch.cuda.default_generators[self._X.device.index]
                self._random_state = gen.manual_seed(value)
            else:
                _warn(
                    Slisemap.random_state,
                    "Unknown device, setting the global seed insted",
                )
                torch.random.manual_seed(value)
                self._random_state = None

    random_state = property(fset=random_state, doc=random_state.__doc__)

    @property
    def X(self) -> torch.Tensor:
        # Get the data matrix as a `torch.Tensor`
        # Deprecated (1.2), use `get_X(numpy=False)` instead!
        _deprecated(Slisemap.X, Slisemap.get_X)
        return self._X

    @property
    def Y(self) -> torch.Tensor:
        # Target matrix as a `torch.Tensor`.
        # Deprecated (1.2), use `get_Y(numpy=False)` instead!
        _deprecated(Slisemap.Y, Slisemap.get_Y)
        return self._Y

    @property
    def Z(self) -> torch.Tensor:
        # Normalised embedding matrix as a `torch.Tensor`.
        # Deprecated (1.2), use `get_Z(numpy=False)` instead!
        _deprecated(Slisemap.Z, Slisemap.get_Z)
        return self._Z

    @property
    def B(self) -> torch.Tensor:
        # Coefficient matrix for the local models as a `torch.Tensor`.
        # Deprecated (1.2), use `get_B(numpy=False)` instead!
        _deprecated(Slisemap.B, Slisemap.get_B)
        return self._B

    @property
    def tensorargs(self) -> Dict[str, Any]:
        # When creating a new `torch.Tensor` add these keyword arguments to match the `dtype` and `device` of this Slisemap object.
        return dict(device=self._X.device, dtype=self._X.dtype)

    def cuda(self, **kwargs):
        """Move the tensors to CUDA memory (and run the calculations there).
        Note that this resets the random state.

        Keyword Args:
            **kwargs: Optional arguments to `torch.Tensor.cuda`
        """
        X = self._X.cuda(**kwargs)
        self._X = X
        self._Y = self._Y.cuda(**kwargs)
        self._Z = self._Z.detach().cuda(**kwargs)
        self._B = self._B.detach().cuda(**kwargs)
        self.random_state = self._rs0
        self._loss = None  # invalidate cached loss function

    def cpu(self, **kwargs):
        """Move the tensors to CPU memory (and run the calculations there).
        Note that this resets the random state.

        Keyword Args:
            **kwargs: Optional arguments to `torch.Tensor.cpu`
        """
        X = self._X.cpu(**kwargs)
        self._X = X
        self._Y = self._Y.cpu(**kwargs)
        self._Z = self._Z.detach().cpu(**kwargs)
        self._B = self._B.detach().cpu(**kwargs)
        self.random_state = self._rs0
        self._loss = None  # invalidate cached loss function

    def _get_loss_fn(
        self, individual: bool = False
    ) -> Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ]:
        """Returns the Slisemap loss function.
        This function JITs and caches the loss function for efficiency.

        Args:
            individual: Make a loss function for individual losses. Defaults to False.

        Returns:
            Loss function: `f(X, Y, B, Z) -> loss`.
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

    def get_loss_fn(
        self, individual: bool = False
    ) -> Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ]:
        """Returns the Slisemap loss function.
        This function JITs and caches the loss function for efficiency.

        Args:
            individual: Make a loss function for individual losses. Defaults to False.

        Returns:
            Loss function: `f(X, Y, B, Z) -> loss`.

        Deprecated:
            1.2: The functions has been renamed, use `_gets_loss_fn` instead!
        """
        _deprecated(Slisemap.get_loss_fn, Slisemap._get_loss_fn)
        return self._get_loss_fn(individual)

    def _as_new_X(
        self, X: Union[None, np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        if X is None:
            return self._X
        tensorargs = self.tensorargs
        X = torch.atleast_2d(to_tensor(X, **tensorargs)[0])
        if self._intercept and X.shape[1] == self.m - 1:
            X = torch.cat([X, torch.ones((X.shape[0], 1), **tensorargs)], 1)
        _assert(
            X.shape[1] == self.m,
            f"X has the wrong shape {X.shape} != {(X.shape[0], self.m)}",
            Slisemap._as_new_X,
        )
        return X

    def _as_new_Y(
        self,
        Y: Union[None, float, np.ndarray, torch.Tensor] = None,
        n: Optional[int] = None,
    ) -> torch.Tensor:
        if Y is None:
            return self._Y
        Y = to_tensor(Y, **self.tensorargs)[0]
        if len(Y.shape) < 2:
            Y = torch.reshape(Y, (-1, self.o))
        if n is None:
            n = Y.shape[0]
        _assert(
            Y.shape == (n, self.o),
            f"Y has the wrong shape {Y.shape} != {(n, self.o)}",
            Slisemap._as_new_Y,
        )
        return Y

    def get_Z(
        self, scale: bool = True, rotate: bool = False, numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """Get the Z matrix

        Args:
            scale: Scale the returned `Z` to match self.radius. Defaults to True.
            rotate: Rotate the returned `Z` so that the first dimension is the major axis. Defaults to False.
            numpy: Return the matrix as a numpy (True) or pytorch (False) matrix. Defaults to True.

        Returns:
           : The `Z` matrix.
        """
        self._normalise()
        if scale and self.radius > 0:
            Z = self._Z * self.radius
        else:
            Z = self._Z
        if rotate:
            Z = Z @ PCA_rotation(Z, center=False)
        return tonp(Z) if numpy else Z

    def get_B(self, numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Get the B matrix

        Args:
            numpy: Return the matrix as a numpy (True) or pytorch (False) matrix. Defaults to True.

        Returns:
           : The `B` matrix.
        """
        return tonp(self._B) if numpy else self._B

    def get_D(self, numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Get the embedding distance matrix

        Args:
            numpy: Return the matrix as a numpy (True) or pytorch (False) matrix. Defaults to True.

        Returns:
           : The `D` matrix.
        """
        Z = self.get_Z(rotate=False, scale=True, numpy=False)
        D = self._distance(Z, Z)
        return tonp(D) if numpy else D

    def get_L(
        self,
        X: Union[None, np.ndarray, torch.Tensor] = None,
        Y: Union[None, float, np.ndarray, torch.Tensor] = None,
        numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Get the loss matrix: [B.shape[0], X.shape[0]].

        Args:
            X: Optional replacement for the training X. Defaults to None.
            Y: Optional replacement for the training Y. Defaults to None.
            numpy: Return the matrix as a numpy (True) or pytorch (False) matrix. Defaults to True.

        Returns:
           : The `L` matrix.
        """
        X = self._as_new_X(X)
        Y = self._as_new_Y(Y, X.shape[0])
        L = self.local_loss(self.local_model(X, self._B), Y, self._B)
        return tonp(L) if numpy else L

    def get_W(self, numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Get the weight matrix.

        Args:
            numpy: Return the matrix as a numpy.ndarray instead of a torch.Tensor. Defaults to True.

        Returns:
           : The `W` matrix.
        """
        W = self.kernel(self.get_D(numpy=False))
        return tonp(W) if numpy else W

    def get_X(
        self, numpy: bool = True, intercept: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """Get the data matrix.

        Args:
            numpy: Return the matrix as a numpy.ndarray instead of a torch.Tensor. Defaults to True.
            intercept: Include the intercept column (if `self.intercept == True`). Defaults to True.

        Returns:
           : The `X` matrix.
        """
        X = self._X if intercept or not self._intercept else self._X[:, :-1]
        return tonp(X) if numpy else X

    def get_Y(
        self, numpy: bool = True, ravel: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """Get the target matrix.

        Args:
            numpy: Return the matrix as a numpy.ndarray instead of a torch.Tensor. Defaults to True.
            ravel: Remove the second dimension if it is singular (i.e. turn it into a vector). Defaults to False.

        Returns:
           : The `Y` matrix.
        """
        Y = self._Y.ravel() if ravel else self._Y
        return tonp(Y) if numpy else Y

    def value(
        self, individual: bool = False, numpy: bool = True
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """Calculate the loss value.

        Args:
            individual: Give loss individual loss values for the data points. Defaults to False.
            numpy: Return the loss as a numpy.ndarray or float instead of a torch.Tensor. Defaults to True.

        Returns:
            The loss value(s).
        """
        loss = self._get_loss_fn(individual)
        loss = loss(X=self._X, Y=self._Y, B=self._B, Z=self._Z)
        if individual:
            return tonp(loss) if numpy else loss
        else:
            return loss.cpu().item() if numpy else loss

    def entropy(
        self, aggregate: bool = True, numpy: bool = True
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """Compute row-wise entropy of the `W` matrix induced by `Z`.

        Args:
            aggregate: Aggregate the row-wise entropies into one scalar. Defaults to True.
            numpy: Return a `numpy.ndarray` or `float` instead of a `torch.Tensor`. Defaults to True.

        Returns:
            The entropy.

        Deprecated:
            1.4: Use [slisemap.metrics.entropy][slisemap.metrics.entropy] instead.
        """
        _deprecated(Slisemap.entropy, "slisemap.metrics.entropy")
        from slisemap.metrics import entropy

        return entropy(self, aggregate, numpy)

    def lbfgs(
        self,
        max_iter: int = 500,
        verbose: bool = False,
        *,
        only_B: bool = False,
        **kwargs,
    ) -> float:

        """Optimise Slisemap using LBFGS.

        Args:
            max_iter: Maximum number of LBFGS iterations. Defaults to 500.
            verbose: Print status messages. Defaults to False.
        Keyword Args:
            only_B: Only optimise B. Defaults to False.
            **kwargs: Optional keyword arguments to LBFGS.

        Returns:
            The loss value.
        """
        Z = self._Z.detach().clone().requires_grad_(True)
        B = self._B.detach().clone().requires_grad_(True)

        loss_ = self._get_loss_fn()
        loss_fn = lambda: loss_(self._X, self._Y, B, Z)
        pre_loss = loss_fn().cpu().detach().item()
        LBFGS(
            loss_fn,
            [B] if only_B else [Z, B],
            max_iter=max_iter,
            verbose=verbose,
            **kwargs,
        )
        post_loss = loss_fn().cpu().detach().item()

        if np.isnan(post_loss):
            _warn(
                "An LBFGS optimisation resulted in `nan` (try strengthening the regularisation or reducing the radius)",
                Slisemap.lbfgs,
            )
            # Some datasets (with logistic local models) are initially numerically unstable.
            # Just running LBFGS for one iteration seems to avoid those issues.
            Z = self._Z.detach().requires_grad_(True)
            B = self._B.detach().requires_grad_(True)
            LBFGS(
                loss_fn,
                [B] if only_B else [Z, B],
                max_iter=1,
                verbose=verbose,
                **kwargs,
            )
            post_loss = loss_fn().cpu().detach().item()
        if post_loss < pre_loss:
            self._Z = Z.detach()
            self._B = B.detach()
            self._normalise()
            return post_loss
        else:
            if verbose:
                print("Slisemap.lbfgs: No improvement found")
            return pre_loss

    def escape(
        self,
        force_move: bool = True,
        escape_fn: Callable = escape_neighbourhood,
        noise: float = 0.0,
    ):
        """Try to escape a local optimum by moving the items (embedding and local model) to the neighbourhoods best suited for them.
        This is done by finding another item (in the optimal neighbourhood) and copying its values for Z and B.

        Args:
            force_move: Do not allow the items to pair with themselves. Defaults to True.
            escape_fn: Escape function (see [slisemap.escape][]). Defaults to [escape_neighbourhood][slisemap.escape.escape_neighbourhood].
            noise: Scale of the noise added to the embedding matrix if it looses rank after an escape (recommended for gradient based optimisers). Defaults to 0.0.
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
        if noise > 0.0:
            rank = torch.linalg.matrix_rank(self._Z - torch.mean(self._Z, 0, True))
            if rank.item() < min(*self._Z.shape):
                self._Z = torch.normal(self._Z, noise, generator=self._random_state)
        self._normalise()

    def _normalise(self):
        """Normalise Z."""
        if self.radius > 0:
            scale = torch.sqrt(torch.sum(self._Z**2) / self.n)
            if not torch.allclose(scale, torch.ones_like(scale)):
                self._Z *= 1 / (scale + 1e-8)

    def optimise(
        self,
        patience: int = 2,
        max_escapes: int = 100,
        max_iter: int = 500,
        escape_fn: Callable = escape_neighbourhood,
        verbose: Literal[0, 1, 2] = 0,
        noise: float = 1e-4,
        only_B: bool = False,
        **kwargs,
    ) -> float:
        """Optimise Slisemap by alternating between [self.lbfgs()][slisemap.slisemap.Slisemap.lbfgs] and [self.escape()][slisemap.slisemap.Slisemap.escape] until convergence.

        Args:
            patience: Number of escapes without improvement before stopping. Defaults to 2.
            max_escapes: Maximum number of escapes. Defaults to 100.
            max_iter: Maximum number of LBFGS iterations per round. Defaults to 500.
            escape_fn: Escape function (see [slisemap.escape][]). Defaults to [escape_neighbourhood][slisemap.escape.escape_neighbourhood].
            verbose: Print status messages (0: no, 1: some, 2: all). Defaults to 0.
            noise: Scale of the noise added to the embedding matrix if it looses rank after an escape. Defaults to 1e-4.
            only_B: Only optimise the local models, not the embedding. Defaults to False.
        Keyword Args:
            **kwargs: Optional keyword arguments to Slisemap.lbfgs.

        Returns:
            The loss value.
        """
        loss = np.repeat(np.inf, 2)
        loss[0] = self.lbfgs(
            max_iter=max_iter,
            only_B=True,
            increase_tolerance=not only_B,
            verbose=verbose > 1,
            **kwargs,
        )
        if verbose:
            i = 0
            print(f"Slisemap.optimise LBFGS  {i:2d}: {loss[0]:.2f}")
        if only_B:
            return loss[0]
        cc = CheckConvergence(patience, max_escapes)
        while not cc.has_converged(loss, self.copy, verbose=verbose > 1):
            self.escape(escape_fn=escape_fn, noise=noise)
            loss[1] = self.value()
            if verbose:
                print(f"Slisemap.optimise Escape {i:2d}: {loss[1]:.2f}")
            loss[0] = self.lbfgs(
                max_iter=max_iter,
                increase_tolerance=True,
                verbose=verbose > 1,
                **kwargs,
            )
            if verbose:
                i += 1
                print(f"Slisemap.optimise LBFGS  {i:2d}: {loss[0]:.2f}")
        self._Z = cc.optimal._Z
        self._B = cc.optimal._B
        loss = self.lbfgs(
            max_iter=max_iter * 2,
            increase_tolerance=False,
            verbose=verbose > 1,
            **kwargs,
        )
        if verbose:
            print(f"Slisemap.optimise Final    : {loss:.2f}")
        return loss

    optimize = optimise

    def fit_new(
        self,
        Xnew: Union[np.ndarray, torch.Tensor],
        ynew: Union[float, np.ndarray, torch.Tensor],
        optimise: bool = True,
        between: bool = True,
        escape_fn: Callable = escape_neighbourhood,
        loss: bool = False,
        verbose: bool = False,
        numpy: bool = True,
        **kwargs,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    ]:
        """Generate embedding(s) and model(s) for new data item(s).

        This works as follows:
            1. Find good initial embedding(s) and local model(s) using the escape_fn.
            2. Optionally finetune the embedding(s) and model(s) using LBFG.

        Args:
            Xnew: New data point(s).
            ynew: New target(s).
            optimise: Should the embedding and model be optimised (after finding the neighbourhood). Defaults to True.
            between: If `optimise=True`, should the new points affect each other? Defaults to True.
            escape_fn: Escape function (see [slisemap.escape][]). Defaults to [escape_neighbourhood][slisemap.escape.escape_neighbourhood].
            loss: Return a vector of individual losses for the new items. Defaults to False.
            verbose: Print status messages. Defaults to False.
            numpy: Return the results as numpy (True) or pytorch (False) matrices. Defaults to True.
        Keyword Args:
            **kwargs: Optional keyword arguments to LBFGS.

        Returns:
            Bnew: Local model coefficients for the new data.
            Znew: Embedding(s) for the new data.
            loss: Individual losses if `loss=True`.
        """
        Xnew = self._as_new_X(Xnew)
        n = Xnew.shape[0]
        ynew = self._as_new_Y(ynew, n)
        if verbose:
            print("Escaping the new data")
        Bnew, Znew = escape_fn(
            X=Xnew,
            Y=ynew,
            B=self._B,
            Z=self._Z,
            local_model=self.local_model,
            local_loss=self.local_loss,
            distance=self.distance,
            kernel=self.kernel,
            radius=self.radius,
            force_move=False,
            Xold=self._X,
            Yold=self._Y,
            jit=self.jit,
        )
        if verbose:
            Zrad = torch.sqrt(torch.sum(Znew**2) / n).cpu().detach().item()
            print("  radius(Z_new) =", Zrad)

        if optimise:
            if verbose:
                print("Optimising the new data")
            lf, set_new = make_marginal_loss(
                X=self._X,
                Y=self._Y,
                B=self._B,
                Z=self._Z,
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
            if verbose:
                Zrad = torch.sqrt(torch.sum(Znew**2) / n).cpu().detach().item()
                print("  radius(Z_new) =", Zrad)
        if self.radius > 0:
            if verbose:
                print("Normalising the solution")
            norm = self.radius / (torch.sqrt(torch.sum(self._Z**2) / self.n) + 1e-8)
            Zout = Znew * norm
            if verbose:
                Zrad = torch.sqrt(torch.sum(Zout**2) / n).cpu().detach().item()
                print("  radius(Z_new) =", Zrad)
        else:
            Zout = Znew
        if loss:
            if verbose:
                print("Calculating individual losses")
            lf = self._get_loss_fn(individual=True)
            if between:
                loss = lf(
                    X=torch.cat((self._X, Xnew), 0),
                    Y=torch.cat((self._Y, ynew), 0),
                    B=torch.cat((self._B, Bnew), 0),
                    Z=torch.cat((self._Z, Znew), 0),
                )[self.n :]
            else:
                if self._jit:
                    lf = torch.jit.trace(lf, (Xnew[:1], ynew[:1], Bnew[:1], Znew[:1]))
                loss = torch.zeros(n, **self.tensorargs)
                for j in range(n):
                    loss[j] = lf(
                        X=torch.cat((self._X, Xnew[None, j]), 0),
                        Y=torch.cat((self._Y, ynew[None, j]), 0),
                        B=torch.cat((self._B, Bnew[None, j]), 0),
                        Z=torch.cat((self._Z, Znew[None, j]), 0),
                    )[-1]
            if verbose:
                print("  mean(loss) =", loss.detach().mean().cpu().item())
            return (tonp(Bnew), tonp(Zout), tonp(loss)) if numpy else (Bnew, Zout, loss)
        else:
            return (tonp(Bnew), tonp(Zout)) if numpy else (Bnew, Zout)

    def predict(
        self,
        X: Union[None, np.ndarray, torch.Tensor] = None,
        B: Union[None, np.ndarray, torch.Tensor] = None,
        Z: Union[None, np.ndarray, torch.Tensor] = None,
        numpy: bool = True,
        *_,
        Xnew=None,
        Znew=None,
        **kwargs,
    ) -> np.ndarray:
        """Predict new outcomes when the data and embedding or local model is known.

        If the local models are known they are used, otherwise the embeddings are used to find new local models.

        Args:
            X: Data matrix (set to None to use the training data). Defaults to None.
            B: Coefficient matrix. Defaults to None.
            Z: Embedding matrix. Defaults to None.
            numpy: Return the result as a numpy (True) or a pytorch (False) matrix. Defaults to True.
        Keyword Args:
            **kwargs: Optional keyword arguments to LBFGS.

        Returns:
            Prediction matrix.

        Deprecated:
            1.4: Renamed Xnew, Znew to X, Z and added optional B.
        """
        if Xnew is not None:
            X = Xnew
            _deprecated(
                "Parameter 'Xnew' in Slisemap.predict",
                "parameter 'X' in Slisemap.predict",
            )
        if Znew is not None:
            Z = Znew
            _deprecated(
                "Parameter 'Znew' in Slisemap.predict",
                "parameter 'Z' in Slisemap.predict",
            )
        if X is None and B is None and Z is None:
            X = self._X
            B = self._B
        _assert(
            B is not None or Z is not None,
            "Either B or Z must be given",
            Slisemap.predict,
        )
        X = self._as_new_X(X)
        if B is not None:
            B = torch.atleast_2d(to_tensor(B, **self.tensorargs)[0])
            if B.shape[0] == 1:
                yhat = self.local_model(X, B)[0, ...]
            else:
                _assert(
                    X.shape[0] == B.shape[0],
                    f"X and B must have the same number of rows: {X.shape[0]} != {B.shape[0]}",
                    Slisemap.predict,
                )
                yhat = local_predict(X, B, self.local_model)
        else:
            Z = torch.atleast_2d(to_tensor(Z, **self.tensorargs)[0])
            _assert(
                Z.shape == (X.shape[0], self.d),
                f"Z has the wrong shape {Z.shape} != {(X.shape[0], self.d)}",
                Slisemap.predict,
            )
            D = self._distance(Z, self._Z)
            W = self.kernel(D)
            B = self._B[torch.argmin(D, 1)].clone().requires_grad_(True)
            yhat = lambda: (
                torch.sum(W * self.local_loss(self.local_model(self._X, B), self._Y, B))
                + self.lasso * torch.sum(torch.abs(B))
                + self.ridge * torch.sum(B**2)
            )
            LBFGS(yhat, [B], **kwargs)
            yhat = local_predict(X, B, self.local_model)
        return tonp(yhat) if numpy else yhat

    def copy(self) -> "Slisemap":
        """Make a copy of this Slisemap that references as much of the same torch-data as possible.

        Returns:
            An almost shallow copy of this Slisemap object.
        """
        other = copy(self)  # Shallow copy!
        # Deep copy these:
        other._B = other._B.clone().detach()
        other._Z = other._Z.clone().detach()
        return other

    def restore(self):
        """Reset B and Z (and random_state) to their initial values B0 and Z0 (and _rs0)."""
        self._Z = self._Z0.clone().detach()
        self._B = self._B0.clone().detach()
        self.random_state = self._rs0

    def save(
        self, f: Union[str, PathLike, BinaryIO], any_extension: bool = False, **kwargs
    ):
        """Save the Slisemap object to a file.

        This method uses `torch.save` (which uses `pickle` for the non-pytorch properties).
        This means that lambda-functions are not supported (unless a custom pickle module is used, see `torch.save`).

        Note that the random state is not saved, only the initial seed (if set).

        The default file extension is ".sm".

        Args:
            f: Either a Path-like object or a (writable) File-like object.
            any_extension: Do not check the file extension. Defaults to False.
        Keyword Args:
            **kwargs: Parameters forwarded to `torch.save`.
        """
        if not any_extension and isinstance(f, (str, PathLike)):
            if not str(f).endswith(".sm"):
                _warn(
                    "When saving Slisemap objects, consider using the '.sm' extension for consistency.",
                    Slisemap.save,
                )
        loss = self._loss
        prng = self._random_state
        try:
            self.metadata.root = None
            self._B = self._B.detach()
            self._Z = self._Z.detach()
            self._loss = None
            self._random_state = None
            torch.save(self, f, **kwargs)
        finally:
            self.metadata.root = self
            self._loss = loss
            self._random_state = prng

    @classmethod
    def load(
        cls,
        f: Union[str, PathLike, BinaryIO],
        device: Union[None, str, torch.device] = None,
        map_location: Optional[Any] = None,
        **kwargs,
    ) -> "Slisemap":
        """Load a Slisemap object from a file.

        This function uses `torch.load`, so the tensors are restored to their previous devices.
        Use `device="cpu"` to avoid assuming that the same device exists.
        This is useful if the Slisemap object has been trained on a GPU, but the current computer lacks a GPU.

        Note that this is a classmethod, use it with: `Slisemap.load(...)`.

        Args:
            f: Either a Path-like object or a (readable) File-like object.
            device: Device to load the tensors to (or the original if None). Defaults to None.
            map_location: The same as `device` (this is the name used by `torch.load`). Defaults to None.
        Keyword Args:
            **kwargs: Parameters forwarded to `torch.load`.

        Returns:
            The loaded Slisemap object.
        """
        if device is None:
            device = map_location
        sm = torch.load(f, map_location=device, **kwargs)
        sm.random_state = sm._rs0
        try:  # Backwards compatibility
            sm.metadata.root = sm
        except AttributeError:
            sm.metadata = Metadata(sm)
        return sm

    def get_model_clusters(
        self,
        clusters: int,
        B: Optional[np.ndarray] = None,
        random_state: int = 42,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster the local model coefficients using k-means (from scikit-learn).
        This method (with a fixed random seed) is used for plotting Slisemap solutions.

        Args:
            clusters: Number of clusters.
            B: B matrix. Defaults to self.get_B().
            random_state: random_state for the KMeans clustering. Defaults to 42.
        Keyword Args:
            **kwargs: Additional arguments to `sklearn.KMeans`.

        Returns:
            labels: Vector of cluster labels.
            centres: Matrix of cluster centres.
        """
        B = B if B is not None else self.get_B()
        km = KMeans(clusters, random_state=random_state, **kwargs).fit(B)
        # Sort according to value for the most influential coefficient
        influence = (
            km.cluster_centers_.var(0)
            + np.abs(km.cluster_centers_).mean(0)
            + np.abs(km.cluster_centers_).max(0)
        )
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
        jitter: Union[float, np.ndarray] = 0.0,
        B: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
        show: bool = True,
        **kwargs,
    ) -> Optional[Figure]:
        """Plot the Slisemap solution using seaborn.

        Args:
            title: Title of the plot. Defaults to "".
            variables: List of variable names. Defaults to None. **DEPRECATED**
            targets: Target name(s). Defaults to None. **DEPRECATED**
            clusters: Can be None (plot individual losses), an int (plot k-means clusters of B), or an array of known cluster id:s. Defaults to None.
            bars: If the clusters are from k-means, plot the local models in a bar plot. If `bar` is an int then only plot the most influential variables. Defaults to False.
            jitter: Add random (normal) noise to the embedding, or a matrix with pre-generated noise matching Z. Defaults to 0.0.
            B: Override self.get_B() in the plot. Defaults to None. **DEPRECATED**
            Z: Override self.get_Z() in the plot. Defaults to None. **DEPRECATED**
            show: Show the plot. Defaults to True.
        Keyword Args:
            **kwargs: Additional arguments to `plt.subplots`.

        Returns:
            `matplotlib.figure.Figure` if `show=False`.

        Deprecated:
            1.3: Parameter `variables`, use `metadata.set_variables()` instead!
            1.3: Parameter `targets`, use `metadata.set_targets()` instead!
            1.3: Parameter `B`.
            1.3: Parameter `Z`.
        """
        if Z is None:
            Z = self.get_Z(rotate=True)
        else:
            _deprecated("Parameter 'Z' in Slisemap.plot")
        if B is None:
            B = self.get_B()
        else:
            _deprecated("Parameter 'B' in Slisemap.plot")
        Z_names = self.metadata.get_dimensions(long=True)
        if variables is not None:
            _deprecated(
                "Parameter 'variables' in 'Slisemap.plot'",
                "'Slisemap.metadata.set_variables'",
            )
            coefficients = _expand_variable_names(
                variables, self.intercept, self.m, targets, self.q
            )
        else:
            coefficients = self.metadata.get_coefficients()
        if targets is not None:
            _deprecated(
                "Parameter 'targets' in 'Slisemap.plot'",
                "'Slisemap.metadata.set_targets'",
            )

        kwargs.setdefault("figsize", (12, 6))
        fig, (ax1, ax2) = plt.subplots(1, 2, **kwargs)
        if clusters is None:
            _assert(not bars, "`bars!=False` requires `clusters`", Slisemap.plot)
            if Z.shape[0] == self._Z.shape[0]:
                yhat = self.predict(numpy=False)
                L = tonp(self.local_loss(yhat, self._Y, self._B)).ravel()
            else:
                L = None
            plot_embedding(
                Z,
                Z_names,
                jitter=jitter,
                color=L,
                color_name=None if L is None else "Local loss",
                color_norm=None if L is None else tuple(np.quantile(L, (0.0, 0.95))),
                ax=ax1,
            )
            B = B[np.argsort(Z[:, 0])]
            plot_matrix(B, coefficients, ax=ax2)
        else:
            if isinstance(clusters, int):
                clusters, centers = self.get_model_clusters(clusters, B)
            else:
                cl = np.sort(np.unique(clusters))
                centers = np.zeros((cl.max() + 1, B.shape[1]))
                for c in cl:
                    centers[c, :] = np.mean(B[clusters == c, :], 0)
            plot_embedding(Z, Z_names, jitter=jitter, clusters=clusters, ax=ax1)
            if bars:
                plot_barmodels(B, clusters, centers, coefficients, bars=bars, ax=ax2)
            else:
                plot_matrix(centers, coefficients, ax=ax2)
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
        jitter: Union[float, np.ndarray] = 0.0,
        col_wrap: int = 4,
        selection: bool = True,
        legend_inside: bool = True,
        Z: Optional[np.ndarray] = None,
        show: bool = True,
        **kwargs,
    ) -> Optional[sns.FacetGrid]:
        """Plot fidelities for alternative locations for the selected item(s).
        Indicate the selected item(s) either via `X` and `Y` or via `index`.

        Args:
            X: Data matrix for the selected data item(s). Defaults to None.
            Y: Response matrix for the selected data item(s). Defaults to None.
            index: Index/indices of the selected data item(s). Defaults to None.
            title: Title of the plot. Defaults to "".
            jitter: Add random (normal) noise to the embedding, or a matrix with pre-generated noise matching Z. Defaults to 0.0.
            col_wrap: Maximum number of columns. Defaults to 4.
            selection: Mark the selected data item(s), if index is given. Defaults to True.
            legend_inside: Move the legend inside the grid (if there is an empty cell). Defaults to True.
            Z: Override `self.get_Z()` in the plot. Defaults to None. **DEPRECATED**
            show: Show the plot. Defaults to True.
        Keyword Args:
            **kwargs: Additional arguments to seaborn.relplot.

        Returns:
            `seaborn.FacetGrid` if `show=False`.

        Deprecated:
            1.3: Parameter `Z`.
        """
        if Z is not None:
            _deprecated("Parameter 'Z' in Slisemap.plot_position")
        else:
            Z = self.get_Z(rotate=True)
        if index is None:
            _assert(
                X is not None and Y is not None,
                "Either index or X and Y must be given",
                Slisemap.plot_position,
            )
            L = self.get_L(X, Y)
        else:
            if isinstance(index, int):
                index = [index]
            L = self.get_L()[:, index]
        kwargs.setdefault("palette", "crest")
        hue_norm = tuple(np.quantile(L, (0.0, 0.95)))
        g = plot_embedding_facet(
            self.get_Z(rotate=True),
            self.metadata.get_dimensions(long=True),
            L,
            range(L.shape[1]),
            legend_title="Local loss",
            hue_norm=hue_norm,
            jitter=jitter,
            col_wrap=min(col_wrap, L.shape[1]),
            legend=False,
            **kwargs,
        )
        g.set_titles("")
        plt.suptitle(title)
        plot_position_legend(
            g, index if selection else None, hue_norm, legend_inside, kwargs["palette"]
        )
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
        unscale: bool = True,
        scatter: bool = False,
        jitter: float = 0.0,
        col_wrap: int = 4,
        legend_inside: bool = True,
        B: Optional[np.ndarray] = None,
        show: bool = True,
        **kwargs,
    ) -> Optional[sns.FacetGrid]:
        """Plot the distribution of the variables, either as density plots (with clusters) or as scatterplots.

        Args:
            title: Title of the plot. Defaults to "".
            X: Override self.get_X(). Defaults to None. **DEPRECATED**
            Y: Override self.get_Y(). Defaults to None. **DEPRECATED**
            variables: List of variable names. Defaults to None. **DEPRECATED**
            targets: Target name(s). Defaults to None. **DEPRECATED**
            clusters: Number of cluster or vector of cluster labels. Defaults to None.
            scatter: Use scatterplots instead of density plots (clusters are ignored). Defaults to False.
            unscale: Unscale `X` and `Y` if scaling metadata has been given (see `Slisemap.metadata.set_scale_X`). Defaults to True.
            jitter: Add jitter to the scatterplots. Defaults to 0.0.
            col_wrap: Maximum number of columns. Defaults to 4.
            legend_inside: Move the legend inside the grid (if there is an empty cell). Defaults to True.
            B: Override self.get_B() when finding the clusters (only used if clusters is an int). Defaults to None. **DEPRECATED**
            show: Show the plot. Defaults to True.
        Keyword Args:
            **kwargs: Additional arguments to seaborn.relplot.

        Returns:
            `seaborn.FacetGrid` if `show=False`.

        Deprecated:
            1.3: Parameter `variables`, use `metadata.set_variables()` instead!
            1.3: Parameter `targets`, use `metadata.set_targets()` instead!
            1.3: Parameter `X`, use `metadata.set_scale_X()` instead (to automatically unscale)!
            1.3: Parameter `Y`, use `metadata.set_scale_Y()` instead (to automatically unscale)!
            1.3: Parameter `B`.
        """
        if X is None:
            X = self.get_X(intercept=False)
        else:
            _deprecated("Parameter 'X' in Slisemap.plot_dist")
        if Y is None:
            Y = self.get_Y()
        else:
            _deprecated("Parameter 'Y' in Slisemap.plot_dist")
            Y = np.reshape(Y, (X.shape[0], -1))
        if unscale:
            X = self.metadata.unscale_X(X)
            Y = self.metadata.unscale_Y(Y)
        if variables is None:
            variables = self.metadata.get_variables(intercept=False)
        else:
            _deprecated(
                "Parameter 'variables' in 'Slisemap.plot_dist'",
                "'Slisemap.metadata.set_variables'",
            )
        if targets is not None:
            _deprecated(
                "Parameter 'targets' in 'Slisemap.plot_dist'",
                "'Slisemap.metadata.set_targets'",
            )
            if isinstance(targets, str):
                targets = [targets]
        else:
            targets = self.metadata.get_targets()
        if B is not None:
            _deprecated("Parameter 'B' in Slisemap.plot_dist")

        data = np.concatenate((X, Y), 1)
        labels = self.metadata.get_variables(False) + self.metadata.get_targets()
        if X.shape[0] == self.n:
            L = tonp(self.local_loss(self.predict(numpy=False), self._Y, self._B))
            data = np.concatenate((data, L), 1)
            labels.append("Local loss")
        if scatter:
            g = plot_embedding_facet(
                self.get_Z(rotate=True),
                self.metadata.get_dimensions(long=True),
                data,
                labels,
                jitter=jitter,
                col_wrap=col_wrap,
                **kwargs,
            )
        else:
            if isinstance(clusters, int):
                clusters, _ = self.get_model_clusters(clusters, B)
            elif clusters is None:
                legend_inside = False
            g = plot_density_facet(
                data, labels, clusters=clusters, col_wrap=col_wrap, **kwargs
            )
        plt.suptitle(title)
        if legend_inside:
            legend_inside_facet(g)
        else:
            g.tight_layout()
        if show:
            plt.show()
        else:
            return g
