"""Module that contains the `Slisemap` class."""

import lzma
from copy import copy
from os import PathLike
from timeit import default_timer as timer
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
    ALocalModel,
    LinearRegression,
    LocalModelCollection,
    identify_local_model,
    local_predict,
)
from slisemap.plot import (
    _expand_variable_names,
    plot_dist,
    plot_position,
    plot_solution,
)
from slisemap.utils import (
    LBFGS,
    CallableLike,
    CheckConvergence,
    Metadata,
    PCA_rotation,
    ToTensor,
    _assert,
    _assert_shape,
    _deprecated,
    _warn,
    global_model,
    softmax_row_kernel,
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
        regularisation: Additional regularisation function.
        distance: Distance function.
        kernel: Kernel function.
        jit: Just-In-Time compile the loss function for increased performance (see `torch.jit.trace` for caveats).
        metadata: A dictionary for storing variable names and other metadata (see [slisemap.utils.Metadata][]).
    """

    # Make Python faster and safer by not creating a Slisemap.__dict__
    __slots__ = (
        "_X",
        "_Y",
        "_Z",
        "_B",
        "_radius",
        "_lasso",
        "_ridge",
        "_z_norm",
        "_intercept",
        "_local_model",
        "_local_loss",
        "_regularisation",
        "_loss",
        "_distance",
        "_kernel",
        "_jit",
        "metadata",
        "_Z0",  # deprecated
        "_B0",  # deprecated
        "_random_state",  # deprecated
    )

    def __init__(
        self,
        X: ToTensor,
        y: ToTensor,
        radius: float = 3.5,
        d: int = 2,
        lasso: Optional[float] = None,
        ridge: Optional[float] = None,
        z_norm: float = 0.01,
        intercept: bool = True,
        local_model: Union[
            LocalModelCollection, CallableLike[ALocalModel.predict]
        ] = LinearRegression,
        local_loss: Optional[CallableLike[ALocalModel.loss]] = None,
        coefficients: Union[None, int, CallableLike[ALocalModel.coefficients]] = None,
        regularisation: Union[None, CallableLike[ALocalModel.regularisation]] = None,
        distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.cdist,
        kernel: Callable[[torch.Tensor], torch.Tensor] = softmax_row_kernel,
        B0: Optional[ToTensor] = None,
        Z0: Optional[ToTensor] = None,
        jit: bool = True,
        random_state: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        cuda: Optional[bool] = None,
    ) -> None:
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
            regularisation: Additional regularisation method (see [slisemap.local_models.identify_local_model][]). Defaults to None.
            distance: Distance function. Defaults to `torch.cdist` (Euclidean distance).
            kernel: Kernel function. Defaults to [softmax_row_kernel][slisemap.utils.softmax_row_kernel].
            B0: Initial value for B (random if None). Defaults to None.
            Z0: Initial value for Z (PCA if None). Defaults to None.
            jit: Just-In-Time compile the loss function for increased performance (see `torch.jit.trace` for caveats). Defaults to True.
            random_state: Set an explicit seed for the random number generator (i.e. `torch.manual_seed`). Defaults to None.
            dtype: Floating type. Defaults to `torch.float32`.
            device: Torch device. Defaults to None.
            cuda: Use cuda if available. Defaults to True, if the data is large enough.

        Deprecated:
            1.6: Use `device` instead of `cuda` to force a specific device.
            1.6: The `random_state` has been moved to the escape function.
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
        local_model, local_loss, coefficients, regularisation = identify_local_model(
            local_model, local_loss, coefficients, regularisation
        )
        self.lasso = 0.0 if lasso is None else lasso
        self.ridge = 0.0 if ridge is None else ridge
        self.kernel = kernel
        self.distance = distance
        self.local_model = local_model
        self.local_loss = local_loss
        self.regularisation = regularisation
        self.z_norm = z_norm
        self.radius = radius
        self._intercept = intercept
        self._jit = jit
        self.metadata: Metadata = Metadata(self)

        if cuda is not None:
            _deprecated("cuda", "device")
        if device is None:
            if cuda is None and isinstance(X, torch.Tensor):
                device = X.device
            elif cuda is True:
                device = torch.device("cuda")
        tensorargs = {"device": device, "dtype": dtype}

        self._X, X_rows, X_columns = to_tensor(X, **tensorargs)
        if intercept:
            self._X = torch.cat((self._X, torch.ones_like(self._X[:, :1])), 1)
        n, m = self._X.shape
        self.metadata.set_variables(X_columns, intercept)

        self._Y, Y_rows, Y_columns = to_tensor(y, **tensorargs)
        self.metadata.set_targets(Y_columns)
        if len(self._Y.shape) == 1:
            self._Y = self._Y[:, None]
        _assert_shape(self._Y, (n, self._Y.shape[1]), "Y", Slisemap)

        if random_state is not None:
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
            _assert_shape(self._Z0, (n, d), "Z0", Slisemap)
        if radius > 0:
            norm = 1 / (torch.sqrt(torch.sum(self._Z0**2) / self._Z0.shape[0]) + 1e-8)
            self._Z0 = self._Z0 * norm
        self._Z = self._Z0.detach().clone()

        if callable(coefficients):
            coefficients = coefficients(self._X, self._Y)
        if B0 is None:
            B0 = global_model(
                X=self._X,
                Y=self._Y,
                local_model=self.local_model,
                local_loss=self.local_loss,
                coefficients=coefficients,
                lasso=self.lasso,
                ridge=self.ridge,
            ).detach()
            if not torch.all(torch.isfinite(B0)):
                _warn(
                    "Optimising a global model as initialisation resulted in non-finite values. Consider using stronger regularisation (increase `lasso` or `ridge`).",
                    Slisemap,
                )
                B0 = torch.zeros_like(B0)
            self._B0 = B0.expand((n, coefficients))
            B_rows = None
        else:
            self._B0, B_rows, B_columns = to_tensor(B0, **tensorargs)
            if self._B0.shape[0] == 1:
                self._B0 = self._B0.expand((self.n, coefficients))
            _assert_shape(self._B0, (n, coefficients), "B0", Slisemap)
            self.metadata.set_coefficients(B_columns)
        self._B = self._B0.clone()
        self.metadata.set_rows(X_rows, Y_rows, B_rows, Z_rows)

        if (
            device is None
            and self.n**2 * self.m * self.o > 1_000_000
            and torch.cuda.is_available()
        ):
            self.cuda()

    @property
    def n(self) -> int:
        """The number of data items."""
        return self._X.shape[0]

    @property
    def m(self) -> int:
        """The number of variables (including potential intercept)."""
        return self._X.shape[1]

    @property
    def o(self) -> int:
        """The number of target variables (i.e. the number of classes)."""
        return self._Y.shape[-1]

    @property
    def d(self) -> int:
        """The number of embedding dimensions."""
        return self._Z.shape[1]

    @d.setter
    def d(self, value: int) -> None:
        # Deprecated since 1.6.
        _assert(
            value > 0, "The number of embedding dimensions must be positive", Slisemap.d
        )
        _deprecated("Set `Slisemap.d`", "Create a new Slisemap instead")
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
        """The number of local model coefficients."""
        return self._B.shape[1]

    @property
    def intercept(self) -> bool:
        """Is an intercept column added to the data?."""
        return self._intercept

    @property
    def radius(self) -> float:
        """The radius of the embedding."""
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        if self._radius != value:
            _assert(value >= 0, "radius must not be negative", Slisemap.radius)
            self._radius = value
            self._loss = None  # invalidate cached loss function

    @property
    def lasso(self) -> float:
        """Lasso regularisation strength."""
        return self._lasso

    @lasso.setter
    def lasso(self, value: float) -> None:
        if self._lasso != value:
            _assert(value >= 0, "lasso must not be negative", Slisemap.lasso)
            self._lasso = value
            self._loss = None  # invalidate cached loss function

    @property
    def ridge(self) -> float:
        """Ridge regularisation strength."""
        return self._ridge

    @ridge.setter
    def ridge(self, value: float) -> None:
        if self._ridge != value:
            _assert(value >= 0, "ridge must not be negative", Slisemap.ridge)
            self._ridge = value
            self._loss = None  # invalidate cached loss function

    @property
    def z_norm(self) -> float:
        """Z normalisation regularisation strength."""
        return self._z_norm

    @z_norm.setter
    def z_norm(self, value: float) -> None:
        if self._z_norm != value:
            _assert(value >= 0, "z_norm must not be negative", Slisemap.z_norm)
            self._z_norm = value
            self._loss = None  # invalidate cached loss function

    @property
    def local_model(self) -> CallableLike[ALocalModel.predict]:
        """Local model prediction function. Takes in X[n, m] and B[n, q], and returns Ytilde[n, n, o]."""
        return self._local_model

    @local_model.setter
    def local_model(self, value: CallableLike[ALocalModel.predict]) -> None:
        if self._local_model != value:
            _assert(
                callable(value), "local_model must be callable", Slisemap.local_model
            )
            self._local_model = value
            self._loss = None  # invalidate cached loss function

    @property
    def local_loss(self) -> CallableLike[ALocalModel.loss]:
        """Local model loss function. Takes in Ytilde[n, n, o] and Y[n, o] and returns L[n, n]."""
        return self._local_loss

    @local_loss.setter
    def local_loss(self, value: CallableLike[ALocalModel.loss]) -> None:
        if self._local_loss != value:
            _assert(callable(value), "local_loss must be callable", Slisemap.local_loss)
            self._local_loss = value
            self._loss = None  # invalidate cached loss function

    @property
    def regularisation(self) -> CallableLike[ALocalModel.regularisation]:
        """Regularisation function. Takes in X, Y, Bp, Z, and Ytilde and returns an additional loss scalar."""
        return self._regularisation

    @regularisation.setter
    def regularisation(self, value: CallableLike[ALocalModel.regularisation]) -> None:
        if self._regularisation != value:
            _assert(
                callable(value),
                "regularisation function must be callable",
                Slisemap.regularisation,
            )
            self._regularisation = value
            self._loss = None  # invalidate cached loss function

    @property
    def distance(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Distance function. Takes in Z[n1, d] and Z[n2, d], and returns D[n1, n2]."""
        return self._distance

    @distance.setter
    def distance(
        self, value: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> None:
        if self._distance != value:
            _assert(callable(value), "distance must be callable", Slisemap.distance)
            self._distance = value
            self._loss = None  # invalidate cached loss function

    @property
    def kernel(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Kernel function. Takes in D[n, n] and returns W[n, n]."""
        return self._kernel

    @kernel.setter
    def kernel(self, value: Callable[[torch.Tensor], torch.Tensor]) -> None:
        if self._kernel != value:
            _assert(callable(value), "kernel must be callable", Slisemap.kernel)
            self._kernel = value
            self._loss = None  # invalidate cached loss function

    @property
    def jit(self) -> bool:
        """Just-In-Time compile the loss function?."""
        return self._jit

    @jit.setter
    def jit(self, value: bool) -> None:
        if self._jit != value:
            self._jit = value
            self._loss = None  # invalidate cached loss function

    def random_state(self, value: Optional[int]) -> None:
        """Set the seed for the random number generator specific for this object (None reverts to the global `torch` PRNG).

        Deprecated:
            1.6: Use `Slisemap.escape(random_state=...)` instead.
        """
        _deprecated(Slisemap.random_state, "Slisemap.escape(random_state=...)")
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
    def tensorargs(self) -> Dict[str, Any]:
        """When creating a new `torch.Tensor` add these keyword arguments to match the `dtype` and `device` of this Slisemap object."""
        return {"device": self._X.device, "dtype": self._X.dtype}

    def cuda(self, **kwargs: Any) -> None:
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
        self._loss = None  # invalidate cached loss function

    def cpu(self, **kwargs: Any) -> None:
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
        self._loss = None  # invalidate cached loss function

    def _get_loss_fn(
        self, individual: bool = False
    ) -> Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ]:
        """Return the Slisemap loss function.

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
                regularisation=self.regularisation,
            )
            # JITting the loss function improves the performance
            if self._jit:
                self._loss = torch.jit.trace(
                    self._loss, (self._X[:1], self._Y[:1], self._B[:1], self._Z[:1])
                )
        return self._loss

    def _as_new_X(self, X: Optional[ToTensor] = None) -> torch.Tensor:
        if X is None:
            return self._X
        X = torch.atleast_2d(to_tensor(X, **self.tensorargs)[0])
        if self._intercept and X.shape[1] == self.m - 1:
            X = torch.cat((X, torch.ones_like(X[:, :1])), 1)
        _assert_shape(X, (X.shape[0], self.m), "X", Slisemap._as_new_X)
        return X

    def _as_new_Y(self, Y: Optional[ToTensor] = None, n: int = -1) -> torch.Tensor:
        if Y is None:
            return self._Y
        Y = to_tensor(Y, **self.tensorargs)[0]
        if len(Y.shape) < 2:
            Y = torch.reshape(Y, (n, self.o))
        _assert_shape(Y, (n if n > 0 else Y.shape[0], self.o), "Y", Slisemap._as_new_Y)
        return Y

    def get_Z(
        self, scale: bool = True, rotate: bool = False, numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """Get the Z matrix.

        Args:
            scale: Scale the returned `Z` to match self.radius. Defaults to True.
            rotate: Rotate the returned `Z` so that the first dimension is the major axis. Defaults to False.
            numpy: Return the matrix as a numpy (True) or pytorch (False) matrix. Defaults to True.

        Returns:
           The `Z` matrix.
        """
        self._normalise()
        Z = self._Z * self.radius if scale and self.radius > 0 else self._Z
        if rotate:
            Z = Z @ PCA_rotation(Z, center=False)
        return tonp(Z) if numpy else Z

    def get_B(self, numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Get the B matrix.

        Args:
            numpy: Return the matrix as a numpy (True) or pytorch (False) matrix. Defaults to True.

        Returns:
           The `B` matrix.
        """
        return tonp(self._B) if numpy else self._B

    def get_D(self, numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Get the embedding distance matrix.

        Args:
            numpy: Return the matrix as a numpy (True) or pytorch (False) matrix. Defaults to True.

        Returns:
           The `D` matrix.
        """
        Z = self.get_Z(rotate=False, scale=True, numpy=False)
        D = self._distance(Z, Z)
        return tonp(D) if numpy else D

    def get_L(
        self,
        X: Optional[ToTensor] = None,
        Y: Optional[ToTensor] = None,
        numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Get the loss matrix: [B.shape[0], X.shape[0]].

        Args:
            X: Optional replacement for the training X. Defaults to None.
            Y: Optional replacement for the training Y. Defaults to None.
            numpy: Return the matrix as a numpy (True) or pytorch (False) matrix. Defaults to True.

        Returns:
           The `L` matrix.
        """
        X = self._as_new_X(X)
        Y = self._as_new_Y(Y, X.shape[0])
        L = self.local_loss(self.local_model(X, self._B), Y)
        return tonp(L) if numpy else L

    def get_W(self, numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Get the weight matrix.

        Args:
            numpy: Return the matrix as a numpy.ndarray instead of a torch.Tensor. Defaults to True.

        Returns:
           The `W` matrix.
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
           The `X` matrix.
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
           The `Y` matrix.
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
        """Compute row-wise entropy of the `W` matrix induced by `Z`. **DEPRECATED**.

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
        **kwargs: Any,
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
        loss_fn = lambda: loss_(self._X, self._Y, B, Z)  # noqa: E731
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
        lerp: float = 0.95,
        noise: float = 0.0,
        random_state: int = 42,
    ) -> None:
        """Try to escape a local optimum by moving the items (embedding and local model) to the neighbourhoods best suited for them.

        This is done by finding another item (in the optimal neighbourhood) and copying its values for Z and B.

        Args:
            force_move: Do not allow the items to pair with themselves. Defaults to True.
            escape_fn: Escape function (see [slisemap.escape][]). Defaults to [escape_neighbourhood][slisemap.escape.escape_neighbourhood].
            lerp: Linear interpolation between the old (0.0) and the new (1.0) embedding position. Defaults to 0.95.
            noise: Scale of the noise added to the embedding matrix if it looses rank after an escape (recommended for gradient based optimisers). Defaults to 1e-4.
            random_state: Seed for the random generator if `noise > 0.0`. Defaults to 42.
        """
        if lerp <= 0.0:
            _warn("Escaping with `lerp <= 0` does nothing!", Slisemap.escape)
            return
        B, Z = escape_fn(
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
        if lerp >= 1.0:
            self._B, self._Z = B, Z
        else:
            self._B = (1.0 - lerp) * self._B + lerp * B
            self._Z = (1.0 - lerp) * self._Z + lerp * Z
        if noise > 0.0:
            rank = torch.linalg.matrix_rank(self._Z - torch.mean(self._Z, 0, True))
            if rank.item() < min(*self._Z.shape):
                generator = torch.Generator(self._Z.device).manual_seed(random_state)
                self._Z = torch.normal(self._Z, noise, generator=generator)
        self._normalise()

    def _normalise(self) -> None:
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
        verbose: Literal[0, 1, 2] = 0,
        only_B: bool = False,
        escape_kws: Dict[str, object] = {},
        *,
        escape_fn: Optional[CallableLike[escape_neighbourhood]] = None,
        noise: Optional[float] = None,
        **kwargs: Any,
    ) -> float:
        """Optimise Slisemap by alternating between [self.lbfgs()][slisemap.slisemap.Slisemap.lbfgs] and [self.escape()][slisemap.slisemap.Slisemap.escape] until convergence.

        Statistics for the optimisation can be found in `self.metadata["optimize_time"]` and `self.metadata["optimize_loss"]`.

        Args:
            patience: Number of escapes without improvement before stopping. Defaults to 2.
            max_escapes: Maximum numbers optimisation rounds. Defaults to 100.
            max_iter: Maximum number of LBFGS iterations per round. Defaults to 500.
            verbose: Print status messages (0: no, 1: some, 2: all). Defaults to 0.
            only_B: Only optimise the local models, not the embedding. Defaults to False.
            escape_kws: Optional keyword arguments to [self.escape()][slisemap.slisemap.Slisemap.escape]. Defaults to {}.

        Keyword Args:
            escape_fn: Escape function (see [slisemap.escape][]). Defaults to [escape_neighbourhood][slisemap.escape.escape_neighbourhood].
            noise: Scale of the noise added to the embedding matrix if it looses rank after an escape.
            **kwargs: Optional keyword arguments to Slisemap.lbfgs.

        Returns:
            The loss value.

        Deprecated:
            1.6: The `noise` argument, use `escape_kws={"noise": noise}` instead.
            1.6: The `escape_fn` argument, use `escape_kws={"escape_fn": escape_fn}` instead.
        """
        if noise is not None:
            _deprecated(
                "Slisemap.optimise(noise=noise, ...)",
                'Slisemap.optimise(escape_kws={"noise":noise}, ...)',
            )
            escape_kws.setdefault("noise", noise)
        if escape_fn is not None:
            _deprecated(
                "Slisemap.optimise(escape_fn=escape_fn, ...)",
                'Slisemap.optimise(escape_kws={"escape_fn":escape_fn}, ...)',
            )
            escape_kws.setdefault("escape_fn", escape_fn)
        loss = np.repeat(np.inf, 2)
        time = timer()
        loss[0] = self.lbfgs(
            max_iter=max_iter,
            only_B=True,
            increase_tolerance=not only_B,
            verbose=verbose > 1,
            **kwargs,
        )
        history = [loss[0]]
        if verbose:
            i = 0
            print(f"Slisemap.optimise LBFGS  {i:2d}: {loss[0]:.2f}")
        if only_B:
            self.metadata["optimize_time"] = timer() - time
            self.metadata["optimize_loss"] = history
            return loss[0]
        cc = CheckConvergence(patience, max_escapes)
        while not cc.has_converged(loss, self.copy, verbose=verbose > 1):
            self.escape(random_state=cc.iter, **escape_kws)
            loss[1] = self.value()
            if verbose:
                print(f"Slisemap.optimise Escape {i:2d}: {loss[1]:.2f}")
            loss[0] = self.lbfgs(
                max_iter=max_iter,
                increase_tolerance=True,
                verbose=verbose > 1,
                **kwargs,
            )
            history.append(loss[1])
            history.append(loss[0])
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
        history.append(loss)
        self.metadata["optimize_time"] = timer() - time
        self.metadata["optimize_loss"] = history
        if verbose:
            print(f"Slisemap.optimise Final    : {loss:.2f}")
        return loss

    optimize = optimise

    def fit_new(
        self,
        Xnew: ToTensor,
        ynew: ToTensor,
        optimise: bool = True,
        between: bool = True,
        escape_fn: Callable = escape_neighbourhood,
        loss: bool = False,
        verbose: bool = False,
        numpy: bool = True,
        **kwargs: Any,
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
                    LBFGS(lambda: lf(Bi, Zi), [Bi, Zi], **kwargs)  # noqa: B023
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

    def predict(  # noqa: D417
        self,
        X: Optional[ToTensor] = None,
        B: Optional[ToTensor] = None,
        Z: Optional[ToTensor] = None,
        numpy: bool = True,
        *,
        Xnew: Optional[ToTensor] = None,
        Znew: Optional[ToTensor] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Predict new outcomes when the data and embedding or local model is known.

        If the local models `B` are known they are used.
        If the embeddings `Z` are known they are used to find new local models.
        Ohterwise the closest training X gives the `B`.

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
            1.4: Renamed Xnew, Znew to X, Z.
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
        if X is None:
            X = self._X
            if B is None and Z is None:
                B = self._B
        else:
            X = self._as_new_X(X)
            if B is None and Z is None:
                D = torch.cdist(X, self._X)
                B = self._B[D.argmin(1), :]
        if B is None:
            Z = torch.atleast_2d(to_tensor(Z, **self.tensorargs)[0])
            _assert_shape(Z, (X.shape[0], self.d), "Z", Slisemap.predict)
            D = self._distance(Z, self._Z)
            W = self.kernel(D)
            B = self._B[torch.argmin(D, 1)].clone().requires_grad_(True)
            yhat = lambda: (  # noqa: E731
                torch.sum(W * self.local_loss(self.local_model(self._X, B), self._Y))
                + self.lasso * torch.sum(torch.abs(B))
                + self.ridge * torch.sum(B**2)
            )
            LBFGS(yhat, [B], **kwargs)
        else:
            B = torch.atleast_2d(to_tensor(B, **self.tensorargs)[0])
            _assert_shape(B, (X.shape[0], self.q), "B", Slisemap.predict)
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

    def restore(self) -> None:
        """Reset B and Z to their initial values B0 and Z0.

        Deprecated:
            1.6: Use `Slisemap.copy` before any optimisation instead.
        """
        _deprecated(Slisemap.restore, Slisemap.copy)
        self._Z = self._Z0.clone().detach()
        self._B = self._B0.clone().detach()

    def save(
        self,
        f: Union[str, PathLike, BinaryIO],
        any_extension: bool = False,
        compress: Union[bool, int] = True,
        **kwargs: Any,
    ) -> None:
        """Save the Slisemap object to a file.

        This method uses `torch.save` (which uses `pickle` for the non-pytorch properties).
        This means that lambda-functions are not supported (unless a custom pickle module is used, see `torch.save`).

        Note that the random state is not saved, only the initial seed (if set).

        The default file extension is ".sm".

        Args:
            f: Either a Path-like object or a (writable) File-like object.
            any_extension: Do not check the file extension. Defaults to False.
            compress: Compress the file with LZMA. Either a bool or a compression preset [0, 9]. Defaults to True.

        Keyword Args:
            **kwargs: Parameters forwarded to `torch.save`.
        """
        if not any_extension and isinstance(f, (str, PathLike)):  # noqa: SIM102
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
            if isinstance(compress, int) and compress > 0:
                with lzma.open(f, "wb", preset=compress) as f2:
                    torch.save(self, f2, **kwargs)
            elif compress:
                with lzma.open(f, "wb") as f2:
                    torch.save(self, f2, **kwargs)
            else:
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
        *,
        map_location: Optional[object] = None,
        **kwargs: Any,
    ) -> "Slisemap":
        """Load a Slisemap object from a file.

        This function uses `torch.load`, so the tensors are restored to their previous devices.
        Use `device="cpu"` to avoid assuming that the same device exists.
        This is useful if the Slisemap object has been trained on a GPU, but the current computer lacks a GPU.

        Note that this is a classmethod, use it with: `Slisemap.load(...)`.

        SAFETY: This function is based on `torch.load` which (by default) uses `pickle`.
        Do not use `Slisemap.load` on untrusted files, since `pickle` can run arbitrary Python code.

        Args:
            f: Either a Path-like object or a (readable) File-like object.
            device: Device to load the tensors to (or the original if None). Defaults to None.

        Keyword Args:
            map_location: The same as `device` (this is the name used by `torch.load`). Defaults to None.
            **kwargs: Parameters forwarded to `torch.load`.

        Returns:
            The loaded Slisemap object.
        """
        if device is None:
            device = map_location
        try:
            with lzma.open(f, "rb") as f2:
                sm = torch.load(f2, map_location=device, **kwargs)
        except lzma.LZMAError:
            sm: Slisemap = torch.load(f, map_location=device, **kwargs)
        return sm

    def __setstate__(self, data: Any) -> None:
        # Handling loading of Slisemap objects from older versions
        if not isinstance(data, dict):
            data = next(d for d in data if isinstance(d, dict))
        for k, v in data.items():
            try:
                setattr(self, k, v)
            except AttributeError as e:
                _warn(e, Slisemap.__setstate__)
        if isinstance(getattr(self, "metadata", {}), Metadata):
            self.metadata.root = self
        else:
            self.metadata = Metadata(self, **getattr(self, "metadata", {}))
        if not hasattr(self, "_regularisation"):
            self._regularisation = ALocalModel.regularisation

    def get_model_clusters(
        self,
        clusters: int,
        B: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster the local model coefficients using k-means (from scikit-learn).

        This method (with a fixed random seed) is used for plotting Slisemap solutions.

        Args:
            clusters: Number of clusters.
            B: B matrix. Defaults to `self.get_B()`.
            Z: Z matrix. Defaults to `self.get_Z(rotate=True)`.
            random_state: random_state for the KMeans clustering. Defaults to 42.

        Keyword Args:
            **kwargs: Additional arguments to `sklearn.KMeans`.

        Returns:
            labels: Vector of cluster labels.
            centres: Matrix of cluster centres.
        """
        B = B if B is not None else self.get_B()
        Z = Z if Z is not None else self.get_Z(rotate=True)
        km = KMeans(clusters, random_state=random_state, **kwargs).fit(B)
        ord = np.argsort([Z[km.labels_ == k, 0].mean() for k in range(clusters)])
        return np.argsort(ord)[km.labels_], km.cluster_centers_[ord]

    def plot(
        self,
        title: str = "",
        clusters: Union[None, int, np.ndarray] = None,
        bars: Union[bool, int, Sequence[str]] = True,
        jitter: Union[float, np.ndarray] = 0.0,
        show: bool = True,
        bar: Union[None, bool, int] = None,
        *,
        B: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
        variables: Optional[Sequence[str]] = None,
        targets: Union[None, str, Sequence[str]] = None,
        **kwargs: Any,
    ) -> Optional[Figure]:
        """Plot the Slisemap solution using seaborn.

        Args:
            title: Title of the plot. Defaults to "".
            clusters: Can be None (plot individual losses), an int (plot k-means clusters of B), or an array of known cluster id:s. Defaults to None.
            bars: Plot the local models in a bar plot. Either an int (to only plot the most influential variables), a list of variables, or a bool. Defaults to True.
            jitter: Add random (normal) noise to the embedding, or a matrix with pre-generated noise matching Z. Defaults to 0.0.
            show: Show the plot. Defaults to True.
            bar: Alternative spelling for `bars`. Defaults to None.

        Keyword Args:
            B: Override self.get_B() in the plot. Defaults to None. **DEPRECATED**
            Z: Override self.get_Z() in the plot. Defaults to None. **DEPRECATED**
            variables: List of variable names. Defaults to None. **DEPRECATED**
            targets: Target name(s). Defaults to None. **DEPRECATED**
            **kwargs: Additional arguments to [plot_solution][slisemap.plot.plot_solution] and `plt.subplots`.

        Returns:
            `matplotlib.figure.Figure` if `show=False`.

        Deprecated:
            1.3: Parameter `variables`, use `metadata.set_variables()` instead!
            1.3: Parameter `targets`, use `metadata.set_targets()` instead!
            1.3: Parameter `B`.
            1.3: Parameter `Z`.
        """
        if bar is not None:
            bars = bar
        if Z is None:
            Z = self.get_Z(rotate=True)
        else:
            _deprecated("Parameter 'Z' in Slisemap.plot")
        if B is None:
            B = self.get_B()
        else:
            _deprecated("Parameter 'B' in Slisemap.plot")
        dimensions = self.metadata.get_dimensions(long=True)
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

        loss = None
        centers = None
        if clusters is None:
            if Z.shape[0] == self._Z.shape[0]:
                loss = tonp(self.local_loss(self.predict(numpy=False), self._Y))
        else:
            if isinstance(clusters, int):
                clusters, centers = self.get_model_clusters(clusters, B, Z)
            else:
                clusters = np.asarray(clusters)
                centers = np.stack(
                    [np.mean(B[clusters == c, :], 0) for c in np.unique(clusters)], 0
                )
        fig = plot_solution(
            Z=Z,
            B=B,
            loss=loss,
            clusters=clusters,
            centers=centers,
            coefficients=coefficients,
            dimensions=dimensions,
            title=title,
            bars=bars,
            jitter=jitter,
            **kwargs,
        )
        if show:
            plt.show()
        else:
            return fig

    def plot_position(
        self,
        X: Optional[ToTensor] = None,
        Y: Optional[ToTensor] = None,
        index: Union[None, int, Sequence[int]] = None,
        title: str = "",
        jitter: Union[float, np.ndarray] = 0.0,
        selection: bool = True,
        legend_inside: bool = True,
        show: bool = True,
        *,
        Z: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Optional[sns.FacetGrid]:
        """Plot local losses for alternative locations for the selected item(s).

        Indicate the selected item(s) either via `X` and `Y` or via `index`.

        Args:
            X: Data matrix for the selected data item(s). Defaults to None.
            Y: Response matrix for the selected data item(s). Defaults to None.
            index: Index/indices of the selected data item(s). Defaults to None.
            title: Title of the plot. Defaults to "".
            jitter: Add random (normal) noise to the embedding, or a matrix with pre-generated noise matching Z. Defaults to 0.0.
            selection: Mark the selected data item(s), if index is given. Defaults to True.
            legend_inside: Move the legend inside the grid (if there is an empty cell). Defaults to True.
            show: Show the plot. Defaults to True.

        Keyword Args:
            Z: Override `self.get_Z()` in the plot. Defaults to None. **DEPRECATED**
            **kwargs: Additional arguments to `seaborn.relplot`.

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
            L = self.get_L(X=X, Y=Y)
        else:
            if isinstance(index, int):
                index = [index]
            L = self.get_L()[:, index]
        g = plot_position(
            Z=Z,
            L=L,
            Zs=Z[index, :] if selection and index is not None else None,
            dimensions=self.metadata.get_dimensions(long=True),
            title=title,
            jitter=jitter,
            legend_inside=legend_inside,
            **kwargs,
        )
        if show:
            plt.show()
        else:
            return g

    def plot_dist(
        self,
        title: str = "",
        clusters: Union[None, int, np.ndarray] = None,
        unscale: bool = True,
        scatter: bool = False,
        jitter: float = 0.0,
        legend_inside: bool = True,
        show: bool = True,
        *,
        X: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        variables: Optional[List[str]] = None,
        targets: Union[None, str, Sequence[str]] = None,
        **kwargs: Any,
    ) -> Optional[sns.FacetGrid]:
        """Plot the distribution of the variables, either as density plots (with clusters) or as scatterplots.

        Args:
            title: Title of the plot. Defaults to "".
            clusters: Number of cluster or vector of cluster labels. Defaults to None.
            scatter: Use scatterplots instead of density plots (clusters are ignored). Defaults to False.
            unscale: Unscale `X` and `Y` if scaling metadata has been given (see `Slisemap.metadata.set_scale_X`). Defaults to True.
            jitter: Add jitter to the scatterplots. Defaults to 0.0.
            legend_inside: Move the legend inside the grid (if there is an empty cell). Defaults to True.
            show: Show the plot. Defaults to True.

        Keyword Args:
            X: Override self.get_X(). Defaults to None. **DEPRECATED**
            Y: Override self.get_Y(). Defaults to None. **DEPRECATED**
            B: Override self.get_B() when finding the clusters (only used if clusters is an int). Defaults to None. **DEPRECATED**
            variables: List of variable names. Defaults to None. **DEPRECATED**
            targets: Target name(s). Defaults to None. **DEPRECATED**
            **kwargs: Additional arguments to `seaborn.relplot` or `seaborn.scatterplot`.

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
        if isinstance(X, torch.Tensor):
            X = tonp(X)
        if isinstance(Y, torch.Tensor):
            Y = tonp(Y)
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
        if isinstance(clusters, int):
            clusters, _ = self.get_model_clusters(clusters, B)
        loss = tonp(self.local_loss(self.predict(numpy=False), self._Y))

        g = plot_dist(
            X=X,
            Y=Y,
            Z=self.get_Z(),
            loss=loss,
            variables=self.metadata.get_variables(False),
            targets=self.metadata.get_targets(),
            dimensions=self.metadata.get_dimensions(long=True),
            title=title,
            clusters=clusters,
            scatter=scatter,
            jitter=jitter,
            legend_inside=legend_inside,
            **kwargs,
        )
        if show:
            plt.show()
        else:
            return g


def make_loss(
    local_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    local_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.cdist,
    kernel: Callable[[torch.Tensor], torch.Tensor] = softmax_row_kernel,
    radius: float = 3.5,
    lasso: float = 0.0,
    ridge: float = 0.0,
    z_norm: float = 1.0,
    individual: bool = False,
    regularisation: Optional[Callable] = None,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    r"""Create a loss function for Slisemap to optimise.

    Args:
        local_model: Prediction function for the local models.
        local_loss: Loss function for the local models.
        distance: Embedding distance function. Defaults to `torch.cdist` (Euclidean distance).
        kernel: Kernel for embedding distances, Defaults to `softmax_kernel`.
        radius: For enforcing the radius of Z. Defaults to 3.5.
        lasso: Lasso-regularisation coefficient for B ($\lambda_{lasso} * ||B||_1$). Defaults to 0.0.
        ridge: Ridge-regularisation coefficient for B ($\lambda_{ridge} * ||B||_2$). Defaults to 0.0.
        z_norm: Z normalisation regularisation coefficient ($\\lambda_{norm} * (sum(Z^2)-n)^2$). Defaults to 1.0.
        individual: Return individual (row-wise) losses. Defaults to False.
        regularisation: Additional loss function. Defaults to None.

    Returns:
        Loss function for SLISEMAP
    """
    dim = 1 if individual else ()
    if individual and z_norm > 0:
        _warn(
            "The Z normalisation is added to every individual loss if z_norm > 0",
            make_loss,
        )

    def loss_fn(
        X: torch.Tensor,
        Y: torch.Tensor,
        B: torch.Tensor,
        Z: torch.Tensor,
    ) -> torch.Tensor:
        """Slisemap loss function.

        Args:
            X: Data matrix [n, m].
            Y: Target matrix [n, k].
            B: Local models [n, p].
            Z: Embedding matrix [n, d].

        Returns:
            The loss value.
        """
        if radius > 0:
            Zss = torch.sum(Z**2)
            Z = Z * (radius / (torch.sqrt(Zss / Z.shape[0]) + 1e-8))
        D = distance(Z, Z)
        Ytilde = local_model(X, B)
        L = local_loss(Ytilde, Y)
        loss = torch.sum(kernel(D) * L, dim=dim)
        if lasso > 0:
            loss += lasso * torch.sum(B.abs(), dim=dim)
        if ridge > 0:
            loss += ridge * torch.sum(B**2, dim=dim)
        if z_norm > 0 and radius > 0:
            loss += z_norm * (Zss - Z.shape[0]) ** 2
        if regularisation is not None:
            loss += regularisation(X, Y, B, Z, Ytilde)
        return loss

    return loss_fn


def make_marginal_loss(
    X: torch.Tensor,
    Y: torch.Tensor,
    B: torch.Tensor,
    Z: torch.Tensor,
    Xnew: torch.Tensor,
    Ynew: torch.Tensor,
    local_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    local_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.cdist,
    kernel: Callable[[torch.Tensor], torch.Tensor] = softmax_row_kernel,
    radius: float = 3.5,
    lasso: float = 0.0,
    ridge: float = 0.0,
    jit: bool = True,
) -> Tuple[
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor, torch.Tensor], None],
]:
    r"""Create a loss for adding new points with Slisemap.

    Args:
        X: The existing data matrix [n_old, m].
        Y: The existing target matrix [n_old, k].
        B: The fitted models [n_old, p].
        Z: The fitted embedding [n_old, d].
        Xnew: The new data matrix [n_new, m].
        Ynew: The new target matrix [n_new, k].
        local_model: Prediction function for the local models.
        local_loss: Loss function for the local models.
        distance: Embedding distance function. Defaults to `torch.cdist` (Euclidean distance).
        kernel: Kernel for embedding distances, Defaults to `softmax_kernel`.
        radius: For enforcing the radius of Z. Defaults to 3.5.
        lasso: Lasso-regularisation coefficient for B ($\lambda_{lasso} * ||B||_1$). Defaults to 0.0.
        ridge: Ridge-regularisation coefficient for B ($\lambda_{ridge} * ||B||_2$). Defaults to 0.0.
        jit: Just-In-Time compile the loss function. Defaults to True.

    Returns:
        loss: A marginal loss function that takes Bnew [n_new, p] and Znew [n_new, d].
        set_new: A function for changing the Xnew [n_new, m] and Ynew [n_new, k].
    """
    Xcomb = torch.cat((X, Xnew), 0)
    Ycomb = torch.cat((Y, Ynew), 0)
    Nold = X.shape[0]
    L0 = local_loss(local_model(Xcomb, B), Ycomb)  # Nold x Ncomb
    D0 = distance(Z, Z)  # Nold x Nold

    def set_new(Xnew: torch.Tensor, Ynew: torch.Tensor) -> None:
        """Set the Xnew and Ynew for the generated marginal Slisemap loss function.

        Args:
            Xnew: New data matrix [n_new, m].
            Ynew: New target matrix [n_new, k].
        """
        nonlocal Xcomb, Ycomb, L0
        Xcomb[Nold:] = Xnew
        Ycomb[Nold:] = Ynew
        L0[:, Nold:] = local_loss(local_model(Xnew, B), Ynew)

    if radius > 0:
        Zss0 = torch.sum(Z**2)

    def loss(Bnew: torch.Tensor, Znew: torch.Tensor) -> torch.Tensor:
        """Marginal Slisemap loss.

        Args:
            Bnew: New local models [n_new, p].
            Znew: New embedding matrix [n_new, d].

        Returns:
            The marginal loss value.
        """
        L1 = local_loss(local_model(Xcomb, Bnew), Ycomb)  # Nnew x Ncomb
        L = torch.cat((L0, L1), 0)  # Ncomb x Ncomb

        D1 = distance(Znew, Z)  # Nnew x Nold
        D2 = distance(Znew, Znew)  # Nnew x Nnew
        D3 = D1.transpose(0, 1)
        D = torch.cat(
            (torch.cat((D0, D1), 0), torch.cat((D3, D2), 0)), 1
        )  # Ncomb x Ncomb
        if radius > 0:
            Zss = Zss0 + torch.sum(Znew**2)
            Ncomb = Z.shape[0] + Znew.shape[0]
            norm = radius / (torch.sqrt(Zss / Ncomb) + 1e-8)
            D = D * norm

        kD = kernel(D)
        a = torch.sum(kD * L)
        if lasso > 0:
            a += lasso * torch.sum(Bnew.abs())
        if ridge > 0:
            a += ridge * torch.sum(Bnew**2)
        return a

    if jit:
        Nnew = Xnew.shape[0]
        loss = torch.jit.trace(loss, (B[:1].expand(Nnew, -1), Z[:1].expand(Nnew, -1)))
    return loss, set_new
