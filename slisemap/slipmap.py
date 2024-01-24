"""
Prototype version of Slisemap.

Instead of giving every data item its own local model we have a fixed grid of
prototypes, where each prototype has a local model. This improves the scaling
from quadratic to linear.
"""

from copy import copy
import lzma
from os import PathLike
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
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

from slisemap.local_models import (
    ALocalModel,
    LinearRegression,
    local_predict,
    LocalModelCollection,
    identify_local_model,
)
from slisemap.loss import softmax_column_kernel, squared_distance
from slisemap.slisemap import Slisemap
from slisemap.plot import (
    plot_dist,
    plot_position,
    plot_prototypes,
    plot_solution,
)
from slisemap.utils import (
    LBFGS,
    CheckConvergence,
    CallableLike,
    PCA_rotation,
    _assert,
    _warn,
    global_model,
    tonp,
    Metadata,
    make_grid,
)


class Slipmap:
    """__Slipmap__: Faster and more robust `[Slisemap][slisemap.slisemap.Slisemap]`.

    This class contains the data and the parameters needed for finding a Slipmap solution.
    It also contains the solution (remember to [optimise()][slisemap.slipmap.Slipmap.optimize] first) in the form of an embedding matrix, see [get_Z()][slisemap.slipmap.Slipmap.get_Z], and a matrix of coefficients for the local model, see [get_Bp()][slisemap.slipmap.Slipmap.get_Bp].
    Other methods of note are the various plotting methods, the [save()][slisemap.slipmap.Slipmap.save] method, and the [predict()][slisemap.slipmap.Slipmap.predict] method.

    The use of some regularisation is highly recommended. Slipmap comes with built-in lasso/L1 and ridge/L2 regularisation (if these are used it is also a good idea to normalise the data in advance).

    Attributes:
        n: The number of data items (`X.shape[0]`).
        m: The number of variables (`X.shape[1]`).
        o: The number of targets (`Y.shape[1]`).
        d: The number of embedding dimensions (`Z.shape[1]`).
        p: The number of prototypes (`Zp.shape[1]`).
        q: The number of coefficients (`Bp.shape[1]`).
        intercept: Has an intercept term been added to `X`.
        radius: The radius of the embedding.
        lasso: Lasso regularisation coefficient.
        ridge: Ridge regularisation coefficient.
        local_model: Local model prediction function (see [slisemap.local_models][]).
        local_loss: Local model loss function (see [slisemap.local_models][]).
        regularisation: Additional regularisation function.
        distance: Distance function.
        kernel: Kernel function.
        jit: Just-In-Time compile the loss function for increased performance (see `torch.jit.trace` for caveats).
        metadata: A dictionary for storing variable names and other metadata (see [slisemap.utils.Metadata][]).
    """

    # Make Python faster and safer by not creating a Slipmap.__dict__
    __slots__ = (
        "_X",
        "_Y",
        "_Z",
        "_Bp",
        "_Zp",
        "_radius",
        "_lasso",
        "_ridge",
        "_intercept",
        "_local_model",
        "_local_loss",
        "_regularisation",
        "_loss",
        "_distance",
        "_kernel",
        "_jit",
        "metadata",
    )

    def __init__(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        radius: float = 2.0,
        d: int = 2,
        lasso: Optional[float] = None,
        ridge: Optional[float] = None,
        intercept: bool = True,
        local_model: Union[
            LocalModelCollection, CallableLike[ALocalModel.predict]
        ] = LinearRegression,
        local_loss: Optional[CallableLike[ALocalModel.loss]] = None,
        coefficients: Union[None, int, CallableLike[ALocalModel.coefficients]] = None,
        regularisation: Union[None, CallableLike[ALocalModel.regularisation]] = None,
        distance: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = squared_distance,
        kernel: Callable[[torch.Tensor], torch.Tensor] = softmax_column_kernel,
        Z0: Union[None, np.ndarray, torch.Tensor] = None,
        Bp0: Union[None, np.ndarray, torch.Tensor] = None,
        Zp0: Union[None, np.ndarray, torch.Tensor] = None,
        prototypes: Union[int, float] = 1.0,
        jit: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        """Create a Slipmap object.

        Args:
            X: Data matrix.
            y: Target vector or matrix.
            radius: The radius of the embedding Z. Defaults to 2.0.
            d: The number of embedding dimensions. Defaults to 2.
            lasso: Lasso regularisation coefficient. Defaults to 0.0.
            ridge: Ridge regularisation coefficient. Defaults to 0.0.
            intercept: Should an intercept term be added to `X`. Defaults to True.
            local_model: Local model prediction function (see [slisemap.local_models.identify_local_model][]). Defaults to [LinearRegression][slisemap.local_models.LinearRegression].
            local_loss: Local model loss function (see [slisemap.local_models.identify_local_model][]). Defaults to None.
            coefficients: The number of local model coefficients (see [slisemap.local_models.identify_local_model][]). Defaults to None.
            regularisation: Additional regularisation method (see [slisemap.local_models.identify_local_model][]). Defaults to None.
            distance: Distance function. Defaults to [squared_distance][slisemap.loss.squared_distance].
            kernel: Kernel function. Defaults to [softmax_column_kernel][slisemap.loss.softmax_column_kernel].
            Z0: Initial embedding for the data. Defaults to PCA.
            Bp0: Initial coefficients for the local models. Defaults to None.
            Zp0: Initial embedding for the prototypes. Defaults to `[make_grid][slisemap.utils.make_grid](prototypes)`.
            prototypes: Number of prototypes (if > 6) or prototype density (if < 6.0). Defaults to 1.0.
            jit: Just-In-Time compile the loss function for increased performance (see `torch.jit.trace` for caveats). Defaults to True.
            dtype: Floating type. Defaults to `torch.float32`.
            device: Torch device. Defaults to None.
        """
        for s in Slipmap.__slots__:
            # Initialise all attributes (to avoid attribute errors)
            setattr(self, s, None)
        if lasso is None and ridge is None:
            _warn(
                "Consider using regularisation!\n"
                + "\tRegularisation is important for handling small neighbourhoods, and also makes the local models more local."
                + " Lasso (l1) and ridge (l2) regularisation is built-in, via the parameters ``lasso`` and ``ridge``."
                + " Set ``lasso=0`` to disable this warning (if no regularisation is really desired).",
                Slipmap,
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
        self._radius = radius
        self._intercept = intercept
        self._jit = jit

        if Zp0 is None:
            if prototypes < 6.0:
                # Interpret prototypes as a density (prototypes per unit square)
                prototypes = radius**2 * 2 * np.pi * prototypes
            Zp0 = make_grid(prototypes, d=d)
        o = Zp0.shape[0]
        _assert(
            Zp0.shape == (o, d),
            f"Zp0 has the wrong shape: {Zp0.shape} != ({o}, {d})",
            Slipmap,
        )

        if device is None:
            if isinstance(X, torch.Tensor):
                device = X.device
            else:
                scale = X.shape[0] * X.shape[1] * Zp0.shape[0]
                scale *= 1 if len(y.shape) < 2 else y.shape[1]
                cuda = scale > 500_000 and torch.cuda.is_available()
                device = torch.device("cuda" if cuda else "cpu")
        tensorargs = {"device": device, "dtype": dtype}

        self._X = torch.as_tensor(X, **tensorargs)
        if intercept:
            self._X = torch.cat((self._X, torch.ones_like(self._X[:, :1])), 1)
        n, m = self._X.shape

        self._Y = torch.as_tensor(y, **tensorargs)
        _assert(
            self._Y.shape[0] == n,
            f"The length of y must match X: {self._Y.shape[0]} != {n}",
            Slipmap,
        )
        if len(self._Y.shape) == 1:
            self._Y = self._Y[:, None]

        if Z0 is None:
            Z0 = self._X @ PCA_rotation(self._X, d)
            if Z0.shape[1] < d:
                _warn(
                    "The number of embedding dimensions is larger than the number of data dimensions",
                    Slipmap,
                )
                Z0fill = torch.zeros(size=[n, d - Z0.shape[1]], **tensorargs)
                Z0 = torch.cat((Z0, Z0fill), 1)
        else:
            Z0 = torch.as_tensor(Z0, **tensorargs)
            _assert(
                Z0.shape == (n, d),
                f"Z0 has the wrong shape: {Z0.shape} != ({n}, {d})",
                Slipmap,
            )
        self._Z = Z0
        self._Zp = torch.as_tensor(Zp0, **tensorargs)
        self._normalise(True)

        if callable(coefficients):
            coefficients = coefficients(self._X, self._Y)
        if Bp0 is None:
            Bp0 = global_model(
                X=self._X,
                Y=self._Y,
                local_model=self.local_model,
                local_loss=self.local_loss,
                coefficients=coefficients,
                lasso=self.lasso,
                ridge=self.ridge,
            )
            if not torch.all(torch.isfinite(Bp0)):
                _warn(
                    "Optimising a global model as initialisation resulted in non-finite values. Consider using stronger regularisation (increase ``lasso`` or ``ridge``).",
                    Slipmap,
                )
                Bp0 = torch.zeros_like(Bp0)
            Bp0 = Bp0.expand((o, coefficients)).clone()
        else:
            Bp0 = torch.as_tensor(Bp0, **tensorargs)
            _assert(
                len(Bp0.shape) > 1, "Bp0 must have more than one dimension", Slipmap
            )
            if Bp0.shape[0] == 1:
                Bp0 = Bp0.expand((o, coefficients)).clone()
            else:
                _assert(
                    Bp0.shape == (o, coefficients),
                    f"Bp0 has the wrong shape: {Bp0.shape} != ({o}, {coefficients})",
                    Slipmap,
                )
        self._Bp = Bp0
        self.metadata = Metadata(self)

    @property
    def n(self) -> int:
        # The number of data items
        return self._X.shape[0]

    @property
    def m(self) -> int:
        # The number of variables (including potential intercept)
        return self._X.shape[1]

    @property
    def o(self) -> int:
        # The number of target variables (i.e. the number of classes)
        return self._Y.shape[-1]

    @property
    def d(self) -> int:
        # The number of embedding dimensions
        return self._Z.shape[1]

    @property
    def p(self) -> int:
        # The number of prototypes
        return self._Zp.shape[0]

    @property
    def q(self) -> int:
        # The number of local model coefficients
        return self._Bp.shape[1]

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
            _assert(value >= 0, "radius must not be negative", Slipmap.radius)
            self._radius = value
            self._normalise(True)
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
    def local_model(self) -> CallableLike[ALocalModel.predict]:
        # Local model prediction function. Takes in X[n, m] and B[n, q], and returns Ytilde[n, n, o]
        return self._local_model

    @local_model.setter
    def local_model(self, value: CallableLike[ALocalModel.predict]):
        if self._local_model != value:
            _assert(
                callable(value), "local_model must be callable", Slisemap.local_model
            )
            self._local_model = value
            self._loss = None  # invalidate cached loss function

    @property
    def local_loss(self) -> CallableLike[ALocalModel.loss]:
        # Local model loss function. Takes in Ytilde[n, n, o] and Y[n, o] and returns L[n, n]
        return self._local_loss

    @local_loss.setter
    def local_loss(self, value: CallableLike[ALocalModel.loss]):
        if self._local_loss != value:
            _assert(callable(value), "local_loss must be callable", Slisemap.local_loss)
            self._local_loss = value
            self._loss = None  # invalidate cached loss function

    @property
    def regularisation(self) -> CallableLike[ALocalModel.regularisation]:
        # Regularisation function. Takes in X, Y, Bp, Z, and Ytilde and returns an additional loss scalar
        return self._regularisation

    @regularisation.setter
    def regularisation(self, value: CallableLike[ALocalModel.regularisation]):
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

    def get_Z(self, numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Get the Z matrix (the embedding for all data items).

        Args:
            numpy: Return the matrix as a numpy (True) or pytorch (False) matrix. Defaults to True.

        Returns:
            The Z matrix `[n, d]`.
        """
        self._normalise()
        return tonp(self._Z) if numpy else self._Z

    def get_B(self, numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Get the B matrix (the coefficients of the closest local model for all data items).

        Args:
            numpy: Return the matrix as a numpy (True) or pytorch (False) matrix. Defaults to True.

        Returns:
            The B matrix `[n, q]`.
        """
        B = self._Bp[self.get_closest(numpy=False)]
        return tonp(B) if numpy else B

    def get_Zp(self, numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Get the Zp matrix (the embedding for the prototypes).

        Args:
            numpy: Return the matrix as a numpy (True) or pytorch (False) matrix. Defaults to True.

        Returns:
            The Zp matrix `[p, d]`.
        """
        return tonp(self._Zp) if numpy else self._Zp

    def get_Bp(self, numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Get the Bp matrix (the local model coefficients for the prototypes).

        Args:
            numpy: Return the matrix as a numpy (True) or pytorch (False) matrix. Defaults to True.

        Returns:
            The Bp matrix `[p, q]`.
        """
        return tonp(self._Bp) if numpy else self._Bp

    def get_X(
        self, intercept: bool = True, numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """Get the data matrix

        Args:
            intercept: Include the intercept column (if ``self.intercept == True``). Defaults to True.
            numpy: Return the matrix as a numpy.ndarray instead of a torch.Tensor. Defaults to True.

        Returns:
            The X matrix `[n, m]`.
        """
        X = self._X if intercept or not self._intercept else self._X[:, :-1]
        return tonp(X) if numpy else X

    def get_Y(
        self, ravel: bool = False, numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """Get the target matrix

        Args:
            ravel: Remove the second dimension if it is singular (i.e. turn it into a vector). Defaults to False.
            numpy: Return the matrix as a numpy.ndarray instead of a torch.Tensor. Defaults to True.

        Returns:
            The Y matrix `[n, o]`.
        """
        Y = self._Y.ravel() if ravel else self._Y
        return tonp(Y) if numpy else Y

    def get_D(
        self,
        proto_rows: bool = True,
        proto_cols: bool = False,
        Z: Optional[torch.Tensor] = None,
        numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Get the embedding distance matrix

        Args:
            proto_rows: Calculate the distances with the prototype embeddings on the rows. Defaults to True.
            proto_cols: Calculate the distances with the prototype embeddings on the columns. Defaults to False.
            Z: Optional replacement for the training Z. Defaults to None.
            numpy: Return the matrix as a numpy (True) or pytorch (False) matrix. Defaults to True.

        Returns:
            The D matrix `[n or p, n or p]`.
        """
        if proto_rows and proto_cols:
            D = self._distance(self._Zp, self._Zp)
        else:
            Z = self.get_Z(numpy=False) if Z is None else Z
            if proto_rows:
                D = self._distance(self._Zp, Z)
            elif proto_cols:
                D = self._distance(Z, self._Zp)
            else:
                D = self._distance(Z, Z)
        return tonp(D) if numpy else D

    def get_W(
        self,
        proto_rows: bool = True,
        proto_cols: bool = False,
        Z: Optional[torch.Tensor] = None,
        numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Get the weight matrix

        Args:
            proto_rows: Calculate the weights with the prototype embeddings on the rows. Defaults to True.
            proto_cols: Calculate the weights with the prototype embeddings on the columns. Defaults to False.
            Z: Optional replacement for the training Z. Defaults to None.
            numpy: Return the matrix as a numpy.ndarray instead of a torch.Tensor. Defaults to True.

        Returns:
            The W matrix `[n or p, n or p]`.
        """
        D = self.get_D(numpy=False, proto_rows=proto_rows, proto_cols=proto_cols, Z=Z)
        W = self.kernel(D)
        return tonp(W) if numpy else W

    def get_L(
        self,
        X: Union[None, np.ndarray, torch.Tensor] = None,
        Y: Union[None, float, np.ndarray, torch.Tensor] = None,
        numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Get the loss matrix.

        Args:
            X: Optional replacement for the training X. Defaults to None.
            Y: Optional replacement for the training Y. Defaults to None.
            numpy: Return the matrix as a numpy (True) or pytorch (False) matrix. Defaults to True.

        Returns:
            The L matrix `[p, n]`.
        """
        X = self._as_new_X(X)
        Y = self._as_new_Y(Y, X.shape[0])
        L = self.local_loss(self.local_model(X, self._Bp), Y)
        return tonp(L) if numpy else L

    def get_closest(
        self, Z: Optional[torch.Tensor] = None, numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """Get the closest prototype for each data item.

        Args:
            Z: Optional replacement for the training Z. Defaults to None.
            numpy: Return the vector as a numpy (True) or pytorch (False) array. Defaults to True.

        Returns:
            Index vector `[n]`.
        """
        D = self.get_D(numpy=False, Z=Z, proto_rows=True, proto_cols=False)
        index = torch.argmin(D, 0)
        return tonp(index) if numpy else index

    def _as_new_X(
        self, X: Union[None, np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        if X is None:
            return self._X
        tensorargs = self.tensorargs
        X = torch.atleast_2d(torch.as_tensor(X, **tensorargs))
        if self._intercept and X.shape[1] == self.m - 1:
            X = torch.cat([X, torch.ones((X.shape[0], 1), **tensorargs)], 1)
        _assert(
            X.shape[1] == self.m,
            f"X has the wrong shape {X.shape} != {(X.shape[0], self.m)}",
            Slipmap._as_new_X,
        )
        return X

    def _as_new_Y(
        self, Y: Union[None, float, np.ndarray, torch.Tensor] = None, n: int = -1
    ) -> torch.Tensor:
        if Y is None:
            return self._Y
        Y = torch.as_tensor(Y, **self.tensorargs)
        if len(Y.shape) < 2:
            Y = torch.reshape(Y, (n, self.o))
        _assert(
            Y.shape[1] == self.o,
            f"Y has the wrong shape {Y.shape} != {(Y.shape[0], self.o)}",
            Slipmap._as_new_Y,
        )
        return Y

    @property
    def tensorargs(self) -> Dict[str, Any]:
        # When creating a new `torch.Tensor` add these keyword arguments to match the `dtype` and `device` of this Slisemap object.
        return dict(device=self._X.device, dtype=self._X.dtype)

    def cuda(self, **kwargs):
        """Move the tensors to CUDA memory (and run the calculations there).
        Note that this resets the random state.

        Args:
            **kwargs: Optional arguments to ``torch.Tensor.cuda``
        """
        X = self._X.cuda(**kwargs)
        self._X = X
        self._Y = self._Y.cuda(**kwargs)
        self._Z = self._Z.cuda(**kwargs)
        self._Bp = self._Bp.cuda(**kwargs)
        self._Zp = self._Zp.cuda(**kwargs)
        self._loss = None  # invalidate cached loss function

    def cpu(self, **kwargs):
        """Move the tensors to CPU memory (and run the calculations there).
        Note that this resets the random state.

        Args:
            **kwargs: Optional arguments to ``torch.Tensor.cpu``
        """
        X = self._X.cpu(**kwargs)
        self._X = X
        self._Y = self._Y.cpu(**kwargs)
        self._Z = self._Z.cpu(**kwargs)
        self._Bp = self._Bp.cpu(**kwargs)
        self._Zp = self._Zp.cpu(**kwargs)
        self._loss = None  # invalidate cached loss function

    def copy(self) -> "Slipmap":
        """Make a copy of this Slipmap that references as much of the same torch-data as possible.

        Returns:
            An almost shallow copy of this Slipmap object.
        """
        other = copy(self)  # Shallow copy!
        # Deep copy these:
        other._Z = other._Z.clone().detach()
        other._Bp = other._Bp.clone().detach()
        return other

    @classmethod
    def convert(cls, sm: Slisemap, keep_kernel: bool = False, **kwargs) -> "Slipmap":
        """Converts a Slisemap object into a Slipmap object.

        Args:
            sm: Slisemap object.
            keep_kernel: Use the kernel from the Slisemap object. Defaults to False.
            **kwargs: Other parameters forwarded to Slipmap.

        Returns:
            Slipmap object for the same data as the Slisemap object.
        """
        if keep_kernel:
            kwargs["kernel"] = sm.kernel
        kwargs.setdefault("radius", sm.radius)
        kwargs.setdefault("d", sm.d)
        kwargs.setdefault("lasso", sm.lasso)
        kwargs.setdefault("ridge", sm.ridge)
        kwargs.setdefault("intercept", sm.intercept)
        kwargs.setdefault("local_model", (sm.local_model, sm.local_loss, sm.q))
        kwargs.setdefault("distance", sm.distance)
        kwargs.setdefault("Z0", sm.get_Z(scale=False, rotate=True, numpy=False))
        kwargs.setdefault("jit", sm.jit)
        B = sm.get_B(False)
        sp = Slipmap(
            X=sm.get_X(numpy=False, intercept=False),
            y=sm.get_Y(numpy=False),
            Bp0=B[:1, ...],
            **kwargs,
            **sm.tensorargs,
        )
        D = sp.get_D(numpy=False, proto_rows=True, proto_cols=False)
        sp._Bp[...] = B[torch.argmin(D, 1), ...]
        return sp

    def into(self, keep_kernel: bool = False) -> Slisemap:
        """Converts a Slipmap object into a Slisemap object.

        Args:
            keep_kernel: Use the kernel from the Slipmap object. Defaults to False.

        Returns:
            Slisemap object for the same data as the Slipmap object.
        """
        kwargs = {}
        if keep_kernel:
            kwargs["kernel"] = self.kernel
        return Slisemap(
            X=self.get_X(numpy=False, intercept=False),
            y=self.get_Y(numpy=False),
            radius=self.radius,
            d=self.d,
            lasso=self.lasso,
            ridge=self.ridge,
            intercept=self.intercept,
            local_model=self.local_model,
            local_loss=self.local_loss,
            coefficients=self.q,
            distance=self.distance,
            B0=self.get_B(numpy=False),
            Z0=self.get_Z(numpy=False),
            jit=self.jit,
            **{**self.tensorargs, **kwargs},
        )

    def save(
        self,
        f: Union[str, PathLike, BinaryIO],
        any_extension: bool = False,
        compress: Union[bool, int] = True,
        **kwargs,
    ):
        """Save the Slipmap object to a file.

        This method uses ``torch.save`` (which uses ``pickle`` for the non-pytorch properties).
        This means that lambda-functions are not supported (unless a custom pickle module is used, see ``torch.save``).

        Note that the random state is not saved, only the initial seed (if set).

        The default file extension is ".sp".

        Args:
            f: Either a Path-like object or a (writable) File-like object.
            any_extension: Do not check the file extension. Defaults to False.
            **kwargs: Parameters forwarded to ``torch.save``.
        """
        if not any_extension and isinstance(f, (str, PathLike)):
            if not str(f).endswith(".sp"):
                _warn(
                    "When saving Slipmap objects, consider using the '.sp' extension for consistency.",
                    Slipmap.save,
                )
        loss = self._loss
        try:
            self.metadata.root = None
            self._Z = self._Z.detach()
            self._Bp = self._Bp.detach()
            self._loss = None
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

    @classmethod
    def load(
        cls,
        f: Union[str, PathLike, BinaryIO],
        device: Union[None, str, torch.device] = None,
        map_location: Optional[Any] = None,
        **kwargs,
    ) -> "Slipmap":
        """Load a Slipmap object from a file.

        This function uses ``torch.load``, so the tensors are restored to their previous devices.
        Use ``device="cpu"`` to avoid assuming that the same device exists.
        This is useful if the Slipmap object has been trained on a GPU, but the current computer lacks a GPU.

        Note that this is a classmethod, use it with: ``Slipmap.load(...)``.

        SAFETY: This function is based on `torch.load` which (by default) uses `pickle`.
        Do not use `Slipmap.load` on untrusted files, since `pickle` can run arbitrary Python code.

        Args:
            f: Either a Path-like object or a (readable) File-like object.
            device: Device to load the tensors to (or the original if None). Defaults to None.
            map_location: The same as `device` (this is the name used by `torch.load`). Defaults to None.
        Keyword Args:
            **kwargs: Parameters forwarded to `torch.load`.

        Returns:
            The loaded Slipmap object.
        """
        if device is None:
            device = map_location
        try:
            with lzma.open(f, "rb") as f2:
                sm = torch.load(f2, map_location=device, **kwargs)
        except lzma.LZMAError:
            sm: Slipmap = torch.load(f, map_location=device, **kwargs)
        return sm

    def __setstate__(self, data):
        # Handling loading of Slipmap objects from older versions
        if not isinstance(data, dict):
            data = next(d for d in data if isinstance(d, dict))
        for k, v in data.items():
            try:
                setattr(self, k, v)
            except AttributeError as e:
                _warn(e, Slipmap.__setstate__)
        if isinstance(getattr(self, "metadata", {}), Metadata):
            self.metadata.root = self
        else:
            self.metadata = Metadata(self, **getattr(self, "metadata", {}))
        if not hasattr(self, "_regularisation"):
            self._regularisation = ALocalModel.regularisation

    def _get_loss_fn(
        self, individual: bool = False
    ) -> Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        """Returns the Slipmap loss function.
        This function JITs and caches the loss function for efficiency.

        Args:
            individual: Make a loss function for individual losses. Defaults to False.
        Returns:
            Loss function `(X, Y, Z, Bp, Zp) -> loss`.
        """
        if not individual and self._loss is not None:
            return self._loss

        def loss(X, Y, Z, Bp, Zp):
            """Slipmap loss function.

            Args:
                X: Data matrix [n, m].
                Y: Target matrix [n, k].
                Z: Embedding matrix [n, d].
                Bp: Local models [o, p].
                Zp: Prototype embeddings [o, d].

            Returns:
                The loss value.
            """
            if self.radius > 0.0:
                epsilon = self.radius * torch.finfo(Z.dtype).eps / 2
                scale = torch.sqrt(torch.sum(Z**2) / Z.shape[0])
                Z = Z * (self.radius / (scale + epsilon))
            W = self.kernel(self.distance(Zp, Z))
            Ytilde = self.local_model(X, Bp)
            L = self.local_loss(Ytilde, Y)
            loss = torch.sum(W * L, dim=0 if individual else ())
            if not individual and self.lasso > 0.0:
                loss += self.lasso * torch.sum(torch.abs(Bp))
            if not individual and self.ridge > 0.0:
                loss += self.ridge * torch.sum(Bp**2)
            if not individual and self.radius > 0.0:
                loss += 1e-4 * (scale - self.radius) ** 2
            if not individual:
                loss += self.regularisation(self._X, self._Y, Z, self._Bp, Ytilde)
            return loss

        if individual:
            return loss
        if self._jit:
            # JITting the loss function improves the performance
            ex = (self._X[:1], self._Y[:1], self._Z[:1], self._Bp[:1], self._Zp[:1])
            loss = torch.jit.trace(loss, ex)
        # Caching the loss function
        self._loss = loss
        return self._loss

    def value(
        self, individual: bool = False, numpy: bool = True
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """Calculate the loss value.

        Args:
            individual: Give loss individual loss values for the data points. Defaults to False.

        Returns:
            The loss value(s).
        """
        loss = self._get_loss_fn(individual)
        loss = loss(X=self._X, Y=self._Y, Z=self._Z, Bp=self._Bp, Zp=self._Zp)
        if individual:
            return tonp(loss) if numpy else loss
        else:
            return loss.cpu().item() if numpy else loss

    def _normalise(self, both: bool = False):
        """Normalise Z."""
        if self.radius > 0:
            epsilon = self.radius * torch.finfo(self._Z.dtype).eps / 2
            if both:  # Normalise the prototype embedding
                scale = torch.sqrt(torch.sum(self._Zp**2) / self.p)
                proto_rad = self.radius * np.sqrt(2)
                if not np.allclose(proto_rad, scale.cpu().item()):
                    self._Zp = self._Zp * (proto_rad / (scale + epsilon))
            z_sum = torch.sum(self._Z**2, 1, True)
            scale = torch.sqrt(z_sum.mean())
            if not np.allclose(scale.cpu().item(), self.radius):
                self._Z *= self.radius / (scale + epsilon)

    def lbfgs(
        self,
        max_iter: int = 500,
        verbose: bool = False,
        *,
        only_B: bool = False,
        only_Z: bool = False,
        **kwargs,
    ) -> float:
        """Optimise Slipmap using LBFGS.

        Args:
            max_iter: Maximum number of LBFGS iterations. Defaults to 500.
            verbose: Print status messages. Defaults to False.
            only_B: Only optimise Bp. Defaults to False.
            only_Z: Only optimise Z. Defaults to False.
            **kwargs: Optional keyword arguments to LBFGS.

        Returns:
            The loss value.
        """
        if only_B == only_Z:
            only_B = only_Z = True
        Bp = self._Bp
        Z = self._Z
        if only_B:
            Bp = Bp.clone().requires_grad_(True)
        if only_Z:
            Z = Z.clone().requires_grad_(True)

        loss_ = self._get_loss_fn()
        loss_fn = lambda: loss_(self._X, self._Y, Z, Bp, self._Zp)
        pre_loss = loss_fn().cpu().detach().item()

        opt = [Bp] if not only_Z else ([Z] if not only_B else [Z, Bp])
        LBFGS(loss_fn, opt, max_iter=max_iter, verbose=verbose, **kwargs)
        post_loss = loss_fn().cpu().detach().item()

        if post_loss < pre_loss:
            if only_Z:
                self._Z = Z.detach()
                self._normalise()
            if only_B:
                self._Bp = Bp.detach()
            return post_loss
        else:
            if verbose:
                print("Slipmap.lbfgs: No improvement found")
            return pre_loss

    def escape(self, lerp: float = 1.0, outliers: bool = True, B_iter: int = 10):
        """Escape from a local optimum by moving each data item embedding towards the most suitable prototype embedding.

        Args:
            lerp: Linear interpolation between the old (0.0) and the new (1.0) embedding position. Defaults to 1.0.
            outliers: Check for and reset embeddings outside the prototype grid. Defaults to True.
            B_iter: Optimise B for `B_iter` number of LBFGS iterations. Set `B_iter=0` to disable. Defaults to 10.
        """
        if lerp <= 0.0:
            _warn("Escaping with `lerp <= 0` does nothing!", Slipmap.escape)
            return
        L = self.get_L(numpy=False)
        W = self.get_W(numpy=False, proto_rows=True, proto_cols=True)
        index = torch.argmin(W @ L, 0)
        if lerp >= 1.0:
            self._Z = self._Zp[index].clone()
        else:
            if outliers:  # Check for and reset outliers in the embedding
                scale = torch.sum(self._Z**2, 1)
                radius = torch.max(torch.sum(self._Zp**2, 1))
                if torch.any(scale >= radius):
                    self._Z[scale >= radius] = 0.0
            self._Z = (1.0 - lerp) * self._Z + lerp * self._Zp[index]
        self._normalise()
        if B_iter > 0:
            self.lbfgs(max_iter=B_iter, only_B=True)

    def optimize(
        self,
        patience: int = 2,
        max_escapes: int = 100,
        max_iter: int = 500,
        only_B: bool = False,
        lerp: float = 0.9,
        verbose: Literal[0, 1, 2] = 0,
        **kwargs,
    ) -> float:
        """Optimise Slipmap by alternating between [Slipmap.lbfgs][slisemap.slipmap.Slipmap.lbfgs] and [Slipmap.escape][slisemap.slipmap.Slipmap.escape] until convergence.

        Args:
            patience: Number of escapes without improvement before stopping. Defaults to 2.
            max_escapes: Maximum number of escapes. Defaults to 100.
            max_iter: Maximum number of LBFGS iterations per round. Defaults to 500.
            only_B: Only optimise the local models, not the embedding. Defaults to False.
            lerp: Linear interpolation when escaping (see `Slipmap.escape`). Defaults to 0.9.
            verbose: Print status messages (0: no, 1: some, 2: all). Defaults to 0.
            **kwargs: Optional keyword arguments to `Slipmap.lbfgs`.

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
            print(f"Slipmap.optimise LBFGS  {0:2d}: {loss[0]:.2f}")
        if only_B:
            return loss[0]
        cc = CheckConvergence(patience, max_escapes)
        while not cc.has_converged(loss, self.copy, verbose=verbose > 1):
            self.escape(lerp=lerp)
            loss[1] = self.value()
            if verbose:
                print(f"Slipmap.optimise Escape {i:2d}: {loss[1]:.2f}")
            loss[0] = self.lbfgs(
                max_iter, increase_tolerance=True, verbose=verbose > 1, **kwargs
            )
            if verbose:
                i += 1
                print(f"Slipmap.optimise LBFGS  {i:2d}: {loss[0]:.2f}")
        self._Z = cc.optimal._Z
        self._Bp = cc.optimal._Bp
        loss = self.lbfgs(
            max_iter * 2, increase_tolerance=False, verbose=verbose > 1, **kwargs
        )
        if verbose:
            print(f"Slipmap.optimise Final    : {loss:.2f}")
        return loss

    optimise = optimize

    def predict(
        self,
        Xnew: Union[np.ndarray, torch.Tensor],
        weighted: bool = True,
        numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Predict the outcome for new data items.
        This function uses the nearest neighbour in X space to find the embedding.
        Then the prediction is made with the local model (of the closest prototype).

        Args:
            Xnew: Data matrix.
            weighted: Use a weighted model instead of just the nearest. Defaults to True
            numpy: Return the predictions as a `numpy.ndarray` instead of `torch.Tensor`. Defaults to True.

        Returns:
            Predicted Y:s.
        """
        Xnew = self._as_new_X(Xnew)
        xnn = torch.cdist(Xnew, self._X).argmin(1)
        if weighted:
            Y = self.local_model(Xnew, self._Bp)
            D = self.get_D(True, False, numpy=False)[:, xnn]
            W = softmax_column_kernel(D)
            Y = torch.sum(W[..., None] * Y, 0)
        else:
            B = self.get_B(False)[xnn, :]
            Y = local_predict(Xnew, B, self.local_model)
        return tonp(Y) if numpy else Y

    def get_model_clusters(
        self,
        clusters: int,
        B: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster the local model coefficients using k-means (from scikit-learn).
        This method (with a fixed random seed) is used for plotting Slipmap solutions.

        Args:
            clusters: Number of clusters.
            B: B matrix. Defaults to `self.get_B()`.
            Z: Z matrix. Defaults to `self.get_Z()`.
            random_state: random_state for the KMeans clustering. Defaults to 42.

        Returns:
            labels: Vector of cluster labels.
            centres: Matrix of cluster centres.
        """
        B = B if B is not None else self.get_B()
        Z = Z if Z is not None else self.get_Z()
        km = KMeans(clusters, random_state=random_state).fit(B)
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
        **kwargs,
    ) -> Optional[Figure]:
        """Plot the Slipmap solution using seaborn.

        Args:
            title: Title of the plot. Defaults to "".
            clusters: Can be None (plot individual losses), an int (plot k-means clusters of Bp), or an array of known cluster id:s. Defaults to None.
            bars: If `clusters is not None`, plot the local models in a bar plot. If ``bar`` is an int then only plot the most influential variables. Defaults to True.
            jitter: Add random (normal) noise to the embedding, or a matrix with pre-generated noise matching Z. Defaults to 0.0.
            show: Show the plot. Defaults to True.
            bar: Alternative spelling for `bars`. Defaults to None.
            **kwargs: Additional arguments to [plot_solution][slisemap.plot.plot_solution] and `plt.subplots`.

        Returns:
            Matplotlib figure if `show=False`.
        """
        if bar is not None:
            bars = bar
        B = self.get_B()
        Z = self.get_Z()
        if clusters is None:
            loss = tonp(self.local_loss(self.predict(self._X, numpy=False), self._Y))
            clusters = None
            centers = None
        else:
            loss = None
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
            coefficients=self.metadata.get_coefficients(),
            dimensions=self.metadata.get_dimensions(long=True),
            title=title,
            bars=bars,
            jitter=jitter,
            **kwargs,
        )
        plot_prototypes(self.get_Zp(), fig.axes[0])
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
        legend_inside: bool = True,
        show: bool = True,
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
            legend_inside: Move the legend inside the grid (if there is an empty cell). Defaults to True.
            show: Show the plot. Defaults to True.
        Keyword Args:
            **kwargs: Additional arguments to `seaborn.relplot`.

        Returns:
            `seaborn.FacetGrid` if `show=False`.
        """
        if index is None:
            _assert(
                X is not None and Y is not None,
                "Either index or X and Y must be given",
                Slipmap.plot_position,
            )
            L = self.get_L(X=X, Y=Y)
        else:
            if isinstance(index, int):
                index = [index]
            L = self.get_L()[:, index]
        g = plot_position(
            Z=self.get_Zp(),
            L=L,
            Zs=self.get_Z()[index, :] if index is not None else None,
            dimensions=self.metadata.get_dimensions(long=True),
            title=title,
            jitter=jitter,
            legend_inside=legend_inside,
            marker_size=6.0,
            **kwargs,
        )
        # plot_prototypes(self.get_Zp(), *g.axes.flat)
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
            **kwargs: Additional arguments to `seaborn.relplot` or `seaborn.scatterplot`.

        Returns:
            `seaborn.FacetGrid` if `show=False`.
        """
        X = self.get_X(intercept=False)
        Y = self.get_Y()
        if unscale:
            X = self.metadata.unscale_X(X)
            Y = self.metadata.unscale_Y(Y)
        loss = tonp(self.local_loss(self.predict(self._X, numpy=False), self._Y))
        if isinstance(clusters, int):
            clusters, _ = self.get_model_clusters(clusters)
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
        if scatter:
            plot_prototypes(self.get_Zp(), *g.axes.flat)
        if show:
            plt.show()
        else:
            return g
