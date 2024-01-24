"""
This module contains the SLISEMAP loss functions.
"""

from typing import Callable, Optional, Tuple

import torch

from slisemap.utils import _warn, _deprecated


def softmax_row_kernel(D: torch.Tensor) -> torch.Tensor:
    """Kernel function that applies softmax on the rows.

    Args:
        D: Distance matrix.

    Returns:
        Weight matrix.
    """
    return torch.softmax(-D, 1)


def softmax_column_kernel(D: torch.Tensor) -> torch.Tensor:
    """Kernel function that applies softmax on the columns.

    Args:
        D: Distance matrix.

    Returns:
        Weight matrix.
    """
    return torch.softmax(-D, 0)


def squared_distance(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Distance function that returns the squared euclidean distances.

    Args:
        A: The first matrix [n1, d].
        B: The second matrix [n2, d].

    Returns:
        Distance matrix [n1, n2].
    """
    return torch.sum((A[:, None, ...] - B[None, ...]) ** 2, -1)


def softmax_kernel(D: torch.Tensor) -> torch.Tensor:
    _deprecated(softmax_kernel, softmax_row_kernel)
    return softmax_row_kernel(D)


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
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,], torch.Tensor,]:
    """Create a loss function for Slisemap to optimise

    Args:
        local_model: Prediction function for the local models.
        local_loss: Loss function for the local models.
        distance: Embedding distance function. Defaults to `torch.cdist` (Euclidean distance).
        kernel: Kernel for embedding distances, Defaults to `softmax_kernel`.
        radius: For enforcing the radius of Z. Defaults to 3.5.
        lasso: Lasso-regularisation coefficient for B ($\\lambda_{lasso} * ||B||_1$). Defaults to 0.0.
        ridge: Ridge-regularisation coefficient for B ($\\lambda_{ridge} * ||B||_2$). Defaults to 0.0.
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
    """Create a loss for adding new points with Slisemap.

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
        lasso: Lasso-regularisation coefficient for B ($\\lambda_{lasso} * ||B||_1$). Defaults to 0.0.
        ridge: Ridge-regularisation coefficient for B ($\\lambda_{ridge} * ||B||_2$). Defaults to 0.0.
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

    def set_new(Xnew: torch.Tensor, Ynew: torch.Tensor):
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
            B: New local models [n_new, p].
            Z: New embedding matrix [n_new, d].

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
