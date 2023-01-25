"""
This module contains alternative escape heuristics.
"""

from typing import Callable, Optional, Tuple

import numpy as np
import torch

from slisemap.utils import _assert


def escape_neighbourhood(
    X: torch.Tensor,
    Y: torch.Tensor,
    B: torch.Tensor,
    Z: torch.Tensor,
    local_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    local_loss: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    kernel: Callable[[torch.Tensor], torch.Tensor],
    radius: float = 3.5,
    force_move: bool = False,
    **_,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Try to escape a local optimum by moving the data items.
    Move the data items to the neighbourhoods (embedding and local model) best suited for them.
    This is done by finding another item (in the optimal neighbourhood) and copying its values for Z and B.

    Args:
        X: Data matrix.
        Y: Target matrix.
        B: Local models.
        Z: Embedding matrix.
        local_model: Prediction function for the local models.
        local_loss: Loss function for the local models.
        distance: Embedding distance function.
        kernel: Kernel for embedding distances.
        radius: For enforcing the radius of Z. Defaults to 3.5.
        force_move: Do not allow the items to pair with themselves. Defaults to True.

    Returns:
        B: Escaped `B`.
        Z: Escaped `Z`.
    """
    L = local_loss(local_model(X, B), Y, B)
    if radius > 0:
        Z2 = Z * (radius / (torch.sqrt(torch.sum(Z**2) / Z.shape[0]) + 1e-8))
        D = distance(Z2, Z2)
    else:
        D = distance(Z, Z)
    W = kernel(D)
    # K = torch.zeros_like(L)
    # for i in range(L.shape[1]):
    #     K[:, i] = torch.sum(W * L[:, i].ravel()[None, :], 1)
    # K = torch.sum(W[:, :, None] * L[None, :, :], 1)
    K = W @ L
    if force_move:
        _assert(
            K.shape[0] == K.shape[1],
            "force_move only works if (X, Y) corresponds to (B, Z)",
            escape_neighbourhood,
        )
        K.fill_diagonal_(np.inf)
    index = torch.argmin(K, 0)
    return B.detach()[index].clone(), Z.detach()[index].clone()


def escape_greedy(
    X: torch.Tensor,
    Y: torch.Tensor,
    B: torch.Tensor,
    Z: torch.Tensor,
    local_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    local_loss: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    kernel: Callable[[torch.Tensor], torch.Tensor],
    radius: float = 3.5,
    force_move: bool = False,
    **_,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Try to escape a local optimum by moving the data items.
    Move the data items to a locations with optimal local models.
    This is done by finding another item (with an optimal local model) and copying its values for Z and B.

    Args:
        X: Data matrix.
        Y: Target matrix.
        B: Local models.
        Z: Embedding matrix.
        local_model: Prediction function for the local models.
        local_loss: Loss function for the local models.
        distance: Embedding distance function.
        kernel: Kernel for embedding distances.
        radius: For enforcing the radius of Z. Defaults to 3.5.
        force_move: Do not allow the items to pair with themselves. Defaults to True.

    Returns:
        B: Escaped `B`.
        Z: Escaped `Z`.
    """
    L = local_loss(local_model(X, B), Y, B)
    if force_move:
        _assert(
            L.shape[0] == L.shape[1],
            "force_move only works if (X, Y) corresponds to (B, Z)",
            escape_greedy,
        )
        L.fill_diagonal_(np.inf)
    index = torch.argmin(L, 0)
    return B.detach()[index].clone(), Z.detach()[index].clone()


def escape_combined(
    X: torch.Tensor,
    Y: torch.Tensor,
    B: torch.Tensor,
    Z: torch.Tensor,
    local_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    local_loss: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    kernel: Callable[[torch.Tensor], torch.Tensor],
    radius: float = 3.5,
    force_move: bool = False,
    **_,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Try to escape a local optimum by moving the data items.
    Move the data items to the neighbourhoods (embedding and local model) best suited for them.
    This is done by finding another item (in the optimal neighbourhood) and copying its values for Z and B.

    This is a combination of escape_neighbourhood and escape_greedy.

    Args:
        X: Data matrix.
        Y: Target matrix.
        B: Local models.
        Z: Embedding matrix.
        local_model: Prediction function for the local models.
        local_loss: Loss function for the local models.
        distance: Embedding distance function.
        kernel: Kernel for embedding distances.
        radius: For enforcing the radius of Z. Defaults to 3.5.
        force_move: Do not allow the items to pair with themselves. Defaults to True.

    Returns:
        B: Escaped `B`.
        Z: Escaped `Z`.
    """
    L = local_loss(local_model(X, B), Y, B)
    if radius > 0:
        Z2 = Z * (radius / (torch.sqrt(torch.sum(Z**2) / Z.shape[0]) + 1e-8))
        D = distance(Z2, Z2)
    else:
        D = distance(Z, Z)
    W = kernel(D)
    K = W @ L + L
    if force_move:
        _assert(
            K.shape[0] == K.shape[1],
            "force_move only works if (X, Y) corresponds to (B, Z)",
            escape_combined,
        )
        K.fill_diagonal_(np.inf)
    index = torch.argmin(K, 0)
    return B.detach()[index].clone(), Z.detach()[index].clone()


def escape_marginal(
    X: torch.Tensor,
    Y: torch.Tensor,
    B: torch.Tensor,
    Z: torch.Tensor,
    local_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    local_loss: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    kernel: Callable[[torch.Tensor], torch.Tensor],
    radius: float = 3.5,
    force_move: bool = False,
    Xold: Optional[torch.Tensor] = None,
    Yold: Optional[torch.Tensor] = None,
    jit: bool = True,
    **_,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Try to escape a local optimum by moving the data items.
    Move the data items to locations with optimal marginal losses.
    This is done by finding another item (where the marginal loss is optimal) and copying its values for Z and B.

    This might produce better results than `escape_neighbourhood`, but is really slow.

    Args:
        X: Data matrix.
        Y: Target matrix.
        B: Local models.
        Z: Embedding matrix.
        local_model: Prediction function for the local models.
        local_loss: Loss function for the local models.
        distance: Embedding distance function.
        kernel: Kernel for embedding distances.
        radius: For enforcing the radius of Z. Defaults to 3.5.
        force_move: Do not allow the items to pair with themselves. Defaults to True.
        jit: Just-In-Time compile the loss function. Defaults to True.

    Returns:
        B: Escaped `B`.
        Z: Escaped `Z`.
    """
    from slisemap.slisemap import make_marginal_loss

    _assert(
        not force_move or X.shape[0] == Z.shape[0],
        "force_move only works if (X, Y) corresponds to (B, Z)",
        escape_marginal,
    )
    _assert(
        Xold is not None or X.shape[0] == Z.shape[0],
        "(Xold Yold) is required if (X, Y) does not correspond to (B, Z)",
        escape_marginal,
    )
    if radius > 0:
        Z2 = Z * (radius / (torch.sqrt(torch.sum(Z**2) / Z.shape[0]) + 1e-8))
    else:
        Z2 = Z
    lf, set_new = make_marginal_loss(
        X=X if Xold is None else Xold,
        Y=Y if Yold is None else Yold,
        B=B,
        Z=Z2,
        Xnew=X[:1],
        Ynew=Y[:1],
        local_model=local_model,
        local_loss=local_loss,
        distance=distance,
        kernel=kernel,
        radius=0.0,
        lasso=0.0,
        ridge=0.0,
        jit=jit and (X.shape[0] > 10 if Xold is None else Xold.shape[0] > 10),
    )
    index = []
    for i in range(X.shape[0]):
        set_new(X[i : i + 1], Y[i : i + 1])
        index.append(i)
        best = np.inf
        for j in range(Z.shape[0]):
            if force_move and i == j:
                continue
            l = lf(B[j : j + 1], Z2[j : j + 1])
            if l < best:
                best = l
                index[-1] = j
    return B.detach()[index].clone(), Z.detach()[index].clone()
