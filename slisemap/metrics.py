"""
    This module contains functions that can be used to evaluate SLISEMAP solutions.
    The functions take a solution (plus other arguments) and returns a single float.
    This float should either be minimised or maximised for best results (see individual functions).
"""

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from slisemap.slisemap import Slisemap
from slisemap.utils import _tonp


def _non_crashing_median(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return np.nan
    else:
        return torch.median(x).cpu().item()


def nanmean(x) -> float:
    mask = np.isfinite(x)
    if np.all(mask):
        return np.mean(x)
    elif not np.any(mask):
        return np.nan
    else:
        return np.mean(x[mask])


def euclidean_nearest_neighbours(
    D: torch.Tensor, index: int, k: Union[int, float] = 0.1, include_self: bool = True
) -> torch.LongTensor:
    """Find the k nearest neighbours using euclidean distance

    Args:
        D (torch.Tensor): Distance matrix.
        index (int): The item (row), for which to find neighbours.
        k (int, optional): The number of neighbours to find. Defaults to 0.1.
        include_self (bool, optional): include the item in its' neighbourhood. Defaults to True.

    Returns:
        torch.LongTensor: Vector of indices for the neighbours.
    """
    if isinstance(k, float) and 0 < k <= 1:
        k = int(k * D.shape[0])
    if include_self:
        return torch.argsort(D[index])[:k]
    else:
        dist = D[index].clone()
        if not include_self:
            dist[index] = np.inf
            if k == D.shape[0]:
                k -= 1
        return torch.argsort(dist)[:k]


def kernel_neighbours(
    D: torch.Tensor, index: int, epsilon: float = 1.0, include_self: bool = True
) -> torch.LongTensor:
    """Find the neighbours using a softmax kernel

    Args:
        D (torch.Tensor): Distance matrix.
        index (int): The item for which we want to find neighbours.
        epsilon (float, optional): Treshold for selecting the neighbourhood (will be divided by `n`). Defaults to 1.0.
        include_self (bool, optional): include the item in its' neighbourhood. Defaults to True.

    Returns:
        torch.LongTensor: Vector of indices for the neighbours.
    """
    K = torch.nn.functional.softmax(-D[index], 0)
    epsilon2 = epsilon / K.numel()
    if include_self:
        return torch.where(K >= epsilon2)[0]
    else:
        mask = K >= epsilon2
        mask[index] = False
        return torch.where(mask)[0]


def cluster_neighbours(
    D: torch.Tensor,
    index: int,
    clusters: torch.LongTensor,
    include_self: bool = True,
) -> torch.LongTensor:
    """Find the neighbours with given clusters

    Args:
        D (torch.Tensor): Distance matrix (ignored).
        index (int): The item for which we want to find neighbours.
        clusters (torch.LongTensor): Cluster id:s for the data items.
        include_self (bool, optional): include the item in its' neighbourhood. Defaults to True.

    Returns:
        torch.LongTensor: Vector of indices for the neighbours.
    """
    if include_self:
        return torch.where(clusters == clusters[index])[0]
    else:
        mask = clusters == clusters[index]
        mask[index] = False
        return torch.where(mask)[0]


def radius_neighbours(
    D: torch.Tensor,
    index: int,
    radius: Optional[float] = None,
    quantile: float = 0.2,
    include_self: bool = True,
) -> torch.LongTensor:
    """Find the neighbours within a radius

    Args:
        D (torch.Tensor): Distance matrix (ignored).
        index (int): The item for which we want to find neighbours.
        radius (Optional[float], optional): The radius of the neighbourhood. Defaults to None.
        quantile (float, optional): If radius is None then radius is set to the quantile of D. Defaults to 0.2.
        include_self (bool, optional): include the item in its' neighbourhood. Defaults to True.

    Returns:
        torch.LongTensor: Vector of indices for the neighbours.
    """
    if radius is None:
        radius = torch.quantile(D, quantile)
    if include_self:
        return torch.where(D[index] <= radius)[0]
    else:
        mask = D[index] <= radius
        mask[index] = False
        return torch.where(mask)[0]


def _get_neighbours(
    sm: Union[Slisemap, torch.Tensor],
    neighbours: Union[
        None,
        np.ndarray,
        torch.LongTensor,
        Callable[[torch.Tensor, int], torch.LongTensor],
    ],
    full_if_none: bool = False,
    **kwargs,
) -> Callable[[int], torch.LongTensor]:
    """Create a function that takes the index of an item and returns the indices of its neighbours.

    The neighbours parameter is primarily one of:
     - euclidean_nearest_neighbours
     - kernel_neighbours
     - cluster_neighbours
     - radius_neighbours

    Args:
        sm (Union[Slisemap, torch.Tensor]): Trained Slisemap solution or an embedding vector (like Slisemap.Z).
        neighbours (Union[ None, np.ndarray, torch.LongTensor, Callable[[torch.Tensor, int], torch.LongTensor], ]): Either None (return self), a vector of cluster id:s (take neighbours from the same cluter), or a function that gives neighbours (that takes a distance matrix and an index).
        full_if_none (bool, optional): If `neighbours` is None, return the whole dataset. Defaults to False.
        **kwargs: Arguments passed on to `neighbours` (if it is a function).

    Returns:
        Callable[[int], torch.LongTensor]: Function that takes an index and returns neighbour indices.
    """
    if neighbours is None:
        if full_if_none:
            n = sm.n if isinstance(sm, Slisemap) else sm.shape[0]
            return lambda i: torch.arange(n)
        else:
            return lambda i: torch.LongTensor([i])
    if callable(neighbours):
        D = sm.get_D(numpy=False) if isinstance(sm, Slisemap) else torch.cdist(sm, sm)
        return lambda i: neighbours(D, i, **kwargs)
    else:
        neighbours = torch.as_tensor(neighbours)
        return lambda i: cluster_neighbours(None, i, neighbours, **kwargs)


def slisemap_loss(sm: Slisemap) -> float:
    """Evaluate a SLISEMAP solution by calculating the loss.

    Smaller is better.

    Args:
        sm (Slisemap): Trained Slisemap solution.

    Returns:
        float: The loss value.
    """
    return sm.value()


def slisemap_entropy(sm: Slisemap) -> float:
    """Evaluate a SLISEMAP solution by calculating the entropy.

    Args:
        sm (Slisemap): Trained Slisemap solution.

    Returns:
        float: The loss value.
    """
    return sm.entropy()


def fidelity(
    sm: Slisemap,
    neighbours: Union[
        None,
        np.ndarray,
        torch.LongTensor,
        Callable[[torch.Tensor, int], torch.LongTensor],
    ] = None,
    **kwargs,
) -> float:
    """Evaluate a SLISEMAP solution by calculating the fidelity (loss per item/neighbourhood).

    Smaller is better.

    Args:
        sm (Slisemap): Trained Slisemap solution.
        neighbours (Union[ None, np.ndarray, torch.LongTensor, Callable[[torch.Tensor, int], torch.LongTensor], ]): Either None (only corresponding local model), a vector of cluster id:s, or a function that gives neighbours (see `_get_neighbours`).
        **kwargs: Arguments passed on to `neighbours` (if it is a function).

    Returns:
        float: The mean loss.
    """
    neighbours = _get_neighbours(sm, neighbours, full_if_none=False, **kwargs)
    results = np.zeros(sm.n)
    L = sm.get_L(numpy=False)
    for i in range(len(results)):
        ni = neighbours(i)
        if ni.numel() == 0:
            results[i] = np.nan
        else:
            results[i] = torch.mean(L[i, ni]).cpu().detach().item()
    return nanmean(results)


def coverage(
    sm: Slisemap,
    max_loss: float,
    neighbours: Union[
        None,
        np.ndarray,
        torch.LongTensor,
        Callable[[torch.Tensor, int], torch.LongTensor],
    ] = None,
    **kwargs,
) -> float:
    """Evaluate a SLISEMAP solution by calculating the coverage.

    Larger is better.

    Args:
        sm (Slisemap): Trained Slisemap solution.
        max_loss (float): Maximum tolerable loss.
        neighbours (Union[ None, np.ndarray, torch.LongTensor, Callable[[torch.Tensor, int], torch.LongTensor], ], optional): Either None (all), a vector of cluster id:s, or a function that gives neighbours (see `_get_neighbours`).
        **kwargs: Arguments passed on to `neighbours` (if it is a function).

    Returns:
        float: The mean fraction of items within the error bound.
    """
    if torch.all(torch.isnan(sm.B.sum(1))).cpu().item():
        return np.nan
    neighbours = _get_neighbours(sm, neighbours, full_if_none=True, **kwargs)
    results = np.zeros(sm.n)
    L = sm.get_L(numpy=False)
    for i in range(len(results)):
        ni = neighbours(i)
        if ni.numel() == 0:
            results[i] = np.nan
        else:
            results[i] = np.mean(_tonp(L[i, ni] < max_loss))
    return nanmean(results)


def median_loss(
    sm: Slisemap,
    neighbours: Union[
        None,
        np.ndarray,
        torch.LongTensor,
        Callable[[torch.Tensor, int], torch.LongTensor],
    ] = None,
    **kwargs,
) -> float:
    """Evaluate a SLISEMAP solution by calculating the median loss.

    Smaller is better.

    Args:
        sm (Slisemap): Trained Slisemap solution.
        neighbours (Union[ None, np.ndarray, torch.LongTensor, Callable[[torch.Tensor, int], torch.LongTensor], ], optional): Either None (all), a vector of cluster id:s, or a function that gives neighbours (see `_get_neighbours`).
        **kwargs: Arguments passed on to `neighbours` (if it is a function).

    Returns:
        float: The mean median loss.
    """
    neighbours = _get_neighbours(sm, neighbours, full_if_none=True, **kwargs)
    results = np.zeros(sm.n)
    L = sm.get_L(numpy=False)
    for i in range(len(results)):
        ni = neighbours(i)
        if ni.numel() == 0:
            results[i] = np.nan
        else:
            results[i] = _non_crashing_median(L[i, ni])
    return nanmean(results)


def coherence(
    sm: Slisemap,
    neighbours: Union[
        None,
        np.ndarray,
        torch.LongTensor,
        Callable[[torch.Tensor, int], torch.LongTensor],
    ] = None,
    **kwargs,
) -> float:
    """Evaluate a SLISEMAP solution by calculating the coherence (max change in prediction divided by the change in variable values).

    Smaller is better.

    Args:
        sm (Slisemap): Trained Slisemap solution.
        neighbours (Union[ None, np.ndarray, torch.LongTensor, Callable[[torch.Tensor, int], torch.LongTensor], ], optional): Either None (all), a vector of cluster id:s, or a function that gives neighbours (see `_get_neighbours`).
        **kwargs: Arguments passed on to `neighbours` (if it is a function).

    Returns:
        float: The mean coherence.
    """
    neighbours = _get_neighbours(
        sm, neighbours, full_if_none=True, include_self=False, **kwargs
    )
    results = np.zeros(sm.n)
    P = sm.local_model(sm.X, sm.B)
    for i in range(len(results)):
        ni = neighbours(i)
        if ni.numel() == 0:
            results[i] = np.nan
        else:
            dP = torch.sum((P[None, i, i] - P[ni, i] - P[i, ni] + P[ni, ni]) ** 2, 1)
            dX = torch.sum((sm.X[None, i, :] - sm.X[ni, :]) ** 2, 1) + 1e-8
            results[i] = torch.sqrt(torch.max(dP / dX))
    return nanmean(results)


def stability(
    sm: Slisemap,
    neighbours: Union[
        None,
        np.ndarray,
        torch.LongTensor,
        Callable[[torch.Tensor, int], torch.LongTensor],
    ] = None,
    **kwargs,
) -> float:
    """Evaluate a SLISEMAP solution by calculating the stability (max change in the local model divided by the change in variable values).

    Smaller is better.

    Args:
        sm (Slisemap): Trained Slisemap solution.
        neighbours (Union[ None, np.ndarray, torch.LongTensor, Callable[[torch.Tensor, int], torch.LongTensor], ], optional): Either None (all), a vector of cluster id:s, or a function that gives neighbours (see `_get_neighbours`).
        **kwargs: Arguments passed on to `neighbours` (if it is a function).

    Returns:
        float: The mean stability.
    """
    neighbours = _get_neighbours(
        sm, neighbours, full_if_none=True, include_self=False, **kwargs
    )
    results = np.zeros(sm.n)
    for i in range(len(results)):
        ni = neighbours(i)
        if ni.numel() == 0:
            results[i] = np.nan
        else:
            dB = torch.sum((sm.B[None, i, :] - sm.B[ni, :]) ** 2, 1)
            dX = torch.sum((sm.X[None, i, :] - sm.X[ni, :]) ** 2, 1) + 1e-8
            results[i] = torch.sqrt(torch.max(dB / dX))
    return nanmean(results)


def kmeans_matching(
    sm: Slisemap, clusters: Union[int, Sequence[int]] = range(2, 10)
) -> float:
    """Evaluate SLISE by measuring how well clusters in Z and B overlap (using kmeans to find the clusters).
    The overlap is measured by finding the best matching clusters and dividing the size of intersect by the size of the union of each cluster pair.

    Larger is better.

    Args:
        sm (Slisemap): Trained Slisemap solution.
        clusters (Union[int, Sequence[int]], optional): The number of clusters. Defaults to range(2, 10).

    Returns:
        float: mean cluster matching.
    """
    from sklearn.cluster import KMeans
    from scipy.optimize import linear_sum_assignment

    Z = sm.get_Z()
    B = sm.get_B()
    if np.all(np.var(Z, 0) < 1e-8) or np.all(np.var(B, 0) < 1e-8):
        return np.nan  # Do not compare singular clusters
    if not np.all(np.isfinite(Z)) or not np.all(np.isfinite(B)):
        return np.nan
    if isinstance(clusters, int):
        clusters = range(clusters, clusters + 1)
    results = []
    for k in clusters:
        cl_B = KMeans(n_clusters=k, init="k-means++").fit(B)
        cl_Z = KMeans(n_clusters=k, init="k-means++").fit(Z)
        sets_B = [set(np.where(cl_B.labels_ == i)[0]) for i in range(k)]
        sets_Z = [set(np.where(cl_Z.labels_ == i)[0]) for i in range(k)]
        mat = np.zeros((k, k))
        for i, sB in enumerate(sets_B):
            for j, sZ in enumerate(sets_Z):
                mat[i, j] = len(sB.intersection(sZ)) / (len(sB.union(sZ)) + 1e-8)
        # Hungarian algorithm to find the best match between the clusterings
        rows, cols = linear_sum_assignment(mat, maximize=True)
        results.append(mat[rows, cols].mean())
    return nanmean(results)


def cluster_purity(
    sm: Union[Slisemap, torch.Tensor, np.ndarray],
    clusters: Union[np.ndarray, torch.LongTensor],
) -> float:
    """Evaluate a SLISEMAP solution by calculating how many items in the same cluster are neighbours.

    Larger is better.

    Args:
        sm (Union[Slisemap, torch.Tensor, np.ndarray]): Trained Slisemap solution _or_ embedding matrix.
        clusters (Union[np.ndarray, torch.LongTensor]): Cluster ids.

    Returns:
        float: The mean number of items sharing cluster that are neighbours.
    """
    if isinstance(sm, Slisemap):
        Z = sm.get_Z(numpy=False)
    elif isinstance(sm, torch.Tensor):
        Z = sm
    else:
        Z = torch.as_tensor(sm)
    if isinstance(clusters, np.ndarray):
        clusters = torch.as_tensor(clusters, device=Z.device)
    res = np.zeros(Z.shape[0])
    D = torch.cdist(Z, Z)
    for i in range(len(res)):
        mask = clusters[i] == clusters
        knn = euclidean_nearest_neighbours(D, i, mask.sum())
        res[i] = torch.sum(mask[knn]) / knn.shape[0]
    return nanmean(res)


def kernel_purity(
    sm: Slisemap,
    clusters: Union[np.ndarray, torch.LongTensor],
    epsilon: float = 1.0,
    losses: bool = False,
) -> float:
    """Evaluate a SLISEMAP solution by calculating how many neighbours are in the same cluster

    Larger is better.

    Args:
        sm (Slisemap): Trained Slisemap solution.
        clusters (Union[np.ndarray, torch.LongTensor]): Cluster ids.
        epsilon (float, optional): Treshold for being a neighbour (`softmax(D) < epsilon/n`). Defaults to 1.0.
        losses (bool, optional): Use losses instead of embedding distances. Defaults to False.

    Returns:
        float: The mean number of neighbours that are in the same cluster.
    """
    if isinstance(clusters, np.ndarray):
        clusters = torch.tensor(clusters)
    res = np.zeros(sm.n)
    if losses:
        D = sm.get_L(numpy=False)
    else:
        D = sm.get_D(numpy=False)
    for i in range(len(res)):
        mask = clusters[i] == clusters
        neig = kernel_neighbours(D, i, epsilon)
        res[i] = torch.sum(mask[neig]) / neig.numel()
    return nanmean(res)


def recall(sm: Slisemap, epsilon_D: float = 1.0, epsilon_L: float = 1.0) -> float:
    """Evaluate a SLISEMAP solution by calculating the recall.

    We define recall as the intersection between the loss and embedding neighbourhoods divided by the loss neighbourhood.

    Larger is better.

    Args:
        sm (Slisemap): Trained Slisemap solution.
        epsilon_D (float, optional): Treshold for being an embedding neighbour (`softmax(D) < epsilon/n`). Defaults to 1.0.
        epsilon_L (float, optional): Treshold for being a loss neighbour (`softmax(L) < epsilon/n`). Defaults to 1.0.

    Returns:
        float: The mean recall.
    """
    res = np.zeros(sm.n)
    D = sm.get_D(numpy=False)
    L = sm.get_L(numpy=False)
    for i in range(len(res)):
        nL = kernel_neighbours(L, i, epsilon_L)
        if nL.numel() == 0:
            res[i] = np.nan
        else:
            nD = kernel_neighbours(D, i, epsilon_D)
            inter = np.intersect1d(_tonp(nD), _tonp(nL), True)
            res[i] = len(inter) / nL.numel()
    return nanmean(res)


def precision(sm: Slisemap, epsilon_D: float = 1.0, epsilon_L: float = 1.0) -> float:
    """Evaluate a SLISEMAP solution by calculating the recall.

    We define recall as the intersection between the loss and embedding neighbourhoods divided by the embedding neighbourhood.

    Larger is better.

    Args:
        sm (Slisemap): Trained Slisemap solution.
        epsilon_D (float, optional): Treshold for being an embedding neighbour (`softmax(D) < epsilon/n`). Defaults to 1.0.
        epsilon_L (float, optional): Treshold for being a loss neighbour (`softmax(L) < epsilon/n`). Defaults to 1.0.

    Returns:
        float: The mean precision.
    """
    res = np.zeros(sm.n)
    D = sm.get_D(numpy=False)
    L = sm.get_L(numpy=False)
    for i in range(len(res)):
        nD = kernel_neighbours(D, i, epsilon_D)
        if nD.numel() == 0:
            res[i] = np.nan
        else:
            nL = kernel_neighbours(L, i, epsilon_L)
            inter = np.intersect1d(_tonp(nD), _tonp(nL), True)
            res[i] = len(inter) / nD.numel()
    return nanmean(res)


def relevance(sm: Slisemap, pred_fn: Callable, change: float) -> float:
    """Evaluate a SLISEMAP solution by calculating the relevance.

    Smaller is better.

    TODO: This does not (currently) work for multi-class predictions

    Args:
        sm (Slisemap): Trained Slisemap solution.
        pred_fn (Callable): Function that gives y:s for new x:s (the "black box model").
        change (float): How much should the prediction change?

    Returns:
        float: The mean number of mutated variables required to cause a large enough change in the prediction.
    """
    rel = np.ones(sm.n) * sm.m
    for i in range(len(rel)):
        b = sm.B[i, :]
        x = sm.X[i, :]
        y = sm.Y[i, 0]
        xmax = torch.max(sm.X, 0)[0]
        xmin = torch.min(sm.X, 0)[0]
        xinc = torch.where(b > 0, xmax, xmin)
        xdec = torch.where(b < 0, xmax, xmin)
        babs = torch.abs(b)
        for i, bs in enumerate(torch.sort(babs)[0]):
            yinc = pred_fn(torch.where(babs >= bs, xinc, x))
            ydec = pred_fn(torch.where(babs >= bs, xdec, x))
            if yinc - y > change or y - ydec > change:
                rel[i] = i
                break
    return nanmean(rel)


def accuracy(
    sm: Slisemap,
    X: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
    **kwargs,
) -> float:
    """Evaluate a SLISEMAP solution by checking how well the fitted models work on new points

    Args:
        sm (Slisemap): Trained Slisemap solution.
        X (Optional[np.ndarray], optional): New data matrix (uses the training data if None). Defaults to None.
        Y (Optional[np.ndarray], optional): New target matrix (uses the training data if None). Defaults to None.
        **kwargs: Optional keyword arguments to Slisemap.fit_new.

    Returns:
        float: Mean loss for the new points.
    """
    if X is None or Y is None:
        X = sm.get_X(intercept=False, numpy=False)
        Y = sm.get_Y(numpy=False)
    B, Z, loss = sm.fit_new(X, Y, loss=True, **kwargs)
    return loss.mean()
