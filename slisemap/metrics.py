"""
    This module contains functions that can be used to evaluate SLISEMAP solutions.
    The functions take a solution (plus other arguments) and returns a single float.
    This float should either be minimised or maximised for best results (see individual functions).
"""

from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
from sklearn.cluster import KMeans

from slisemap.slisemap import Slisemap
from slisemap.utils import _deprecated, tonp


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
        D: Distance matrix.
        index: The item (row), for which to find neighbours.
        k: The number of neighbours to find. Defaults to 0.1.
        include_self: include the item in its' neighbourhood. Defaults to True.

    Returns:
        Vector of indices for the neighbours.
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
        D: Distance matrix.
        index: The item for which we want to find neighbours.
        epsilon: Treshold for selecting the neighbourhood (will be divided by `n`). Defaults to 1.0.
        include_self: include the item in its' neighbourhood. Defaults to True.

    Returns:
        Vector of indices for the neighbours.
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
        D: Distance matrix (ignored).
        index: The item for which we want to find neighbours.
        clusters: Cluster id:s for the data items.
        include_self: include the item in its' neighbourhood. Defaults to True.

    Returns:
        Vector of indices for the neighbours.
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
        D: Distance matrix (ignored).
        index: The item for which we want to find neighbours.
        radius: The radius of the neighbourhood. Defaults to None.
        quantile: If radius is None then radius is set to the quantile of D. Defaults to 0.2.
        include_self: include the item in its' neighbourhood. Defaults to True.

    Returns:
        Vector of indices for the neighbours.
    """
    if radius is None:
        radius = torch.quantile(D, quantile)
    if include_self:
        return torch.where(D[index] <= radius)[0]
    else:
        mask = D[index] <= radius
        mask[index] = False
        return torch.where(mask)[0]


def get_neighbours(
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
        sm: Trained Slisemap solution or an embedding vector (like Slisemap.Z).
        neighbours: Either None (return self), a vector of cluster id:s (take neighbours from the same cluter), or a function that gives neighbours (that takes a distance matrix and an index).
        full_if_none: If `neighbours` is None, return the whole dataset. Defaults to False.
    Keyword Args:
        **kwargs: Arguments passed on to `neighbours` (if it is a function).

    Returns:
        Function that takes an index and returns neighbour indices.
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
        sm: Trained Slisemap solution.

    Returns:
        The loss value.
    """
    return sm.value()


def entropy(
    sm: Slisemap, aggregate: bool = True, numpy: bool = True
) -> Union[float, np.ndarray, torch.Tensor]:
    """Compute row-wise entropy of the `W` matrix induced by `Z`.

    Args:
        aggregate: Aggregate the row-wise entropies into one scalar. Defaults to True.
        numpy: Return a `numpy.ndarray` or `float` instead of a `torch.Tensor`. Defaults to True.

    Returns:
        The entropy.
    """
    W = sm.get_W(False)
    entropy = -(W * W.log()).sum(dim=1)
    if aggregate:
        entropy = entropy.mean().exp() / sm.n
        return entropy.cpu().item() if numpy else entropy
    else:
        return tonp(entropy) if numpy else entropy


def slisemap_entropy(sm: Slisemap) -> float:
    """Evaluate a SLISEMAP solution by calculating the entropy. **DEPRECATED**

    Args:
        sm: Trained Slisemap solution.

    Returns:
        The embedding entropy.

    Deprecated:
        1.4: Use [entropy][slisemap.metrics.entropy] instead.
    """
    _deprecated(slisemap_entropy, entropy)
    return entropy(sm, aggregate=True, numpy=True)


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
        sm: Trained Slisemap solution.
        neighbours: Either None (only corresponding local model), a vector of cluster id:s, or a function that gives neighbours (see [get_neighbours][slisemap.metrics.get_neighbours]).
    Keyword Args:
        **kwargs: Arguments passed on to `neighbours` (if it is a function).

    Returns:
        The mean loss.
    """
    neighbours = get_neighbours(sm, neighbours, full_if_none=False, **kwargs)
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
        sm: Trained Slisemap solution.
        max_loss: Maximum tolerable loss.
        neighbours: Either None (all), a vector of cluster id:s, or a function that gives neighbours (see [get_neighbours][slisemap.metrics.get_neighbours]).
    Keyword Args:
        **kwargs: Arguments passed on to `neighbours` (if it is a function).

    Returns:
        The mean fraction of items within the error bound.
    """
    if torch.all(torch.isnan(sm._B.sum(1))).cpu().item():
        return np.nan
    neighbours = get_neighbours(sm, neighbours, full_if_none=True, **kwargs)
    results = np.zeros(sm.n)
    L = sm.get_L(numpy=False)
    for i in range(len(results)):
        ni = neighbours(i)
        if ni.numel() == 0:
            results[i] = np.nan
        else:
            results[i] = np.mean(tonp(L[i, ni] < max_loss))
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
        sm: Trained Slisemap solution.
        neighbours: Either None (all), a vector of cluster id:s, or a function that gives neighbours (see [get_neighbours][slisemap.metrics.get_neighbours]).
    Keyword Args:
        **kwargs: Arguments passed on to `neighbours` (if it is a function).

    Returns:
        The mean median loss.
    """
    neighbours = get_neighbours(sm, neighbours, full_if_none=True, **kwargs)
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
        sm: Trained Slisemap solution.
        neighbours: Either None (all), a vector of cluster id:s, or a function that gives neighbours (see [get_neighbours][slisemap.metrics.get_neighbours]).
    Keyword Args:
        **kwargs: Arguments passed on to `neighbours` (if it is a function).

    Returns:
        The mean coherence.
    """
    neighbours = get_neighbours(
        sm, neighbours, full_if_none=True, include_self=False, **kwargs
    )
    results = np.zeros(sm.n)
    P = sm.local_model(sm._X, sm._B)
    for i in range(len(results)):
        ni = neighbours(i)
        if ni.numel() == 0:
            results[i] = np.nan
        else:
            dP = torch.sum((P[None, i, i] - P[ni, i] - P[i, ni] + P[ni, ni]) ** 2, 1)
            dX = torch.sum((sm._X[None, i, :] - sm._X[ni, :]) ** 2, 1) + 1e-8
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
        sm: Trained Slisemap solution.
        neighbours: Either None (all), a vector of cluster id:s, or a function that gives neighbours (see [get_neighbours][slisemap.metrics.get_neighbours]).
    Keyword Args:
        **kwargs: Arguments passed on to `neighbours` (if it is a function).

    Returns:
        The mean stability.
    """
    neighbours = get_neighbours(
        sm, neighbours, full_if_none=True, include_self=False, **kwargs
    )
    results = np.zeros(sm.n)
    for i in range(len(results)):
        ni = neighbours(i)
        if ni.numel() == 0:
            results[i] = np.nan
        else:
            dB = torch.sum((sm._B[None, i, :] - sm._B[ni, :]) ** 2, 1)
            dX = torch.sum((sm._X[None, i, :] - sm._X[ni, :]) ** 2, 1) + 1e-8
            results[i] = torch.sqrt(torch.max(dB / dX))
    return nanmean(results)


def kmeans_matching(
    sm: Slisemap, clusters: Union[int, Sequence[int]] = range(2, 10), **kwargs
) -> float:
    """Evaluate SLISE by measuring how well clusters in Z and B overlap (using kmeans to find the clusters).
    The overlap is measured by finding the best matching clusters and dividing the size of intersect by the size of the union of each cluster pair.

    Larger is better.

    Args:
        sm: Trained Slisemap solution.
        clusters: The number of clusters. Defaults to range(2, 10).
    Keyword Args:
        **kwargs: Additional arguments to `sklearn.KMeans`.

    Returns:
        The mean cluster matching.
    """
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
        cl_B = KMeans(n_clusters=k, **kwargs).fit(B)
        cl_Z = KMeans(n_clusters=k, **kwargs).fit(Z)
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
        sm: Trained Slisemap solution _or_ embedding matrix.
        clusters: Cluster ids.

    Returns:
        The mean number of items sharing cluster that are neighbours.
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
        sm: Trained Slisemap solution.
        clusters: Cluster ids.
        epsilon: Treshold for being a neighbour (`softmax(D) < epsilon/n`). Defaults to 1.0.
        losses: Use losses instead of embedding distances. Defaults to False.

    Returns:
        The mean number of neighbours that are in the same cluster.
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
        sm: Trained Slisemap solution.
        epsilon_D: Treshold for being an embedding neighbour (`softmax(D) < epsilon/n`). Defaults to 1.0.
        epsilon_L: Treshold for being a loss neighbour (`softmax(L) < epsilon/n`). Defaults to 1.0.

    Returns:
        The mean recall.
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
            inter = np.intersect1d(tonp(nD), tonp(nL), True)
            res[i] = len(inter) / nL.numel()
    return nanmean(res)


def precision(sm: Slisemap, epsilon_D: float = 1.0, epsilon_L: float = 1.0) -> float:
    """Evaluate a SLISEMAP solution by calculating the recall.

    We define recall as the intersection between the loss and embedding neighbourhoods divided by the embedding neighbourhood.

    Larger is better.

    Args:
        sm: Trained Slisemap solution.
        epsilon_D: Treshold for being an embedding neighbour (`softmax(D) < epsilon/n`). Defaults to 1.0.
        epsilon_L: Treshold for being a loss neighbour (`softmax(L) < epsilon/n`). Defaults to 1.0.

    Returns:
        The mean precision.
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
            inter = np.intersect1d(tonp(nD), tonp(nL), True)
            res[i] = len(inter) / nD.numel()
    return nanmean(res)


def relevance(sm: Slisemap, pred_fn: Callable, change: float) -> float:
    """Evaluate a SLISEMAP solution by calculating the relevance.

    Smaller is better.

    TODO: This does not (currently) work for multi-class predictions

    Args:
        sm: Trained Slisemap solution.
        pred_fn: Function that gives y:s for new x:s (the "black box model").
        change: How much should the prediction change?

    Returns:
        The mean number of mutated variables required to cause a large enough change in the prediction.
    """
    rel = np.ones(sm.n) * sm.m
    for i in range(len(rel)):
        b = sm._B[i, :]
        x = sm._X[i, :]
        y = sm._Y[i, 0]
        xmax = torch.max(sm._X, 0)[0]
        xmin = torch.min(sm._X, 0)[0]
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
    X: Union[None, np.ndarray, torch.Tensor] = None,
    Y: Union[None, np.ndarray, torch.Tensor] = None,
    fidelity: bool = True,
    optimise: bool = False,
    **kwargs,
) -> float:
    """Evaluate a SLISEMAP solution by checking how well the fitted models work on new points

    Args:
        sm: Trained Slisemap solution.
        X: New data matrix (uses the training data if None). Defaults to None.
        Y: New target matrix (uses the training data if None). Defaults to None.
        fidelity: Return the mean local loss (fidelity) instead of the mean embedding weighted loss. Defaults to True.
    Keyword Args:
        **kwargs: Optional keyword arguments to [Slisemap.fit_new][slisemap.slisemap.Slisemap.fit_new].

    Returns:
        Mean loss for the new points.
    """
    if X is None or Y is None:
        X = sm.get_X(intercept=False, numpy=False)
        Y = sm.get_Y(numpy=False)
    if not fidelity:
        loss = sm.fit_new(X, Y, loss=True, optimise=optimise, numpy=False, **kwargs)[2]
        return loss.mean().cpu().item()
    else:
        X = sm._as_new_X(X)
        Y = sm._as_new_Y(Y, X.shape[0])
        B, _ = sm.fit_new(X, Y, loss=False, optimise=optimise, numpy=False, **kwargs)
        return sm.local_loss(sm.predict(X, B, numpy=False), Y, B).mean().cpu().item()
