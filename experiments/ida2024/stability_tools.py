import numpy as np
from slisemap import Slisemap
from scipy.stats import gaussian_kde
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch
from typing import Optional, Callable
from pathlib import Path
import sys

root = str(Path(__file__).parent.parent.parent.absolute())
if root not in sys.path:
    sys.path.insert(0, root)
from slisemap.slipmap import Slipmap


def absolute_regression_loss(
    Ytilde: torch.Tensor, Y: torch.Tensor, B: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Calculate absolute regression loss."""
    return (Ytilde - Y.expand(Ytilde.shape)).abs().sum(dim=-1)


def intercluster_loss(
    X1: torch.Tensor,
    X2: torch.Tensor,
    B1: torch.Tensor,
    B2: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    loss_function,
    loss_difference=True,
):
    """Calculate loss between two cluster prototypes.

    Args:
        X1: X matrix for cluster 1
        X2: X matrix for cluster 2
        B1: B matrix for cluster 1
        B2: B matrix for cluster 2
        Y1: Y matrix for cluster 1
        Y2: Y matrix for cluster 2
        loss_function: a loss metric
        loss_difference:  flags whether the desired measure is the sum in mean
            loss or the change in loss

    Returns:
        Sum of mean losses for prototype for cluster 1 with Y from 2 and vice
        versa.
    """
    yhat12 = torch.atleast_2d(B1 @ X2.T).T
    yhat21 = torch.atleast_2d(B2 @ X1.T).T
    yhat11 = torch.atleast_2d(B1 @ X1.T).T
    yhat22 = torch.atleast_2d(B2 @ X2.T).T
    loss_12 = loss_function(yhat12, y2).mean()
    loss_21 = loss_function(yhat21, y1).mean()
    loss_11 = loss_function(yhat11, y1).mean()
    loss_22 = loss_function(yhat22, y2).mean()
    loss_val = abs(loss_12 - loss_22 * loss_difference) + abs(
        loss_21 - loss_11 * loss_difference
    )
    return loss_val


def generate_intercluster_loss_table(
    sm1: Slisemap,
    sm2: Slisemap,
    clusters: int,
    loss_function="SLISEMAP",
    loss_difference=True,
    **kwargs,
):
    """Generate intercluster loss table for two SLISEMAP objects.

    Args:
        sm1: SLISEMAP object
        sm2: SLISEMAP object
        clusters: number of cluster to generate
        loss_function: 'SLISEMAP' or function
            The loss function to be used. Default is 'SLISEMAP' which
            translates to the local loss function of sm_x

    Returns:
        np.ndarray where elements [i, j] correspond to intercluster loss for
        cluster i in sm1 and cluster j in sm2.
    """
    if loss_function == "SLISEMAP":
        loss_function = sm1.local_loss
    labels1, B1s = sm1.get_model_clusters(clusters=clusters, **kwargs)
    labels2, B2s = sm2.get_model_clusters(clusters=clusters, **kwargs)
    loss_table = np.zeros((clusters, clusters))
    for i in range(clusters):
        for j in range(clusters):
            X1 = sm1.get_X(numpy=False)[np.where(labels1 == i)]
            Y1 = sm1.get_Y(numpy=False)[np.where(labels1 == i)]
            B1 = torch.tensor(B1s[i]).to(device=X1.device)
            X2 = sm2.get_X(numpy=False)[np.where(labels2 == j)]
            Y2 = sm2.get_Y(numpy=False)[np.where(labels2 == j)]
            B2 = torch.tensor(B2s[j]).to(device=X2.device)
            loss_table[i, j] = intercluster_loss(
                X1, X2, B1, B2, Y1, Y2, loss_function, loss_difference
            )
    return loss_table


def match_clusters_by_loss(
    sm1: Slisemap,
    sm2: Slisemap,
    clusters: int,
    loss_function: Callable[[torch.Tensor, torch.Tensor], float] = "SLISEMAP",
    loss_difference=True,
    **kwargs,
):
    """Generate best matching for clusters between two SLISEMAP objects
       based on intercluster loss.
    Args:
        sm1: SLISEMAP object
        sm2: SLISEMAP object
        clusters: number of cluster to generate
        loss_function: 'SLISEMAP' or function
            The loss function to be used. Default is 'SLISEMAP' which
            translates to the local loss function of sm_x
        loss_difference: flag whether to match clusters based on total loss
            or the difference between intercluster loss and the cluster's own
            loss

    Returns:
        Sum of the loss table
        Ideal matching as a tuple of two vectors where pair i can be found as
        matching[0][i], matching[1][i]
        The loss table: np.ndarray where elements [i, j] correspond to
        intercluster loss for cluster i in sm1 and cluster j in sm2.
    """
    loss_table = generate_intercluster_loss_table(
        sm1, sm2, clusters, loss_function, loss_difference, **kwargs
    )
    matching = linear_sum_assignment(loss_table)
    return np.sum(loss_table[matching]), matching, loss_table


def generate_intercluster_membership_table(
    sm1: Slisemap, sm2: Slisemap, clusters: int, include_y=True, jaccard=False, **kwargs
):
    """Generate intercluster loss table for two SLISEMAP objects.

    Args:
        sm1: SLISEMAP object
        sm2: SLISEMAP object
        clusters: number of cluster to generate
        include_y: flag whether or not to include y to the points

    Returns:
        np.ndarray where elements [i, j] correspond to the either the
        Jaccard distance between the clusters or simply the intersection size.
    """
    labels1, _ = sm1.get_model_clusters(clusters=clusters, **kwargs)
    labels2, _ = sm2.get_model_clusters(clusters=clusters, **kwargs)
    intersection_table = np.zeros((clusters, clusters))
    for i in range(clusters):
        for j in range(clusters):
            X1 = sm1.get_X(intercept=False)[np.where(labels1 == i)]
            X2 = sm2.get_X(intercept=False)[np.where(labels2 == j)]
            if include_y:
                Y1 = sm1.get_Y()[np.where(labels1 == i)]
                Y2 = sm2.get_Y()[np.where(labels2 == j)]
                X1 = np.hstack([X1, Y1])
                X2 = np.hstack([X2, Y2])
            s1 = set([tuple(x) for x in X1])
            s2 = set([tuple(x) for x in X2])
            if jaccard:
                intersection_table[i, j] = 1 - (len(s1 & s2) / len(s1 | s2))
            else:
                intersection_table[i, j] = len(s1 & s2)
    return intersection_table


def match_clusters_by_prototypes(sm1: Slisemap, sm2: Slisemap, clusters: int, **kwargs):
    _, B1 = sm1.get_model_clusters(clusters=clusters, **kwargs)
    _, B2 = sm2.get_model_clusters(clusters=clusters, **kwargs)
    prototype_distances = np.zeros((clusters, clusters))
    for i in range(clusters):
        for j in range(clusters):
            prototype_distances[i, j] = np.sqrt(np.sum(np.square(B1[i] - B2[j])))
    matching = linear_sum_assignment(prototype_distances)
    return prototype_distances[matching], matching, prototype_distances


def match_clusters_by_intersection(
    sm1: Slisemap,
    sm2: Slisemap,
    clusters: int,
    normalise=None,
    include_y=True,
    **kwargs,
):
    """Generate best matching for clusters between two SLISEMAP objects
       based on number of shared X points.
    Args:
        sm1: SLISEMAP object
        sm2: SLISEMAP object
        clusters: number of cluster to generate
        normalise: str or floar
            Normalisation setting. Options include 'sm1', 'sm2' or 'jaccard'.
            If normalise='sm1' or 'sm2', normalise the intersection table
            by dividing it by the sample size of the corresponding SLISEMAP
            object.
            If normalise='jaccard', instead of simple intersections, calculate
            the Jaccard distance between the clusters and choose ideal matching
            based on minimum distance instead of maximum intersection size.

    Returns:
        Ideal matching as a tuple of two vectors where pair i can be found as
        matching[0][i], matching[1][i]
        The intersection table: np.ndarray where elements [i, j] correspond to
        size of intersection of points for cluster i in sm1 and cluster j in
        sm2.
    """
    jaccard = False
    if normalise is None:
        norm_value = 1.0
    elif normalise == "sm1":
        norm_value = sm1.n
    elif normalise == "sm2":
        norm_value = sm2.n
    elif normalise == "jaccard":
        jaccard = True
    else:
        norm_value = normalise
    intersection_table = generate_intercluster_membership_table(
        sm1, sm2, clusters, include_y=include_y, jaccard=jaccard, **kwargs
    )
    if not jaccard:
        intersection_table /= norm_value
    matching = linear_sum_assignment(intersection_table, maximize=not jaccard)
    return matching, intersection_table


def get_X_by_cluster(sm: Slisemap, clusters: int, unscale=True, **kwargs):
    """Get X and B matrices from a SLISEMAP object by cluster."""
    labels, B = sm.get_model_clusters(clusters, **kwargs)
    if unscale:
        X = sm.metadata.unscale_X()
    else:
        X = sm.get_X()
    out = []
    for i in range(clusters):
        label_indices = np.where(labels == i)[0]
        out.append(X[label_indices, :])
    return out, B


def get_sm_dataframe(sm: Slisemap):
    """Get SLISEMAP training data and local model coefficients as a
       pd.DataFrame.

    Args:
        sm: Slisemap object
    Returns:
        pd.DataFrame with X, Y, and B as columns.
    """
    X = sm.get_X(intercept=False)
    Y = sm.get_Y()
    B = sm.get_B()
    df = pd.DataFrame(np.hstack([X, Y, B]))
    df.columns = (
        [f"x_{i+1}" for i in range(X.shape[1])]
        + [f"y_{i+1}" for i in range(Y.shape[1])]
        + [f"b_{i+1}" for i in range(B.shape[1])]
    )
    return df


def shared_point_loss(
    sm_x: Slisemap,
    sm_y: Slisemap,
    loss_function="SLISEMAP",
    include_y=False,
    full_loss=False,
):
    """Calculate the local model loss between two SLISEMAP objects.
    This is done by first finding the set of shared X points between the
    objects. Loss is then calculated by comparing the local models from
    SLISEMAP object 1 against the Y matrix of SLISEMAP object 2 and vice
    versa.

    Args:
        sm_x: Slisemap
        sm_y: Slisemap
        loss_function: 'SLISEMAP' or function
            The loss function to be used. Default is 'SLISEMAP' which
            translates to the local loss function of sm_x
        include_y: bool, default False
            Flag whether to include Y vector to X when searching for common
            points. Useful when working with degenerate datasets, where some
            dimensions are removed and thus multiple identical points in X
            can have different Y values.
        full_loss: bool default False
            Flags whether only the total loss is returned or the full loss
            vectors (along with shared indices)

    Returns:
        The total two-way loss for the shared points OR the index vectors of
        those points along with the full loss vectors if full_loss=True.
    """
    if loss_function == "SLISEMAP":
        loss_function = sm_x.local_loss
    i1, i2 = get_common_indices(sm_x, sm_y, include_y=include_y)
    X1 = sm_x.get_X(numpy=False, intercept=sm_y.intercept)[i1, :]
    X2 = sm_y.get_X(numpy=False, intercept=sm_x.intercept)[i2, :]
    B1 = sm_x.get_B(numpy=False)[i1, :]
    B2 = sm_y.get_B(numpy=False)[i2, :]
    Y1 = sm_x.get_Y(numpy=False)[i1, :]
    Y2 = sm_y.get_Y(numpy=False)[i2, :]
    loss21 = loss_function(Y2, sm_x.local_model(X=X2, B=B1)[0])
    loss12 = loss_function(Y1, sm_y.local_model(X=X1, B=B2)[0])
    loss11 = loss_function(Y1, sm_x.local_model(X=X1, B=B1)[0])
    loss22 = loss_function(Y2, sm_y.local_model(X=X2, B=B2)[0])
    if full_loss:
        return i1, i2, loss12, loss21
    else:
        return (loss12 - loss11 + loss21 - loss22).abs().sum().item()


def model_quality(sm: Slisemap):
    """Calculate model quality by comparing the predictions from the model
    to random permutation of predictions."""
    Yhat = sm.predict(numpy=False)
    random_indices = np.random.permutation(range(len(Yhat)))
    Yp = Yhat[random_indices]
    L = sm.local_loss(Yhat, sm.get_Y(numpy=False)).mean()
    L0 = sm.local_loss(Yp, sm.get_Y(numpy=False)).mean()
    return 1 - (L / L0).item()


def model_training_quality(sm: Slisemap):
    """Calculate model quality by comparing it to a model trained on randomly
    permuted data."""
    X = sm.get_X(numpy=False, intercept=False)
    Y = sm.get_Y(numpy=False)
    Yhat = sm.predict(numpy=False)
    random_indices = np.random.permutation(range(sm.n))
    Yp = Y[random_indices]
    sm_P = Slisemap(
        X,
        Yp,
        radius=sm._radius,
        d=sm.get_Z().shape[1],
        lasso=sm.lasso,
        ridge=sm.ridge,
        z_norm=sm._z_norm,
        intercept=sm.intercept,
        local_model=sm._local_model,
        local_loss=sm._local_loss,
    )
    sm_P.optimise()
    L = sm.local_loss(Yhat, Y).mean()
    L0 = sm.local_loss(sm_P.predict(numpy=False), Y).mean()
    return 1 - (L / L0).item()


def get_common_indices(sm1: Slisemap, sm2: Slisemap, include_y=True):
    """Get the indices of shared point between two Slisemap objects.
    The indices are returned ordered such that X1[i1[0]] == X2[i2[0]] etc."""
    X1 = sm1.get_X(numpy=False, intercept=False)
    X2 = sm2.get_X(numpy=False, intercept=False)
    if include_y:
        Y1 = sm1.get_Y(numpy=False)
        Y2 = sm2.get_Y(numpy=False)
        X1 = torch.hstack([X1, Y1])
        X2 = torch.hstack([X2, Y2])
    i1 = [i for i, x in enumerate(X1) if torch.isclose(X2, x).all(axis=1).any()]
    i2 = [None] * len(i1)
    for i in range(len(i1)):
        matching_points = torch.where(torch.isclose(X2, X1[i1[i]]).all(axis=1))[0]
        if len(matching_points) > 1:
            i2[i] = matching_points[0].item()
        else:
            i2[i] = matching_points.item()
    return i1, i2


def min_B_distance(sm1: Slisemap, sm2: Slisemap):
    """Compare the B-space average distance to the minimum distance. Can be thought
    as the most optimistic local model distance formulation.

    Args:
         sm_1: Slisemap
         sm_2: Slisemap
    Returns:
         Minimum distance between local models, normalised by the average distance."""

    B1 = sm1.get_B(numpy=False)
    B2 = sm2.get_B(numpy=False)
    distance_matrix = torch.cdist(B1, B2).cpu()
    min_distances = torch.min(distance_matrix, axis=0)[0]
    mean_distances = torch.mean(distance_matrix, axis=0)
    B_dist = (min_distances / mean_distances).mean()
    return B_dist


def B_distance(
    sm1: Slisemap,
    sm2: Slisemap,
    include_y=True,
    mean=True,
    full_table=False,
    match_by_model=False,
    norm_permutations=100,
    distance_function=None,
):
    """Compare the B-space distance in the set of shared points.

    Args:
        sm1: Slisemap
        sm2: Slisemap
        include_y: bool, optional
            A flag whether to consider Y values when constructing the set of shared
            points. Defaults to True.
        mean: bool, optional
            Whether to return the mean of distances or the sum. Defaults to
            True.
        full_table: bool, optional
            Whether to return the full table of results or just a summary statistic.
            Defaults to False.
        match_by_model: bool, optional
            Whether to calculate the point-wise local model distance (False) or match
            the models optimally based on distance, despite the underlying data items
            (True). Defaults to False.
        norm_permutations: int, optional
            How many permutations should be calculated to normalise the measure between
            0, 1. Defaults to 100.
        distance_function: Callable, optional
            A user-provided distance function. Defaults to Euclidean distance.
    """
    if distance_function is None:
        euc_norm = lambda t1, t2: torch.sqrt(torch.square(t1 - t2).sum(axis=1))
        distance_function = euc_norm

    B1 = sm1.get_B(numpy=False)
    B2 = sm2.get_B(numpy=False)
    if match_by_model:
        # hardcoded to use euclidean distance for now, unfortunately
        distance_matrix = torch.cdist(B1, B2).cpu()
        matching = linear_sum_assignment(distance_matrix, maximize=False)
        B1 = B1[matching[0], :]
        B2 = B2[matching[1], :]
    else:
        i1, i2 = get_common_indices(sm1, sm2, include_y=include_y)
        if len(i1) == 0:
            print("No shared points between Slisemap objects!")
            return np.inf
        B1 = B1[i1, :]
        B2 = B2[i2, :]
    # set norm value
    if norm_permutations is None:
        norm_value = 1.0
    elif match_by_model:
        norm_value = torch.mean(distance_matrix)
    else:
        n = len(i1)
        norm_value = 0
        for i in range(norm_permutations):
            i_randperm = torch.randint(0, n, size=(n,))
            B1_r = B1[i_randperm, :]
            rp_dist = distance_function(B1_r, B2)
            if full_table:
                norm_value += rp_dist / norm_permutations
            elif mean:
                norm_value += rp_dist.mean().item() / norm_permutations
            else:
                norm_value += rp_dist.sum().item() / norm_permutations
    if full_table:
        B_dist = distance_function(B1, B2) / norm_value
        return B_dist
    if mean:
        B_dist = distance_function(B1, B2).mean().item()
        B_dist /= norm_value
        return B_dist
    else:
        B_dist = distance_function(B1, B2).sum().item()
        B_dist /= norm_value
        return B_dist


def points_within_epsilon_ball(
    sm: Slisemap, shared_points: list, point_index: int, epsilon=1.0
):
    """Helper function to find indices of points in a e-radius ball around a given
    point."""
    Z = sm.get_Z(numpy=False)
    z0 = Z[point_index]
    i_epsilon = torch.where(torch.cdist(Z, torch.atleast_2d(z0)) <= epsilon)[0]
    return [x for x in i_epsilon.tolist() if x in shared_points]


def epsilon_ball_distance(
    sm1: Slisemap, sm2: Slisemap, epsilon=1.0, full_table=False, include_y=True
):
    """Find distance between two Slisemap embeddings by comparing the
    intersections of neighborhood sets for points in both embeddings."""
    i1, i2 = get_common_indices(sm1, sm2, include_y=include_y)
    if len(i1) < 1:
        print("No shared points between Slisemap objects!")
        return np.inf
    intersection_sizes = np.empty(len(i1))
    mapping = {i2[i]: i1[i] for i in range(len(i2))}
    for i in range(len(i1)):
        ball1 = points_within_epsilon_ball(sm1, i1, i1[i], epsilon=epsilon)
        ball2 = points_within_epsilon_ball(sm2, i2, i2[i], epsilon=epsilon)
        ball2to1 = [mapping[x] for x in ball2 if x in mapping]
        ball_inter = set(ball1) & set(ball2to1)
        intersection_sizes[i] = len(ball_inter) / len(ball1)
    if full_table:
        return intersection_sizes
    else:
        return 1 - np.sum(intersection_sizes) / len(i1)


def faster_epsilon_ball(
    sm1: Slisemap,
    sm2: Slisemap,
    epsilon=1.0,
    include_y=True,
    minimise=False,
    debug=False,
):
    """Vectorized version of the function above."""
    i1, i2 = get_common_indices(sm1, sm2, include_y=include_y)
    if len(i1) < 1:
        print("No shared points between Slisemap objects!")
        return np.inf
    if debug:
        print(f"Ratio of shared points: {len(i1)/sm1.n}.", flush=True)
    intersection_sizes = np.empty(len(i1))
    if isinstance(sm1, Slisemap):
        D1 = sm1.get_D(numpy=False)
    elif isinstance(sm1, Slipmap):
        D1 = sm1.get_D(proto_rows=False, proto_cols=False, numpy=False)
    if isinstance(sm2, Slisemap):
        D2 = sm2.get_D(numpy=False)
    elif isinstance(sm2, Slipmap):
        D2 = sm2.get_D(proto_rows=False, proto_cols=False, numpy=False)
    if minimise:
        D1 = set_distance(sm1.get_L(numpy=False), D1)
        D2 = set_distance(sm2.get_L(numpy=False), D2)
    D1 = D1[i1, :][:, i1]
    D2 = D2[i2, :][:, i2]
    ball1 = D1 < epsilon
    ball2 = D2 < epsilon
    shared = torch.logical_and(ball1, ball2).sum(axis=0)
    intersection_sizes = shared / torch.logical_or(ball1, ball2).sum(axis=0)
    return (1 - (torch.sum(intersection_sizes) / len(i1))).item()


@torch.jit.script
def set_distance(
    L: torch.Tensor, D: torch.Tensor, threshold: float = 0.9
) -> torch.Tensor:
    """Create a new distance matrix where alternative locations for a point are treated as a "set".
    I.e. when comparing two points the distance for the closest alternative locations are returned.
    As the threshold for alternative locations we use a quantile of the local losses.

    NOTE: This function is O(n^4), but faster in practice

    NOTE: Assumes `torch.all(D.diagonal() == 0.)`

    Args:
        L: Loss matrix, usually from `Slisemap.get_L(numpy=False)`. [n x n]
        D: Original distance matrix, e.g., from `Slisemap.get_D(numpy=False)`. [n x n]
        threshold: Quantile for the local losses to determine alternative locations. Defaults to 0.9.

    Returns:
        Distance matrix where alternative locations may reduce the distances in D.

    Example:
        Dnew = set_distance(sm.get_L(numpy=False), sm.get_D(numpy=False))
        assert torch.all(Dnew <= sm.get_D(numpy=False))
    """
    assert D.shape == (L.shape[0], L.shape[0])
    threshold = torch.quantile(L.diagonal(), threshold)
    Dnew = torch.zeros_like(D)
    sets = L.T <= threshold
    sets = sets.fill_diagonal_(L[0, 0] <= L[0, 0])
    for i in range(D.shape[0]):
        d = D[sets[i], :]
        for j in range(i + 1, D.shape[0]):
            if not sets[i, j] and not sets[j, i]:
                Dnew[i, j] = Dnew[j, i] = torch.min(d[:, sets[j]])
    return Dnew


@torch.jit.script
def set_double_distance(
    L1: torch.Tensor, L2: torch.Tensor, D: torch.Tensor, threshold: float = 0.9
) -> torch.Tensor:
    """Create a new distance matrix for comparing two Slisemaps where alternative locations for a point are treated as a "set".
    I.e. when comparing two points the distance for the closest alternative locations are returned.
    As the threshold for alternative locations we use a quantile of the local losses.

    NOTE: `set_double_distance(L, L, D) == set_distance(L, D)`

    NOTE: This function is O(n^4)

    Args:
        L1: Loss matrix 1, usually from `Slisemap.get_L(numpy=False)`. [n1 x n1]
        L2: Loss matrix 2, usually from `Slisemap.get_L(numpy=False)`. [n2 x n2]
        D: Original distance matrix, e.g., from `Slisemap.get_D(numpy=False)`. [n1 x n2]
        threshold: Quantile for the local losses to determine alternative locations. Defaults to 0.9.

    Returns:
        Distance matrix where alternative locations may reduce the distances in D.

    Example:
        Dnew = set_distance(sm.get_L(numpy=False), sm.get_D(numpy=False))
        assert torch.all(Dnew <= sm.get_D(numpy=False))
    """
    assert D.shape == (L1.shape[0], L2.shape[0])
    threshold1 = torch.quantile(L1.diagonal(), threshold)
    sets1 = L1.T <= threshold1
    sets1 = sets1.fill_diagonal_(L1[0, 0] <= L1[0, 0])
    threshold2 = torch.quantile(L2.diagonal(), threshold)
    sets2 = L2.T <= threshold2
    sets2 = sets2.fill_diagonal_(L2[0, 0] <= L2[0, 0])
    Dnew = torch.zeros_like(D)
    for i in range(D.shape[0]):
        d = D[sets1[i], :]
        for j in range(D.shape[1]):
            Dnew[i, j] = torch.min(d[:, sets2[j]])
    return Dnew


@torch.jit.script
def set_half_distance(
    L: torch.Tensor, D: torch.Tensor, threshold: float = 0.9
) -> torch.Tensor:
    """Create a new distance matrix where alternative locations for a point row-wise are treated as a "set".
    I.e. when comparing two points the distance for the closest alternative location is returned.
    As the threshold for alternative locations we use a quantile of the local losses.

    NOTE: `set_double_distance(1-torch.eye(n), L, D) == set_half_distance(L, D)`

    NOTE: This function is O(n^3)

    Args:
        L2: Loss matrix, usually from `Slisemap.get_L(numpy=False)`. [n2 x n2]
        D: Original distance matrix, e.g., from `Slisemap.get_D(numpy=False)`. [n1 x n2]
        threshold: Quantile for the local losses to determine alternative locations. Defaults to 0.9.

    Returns:
        Distance matrix where alternative locations may reduce the distances in D.

    Example:
        Dnew = set_distance(sm.get_L(numpy=False), sm.get_D(numpy=False))
        assert torch.all(Dnew <= sm.get_D(numpy=False))
    """
    assert D.shape[1] == L.shape[0]
    threshold = torch.quantile(L.diagonal(), threshold)
    sets = L.T <= threshold
    sets = sets.fill_diagonal_(L[0, 0] <= L[0, 0])
    Dnew = torch.zeros_like(D)
    for j in range(D.shape[1]):
        Dnew[:, j] = torch.min(D[:, sets[j]], 1)[0]
    return Dnew


def set_B_distance(
    sm1: Slisemap, sm2: Slisemap, include_y=False, mean=True, full_table=False
):
    """Largely a copy of the set_distance function. Placeholder for now, will need
    to refactor after ECML paper is out."""
    """Compare the B-space distance in the set of shared points."""

    i1, i2 = get_common_indices(sm1, sm2, include_y=include_y)
    if len(i1) == 0:
        print("No shared points between Slisemap objects!")
        return np.inf
    B1 = sm1.get_B(numpy=False)[i1, :]
    B2 = sm2.get_B(numpy=False)[i2, :]
    distance_matrix = torch.cdist(B1, B2).cpu()
    distance_matrix = set_distance(
        sm1.get_L(numpy=False)[i1, :][:, i1], distance_matrix
    )
    norm_value = torch.mean(distance_matrix)
    if mean:
        B_dist = torch.diagonal(distance_matrix).mean().item()
    else:
        B_dist = torch.diagonal(distance_matrix).sum().item()
    B_dist /= norm_value
    return B_dist
