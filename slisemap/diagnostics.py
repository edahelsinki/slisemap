"""
These are diagnostics for identifying potential issues with SLISEMAP solutions.

**Typical usage:**

    sm = Slisemap(...)
    sm.optimise()
    diagnostics = diagnose(sm)
    print_diagnostics(diagnostics)
    plot_diagnostics(sm, diagnostics)
"""
from functools import reduce
from typing import Any, Dict, Optional, Union

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from slisemap.slisemap import Slisemap
from slisemap.utils import dict_array, dict_concat, global_model, tonp


def _size(n: int, part: Union[float, int]) -> int:
    if isinstance(part, int):
        return part
    else:
        return int(n * part)


def _frac(n: int, part: Union[float, int]) -> float:
    if isinstance(part, int):
        return part / n
    else:
        return part


def global_model_losses(
    sm: Slisemap, indices: Optional[np.ndarray] = None, **kwargs: Any
) -> torch.Tensor:
    """Train a global model

    Args:
        sm: Slisemap object.
        indices: Optional subsampling indices. Defaults to None.
    Keyword Args:
        **kwargs: Optional keyword arguments to LBFGS.

    Returns:
        Vector of individual losses for a global model.
    """
    if indices is None:
        X = sm._X
        Y = sm._Y
    else:
        X = sm._X[indices]
        Y = sm._Y[indices]
    B = global_model(
        X=X,
        Y=Y,
        local_model=sm.local_model,
        local_loss=sm.local_loss,
        coefficients=sm.q,
        lasso=sm.lasso,
        ridge=sm.ridge,
    )
    return sm.local_loss(sm.local_model(sm._X, B), sm._Y).detach()


def print_diagnostics(diagnostics: Dict[str, np.ndarray], summary: bool = False):
    """Print diagnostic results.

    Args:
        diagnostics: Dictionary of diagnostic results.
        summary: Print only one summary for all the diagnostics. Defaults to False.
    """
    if summary:
        issues = reduce(lambda a, b: a + b.astype(int), diagnostics.values())
        if np.sum(issues) == 0:
            print(f"All data items passed all the diagnostics!")
        else:
            print(
                f"{np.mean(issues > 0) * 100 :.1f}% of the data items failed at least",
                "one diagnostic and the average failure rate across all diagnostics",
                f"was {np.sum(issues) / (len(issues) * len(diagnostics)) *100:.1f}%",
            )
    else:
        for name, mask in diagnostics.items():
            rate = np.mean(mask) * 100
            if rate == 0:
                print(f"All data items passed the `{name}` diagnostic!")
            else:
                print(f"{rate:.1f}% of the data items failed the `{name}` diagnostic!")


def plot_diagnostics(
    Z: Union[Slisemap, np.ndarray],
    diagnostics: Dict[str, np.ndarray],
    summary: bool = False,
    title: str = "Slisemap Diagnostics",
    show: bool = True,
    **kwargs: Any,
) -> Optional[sns.FacetGrid]:
    """Plot diagnostic results.

    Args:
        Z: The Slisemap object, or embedding matrix.
        diagnostics: Dictionary of diagnostic results.
        summary: Combine multiple diagnostics into one plot. Defaults to False.
        title: Title of the plot. Defaults to "Slisemap Diagnostics".
        show: Show the plot. Defaults to True.
    Keyword Args:
        **kwargs: Additional parameters to `seaborn.relplot`.

    Returns:
        `seaborn.FacetGrid` if `show=False`.
    """
    if isinstance(Z, Slisemap):
        Z = Z.get_Z(rotate=True)
    if len(diagnostics) == 1:
        for name, mask in diagnostics.items():
            df = dict_array({"Z1": Z[:, 0], "Z2": Z[:, 1], name: mask})
            g = sns.relplot(
                data=df, x="Z1", y="Z2", hue=name, style=name, kind="scatter", **kwargs
            )
    elif summary:
        issues = reduce(lambda a, b: a + b.astype(int), diagnostics.values())
        df = dict_array(
            {"Z1": Z[:, 0], "Z2": Z[:, 1], "Problem": issues > 0, "Severity": issues}
        )
        g = sns.relplot(
            data=df,
            x="Z1",
            y="Z2",
            hue="Severity",
            style="Problem",
            kind="scatter",
            **kwargs,
        )
    else:
        if "col_wrap" not in kwargs:
            if len(diagnostics) <= 4:
                kwargs["col_wrap"] = len(diagnostics)
            elif len(diagnostics) < 7 or len(diagnostics) == 9:
                kwargs["col_wrap"] = 3
            else:
                kwargs["col_wrap"] = 4
        df = dict_concat(
            {
                "Z1": Z[:, 0],
                "Z2": Z[:, 1],
                "Problem": mask,
                "Diagnostic": f"{diag} ({np.mean(mask) * 100:.1f} %)",
            }
            for diag, mask in diagnostics.items()
        )
        g = sns.relplot(
            data=df,
            x="Z1",
            y="Z2",
            hue="Problem",
            style="Problem",
            kind="scatter",
            col="Diagnostic",
            **kwargs,
        )
    plt.suptitle(title)
    if show:
        plt.show()
    else:
        return g


def distant_diagnostic(sm: Slisemap, max_distance: float = 10.0) -> np.ndarray:
    """Check if any data item in the embedding is too far way.

    Args:
        sm: Trained Slisemap solution.
        max_distance: Maximum distance from origo in the embedding. Defaults to 10.0.

    Returns:
        Boolean mask of problematic data items.
    """
    return np.sum(sm.get_Z() ** 2, 1) > max_distance**2


def heavyweight_diagnostic(
    sm: Slisemap, min_size: Union[float, int] = 0.1
) -> np.ndarray:
    """Check if any data item has a self-weight that is too large.

    Args:
        sm: Trained Slisemap solution.
        min_size: Miniumum neighbourhood/cluster size (as a fraction or absolute number). Defaults to 0.1.

    Returns:
        Boolean mask of problematic data items.
    """
    return tonp(sm.get_W(numpy=False).diag() > _frac(sm.n, min_size))


def lightweight_diagnostic(
    sm: Slisemap, max_size: Union[float, int] = 0.5
) -> np.ndarray:
    """Check if any data item has a self-weight that is too small

    Args:
        sm: Trained Slisemap solution.
        max_size: Maximum neighbourhood/cluster size (as a fraction or absolute number). Defaults to 0.5.

    Returns:
        Boolean mask of problematic data items.
    """
    return tonp(sm.get_W(numpy=False).diag() < (1 / _size(sm.n, max_size)))


def weight_neighbourhood_diagnostic(
    sm: Slisemap, min_size: Union[float, int] = 0.1, max_size: Union[float, int] = 0.5
) -> np.ndarray:
    """Check if any data item has a neighbourhood that is too small/large by counting the number of non-lightweight neighbours.

    Args:
        sm: Trained Slisemap solution.
        min_size: Miniumum neighbourhood/cluster size (as a fraction or absolute number). Defaults to 0.1.
        max_size: Maximum neighbourhood/cluster size (as a fraction or absolute number). Defaults to 0.5.

    Returns:
        Boolean mask of problematic data items.
    """
    min_size = _size(sm.n, min_size)
    max_size = _size(sm.n, max_size)
    return tonp((sm.get_W(numpy=False) > (1 / max_size)).sum(1) < min_size)


def loss_neighbourhood_diagnostic(
    sm: Slisemap,
    min_size: Union[float, int] = 0.1,
    smoothing: bool = True,
    median: bool = False,
) -> np.ndarray:
    """Check if any data item has a neighbourhood that is too small/large by comparing local losses to global losses.

    Args:
        sm: Trained Slisemap solution.
        min_size: Miniumum neighbourhood/cluster size (as a fraction or absolute number). Defaults to 0.1.
        smoothing: Smooth the sorted losses to avoid sensitivity to outliers. Defaults to True.
        median: Compare against the median global loss instead of the mean global loss. Defaults to False.

    Returns:
        Boolean mask of problematic data items.
    """
    min_size = _size(sm.n, min_size)
    if median:
        gloss = global_model_losses(sm).median().cpu().item()
    else:
        gloss = global_model_losses(sm).mean().cpu().item()
    llosses = sm.get_L(numpy=False)
    order = torch.argsort(sm.get_D(numpy=False))
    result = np.zeros(sm.n, bool)
    if smoothing:
        filter = torch.as_tensor([[[0.25, 0.5, 0.25]]], **sm.tensorargs)
    for i in range(sm.n):
        lli = llosses[i, order[i]]
        if smoothing:
            lli = torch.conv1d(lli[None, None], filter, padding="same")[0, 0]
        first = torch.where(lli > gloss)[0][:1]
        if first.numel() < 1:
            result[i] = lli.numel() < min_size
        else:
            result[i] = (first < min_size).cpu().item()
    return result


def global_loss_diagnostic(
    sm: Slisemap, bootstrap: int = 10, sd: float = 1.0
) -> np.ndarray:
    """Check if any local model is actually a global model.

    Args:
        sm: Trained Slisemap solution.
        bootstrap: Number of (bootstrap) global models to train. Defaults to 10.
        sd: Number of standard deviations from the mean (of global models losses) to consider a local model global. Defaults to 1.0.

    Returns:
        Boolean mask of problematic data items.
    """
    glosses = [
        global_model_losses(sm, np.random.randint(sm.n, size=sm.n)).sum().cpu().item()
        for _ in range(bootstrap)
    ]
    treshold = np.mean(glosses) + np.std(glosses) * sd
    llosses = sm.get_L(numpy=False).sum(1)
    return tonp(llosses < treshold)


def quantile_loss_diagnostic(sm: Slisemap, quantile: float = 0.4) -> np.ndarray:
    """Check if any fidelity is worse than a quantile of all losses.

    Args:
        sm: Trained Slisemap solution.
        quantile: The quantile percentage. Defaults to 0.4.

    Returns:
        Boolean mask of problematic data items.
    """
    L = sm.get_L(numpy=False)
    treshold = torch.quantile(L.ravel(), _frac(sm.n, quantile))
    return tonp(L.diag() > treshold)


def optics_diagnostic(
    sm: Slisemap, min_size: Union[float, int] = 0.1, **kwargs: Any
) -> np.ndarray:
    """Use a clustering method to check for problematic data items in the embedding.

    Args:
        sm: Trained Slisemap solution.
        min_size: Miniumum neighbourhood/cluster size (as a fraction or absolute number). Defaults to 0.1.

    Returns:
        Boolean mask of problematic data items.
    """
    from sklearn.cluster import OPTICS

    optics = OPTICS(min_samples=min_size, metric="euclidean", n_jobs=-1, **kwargs)
    return optics.fit(sm.get_Z()).labels_ < 0


def diagnose(
    sm: Slisemap,
    min_size: Union[float, int] = 0.1,
    max_size: Union[float, int] = 0.5,
    max_distance: float = 10.0,
    conservative: bool = False,
) -> Dict[str, np.ndarray]:
    """Run multiple diagnostics.

    Args:
        sm: Trained Slisemap solution.
        min_size: Miniumum neighbourhood/cluster size (as a fraction or absolute number). Defaults to 0.1.
        max_size: Maximum neighbourhood/cluster size (as a fraction or absolute number). Defaults to 0.5.
        max_distance: Maximum distance from origo in the embedding. Defaults to 10.0.
        conservative: Only run the most conservative diagnostics. Defaults to False.

    Returns:
        Dictionary of the the diagnostics results.
    """
    if conservative:
        return {
            "Distant": distant_diagnostic(sm, max_distance),
            "Heavyweight": heavyweight_diagnostic(sm, min_size),
            "Lightweight": lightweight_diagnostic(sm, max_size),
            "Weight Neighbourhood": weight_neighbourhood_diagnostic(
                sm, min_size, max_size
            ),
            "Quantile Loss": quantile_loss_diagnostic(sm, max_size),
        }
    else:
        return {
            "Distant": distant_diagnostic(sm, max_distance),
            "Heavyweight": heavyweight_diagnostic(sm, min_size),
            "Lightweight": lightweight_diagnostic(sm, max_size),
            "Weight Neighbourhood": weight_neighbourhood_diagnostic(
                sm, min_size, max_size
            ),
            "Loss Neighbourhood": loss_neighbourhood_diagnostic(sm, min_size),
            "Global Loss": global_loss_diagnostic(sm),
            "Clustering": optics_diagnostic(sm, min_size),
            "Quantile Loss": quantile_loss_diagnostic(sm, max_size),
        }
