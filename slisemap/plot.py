"""
Utility functions for plotting
"""

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from slisemap.utils import _assert, _warn, dict_concat


def _expand_variable_names(
    variables: Sequence[str],
    intercept: bool,
    columns: int,
    targets: Union[None, str, Sequence[str]],
    coefficients: int,
) -> List[str]:
    if intercept and len(variables) == columns - 1:
        variables = list(variables) + ["Intercept"]
    if targets and not isinstance(targets, str) and len(targets) > 0:
        if coefficients % len(variables) == 0 and coefficients % len(targets) == 0:
            variables = [f"{t}: {v}" for t in targets for v in variables]
            variables = variables[:coefficients]
    _assert(
        len(variables) == coefficients,
        f"The number of variable names ({len(variables)}) must match the number of coefficients ({coefficients})",
        _expand_variable_names,
    )
    return variables


def _create_legend(
    hue_norm: Tuple[float, float],
    cmap: Any,
    markers: int = 5,
    min_length: int = 3,
    max_digits: int = 6,
) -> Tuple[List[Line2D], List[str]]:
    handles = [
        Line2D([], [], 0, color=cmap(n), marker="o")
        for n in np.linspace(0.0, 1.0, markers)
    ]
    for i in range(max_digits):
        labels = [f"{l:.{i}f}" for l in np.linspace(*hue_norm, markers)]
        length = max(len(l) for l in labels)
        if length < min_length:
            continue
        if len(np.unique(labels)) == markers:
            break
    labels = [f"{l:{length}.{i}f}" for l in np.linspace(*hue_norm, markers)]
    return handles, labels


def legend_inside_facet(grid: sns.FacetGrid):
    """Move the legend to within the facet grid if possible.

    Args:
        grid: Facet grid
    """
    col_wrap = grid._col_wrap
    facets = grid._n_facets
    if col_wrap < facets and facets % col_wrap != 0:
        w = 1 / col_wrap
        h = 1 / ((facets - 1) // col_wrap + 1)
        sns.move_legend(
            grid,
            "center",
            bbox_to_anchor=(1 - w, h * 0.1, w * 0.9, h * 0.9),
            frameon=False,
        )
        plt.tight_layout()


def _prepare_Z(Z, Z_names, jitter, function) -> Tuple[np.ndarray, Tuple[str, str]]:
    if Z.shape[1] > 2:
        _warn("Only the first two dimensions in the embedding are plotted", function)
    elif Z.shape[1] < 2:
        Z = np.concatenate((Z, np.zeros_like(Z)), 1)
        Z_names = (Z_names[0], "")
    _assert(len(Z_names) >= 2, "Requires at least two dimension names", function)
    if not isinstance(jitter, float):
        Z = Z + jitter
    elif jitter > 0:
        Z = np.random.normal(Z, jitter)
    return Z, Z_names


def plot_embedding(
    Z: np.ndarray,
    dimensions: Sequence[str] = ("SLISEMAP 1", "SLISEMAP 2"),
    title: str = "Embedding",
    jitter: Union[float, np.ndarray] = 0.0,
    clusters: Optional[Sequence[int]] = None,
    color: Optional[Sequence[float]] = None,
    color_name: str = "",
    color_norm: Optional[Tuple[float]] = None,
    **kwargs,
) -> plt.Axes:
    """Plot an embedding in a scatterplot.

    Args:
        Z: The embedding.
        dimensions: Dimension names. Defaults to ("SLISEMAP 1", "SLISEMAP 2").
        title: Plot title. Defaults to "Embedding".
        jitter: Jitter amount. Defaults to 0.0.
        clusters: Cluster labels. Defaults to None.
        color: Variable for coloring. Defaults to None.
        color_name: Variable name. Defaults to "".
        color_norm: Color scale limits. Defaults to None.

    Returns:
        The plot.
    """
    Z, dimensions = _prepare_Z(Z, dimensions, jitter, plot_embedding)
    if clusters is None:
        kwargs.setdefault("palette", "crest")
        if color is not None:
            ax = sns.scatterplot(
                x=Z[:, 0],
                y=Z[:, 1],
                hue=color,
                hue_norm=color_norm,
                legend=False,
                **kwargs,
            )
        else:
            ax = sns.scatterplot(x=Z[:, 0], y=Z[:, 1], **kwargs)
    else:
        kwargs.setdefault("palette", "bright")
        ax = sns.scatterplot(
            x=Z[:, 0], y=Z[:, 1], hue=clusters, style=clusters, **kwargs
        )
        color_name = "Cluster"
    if color_norm is not None:
        ax.legend(
            *_create_legend(color_norm, plt.get_cmap(kwargs["palette"]), 5),
            title=color_name,
        )
    elif color_name is not None and color_name != "":
        ax.legend(title=color_name)
    ax.set_xlabel(dimensions[0])
    ax.set_ylabel(dimensions[1])
    ax.axis("equal")
    ax.set_title(title)
    return ax


def plot_matrix(
    B: np.ndarray, coefficients: Sequence[str], palette: str = "RdBu", **kwargs
) -> plt.Axes:
    """Plot local models in a heatmap.

    Args:
        B: Local model coefficients.
        coefficients: Coefficient names.
        palette: `seaborn` palette. Defaults to "RdBu".

    Returns:
        The plot.
    """
    ax = sns.heatmap(B.T, center=0, cmap=palette, robust=True, **kwargs)
    ax.set_yticks(np.arange(len(coefficients)) + 0.5)
    ax.set_yticklabels(coefficients, rotation=0)
    # ax.set_ylabel("Coefficients")
    ax.set_title("Local models")
    return ax


def plot_barmodels(
    B: np.ndarray,
    clusters: np.ndarray,
    centers: np.ndarray,
    coefficients: Sequence[str],
    bars: Union[bool, int] = True,
    palette: str = "bright",
    **kwargs,
) -> plt.Axes:
    """Plot local models in a barplot.

    Args:
        B: Local model coefficients.
        clusters: Cluster labels.
        centers: Cluster centers.
        coefficients: Coefficient names.
        bars: Number of variables to show (or a bool for all). Defaults to True.
        palette: `seaborn` palette. Defaults to "bright".

    Returns:
        The plot.
    """
    if not isinstance(bars, bool):
        influence = np.abs(centers)
        influence = influence.max(0) + influence.mean(0)
        mask = np.argsort(-influence)[:bars]
        coefficients = np.asarray(coefficients)[mask]
        B = B[:, mask]
    ax = sns.barplot(
        y=np.tile(coefficients, B.shape[0]),
        x=B.ravel(),
        hue=np.repeat(clusters, B.shape[1]),
        palette=palette,
        orient="h",
        **kwargs,
    )
    ax.legend().remove()
    lim = np.max(np.abs(ax.get_xlim()))
    ax.set(xlabel=None, ylabel=None, xlim=(-lim, lim))
    # ax.set_ylabel("Coefficients")
    ax.set_title("Local models")
    return ax


def plot_embedding_facet(
    Z: np.ndarray,
    dimensions: Sequence[str],
    data: np.ndarray,
    names: Sequence[str],
    legend_title: str = "Value",
    jitter: Union[float, np.ndarray] = 0.0,
    **kwargs,
) -> sns.FacetGrid:
    """Plot (multiple) embeddings.

    Args:
        Z: Embeddings.
        dimensions: Dimension names.
        data: Data matrix.
        names: Column names.
        legend_title: Legend title. Defaults to "Value".
        jitter: jitter. Defaults to 0.0.

    Returns:
        The plot.
    """
    Z, dimensions = _prepare_Z(Z, dimensions, jitter, plot_embedding_facet)
    df = dict_concat(
        {
            "var": n,
            legend_title: data[:, i],
            dimensions[0]: Z[:, 0],
            dimensions[1]: Z[:, 1],
        }
        for i, n in enumerate(names)
    )
    kwargs.setdefault("palette", "rocket")
    kwargs.setdefault("kind", "scatter")
    g = sns.relplot(
        data=df,
        x=dimensions[0],
        y=dimensions[1],
        hue=legend_title,
        col="var",
        **kwargs,
    )
    g.set_titles("{col_name}")
    return g


def plot_position_legend(
    g: sns.FacetGrid,
    index: Optional[Sequence[int]],
    hue_norm: Tuple[float, float],
    legend_inside: bool = True,
    palette: str = "crest",
):
    """Plot a new legend for the position plot.

    Args:
        g: Facet grid.
        index: Selected data items.
        hue_norm: Limits of the color scale.
        legend_inside: Should the legend be inside the grid if possible. Defaults to True.
        palette: `seaborn` palette. Defaults to "crest".
    """
    col_wrap = g._col_wrap
    facets = g._n_facets
    g.data
    handles, labels = _create_legend(hue_norm, plt.get_cmap(palette), 6)
    legend = {l: h for h, l in zip(handles, labels)}
    inside = legend_inside and col_wrap < facets and facets % col_wrap != 0
    w = 1 / col_wrap
    h = 1 / ((facets - 1) // col_wrap + 1)
    if index is not None:
        size = plt.rcParams["lines.markersize"] ** 2 * 3
        for i, ax in zip(index, g.axes.ravel()):
            ax.scatter(g.data[g._x_var][i], g.data[g._y_var][i], size, "#fd8431", "X")
        g.add_legend(
            legend,
            "Local loss",
            loc="lower center" if inside else "upper right",
            bbox_to_anchor=(1 - w, h * 0.35, w * 0.9, h * 0.6) if inside else None,
        )
        marker = Line2D(
            [], [], linestyle="None", color="#fd8431", marker="X", markersize=5
        )
        g.add_legend(
            {"": marker},
            "Selected",
            loc="upper center" if inside else "lower right",
            bbox_to_anchor=(1 - w, h * 0.05, w * 0.9, h * 0.3) if inside else None,
        )
    else:
        g.add_legend(
            legend,
            "Local loss",
            loc="center" if inside else "center right",
            bbox_to_anchor=(1 - w, 0.05, w * 0.9, h * 0.9) if inside else None,
        )
    if inside:
        plt.tight_layout()
    else:
        g.tight_layout()


def plot_density_facet(
    data: np.ndarray,
    names: Sequence[str],
    clusters: Optional[np.ndarray] = None,
    **kwargs,
) -> sns.FacetGrid:
    """Plot density plots.

    Args:
        data: Data matrix.
        names: Column names.
        clusters: Cluster labels. Defaults to None.

    Returns:
        The plot.
    """
    df = dict_concat(
        {"var": n, "Value": data[:, i], "Cluster": clusters}
        for i, n in enumerate(names)
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
        **kwargs,
    )
    g.set_titles("{col_name}")
    g.set_xlabels("")
    return g
