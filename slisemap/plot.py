"""
Utility functions for plotting
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, figure
from matplotlib.colors import Colormap
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
    cmap: Union[str, Colormap] = "crest",
    markers: int = 5,
    min_digits: int = 3,
    max_digits: int = 6,
) -> Tuple[List[Line2D], List[str]]:
    cmap = plt.get_cmap(cmap)
    handles = [
        Line2D([], [], linewidth=0, color=cmap(n), marker="o")
        for n in np.linspace(0.0, 1.0, markers)
    ]
    values = np.linspace(*hue_norm, markers)
    if max(abs(i) for i in hue_norm) < 10:
        min_digits += 1
    for i in range(max_digits):
        labels = [f"{l:.{i}f}" for l in values]
        length = max(len(l) for l in labels)
        if length < min_digits:
            continue
        if len(np.unique(labels)) == markers:
            if length == min_digits:
                # Do not count '-' for short labels
                if not any(len(l) == length and v >= 0 for v, l in zip(values, labels)):
                    i += 1
            break
    labels = [f"{l:{length}.{i}f}" for l in values]
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
    if not isinstance(jitter, (float, int)):
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
    **kwargs: Any,
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
    Keyword Args:
        **kwargs: Additional arguments to `seaborn.scatterplot`.

    Returns:
        The plot.
    """
    Z, dimensions = _prepare_Z(Z, dimensions, jitter, plot_embedding)
    kwargs.setdefault("rasterized", Z.shape[0] > 2_000)
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
        ax.legend(*_create_legend(color_norm, kwargs["palette"], 5), title=color_name)
    elif color_name is not None and color_name != "":
        ax.legend(title=color_name)
    ax.set_xlabel(dimensions[0])
    ax.set_ylabel(dimensions[1])
    ax.axis("equal")
    ax.set_title(title)
    return ax


def plot_matrix(
    B: np.ndarray,
    coefficients: Sequence[str],
    title: str = "Local models",
    palette: str = "RdBu",
    xlabel: str = "Data items sorted left to right",
    items: Optional[Sequence[str]] = None,
    **kwargs,
) -> plt.Axes:
    """Plot local models in a heatmap.

    Args:
        B: Local model coefficients.
        coefficients: Coefficient names.
        palette: `seaborn` palette. Defaults to "RdBu".
        title: Title of the plot. Defaults to "Local models".
        xlabel: Label for the x-axis. Defaults to "Data items sorted left to right".
        items: Ticklabels for the x-axis. Defaults to None.
    Keyword Args:
        **kwargs: Additional arguments to `seaborn.heatmap`.

    Returns:
        The plot.
    """
    kwargs.setdefault("rasterized", B.shape[0] * B.shape[1] > 20_000)
    ax = sns.heatmap(B.T, center=0, cmap=palette, robust=True, **kwargs)
    ax.set_yticks(np.arange(len(coefficients)) + 0.5)
    ax.set_yticklabels(coefficients, rotation=0)
    ax.set_xlabel(xlabel)
    if items is None:
        ax.set_xticklabels([])
    else:
        ax.set_xticks(np.arange(len(items)) + 0.5)
        ax.set_xticklabels(items)
    ax.set_title(title)
    return ax


def plot_barmodels(
    B: np.ndarray,
    clusters: np.ndarray,
    centers: np.ndarray,
    coefficients: Sequence[str],
    bars: Union[bool, int, Sequence[str]] = True,
    palette: str = "bright",
    xlabel: Optional[str] = "Coefficients",
    title: Optional[str] = "Cluster mean local model",
    **kwargs: Any,
) -> plt.Axes:
    """Plot local models in a barplot.

    Args:
        B: Local model coefficients.
        clusters: Cluster labels.
        centers: Cluster centers.
        coefficients: Coefficient names.
        bars: Number / list of variables to show (or a bool for all). Defaults to True.
        palette: `seaborn` palette. Defaults to "bright".
    Keyword Args:
        **kwargs: Additional arguments to `seaborn.barplot`.

    Returns:
        The plot.
    """
    if isinstance(bars, Sequence):
        mask = [coefficients.index(var) for var in bars]
        coefficients = bars
        B = B[:, mask]
    elif isinstance(bars, bool):
        pass
    elif isinstance(bars, int):
        influence = np.abs(centers)
        influence = influence.max(0) + influence.mean(0)
        mask = np.argsort(-influence)[:bars]
        coefficients = np.asarray(coefficients)[mask]
        B = B[:, mask]
    kwargs.setdefault("rasterized", B.shape[0] * B.shape[1] > 20_000)
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
    ax.set(xlabel=xlabel, ylabel=None, title=title, xlim=(-lim, lim))
    return ax


def plot_embedding_facet(
    Z: np.ndarray,
    dimensions: Sequence[str],
    data: np.ndarray,
    names: Sequence[str],
    legend_title: str = "Value",
    jitter: Union[float, np.ndarray] = 0.0,
    share_hue: bool = True,
    equal_aspect: bool = True,
    **kwargs: Any,
) -> sns.FacetGrid:
    """Plot (multiple) embeddings.

    Args:
        Z: Embeddings.
        dimensions: Dimension names.
        data: Data matrix.
        names: Column names.
        legend_title: Legend title. Defaults to "Value".
        jitter: jitter. Defaults to 0.0.
        equal_aspect: Set equal scale for the axes. Defaults to True.
    Keyword Args:
        **kwargs: Additional arguments to `seaborn.relplot`.

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
    kwargs.setdefault("rasterized", Z.shape[0] > 2_000)
    if share_hue:
        kwargs.setdefault("kind", "scatter")
        g = sns.relplot(
            data=df,
            x=dimensions[0],
            y=dimensions[1],
            hue=legend_title,
            col="var",
            **kwargs,
        )
    else:
        fgkws = kwargs.pop("facet_kws", {})
        fgkws.setdefault("height", 5)
        for k in ("height", "aspect", "col_wrap"):
            if k in kwargs:
                fgkws[k] = kwargs.pop(k)
        fgkws.setdefault("legend_out", False)
        g = sns.FacetGrid(data=df, col="var", hue=legend_title, **fgkws)
        for key, ax in g.axes_dict.items():
            mask = df["var"] == key
            df2 = {k: v[mask] for k, v in df.items()}
            sns.scatterplot(
                data=df2,
                hue=legend_title,
                x=dimensions[0],
                y=dimensions[1],
                ax=ax,
                **kwargs,
            )
    if equal_aspect:
        g.set(aspect="equal")
    g.set_titles("{col_name}")
    return g


def plot_density_facet(
    data: np.ndarray,
    names: Sequence[str],
    clusters: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> sns.FacetGrid:
    """Plot density plots.

    Args:
        data: Data matrix.
        names: Column names.
        clusters: Cluster labels. Defaults to None.
    Keyword Args:
        **kwargs: Additional arguments to `seaborn.displot`.

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
    if clusters is not None:
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


def plot_prototypes(Zp: np.ndarray, *axs: plt.Axes):
    """Draw a grid of prototypes.

    Args:
        Zp: Prototype coordinates.
        *args: Axes to draw on.
    """
    Zp, _ = _prepare_Z(Zp, range(2), 0.0, plot_prototypes)
    for ax in axs:
        ax.scatter(Zp[:, 0], Zp[:, 1], edgecolors="grey", facecolors="none", alpha=0.7)


def plot_solution(
    Z: np.ndarray,
    B: np.ndarray,
    coefficients: Sequence[str],
    dimensions: Sequence[str],
    loss: Optional[np.ndarray] = None,
    clusters: Optional[np.ndarray] = None,
    centers: Optional[np.ndarray] = None,
    title: str = "",
    bars: Union[bool, int, Sequence[str]] = True,
    jitter: Union[float, np.ndarray] = 0.0,
    left_kwargs: Dict[str, object] = {},
    right_kwargs: Dict[str, object] = {},
    **kwargs,
) -> figure.Figure:
    """Plot a Slisemap solution

    Args:
        Z: Embedding matrix.
        B: Local model coefficients.
        coefficients: Coefficient names.
        dimensions: Embedding names.
        loss: Local loss vector. Defaults to None.
        clusters: Cluster labels. Defaults to None.
        centers: Cluster centroids. Defaults to None.
        title: Plot title. Defaults to "".
        bars: Plot coefficients in a barplot instead of a heatmap. Defaults to True.
        jitter: Add noise to the embedding. Defaults to 0.0.
        left_kwargs: Keyword arguments to the left (embedding) plot. Defaults to {}.
        right_kwargs: Keyword arguments to the right (matrix/bar) plot. Defaults to {}.
    Keyword Args:
        **kwargs: Additional arguments to `matplotlib.pyplot.subplots`.

    Returns:
        Figure
    """
    kwargs.setdefault("figsize", (12, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, **kwargs)
    if clusters is None:
        plot_embedding(
            Z,
            dimensions,
            jitter=jitter,
            color=loss.ravel(),
            color_name=None if loss is None else "Local loss",
            color_norm=None if loss is None else tuple(np.quantile(loss, (0.0, 0.95))),
            ax=ax1,
            **left_kwargs,
        )
        B = B[np.argsort(Z[:, 0])]
        plot_matrix(B, coefficients, ax=ax2, **right_kwargs)
    else:
        plot_embedding(
            Z, dimensions, jitter=jitter, clusters=clusters, ax=ax1, **left_kwargs
        )
        if bars:
            plot_barmodels(
                B, clusters, centers, coefficients, bars=bars, ax=ax2, **right_kwargs
            )
        else:
            plot_matrix(
                centers,
                coefficients,
                title="Cluster mean local model",
                xlabel="Cluster",
                items=np.unique(clusters),
                ax=ax2,
                **right_kwargs,
            )
    sns.despine(fig)
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_position(
    Z: np.ndarray,
    L: np.ndarray,
    Zs: Optional[np.ndarray],
    dimensions: Sequence[str],
    title: str = "",
    jitter: Union[float, np.ndarray] = 0.0,
    legend_inside: bool = True,
    marker_size: float = 1.0,
    **kwargs,
) -> sns.FacetGrid:
    """Plot local losses for alternative locations for the selected item(s).

    Args:
        Z: Embedding matrix.
        L: Loss matrix.
        Zs: Selected coordinates.
        dimensions: Embedding names.
        title: Plot title. Defaults to "".
        jitter: Add random noise to the embedding. Defaults to 0.0.
        legend_inside: Move the legend inside the grid (if there is an empty cell). Defaults to True.
        marker_size: Multiply the point size with this value. Defaults to 1.0.
    Keyword Args:
        **kwargs: Additional arguments to `seaborn.relplot`.

    Returns:
        FacetGrid
    """
    kwargs.setdefault("palette", "crest")
    kwargs.setdefault("col_wrap", min(4, L.shape[1]))
    if marker_size != 1.0:
        kwargs.setdefault("s", plt.rcParams["lines.markersize"] * marker_size)
    hue_norm = tuple(np.quantile(L, (0.0, 0.95)))
    g = plot_embedding_facet(
        Z,
        dimensions,
        L,
        range(L.shape[1]),
        legend_title="Local loss",
        hue_norm=hue_norm,
        jitter=jitter,
        legend=False,
        **kwargs,
    )
    g.set_titles("")
    plt.suptitle(title)
    # Legend
    col_wrap = kwargs["col_wrap"]
    facets = g._n_facets
    legend = {l: h for h, l in zip(*_create_legend(hue_norm, kwargs["palette"], 6))}
    inside = legend_inside and col_wrap < facets and facets % col_wrap != 0
    w = 1 / col_wrap
    h = 1 / ((facets - 1) // col_wrap + 1)
    if Zs is not None:
        size = plt.rcParams["lines.markersize"] * 18.0
        for i, ax in enumerate(g.axes.ravel()):
            ax.scatter(Zs[i, 0], Zs[i, 1], size, "#fd8431", "X")
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
    return g


def plot_dist(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    loss: np.ndarray,
    variables: Sequence[str],
    targets: Sequence[str],
    dimensions: Sequence[str],
    title: str = "",
    clusters: Optional[np.ndarray] = None,
    scatter: bool = False,
    jitter: Union[float, np.ndarray] = 0.0,
    legend_inside: bool = True,
    **kwargs,
) -> sns.FacetGrid:
    """Plot the distribution of the variables, either as density plots (with clusters) or as scatterplots.

    Args:
        X: Data matrix.
        Y: Target matrix.
        Z: Embedding matrix.
        loss: Local loss vector.
        variables: Variable names.
        targets: Target names.
        dimensions: Embedding names.
        title: Plot title. Defaults to "".
        clusters: Cluster labels. Defaults to None.
        scatter: Plot a scatterplot instead of a density plot. Defaults to False.
        jitter: Add noise to the embedding. Defaults to 0.0.
        legend_inside: Move the legend inside a facet, if possible.. Defaults to True.
    Keyword Args:
        **kwargs: Additional arguments to `seaborn.relplot` or `seaborn.scatterplot`.

    Returns:
        FacetGrid.
    """
    data = np.concatenate((X, Y, loss.ravel()[:, None]), 1)
    labels = variables + targets + ["Local loss"]
    kwargs.setdefault("col_wrap", 4)
    if scatter:
        g = plot_embedding_facet(
            Z, dimensions, data, labels, jitter=jitter, share_hue=False, **kwargs
        )
    else:
        g = plot_density_facet(data, labels, clusters=clusters, **kwargs)
    plt.suptitle(title)
    if scatter or clusters is None or not legend_inside:
        g.tight_layout()
    else:
        legend_inside_facet(g)
    return g
