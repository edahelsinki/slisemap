"""
Some utility functions for the experiments.
"""


from typing import Dict, Tuple, Union
import seaborn as sns


def paper_theme(
    width: float = 1.0,
    aspect: float = 1.0,
    cols: int = 1,
    rows: int = 1,
    page_width: float = 347.0,
    figsize: bool = False,  # return figsize instead of dict
) -> Union[Dict[str, float], Tuple[float, float]]:
    """Set theme and sizes for plots added to papers.

    Usage:
        sns.relplot(..., **paper_theme(...))
        plt.subplots(..., figsize=paper_theme(..., figsize=True))

    Args:
        width: Fraction of page width. Defaults to 1.0.
        aspect: Aspect ratio of plots. Defaults to 1.0.
        cols: Number of columns in plot. Defaults to 1.
        rows: Number of rows in plot. Defaults to 1.
        page_width: Page width in points. Defaults to 347.0.
        figsize: Return figsize instead of dict (for use with bare pyplot figures). Defaults to False.

    Returns:
        The size.
    """
    if width == 1.0:
        width = 0.99
    scale = page_width / 72.27  # from points to inches
    size = (width * scale, width / cols * rows / aspect * scale)
    sns.set_theme(
        context={k: v * 0.6 for k, v in sns.plotting_context("paper").items()},
        style=sns.axes_style("ticks"),
        palette="bright",
        # font="cmr10",
        rc={
            "figure.figsize": size,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 1e-4,
        },
    )
    if figsize:
        return size
    else:
        return dict(height=width * scale / cols * aspect, aspect=aspect)
