###############################################################################
#
# This experiment is meant to select good radius for the datasets.
# This experiment uses data from `large_evaluation.py` (run that first!).
# Run this script to produce plots from the results:
#   `python experiments/parameter_selection.py`
#
###############################################################################

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
from experiments.large_evaluation import get_results


def plot_parameter_nn(
    df: pd.DataFrame,
    metric: str = "fidelity_nn",
    title: str = "Fidelity",
    pdf: bool = False,
):
    df2 = df[df["method"] == "Slisemap"].copy()
    df2["radius"] = df2["radius"].astype("category")
    df2["radius"].cat.categories = np.round(df2["radius"].cat.categories, 1)
    df2 = pd.melt(
        df2,
        ["data", "radius", "job_index"],
        [c for c in df2.columns if metric in c],
        "NN",
        title,
    )
    df2["Nearest Neighbours"] = df2["NN"].apply(lambda x: int(x[-2:]) / 100)
    g = sns.relplot(
        data=df2,
        x="Nearest Neighbours",
        y=title,
        hue="radius",
        style="radius",
        col="data",
        col_wrap=4,
        kind="line",
        facet_kws=dict(sharey=False, legend_out=False),
        height=3,
        ci=None,
    )
    sns.move_legend(
        g,
        "center",
        bbox_to_anchor=(0.75, 0, 0.25, 1 / np.ceil(len(g.axes) / 4)),
        frameon=False,
    )
    g._legend.set_title("$z_{radius}$")
    g.set_titles("{col_name}")
    if pdf:
        plt.savefig(
            Path(__file__).parent
            / "results"
            / f"parameter_selection_{title.lower()}.pdf"
        )
        plt.close()
    else:
        plt.show()


def plot_parameter(
    df: pd.DataFrame,
    metric: str = "accuracy_new",
    title: str = "Accuracy",
    pdf: bool = False,
):
    df2 = df[df["method"] == "Slisemap"].copy()
    df2 = df2[["data", "radius", metric]].rename(columns={metric: title})
    g = sns.relplot(
        data=df2,
        x="radius",
        y=title,
        col="data",
        col_wrap=4,
        facet_kws=dict(sharey=False),
        kind="line",
        height=3,
    )
    g.set_titles("{col_name}")
    if pdf:
        plt.savefig(
            Path(__file__).parent
            / "results"
            / f"parameter_selection_{title.lower()}.pdf"
        )
        plt.close()
    else:
        plt.show()


def print_parameter(df: pd.DataFrame, metric: str = "accuracy_new"):
    df2 = df[df["method"] == "Slisemap"][["data", "radius", metric]].copy()
    df2 = (
        df2.groupby(["data", "radius"], as_index=False, sort=False)
        .agg("mean")
        .pivot(index="data", columns="radius", values=metric)
    )
    print(
        df2.to_latex(
            float_format="%.3f",
            sparsify=True,
            label="tab:parameter",
            caption="TODO",
            escape=False,
            na_rep="",
            position="ht",
        )
    )


if __name__ == "__main__":
    df = get_results()
    plot_parameter_nn(df, "fidelity_nn", "Fidelity", True)
    plot_parameter_nn(df, "coverage_gl03_nn", "Coverage", True)
