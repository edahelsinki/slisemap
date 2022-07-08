###############################################################################
#
# This experiment creates visualisations for the paper.
# Run this script to produce plots: `python experiments/visualisations.py`.
#
###############################################################################

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import seaborn as sns
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from slisemap import Slisemap
from experiments.data import get_boston
from experiments.utils import paper_theme

RESULTS_DIR = Path(__file__).parent / "results"


def lineup(
    sm: Slisemap,
    radius=[2.0, 3.0, 3.5, 4.0, 5.0, 6.0],
    jitter=0.1,
):
    if isinstance(jitter, float):
        jitter = np.random.normal(0, jitter, sm.get_Z().shape)
    df = []
    for l in radius:
        print("radius:", l)
        smc = sm.copy()
        if np.round(smc.radius, 2) != l:
            smc.radius = l
            smc.optimise()
        Z = smc.get_Z(rotate=True) + jitter
        df.append(pd.DataFrame(dict(x=Z[:, 0], y=Z[:, 1], l=f"$z_{{radius}} = {l:g}$")))
    df = pd.concat(df, ignore_index=True)
    return df


def plot_lineup(df, name, pdf=False):
    g = sns.relplot(
        data=df,
        x="x",
        y="y",
        col="l",
        col_wrap=3,
        kind="scatter",
        **paper_theme(0.8, 1, 3, 2),
    )
    g.set_titles("{col_name}")
    g.set(xlabel=None, ylabel=None)  # , xticklabels=[], yticklabels=[])
    g.tight_layout()
    if pdf:
        plt.savefig(RESULTS_DIR / f"lineup_{name}.pdf")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y, names = get_boston(names=True, remove_B=True)
    Xn, yn = get_boston(normalise=False, remove_B=True)
    sm = Slisemap(X, y, lasso=2e-3)
    print("Optimising, radius:", sm.radius)
    sm.optimise()

    # Points with identical local models will have identical embeddings.
    # Adding jitter makes the true sizes of the clusters easier to see.
    jitter = np.random.normal(0, 0.1, sm.get_Z().shape)
    clusters = 5

    sm.plot(
        clusters=clusters,
        bars=5,
        jitter=jitter,
        variables=names,
        figsize=paper_theme(1, 1, 2, figsize=True),
        show=False,
    )
    plt.savefig(RESULTS_DIR / f"local_models_boston.pdf")
    plt.close()

    g = sm.plot_position(
        index=[1, 3, 6, 10, 13, 8, 21],
        jitter=jitter,
        show=False,
        col_wrap=4,
        **paper_theme(1, 1, 4, 2),
    )
    g.set(xlabel=None, ylabel=None)
    g.axes[-4].tick_params(labelbottom=False)
    plt.tight_layout(rect=g._tight_layout_rect)
    plt.savefig(RESULTS_DIR / "location_boston.pdf")
    plt.close()

    g = sm.plot_dist(
        clusters=clusters,
        variables=names,
        targets="Value in $1000's",
        X=Xn,
        Y=yn,
        col_wrap=5,
        **paper_theme(1, 1, 5, 3),
        show=False,
    )
    g.set(ylabel=None, yticks=[])
    g.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(RESULTS_DIR / "cluster_dist_boston.pdf")
    plt.close()

    lu = lineup(sm, jitter=jitter)
    plot_lineup(lu, "boston", True)
