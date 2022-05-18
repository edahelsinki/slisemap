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

sys.path.append(str(Path(__file__).parent.parent))  # Add the project root to the path
from slisemap import Slisemap
from experiments.data import get_boston


def lineup(
    sm: Slisemap,
    radius=[2.0, 3.0, 3.5, 4.0, 5.0, 6.0],
    name=None,
    pdf=False,
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
    df = pd.concat(df)
    g = sns.relplot(
        data=df, x="x", y="y", col="l", col_wrap=3, kind="scatter", height=3.3
    )
    g.set_titles("{col_name}")
    g.set(xlabel=None, ylabel=None)  # , xticklabels=[], yticklabels=[])
    plt.tight_layout()
    if pdf:
        plt.savefig(Path(__file__).parent / "results" / f"lineup_{name}.pdf")
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
    sm = Slisemap(X, y, lasso=1e-3)
    print("Optimising, radius:", sm.radius)
    sm.optimise()

    # Points with identical local models will have identical embeddings.
    # Adding jitter makes the true sizes of the clusters easier to see.
    jitter = np.random.normal(0, 0.1, sm.get_Z().shape)

    sm.plot(
        clusters=4, bars=5, jitter=jitter, variables=names, figsize=(8, 4), show=False
    )
    plt.savefig(Path(__file__).parent / "results" / f"local_models_boston.pdf")
    plt.close()

    lineup(sm, name="boston", jitter=jitter, pdf=True)

    sm.plot_position(
        index=[1, 3, 6, 10, 13, 8, 21], jitter=jitter, height=3.3, show=False
    )
    plt.savefig(Path(__file__).parent / "results" / "location_boston.pdf")
    plt.close()

    sm.plot_dist(
        clusters=4,
        variables=names,
        targets="Value in $1000's",
        X=Xn,
        Y=yn,
        col_wrap=5,
        height=3,
        show=False,
    )
    plt.savefig(Path(__file__).parent / "results" / "cluster_dist_boston.pdf")
    plt.close()
