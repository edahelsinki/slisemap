###############################################################################
#
# This experiment creates the introductory toy example for the paper.
# Run this script to produce a plot: `python experiments/toy.py`.
#
###############################################################################

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.append(str(Path(__file__).parent.parent))
from slisemap.slisemap import Slisemap

RESULTS_DIR = Path(__file__).parent / "results"


def setseeds(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def toydata(n=99, m=4):
    X = np.random.randn(n, m)
    y = np.amax(X[:, :3], 1)
    c = np.argmax(X[:, :3], 1) + 1
    return X, y, c


def toydata2(X, b=1):
    XX = X[:, :3]
    Z = np.exp(b * (XX - np.max(XX, axis=1, keepdims=True)))
    Z = Z / np.sum(Z, axis=1, keepdims=True)
    y = np.sum(Z * XX, axis=1)
    c = np.argmax(XX, 1) + 1
    return y, c


def jitter(x):
    stdev = 0.01 * (max(x) - min(x))
    return x + np.random.randn(len(x)) * stdev


def plot_scatter(df: pd.DataFrame, file: str):
    g: sns.FacetGrid = sns.relplot(
        data=df,
        x="Z1",
        y="Z2",
        hue="c",
        style="c",
        col="method",
        height=3.5,
        palette="bright",
        kind="scatter",
        facet_kws=dict(sharex=False, sharey=False),
    )
    for ax in g.axes.flat:
        ax.axis("equal")
    g.set_titles("{col_name}")
    g.set(ylabel=None, xlabel=None)
    g.tight_layout()
    plt.savefig(RESULTS_DIR / file)


if __name__ == "__main__":
    setseeds(1)
    X, y, c = toydata()
    y2, c2 = toydata2(X)

    sm = Slisemap(X, y, radius=3.5, intercept=False, lasso=0.0)
    sm.optimise()
    Z = sm.get_Z(rotate=True)

    sm2 = Slisemap(X, y2, radius=3.5, intercept=False, lasso=0.0)
    sm2.optimise()
    Z2 = sm2.get_Z(rotate=True)

    pca = PCA(n_components=2)
    Zpca0 = pca.fit_transform(X)

    df = pd.DataFrame(
        dict(
            Z1=np.concatenate([Zpca0[:, 0], jitter(Z[:, 0])]),
            Z2=np.concatenate([Zpca0[:, 1], jitter(Z[:, 1])]),
            method=np.repeat(["PCA", "SLISEMAP"], len(c)),
            c=pd.Categorical(np.tile(c, 2)),
        )
    )
    plot_scatter(df, "toy.pdf")

    df = pd.DataFrame(
        dict(
            Z1=np.concatenate([Zpca0[:, 0], jitter(Z[:, 0]), jitter(Z2[:, 0])]),
            Z2=np.concatenate([Zpca0[:, 1], jitter(Z[:, 1]), jitter(Z2[:, 1])]),
            method=np.repeat(["PCA", "SLISEMAP (max)", "SLISEMAP (softmax)"], len(c)),
            c=pd.Categorical(np.tile(c, 3)),
        )
    )
    plot_scatter(df, "toy12.pdf")
