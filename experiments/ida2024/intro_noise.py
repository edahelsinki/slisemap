###############################################################################
#
# This experiment compares Slipmap and Slisemap on completely random data.
#
# Run this script to produce a plot of the embeddings:
#   `python experiments/ida2024/noise_clusters.py`
#
###############################################################################

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from project_paths import MANUSCRIPT_DIR

from slisemap import Slisemap
from slisemap.utils import squared_distance
from slisemap.slipmap import Slipmap
from slisemap.plot import plot_embedding
from experiments.utils import paper_theme


def generate_data(n=500, m=10, o=1, noise=False, seed=42):
    print(
        "Generating data", "with pure noise" if noise else "with Y=max(X) + small noise"
    )
    prng = np.random.default_rng(seed)
    X = prng.normal(0, 1, (n, m))
    if noise:
        Y = prng.normal(0, 1, (n, o))
    elif o == 1:
        Y = np.max(X[:, :3], 1) + prng.normal(0, 0.01, n)
    return X, Y


def train_slisemap(X, Y):
    print("Training Slisemap...")
    sm = Slisemap(X, Y, lasso=0.01, random_state=42)
    sm.optimise()
    return sm


def train_slipmap(X, Y):
    print("Training Slipmap...")
    sm = Slipmap(
        X, Y, lasso=0.01, distance=squared_distance, radius=2.5, prototypes=1.0
    )
    sm.optimise()
    return sm


def plot_slisemap(sm, ax, title):
    plot_embedding(
        sm.get_Z(rotate=True), ["", ""], title, color=sm.get_Y().ravel(), ax=ax
    )


def plot_slipmap(sm, ax, title):
    plot_embedding(sm.get_Z(), ["", ""], title, color=sm.get_Y().ravel(), ax=ax)
    Zp = sm.get_Zp()
    ax.scatter(Zp[:, 0], Zp[:, 1], edgecolors="grey", facecolors="none", alpha=0.7)


def plot(smc: Slisemap, spc: Slipmap, smn: Slisemap, spn: Slipmap, pdf: bool = True):
    print("Plotting...")
    if pdf:
        figsize = paper_theme(width=1.0, cols=4, figsize=True)
    else:
        sns.set_theme(context="paper", style=sns.axes_style("ticks"), palette="bright")
        figsize = (16, 6)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
    plot_slisemap(smc, ax1, "Slisemap with little noise")
    plot_slipmap(spc, ax2, "Slipmap with little noise")
    plot_slisemap(smn, ax3, "Slisemap on pure noise")
    plot_slipmap(spn, ax4, "Slipmap on pure noise")
    sns.despine(fig)
    plt.tight_layout()
    if pdf:
        plt.savefig(MANUSCRIPT_DIR / "noise_clusters.pdf")
        plt.close()
        print("Plot saved to", str(MANUSCRIPT_DIR / "noise_clusters.pdf"))
    else:
        plt.show()


if __name__ == "__main__":
    X, Y = generate_data(noise=False)
    smc = train_slisemap(X, Y)
    spc = train_slipmap(X, Y)
    X, Y = generate_data(noise=True)
    smn = train_slisemap(X, Y)
    spn = train_slipmap(X, Y)
    plot(smc, spc, smn, spn)
