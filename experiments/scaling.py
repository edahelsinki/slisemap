###############################################################################
#
# This experiment compares how Slisemap scales on GPU vs CPU.
# This experiment is designed to be run in parallel (e.g. on a computer cluster).
# This experiment requires a GPU (and pytorch with cuda support).
#
# Run this script to perform the experiments, where $index is [1..10]:
#   `python experiments/scaling.py $index cpu`
#   `python experiments/scaling.py $index cuda`
#   `python experiments/scaling.py $index other`
#
# Run this script again without additional arguments to produce a plot from the results:
#   `python experiments/gpu_scaling.py`
#
###############################################################################

import random
import sys
import gc
from glob import glob
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, StrMethodFormatter

if __name__ == "__main__":  # Add the project root to the path
    sys.path.insert(0, str(Path(__file__).parent.parent))
from slisemap import Slisemap
from slisemap.utils import dict_array, dict_append
from experiments.data import get_rsynth
from experiments.large_evaluation import (
    mlle_embedding,
    umap_embedding,
    pca_embedding,
    isomap_embedding,
    tsne_embedding,
)
from experiments.utils import paper_theme


RESULTS_DIR = Path(__file__).parent / "results" / "scaling"


def calculate(device, job_index, n, m):
    seed = 42 + job_index + n * m + m
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    c, X, y, B = get_rsynth(n, m, 3, 0.25, 0.1)
    sm1 = Slisemap(X, y, lasso=0.0001, cuda=device == "cuda")

    time = timer()
    sm1.optimise()
    time = timer() - time
    print(f"{job_index:2d} {device:4}: {n} x {m}  {time:.1f} s", flush=True)

    return dict(
        n=n, m=m, time=time, device=device, method="SLISEMAP", job_index=job_index
    )


def calculate_other(job_index, n, m):
    seed = 42 + job_index + n * m + m
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    c, X, y, B = get_rsynth(n, m, 3, 0.25, 0.1)

    grid = {
        "PCA": pca_embedding,
        "MLLE": mlle_embedding,
        "ISOMAP": isomap_embedding,
        "t-SNE": tsne_embedding,
        "UMAP": umap_embedding,
    }
    times = []
    for key, fn in grid.items():
        time = timer()
        fn(X, y)
        time = timer() - time
        times.append(time)
        print(f"{job_index:2d} {key:6}: {n} x {m}  {time:.1f} s", flush=True)

    return dict(
        n=n,
        m=m,
        time=times,
        device="cpu",
        method=list(grid.keys()),
        job_index=job_index,
    )


def plot_gpu_scaling(df: pd.DataFrame, pdf: bool = False):
    df = df[df["method"] == "SLISEMAP"]
    df = df.rename(columns=dict(time="Time (s)", device="Device"))
    df2 = pd.concat(
        (
            df[df["m"] == df["m"].mode()[0]]
            .rename(columns=dict(n="x"))
            .assign(s=lambda df3: f"m = {df3['m'].iloc[0]}"),
            df[df["n"] == df["n"].mode()[0]]
            .rename(columns=dict(m="x"))
            .assign(s=lambda df3: f"n = {df3['n'].iloc[0]}"),
        ),
        ignore_index=True,
    )
    g = sns.relplot(
        data=df2,
        x="x",
        y="Time (s)",
        hue="Device",
        style="Device",
        col="s",
        facet_kws=dict(sharey="row", sharex="none"),
        kind="line",
        **paper_theme(0.8, 1, 2),
    )
    g.set_titles("{col_name}")
    g.set(yscale="log", xscale="log")
    g.axes[0, 0].set(xlabel="n")
    g.axes[0, 1].set(xlabel="m")
    g.axes[0, 0].yaxis.set_major_formatter(ScalarFormatter())
    g.axes[0, 0].xaxis.set_major_formatter(ScalarFormatter())
    g.axes[0, 1].xaxis.set_major_formatter(ScalarFormatter())
    g.tight_layout()
    if pdf:
        plt.savefig(RESULTS_DIR / ".." / "scaling_gpu.pdf")
        plt.close()
    else:
        plt.show()


def plot_other_scaling(df: pd.DataFrame, pdf: bool = False):
    df = df[df["device"] == "cpu"]
    df = df.rename(columns=dict(time="Time (s)", method="Method"))
    df2 = pd.concat(
        (
            df[df["m"] == df["m"].mode()[0]]
            .rename(columns=dict(n="x"))
            .assign(s=lambda df3: f"m = {df3['m'].iloc[0]}"),
            df[df["n"] == df["n"].mode()[0]]
            .rename(columns=dict(m="x"))
            .assign(s=lambda df3: f"n = {df3['n'].iloc[0]}"),
        ),
        ignore_index=True,
    )
    g: sns.FacetGrid = sns.relplot(
        data=df2,
        x="x",
        y="Time (s)",
        hue="Method",
        style="Method",
        col="s",
        facet_kws=dict(sharey="row", sharex="none"),
        kind="line",
        ci=None,
        **paper_theme(0.8, 1, 2),
    )
    g.set_titles("{col_name}")
    g.set(yscale="log", xscale="log")
    g.axes[0, 0].set(xlabel="n")
    g.axes[0, 1].set(xlabel="m")
    g.axes[0, 0].yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    g.axes[0, 0].xaxis.set_major_formatter(ScalarFormatter())
    g.axes[0, 1].xaxis.set_major_formatter(ScalarFormatter())
    if pdf:
        plt.savefig(RESULTS_DIR / ".." / "scaling_other.pdf")
        plt.close()
    else:
        plt.show()


def get_results(filter=True):
    files = sorted(glob(str(RESULTS_DIR / "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["device"] = df["device"].astype("category")
    index = sorted(np.unique(df["method"], return_index=True)[1])
    methods = df["method"][index]
    df["method"] = df["method"].astype("category").cat.reorder_categories(methods)
    if filter:
        # df = df[(df["n"] != 1000) + (df["m"] < 300)]
        group = df.groupby(["device", "method", "n", "m"], observed=True)
        num = group.count()["time"].max()
        df = group.apply(
            lambda df: df if df.shape[0] > num * 0.8 else None
        ).reset_index()
    return df


if __name__ == "__main__":
    if len(sys.argv) == 1:
        df = get_results()
        plot_gpu_scaling(df, True)
        plot_other_scaling(df, True)
    else:
        sizes = sorted(
            [(1000, i) for i in (5, 10, 20, 40, 80, 160, 320)]
            + [(i, 10) for i in (100, 220, 470, 2200, 4700, 10000)],
            key=lambda x: x[0] ** 2 * x[1],
        )
        job_index = int(sys.argv[1]) - 1
        if len(sys.argv) > 2:
            device = sys.argv[2]
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        out_path = RESULTS_DIR / f"scaling_{job_index:02d}_{device}.parquet"
        if not out_path.exists():
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            print(f"{job_index:02d} {device}: Setup", flush=True)
            results = None
            for n, m in sizes:
                gc.collect()
                if device == "other":
                    res = calculate_other(job_index, n, m)
                else:
                    res = calculate(device, job_index, n, m)
                if results is None:
                    results = dict_array(res)
                else:
                    results = dict_append(results, res)
                df = pd.DataFrame(results)
                df["device"] = df["device"].astype("category")
                df["method"] = df["method"].astype("category")
                df.to_parquet(out_path)
                if np.all(res["time"] > 2000):
                    break
            print(f"{job_index:02d} {device}: Done", flush=True)
