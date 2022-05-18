###############################################################################
#
# This experiment compares how Slisemap scales on GPU vs CPU.
# This experiment is designed to be run in parallel (e.g. on a computer cluster).
# This experiment requires a GPU (and pytorch with cuda support).
#
# Run this script to perform the experiments, where $index is [1..10]:
#   `python experiments/gpu_scaling.py $index cpu`
#   `python experiments/gpu_scaling.py $index cuda`
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

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

sys.path.append(str(Path(__file__).parent.parent))  # Add the project root to the path
from slisemap import Slisemap
from experiments.data import get_rsynth

RESULTS_DIR = Path(__file__).parent / "results" / "gpu"


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

    return dict(n=n, m=m, time=time, device=device, job_index=job_index)


def plot_scaling(df: pd.DataFrame, pdf: bool = False):
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
    )
    g.set
    g.set_titles("{col_name}")
    g.set(yscale="log", xscale="log")
    g.axes[0, 0].set(xlabel="n")
    g.axes[0, 1].set(xlabel="m")
    g.axes[0, 0].yaxis.set_major_formatter(ScalarFormatter())
    g.axes[0, 0].xaxis.set_major_formatter(ScalarFormatter())
    g.axes[0, 1].xaxis.set_major_formatter(ScalarFormatter())
    if pdf:
        plt.savefig(Path(__file__).parent / "results" / "gpu_scaling.pdf")
        plt.close()
    else:
        plt.show()


def get_results(filter=True):
    files = glob(str(RESULTS_DIR / "*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["device"] = df["device"].astype("category")
    if filter:
        df = df[(df["n"] != 1000) + (df["m"] < 300)]
        num = len(files) / len(df["device"].cat.categories)
        filter = (
            df.groupby(["device", "n", "m"])
            .agg(dict(time=lambda x: len(x) > num * 0.8))
            .to_dict()["time"]
        )
        df = df[df.apply(lambda a: filter[(a["device"], a["n"], a["m"])], 1)]
    return df


if __name__ == "__main__":
    if len(sys.argv) == 1:
        df = get_results()
        plot_scaling(df, True)
    else:
        SIZES = [
            [(1000, i) for i in (5, 10, 20, 40, 80, 160)],
            [(i, 10) for i in (100, 220, 470, 2200, 4700, 10000)],
        ]
        job_index = int(sys.argv[1]) - 1
        if len(sys.argv) > 2:
            device = sys.argv[2]
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        out_path = RESULTS_DIR / f"gpu_{job_index:02d}_{device}.parquet"
        if not out_path.exists():
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            print(f"{job_index:02d} {device}: Setup", flush=True)
            results = []
            for sizes in SIZES:
                for n, m in sizes:
                    gc.collect()
                    res = calculate(device, job_index, n, m)
                    results.append(res)
                    df = pd.DataFrame(results)
                    df.to_parquet(out_path)
                    if res["time"] > 2000:
                        break
            print(f"{job_index:02d} {device}: Done", flush=True)
