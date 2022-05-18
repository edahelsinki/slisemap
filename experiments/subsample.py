###############################################################################
#
# This experiment checks how large subset Slisemap really needs.
# This experiment is designed to be run in parallel (e.g. on a computer cluster).
#
# Run this script to perform the experiments, where $index is [1..80]:
#   `python experiments/subsample.py $index`
#
# Run this script again without additional arguments to produce a plot from the results:
#   `python experiments/subsample.py`
#
###############################################################################

import gc
import sys
from glob import glob
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
import seaborn as sns
from matplotlib import pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))  # Add the project root to the path
from slisemap.slisemap import Slisemap
from experiments.large_evaluation import get_data
from slisemap.metrics import accuracy, fidelity, coverage

RESULTS_DIR = Path(__file__).parent / "results" / "subset"


def get_data2(index, n=1000):
    li = index + 60  # avoid RSynth and Boston
    if li >= 130:  # avoid Boston (XAI)
        li += 10
    return get_data(li, n)[:-2]


def distance_to_closest_point(Z1, Z2, second=False):
    D = torch.cdist(Z1, Z2)
    if second:
        D[torch.arange(D.shape[0]), torch.argmin(D, 1)] += np.inf
    return torch.min(D, 1)[0].mean().cpu().item()


def loss_no_reg(sm: Slisemap):
    kD = sm.get_W(numpy=False)
    L = sm.get_L(numpy=False)
    return (torch.sum(kD * L) / L.shape[0]).cpu().item()


def plot_distance(df, pdf=False):
    g = sns.relplot(
        data=df[df["optim"]],
        x="n",
        y="distance",
        col="data",
        hue="old",
        style="individual",
        col_wrap=4,
        height=3,
        kind="line",
    )
    g.set_titles("{col_name}")
    g.set(ylabel="Average Minimum Distance")
    g._legend.set_title(None)
    if pdf:
        plt.savefig(Path(__file__).parent / "results" / "subset_distance.pdf")
        plt.close()
    else:
        plt.show()


def plot_subsample(df, metric="loss", label="Mean Loss", ind=False, pdf=False):
    df2 = df[df["optim"] * (df["old"] + (df["individual"] == ind))].copy()
    df2["set"] = np.array(["Test", "Train"])[df2["old"].values.astype(int)]
    g = sns.relplot(
        data=df2,
        x="n",
        y=metric,
        col="data",
        hue="set",
        style="set",
        col_wrap=4,
        height=3,
        kind="line",
        facet_kws=dict(sharey=False),
    )
    g.set_titles("{col_name}")
    g.set(ylabel=label)
    g._legend.set_title(None)
    if pdf:
        plt.savefig(
            Path(__file__).parent
            / "results"
            / f"subset_{metric}_{'ind' if ind else 'full'}.pdf"
        )
        plt.close()
    else:
        plt.show()


def get_results():
    files = glob(str(RESULTS_DIR / "*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    data_categories = [
        "Air Quality",
        "Air Quality (XAI)",
        "Spam",
        "Spam (XAI)",
        "Higgs",
        "Higgs (XAI)",
        "Covertype",
        "Covertype (XAI)",
    ]
    df["data"] = (
        df["data"]
        .astype("category")
        .cat.set_categories(data_categories)
        .cat.remove_unused_categories()
    )
    df["method"] = df["method"].astype("category")
    return df


def calculate(sm, sm_subset, sm_new, size, job, data):
    sm_subset._X = sm.X[:size]
    sm_subset._Y = sm.Y[:size]
    sm_subset._B = sm.B[:size]
    sm_subset._Z = sm.Z[:size]
    time = timer()
    sm_subset.optimise()
    time = timer() - time
    results = [
        evaluate(
            sm_subset,
            sm_subset,
            time,
            job,
            data,
            True,
            True,
            False,
            sm_subset.value(individual=True),
        )
    ]
    Xnew = sm_new.X[:, : sm_new.m - sm_new.intercept]
    for ind in [True]:  # [True, False]:
        for opt in [True, False]:
            time = timer()
            Bnew, Znew, loss = sm_subset.fit_new(Xnew, sm_new.Y, opt, ind, loss=True)
            time = timer() - time
            sm_new._B = torch.as_tensor(Bnew, **sm.tensorargs)
            sm_new._Z = torch.as_tensor(Znew, **sm.tensorargs)
            results.append(
                evaluate(
                    sm_subset,
                    sm_new,
                    time,
                    job,
                    data,
                    False,
                    opt,
                    ind,
                    loss,
                )
            )
    return results


def evaluate(sm_subset, sm_other, time, job, data, old, optim, individual, loss):
    return dict(
        job=job,
        method="Slisemap",
        data=data,
        n=sm_subset.n,
        m=sm_subset.m,
        old=old,
        optim=optim,
        individual=individual,
        time=time,
        fidelity=fidelity(sm_other),
        loss=sm_other.value(True).mean(),
        loss_nr=loss_no_reg(sm_other),
        distance=distance_to_closest_point(
            sm_other.get_Z(numpy=False),
            sm_subset.get_Z(numpy=False),
            old or not optim,
        ),
        loss_marginal=loss.mean(),
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        df = get_results()
        # plot_subsample(df, "loss_nr", "Mean Loss", ind=False, pdf=True)
        # plot_subsample(df, "loss_nr", "Mean Loss", ind=True, pdf=True)
        # plot_subsample(df, "fidelity", "Fidelity", ind=False, pdf=True)
        plot_subsample(df, "fidelity", "Fidelity", ind=True, pdf=True)
    else:
        job = int(sys.argv[1]) - 1
        sm, data, Xnew, ynew = get_data2(job, 2000)
        out_path = RESULTS_DIR / f"subset_{job:02d}_{data}.parquet"
        if not out_path.exists():
            sm.get_loss_fn()
            results = []
            sm_subset = sm.copy()
            sm_new = sm.copy()
            sm_new.z_norm = 0
            sm_new.radius = 0
            sm_new.get_loss_fn()
            sm_new._X = torch.as_tensor(Xnew[:1000], **sm.tensorargs)
            if sm.intercept:
                sm_new._X = torch.cat(
                    [sm_new.X, torch.ones([sm_new.n, 1], **sm.tensorargs)], 1
                )
            if len(ynew.shape) == 1:
                sm_new._Y = torch.as_tensor(ynew[:1000], **sm.tensorargs)[:, None]
            else:
                sm_new._Y = torch.as_tensor(ynew[:1000], **sm.tensorargs)
            print(f"{job:02d} {data}: Setup", flush=True)
            for i in np.linspace(50, sm.n, 12, dtype=int):
                gc.collect()
                try:
                    results.extend(calculate(sm, sm_subset, sm_new, i, job, data))
                    print(f"{job:02d} {data}:{i:5d} Evaluated", flush=True)
                    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                    df = pd.DataFrame(results)
                    df["method"] = df["method"].astype("category")
                    df["data"] = df["data"].astype("category")
                    df.to_parquet(out_path)
                except Exception as e:
                    print(f"{job:02d} {data}:{i:5d} Error:", e, flush=True)
            print(f"{job:02d} {data}: Done", flush=True)
