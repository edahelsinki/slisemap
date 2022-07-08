###############################################################################
#
# This experiment investigates different number of embedding dimensions.
# This experiment is designed to be run in parallel (e.g. on a computer cluster).
#
# Run this script to perform the experiments, where $index is [1..70]:
#   `python experiments/dimensions.py $index`
#
# Run this script again without additional arguments to produce plots from the results:
#   `python experiments/dimensions.py`
#
###############################################################################

import sys
from glob import glob
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

sys.path.insert(
    0, str(Path(__file__).parent.parent)
)  # Add the project root to the path
from slisemap.metrics import (
    coverage,
    euclidean_nearest_neighbours,
    fidelity,
    radius_neighbours,
)
from slisemap.slisemap import Slisemap
from experiments.large_evaluation import get_data, global_model
from experiments.utils import paper_theme

RESULTS_DIR = Path(__file__).parent / "results" / "dimensions"


def get_data2(index):
    indices = (
        list(range(20, 30))  # RSynth 400x15
        + list(range(50, 60))  # Boston
        + list(range(130, 140))  # Boston XAI
        + list(range(60, 70))  # AQ
        + list(range(70, 80))  # AQ XAI
        + list(range(90, 100))  # Spam XAI
        + list(range(110, 120))  # Higgs XAI
    )
    return get_data(indices[index])[:2]


def matching(sm1: Slisemap, sm2: Slisemap, frac: float = 0.1) -> float:
    assert sm1.n == sm2.n
    D1 = sm1.get_D(numpy=False)
    D2 = sm2.get_D(numpy=False)
    num = int(sm1.n * frac)
    N1 = torch.argsort(D1, 1)[:, :num]
    N2 = torch.argsort(D2, 1)[:, :num]
    NN = torch.cat((N1, N2), 1)
    res = torch.zeros(sm1.n)
    for i in range(sm1.n):
        res[i] = 2 - torch.unique(NN[i], sorted=False).numel() / num
    return res.mean().cpu().item()


def distance(sm1: Slisemap, sm2: Slisemap, frac: float = 0.1):
    assert sm1.n == sm2.n
    D1 = sm1.get_D(numpy=False)
    D2 = sm2.get_D(numpy=False)
    diff = torch.abs(D1 - D2)
    num = int(sm1.n * frac)
    results = np.zeros(sm1.n)
    for i in range(sm1.n):
        dn1 = diff[i][torch.argsort(D1[i])[:num]].mean()
        dn2 = diff[i][torch.argsort(D2[i])[:num]].mean()
        results[i] = ((dn1 + dn2) * 0.5).cpu().detach().item()
    return np.mean(results)


def model_matching(sm1: Slisemap, sm2: Slisemap, frac: float = 0.1) -> Dict[str, float]:
    B1 = sm1.get_B(numpy=False)
    D1 = torch.cdist(B1, B1).ravel()
    B2 = sm2.get_B(numpy=False)
    D2 = torch.cdist(B1, B2).diag()
    mean1 = D1.mean().cpu().item()
    mean2 = D2.mean().cpu().item()
    std1 = D1.std().cpu().item()
    std2 = D2.std().cpu().item()
    median1 = D1.median().cpu().item()
    median2 = D2.median().cpu().item()
    return {
        "mean internal": mean1,
        "std internal": std1,
        "median internal": median1,
        "mean matching": mean2,
        "std matching": std2,
        "median matching": median2,
    }


def get_results():
    files = glob(str(RESULTS_DIR / "*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    data_categories = [
        "Boston",
        "Boston (XAI)",
        "Air Quality",
        "Air Quality (XAI)",
        "Spam (XAI)",
        "Higgs (XAI)",
        "RSynth",
    ]
    df["data"] = (
        df["data"]
        .astype("category")
        .cat.set_categories(data_categories)
        .cat.remove_unused_categories()
    )
    return df


def get_smaller(df: pd.DataFrame) -> pd.DataFrame:
    filter = [
        "Boston",
        "Air Quality (XAI)",
        "Higgs (XAI)",
    ]
    df = df[df["data"].isin(filter) * (df["d"] != 4)].copy()
    df["data"] = df["data"].cat.remove_unused_categories()
    return df


def calculate(
    sm: Slisemap,
    job,
    data,
    ds=[2, 3, 4, 5, 10],
    fs=np.arange(0.05, 0.55, 0.05),
    rs=[2.0, 2.5, 3.0, 3.5, 4.0],
):
    sm.optimise()
    B = global_model(sm)
    global_loss = sm.local_loss(sm.local_model(sm.X, B), sm.Y)
    epsilon = torch.quantile(global_loss, 0.3).cpu().item()
    results = []
    for d in ds:
        sm2 = Slisemap(
            X=sm.X,
            y=sm.Y,
            radius=rs[-1],
            d=d,
            lasso=sm.lasso,
            ridge=sm.ridge,
            z_norm=sm.z_norm,
            intercept=False,
            local_model=sm.local_model,
            local_loss=sm.local_loss,
            coefficients=sm.coefficients,
        )
        for r in rs:
            print(f"{job:02d} {data}: d={d}, r={r:g}", flush=True)
            sm2.radius = r
            # sm2.restore()
            sm2.optimise()
            D = sm2.get_D().ravel()
            mm = model_matching(sm, sm2)
            res = dict(
                job=job,
                data=data,
                n=sm2.n,
                m=sm2.m,
                d=sm2.d,
                radius=sm2.radius,
                fidelity=fidelity(sm2),
                loss=sm2.value(),
                d_mean=np.mean(D),
                d_std=np.std(D),
            )
            keys = list(mm.keys())
            for i, f in enumerate(fs):
                if i < len(keys):
                    res["mm_type"] = keys[i]
                    res["mm_value"] = mm[keys[i]]
                else:
                    res["mm_type"] = ""
                    res["mm_value"] = np.nan
                res["fraction"] = f
                res["matching"] = matching(sm, sm2, f)
                res["distance"] = distance(sm, sm2, f)
                res["fidelity_nn"] = fidelity(
                    sm2, neighbours=euclidean_nearest_neighbours, k=f
                )
                res["fidelity_r"] = fidelity(
                    sm2, neighbours=radius_neighbours, radius=f * r
                )
                res["coverage_nn"] = coverage(
                    sm2, epsilon, neighbours=euclidean_nearest_neighbours, k=f
                )
                res["coverage_r"] = coverage(
                    sm2, epsilon, neighbours=radius_neighbours, radius=f * r
                )
                results.append(res.copy())
    return results


def plot_matching(
    df: pd.DataFrame, metric="matching", title="Fraction Matching", pdf: bool = False
):
    g = sns.relplot(
        data=df,
        x="fraction",
        y=metric,
        hue="d",
        style="d",
        palette="bright",
        col="data",
        col_wrap=4,
        kind="line",
        ci=None,
        facet_kws=dict(sharey=False),
        **paper_theme(0.9, 1, 4, 3),
    )
    if metric == "matching":
        for ax in g.axes.ravel():
            ax.plot([0.05, 0.5], [0.05, 0.5], color="grey")
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.text(
                np.mean(xlim),
                np.mean(xlim),
                "Random",
                ha="center",
                va="center",
                rotation=180 / np.pi * np.arctan2(xlim[1] - xlim[0], ylim[1] - ylim[0]),
                backgroundcolor="white",
            )
    g.set_titles("{col_name}")
    g.set(xlabel="Nearest Neighbours", ylabel=title)
    if pdf:
        plt.savefig(Path(__file__).parent / "results" / f"dimensions_{metric}.pdf")
        plt.close()
    else:
        plt.show()


def plot_metric(df, metric="coverage_nn20", sharey=False):
    sns.relplot(
        data=df,
        x="radius",
        y=metric,
        col="data",
        # col="d",
        hue="d",
        style="d",
        kind="line",
        palette="bright",
        facet_kws=dict(sharey=sharey),
        **paper_theme(0.9, 1, 4, 3),
    )
    plt.show()


def plot_box(df, metric="loss", title="Loss", sharey=False, pdf=False):
    g: sns.FacetGrid = sns.catplot(
        data=df,
        y=metric,
        col="data",
        x="d",
        hue="d",
        kind="box",
        palette="bright",
        col_wrap=4,
        sharey=sharey,
        **paper_theme(1, 1, 4, 2),
    )
    g.set_titles("{col_name}")
    g.set_ylabels(title)
    g.tight_layout(h_pad=0, w_pad=0)
    for ax in g.axes.flat:
        ax.set_ylim((0, ax.get_ylim()[1]))
    if pdf:
        plt.savefig(Path(__file__).parent / "results" / f"dimensions_{metric}.pdf")
        plt.close()
    else:
        plt.show()


def plot_model(df):
    df = df[df["mm_type"].notna()]
    df = df[df["mm_type"].apply(lambda x: "me" in x)]
    g = sns.catplot(
        data=df,
        x="d",
        col="data",
        y="mm_value",
        hue="mm_type",
        kind="box",
        palette="bright",
        col_wrap=4,
        sharey=False,
        **paper_theme(0.9, 1, 4, 3),
    )
    sns.move_legend(
        g,
        "center",
        bbox_to_anchor=(0.75, 0, 0.25, 1 / np.ceil(len(g.axes) / 4)),
        frameon=False,
    )
    plt.show()


def plot_metric_nn(
    df, metric="coverage_nn", title="Coverage", small: bool = False, pdf: bool = False
):
    if small:
        df = get_smaller(df)
    g = sns.relplot(
        data=df,
        x="fraction",
        y=metric,
        hue="radius",
        style="radius",
        col="d",
        row="data",
        kind="line",
        palette="bright",
        facet_kws=dict(sharey="row"),
        ci=None,
        **paper_theme(
            0.95, 1, len(np.unique(df["d"].values)), len(np.unique(df["data"].values))
        ),
    )
    g.set(xlabel="Nearest Neighbours", ylabel=title)
    g.set_titles("{row_name}, d = {col_name}")
    g.tight_layout(w_pad=0)
    if pdf:
        plt.savefig(
            Path(__file__).parent
            / "results"
            / f"dimensions_{title.lower()}{'_b' if small else ''}.pdf"
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        df = get_results()
        # plot_matching(df[df["radius"] == 3.5], "matching", "Fraction matching", True)
        # plot_matching(df[df["radius"] == 3.5], "distance", "Distance", True)
        plot_metric_nn(df, "coverage_nn", "Coverage", pdf=True)
        plot_metric_nn(df, "fidelity_nn", "Fidelity", pdf=True)
        plot_metric_nn(df, "coverage_nn", "Coverage", True, pdf=True)
        plot_box(df[df["radius"] == 3.5], "loss", "Loss", pdf=True)
    else:
        job = int(sys.argv[1]) - 1
        sm, data = get_data2(job)
        out_path = RESULTS_DIR / f"dimensions_{job:02d}_{data}.parquet"
        if not out_path.exists():
            print(f"{job:02d} {data}: Setup", flush=True)
            results = calculate(sm, job, data)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(results)
            df["data"] = df["data"].astype("category")
            df["mm_type"] = df["mm_type"].astype("category")
            df.to_parquet(out_path)
            print(f"{job:02d} {data}: Done", flush=True)
