###############################################################################
#
# This experiment compares the predictions from Slipmap to other methods.
# This experiment is designed to be run in parallel (e.g. on a computer cluster).
#
# Run this script to perform the experiments, where $index is [1..10]:
#   `python experiments/ida2024/predictictions.py $index`
#
# Run this script again without additional arguments to produce a plot from the results:
#   `python experiments/ida2024/predictictions.py`
#
###############################################################################
import gc
import sys
from functools import partial
from glob import glob
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from hyperparameters import (
    discretise,
    get_bb,
    get_data,
    get_knn_params,
    get_slipmap_params,
    get_slisemap_params,
)
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
from project_paths import MANUSCRIPT_DIR, RESULTS_DIR
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

OUTPUT_DIR = RESULTS_DIR / "slipmap_predictive"

from experiments.utils import paper_theme
from slisemap import Slisemap
from slisemap.local_models import LinearRegression, LogisticRegression
from slisemap.slipmap import Slipmap


def get_models(dname, bb, classifier, seed=42):
    lm = LogisticRegression if classifier else LinearRegression

    def slipmap(X, y, weighted=False, row_kernel=False, squared=False, density=False):
        sm = Slipmap(
            X,
            y,
            local_model=lm,
            **get_slipmap_params(dname, bb, weighted, row_kernel, squared, density),
        )
        sm.optimise()
        return partial(sm.predict, weighted=weighted)

    def slisemap(X, y):
        sm = Slisemap(
            X, y, local_model=lm, random_state=seed, **get_slisemap_params(dname, bb)
        )
        sm.optimise()
        return sm.predict

    def knn(X, y, **kwargs):
        if classifier:
            mod = KNeighborsClassifier(**kwargs, n_jobs=-1)
            mod.fit(X, discretise(y))
            return lambda X: np.stack([p[:, -1] for p in mod.predict_proba(X)], -1)
        else:
            mod = KNeighborsRegressor(**kwargs, n_jobs=-1)
            mod.fit(X, y)
            return mod.predict

    return {
        # "Slipmap": partial(slipmap, weighted=False, squared=False),
        # "SlipmapW": partial(slipmap, weighted=True, squared=False),
        # "SlipmapWS": partial(slipmap, weighted=True, squared=True),
        "SlipmapWSD": partial(slipmap, weighted=True, squared=True, density=True),
        # "SlipmapS": partial(slipmap, weighted=False, squared=True),
        # "SlipmapWT": partial(slipmap, weighted=True, row_kernel=True, squared=False),
        "Slisemap": slisemap,
        "Nearest Neighbour": partial(knn, n_neighbors=1),
        # "K Nearest Neighbours": lambda X, y: knn(X, y, **get_knn_params(dname, bb)),
    }


def get_sizes(n: int, num: int = 6, min_size: int = 300, max_size: int = 20_000):
    end = np.log(min(max_size, n * 3 // 4)) - np.log(min_size)
    lin = np.floor(np.exp(np.linspace(0, end, num)) * min_size)
    return lin.astype(np.int32)


def plot(df, pdf=True):
    if pdf:
        rename = {
            "SlipmapWSD": "Slipmap",
            # "SlipmapW": "Slipmap",
            # "SlipmapWSD": "Slipmap²",
            "Slisemap": "Slisemap",
            "Nearest Neighbour": "Nearest\nNeighbour",
        }
        df = df[df["method"].isin(list(rename.keys()))].copy()
        df["Method"] = pd.Categorical(
            df["method"], list(rename.keys()), True
        ).rename_categories(rename)
    else:
        df = df[~df["method"].isin(["K Nearest Neighbours", "Slipmap", "SlipmapT"])]
        df["Method"] = pd.Categorical(df["method"])
    df["data"] = pd.Categorical(df["data"], list(get_data().keys()), True)
    ndata = len(df["data"].cat.categories)
    # targets = {"loss_y": "Real y", "loss_yp": "Predicted y"}
    # df = pd.melt(df, id_vars=[c for c in df.columns if c not in targets])
    # df["\nTarget"] = df["variable"].replace(targets)
    df["\nTarget"] = (df["BB"] == "").replace({True: "Real y", False: "Predicted y"})
    g: sns.FacetGrid = sns.relplot(
        df,
        x="n",
        y="loss_y",
        col="data",
        col_order=list(get_data().keys()),
        col_wrap=(ndata - 1) // 2 + 1,
        hue="Method",
        style="\nTarget",
        kind="line",
        **(paper_theme(cols=(ndata - 1) // 2 + 1, rows=2) if pdf else {}),
        facet_kws={"sharey": False, "sharex": False},
        errorbar=None,
    )
    g.map_dataframe(
        lambda data, **_: plt.axhline(
            data["loss_BB"].mean(), color="black", linestyle="--"
        )
    )
    g.map_dataframe(
        lambda data, **_: plt.text(
            data["n"].median(),
            data["loss_BB"].mean(),
            "\n\n" + data["BB"][data.last_valid_index()],
            horizontalalignment="center",
            verticalalignment="center",
        )
    )
    g.set(xscale="log")
    g.set_axis_labels("Samples", "Loss")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.set_ylim((0, ax.get_ylim()[1]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.ticklabel_format(style="plain", axis="x", useOffset=False)
        ax.xaxis.set_major_locator(LogLocator(subs=(0.3, 1.0)))
    g.tight_layout()
    if pdf:
        plt.savefig(MANUSCRIPT_DIR / "predictive_performance.pdf")
        plt.close()
        print("Figure saved to:", str(MANUSCRIPT_DIR / "predictive_performance.pdf"))
    else:
        plt.show()


def loss(cls: bool, y: np.ndarray, yhat: np.ndarray):
    y = torch.as_tensor(y, dtype=torch.float32)
    yhat = torch.as_tensor(yhat, dtype=torch.float32)
    if len(yhat.shape) == 1:
        yhat = yhat[:, None]
    y = torch.reshape(y, yhat.shape)
    if cls:
        L = LogisticRegression.loss(yhat, y)
    else:
        L = LinearRegression.loss(yhat, y)
    return L.mean().cpu().item()


def evaluate(job):
    print("Job:", job)
    np.random.seed(42 + job)
    torch.manual_seed(42 + job)
    file = OUTPUT_DIR / f"predictive_{job:02d}.parquet"
    file.parent.mkdir(parents=True, exist_ok=True)
    if file.exists():
        results = pd.read_parquet(file)
        # results = results[
        #     (~results["data"].isin(["Higgs"]))
        #     + (~results["method"].isin(["SlipmapWSD"]))
        # ]
    else:
        results = pd.DataFrame()
    for dname, dfn in get_data().items():
        print(f"Loading:    {dname}", flush=True)
        X, y, bb, cls = dfn()
        sizes = get_sizes(X.shape[0])
        ntrain = max(sizes)
        ntest = min(X.shape[0] - ntrain, ntrain)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=ntrain, test_size=ntest, random_state=42 + job
        )
        print(f"Training:   {dname} - {bb}", flush=True)
        bb_pred = get_bb(bb, cls, X_train, y_train, dname)
        p_train = bb_pred(X_train)
        p_test = bb_pred(X_test)
        for train, bb2 in ((y_train, ""), (p_train, bb)):
            for mname, model in get_models(dname, bb2, cls, 42 + job).items():
                for n in sizes:
                    if not results.empty:
                        mask = results["data"] == dname
                        mask &= results["method"] == mname
                        mask &= results["n"] == n
                        mask &= results["BB"] == bb2
                        if mask.any():
                            continue
                    try:
                        print(
                            f"Training:   {dname} - {mname} - {n} - {bb2}", flush=True
                        )
                        gc.collect()
                        time = timer()
                        pred = model(X_train[:n, ...], train[:n, ...])
                        time = timer() - time
                        y_pred = pred(X_test)
                        del pred
                        print(
                            f"Evaluating: {dname} - {mname} - {n} - {bb2}", flush=True
                        )
                        res = {
                            "data": dname,
                            "method": mname,
                            "n": n,
                            "job": job,
                            "time": time,
                            "loss_y": loss(cls, y_test, y_pred),
                            "loss_p": loss(cls, p_test, y_pred),
                            "loss_BB": loss(cls, y_test, p_test),
                            "BB": bb2,
                        }
                        results = pd.concat(
                            (results, pd.DataFrame([res])), ignore_index=True
                        )
                        results.to_parquet(file)
                    except Exception as e:
                        print(
                            f"Failed:     {dname} - {mname} - {n} - {bb2}\n{e}",
                            flush=True,
                        )
    print("Done", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        files = glob(str(OUTPUT_DIR / "*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        # print(df.drop(["n", "job", "BB"], 1).groupby(["data", "method"]).agg("mean"))
        # plot(df, False)
        plot(df)
    else:
        evaluate(int(sys.argv[1]))
