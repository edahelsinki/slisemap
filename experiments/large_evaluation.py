###############################################################################
#
# This experiment is meant to do a large-scale evaluation using multiple methods on multiple datasets with multiple metrics.
# The idea is that these results can be reused for multiple experiments and analyses.
# This experiment is designed to be run in parallel (e.g. on a computer cluster).
#
# Run this script to perform the experiments, where $index is [1..150]:
#   `python experiments/large_evaluation.py $index`
#
# To produce plots and tables see:
#   - experiments/parameter_selection.py
#   - experiments/dr_comparison.py
#   - experiments/escape.py
#
###############################################################################

import gc
import random
import sys
import warnings
from glob import glob
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
import umap  # use the 'umap-learn' package!
from scipy.sparse import SparseEfficiencyWarning
from scipy.special import logit
from sklearn.decomposition import PCA
from sklearn.manifold import (
    MDS,
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
)
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent))
from slisemap.diagnostics import *
from slisemap.escape import *
from slisemap.local_models import *
from slisemap.metrics import *
from slisemap.slisemap import Slisemap
from slisemap.utils import LBFGS, _tonp
from slisemap.utils import global_model as _global_model

from experiments.data import *


RESULTS_DIR = Path(__file__).parent / "results" / "large"


def umap_embedding(X, y):
    # TODO select n_neighbors
    n_neighbors = int(np.sqrt(X.shape[0])) // 2
    fit = umap.UMAP(n_neighbors=n_neighbors)
    Z = fit.fit_transform(X)
    return Z


def umap_sup_embedding(X, y):
    # TODO select n_neighbors
    n_neighbors = int(np.sqrt(X.shape[0])) // 2
    fit = umap.UMAP(n_neighbors=n_neighbors)
    # UMAP has a supervised mode!
    Z = fit.fit_transform(X, y[:, 0])
    return Z


def pca_embedding(X, y):
    pca = PCA(2)
    return pca.fit_transform(X, y)


def tsne_embedding(X, y):
    # TODO select perplexity
    perplexity = int(np.sqrt(X.shape[0]))
    # Future default initialisation for TSNE (PCA scaled by 1e-4)
    init = pca_embedding(X, y)
    init = init * (1e-4 / (np.std(init, 0, keepdims=True) + 1e-8))
    tsne = TSNE(2, perplexity=perplexity, init=init, learning_rate="auto", n_jobs=-1)
    return tsne.fit_transform(X, y)


def mds_embedding(X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        mds = MDS(metric=True, n_init=8, n_jobs=-1)
        return mds.fit_transform(X, y)


def nmmds_embedding(X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        mds = MDS(metric=False, n_init=8, n_jobs=-1)
        return mds.fit_transform(X, y)


def isomap_embedding(X, y):
    # TODO select n_neighbors
    n_neighbors = int(np.sqrt(X.shape[0])) // 2
    isomap = Isomap(n_neighbors=n_neighbors, n_jobs=-1)
    return isomap.fit_transform(X, y)


def lle_embedding(X, y):
    # TODO select n_neighbors
    n_neighbors = int(np.sqrt(X.shape[0])) // 2
    lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_jobs=-1)
    return lle.fit_transform(X, y)


def mlle_embedding(X, y):
    # TODO select n_neighbors
    n_neighbors = int(np.sqrt(X.shape[0])) // 2
    lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, method="modified", n_jobs=-1)
    return lle.fit_transform(X, y)


def spectral_embedding(X, y):
    se = SpectralEmbedding(n_jobs=-1)
    return se.fit_transform(X, y)


def optim_with_embedding(embedding_fn):
    def train_fn(sm: Slisemap):
        sm.z_norm = 0
        X = sm.get_X()
        X = X[:, np.std(X, 0) != 0]
        y = sm.get_Y()
        Z = embedding_fn(X, y)
        sm._Z = torch.as_tensor(Z, **sm.tensorargs)
        B = sm.get_B(numpy=False).requires_grad_(True)
        loss = sm.get_loss_fn()
        LBFGS(lambda: loss(sm.X, sm.Y, B, sm.Z), [B])
        sm._B = B.detach()

    return train_fn


def global_model(sm: Slisemap) -> torch.Tensor:
    return _global_model(
        X=sm.X,
        Y=sm.Y,
        local_model=sm.local_model,
        local_loss=sm.local_loss,
        coefficients=sm.coefficients,
        lasso=sm.lasso,
        ridge=sm.ridge,
    )


def method_generator(c=None, B=None):
    """Generator that generates (train_fn, method_name, method_variant)"""
    for r in [2.0, 3.0, 3.5, 4.0, 5.0, 6.0]:

        def train_fn(sm: Slisemap):
            sm.radius = r
            sm.lbfgs()

        yield train_fn, "Slisemap (no escape)", r
    for r in [2.0, 3.0, 3.5, 4.0, 5.0, 6.0]:

        def train_fn(sm: Slisemap):
            sm.radius = r
            sm.optimise()

        yield train_fn, "Slisemap", r

    yield optim_with_embedding(umap_embedding), "UMAP", 0
    yield optim_with_embedding(umap_sup_embedding), "Supervised UMAP", 0
    yield optim_with_embedding(pca_embedding), "PCA", 0
    yield optim_with_embedding(spectral_embedding), "Spectral Embedding", 0
    yield optim_with_embedding(lle_embedding), "LLE", 0
    yield optim_with_embedding(mlle_embedding), "MLLE", 0
    yield optim_with_embedding(mds_embedding), "MDS", 0
    yield optim_with_embedding(nmmds_embedding), "Non-Metric MDS", 0
    yield optim_with_embedding(isomap_embedding), "Isomap", 0
    yield optim_with_embedding(tsne_embedding), "t-SNE", 0

    def train_fn(sm: Slisemap):
        sm.radius = 0
        sm.z_norm = 0
        B = global_model(sm)
        sm._Z *= 0.0
        sm._B = B.expand(sm.B.shape)

    yield train_fn, "Global", 0

    def train_fn(sm: Slisemap):
        se = SpectralEmbedding(2, n_jobs=-1)
        sm._Z = torch.as_tensor(
            se.fit_transform(_tonp(sm.X)),
            **sm.tensorargs,
        ).contiguous()
        sm.optimise()

    yield train_fn, "Slisemap (spectral Z0)", 0

    def train_fn(sm: Slisemap):
        sm._B = torch.normal(0, 1, size=sm.B.shape, **sm.tensorargs)
        sm.optimise()

    yield train_fn, "Slisemap (random B0)", 0

    def train_fn(sm: Slisemap):
        sm.optimise(escape_fn=escape_greedy)

    yield train_fn, "Slisemap (greedy escape)", 0

    def train_fn(sm: Slisemap):
        sm.optimise(escape_fn=escape_combined)

    yield train_fn, "Slisemap (combined escape)", 0

    # def train_fn(sm: Slisemap):
    #     sm.optimise(escape_fn=escape_marginal)

    # yield train_fn, "Slisemap (marginal escape)", 0

    if B is not None and c is not None:

        def train_fn(sm: Slisemap):
            sm._B = torch.as_tensor(B[c], **sm.tensorargs)
            angles = np.pi * c / (np.max(c) + 1)
            Z = np.stack((np.sin(angles), np.cos(angles)), 1)
            sm._Z = torch.as_tensor(Z, **sm.tensorargs)
            sm.lbfgs()

        yield train_fn, "Slisemap (cheating)", 0


def get_method(index, variant=None, **kwargs):
    if isinstance(index, str):
        for fn, method, var in method_generator(**kwargs):
            if index == method and (variant is None or var == variant):
                return fn, method, variant
    else:
        for i, m in enumerate(method_generator(**kwargs)):
            if i == index:
                return m
    raise IndexError()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_data(index, max_n=1000):
    set_seed(index)
    if index < 50:
        n, m = [(100, 5), (200, 10), (400, 15), (800, 20), (1000, 25)][index // 10]
        c, X, y, B = get_rsynth(n * 2, m, 3, 0.25, 0.1)
        sm = Slisemap(X[:n], y[:n], lasso=0.0)
        return sm, "RSynth", X[n:], y[n:], c[:n], B
    if index < 60:
        X, y = get_boston()
        n = (X.shape[0] * 8) // 10
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=n, shuffle=True
        )
        sm = Slisemap(X_train, y_train, lasso=0.0001)
        return sm, "Boston", X_test, y_test, None, None
    if index < 70:
        X, y = get_airquality()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=max_n, train_size=max_n, shuffle=True
        )
        sm = Slisemap(X_train, y_train, lasso=0.0001)
        return sm, "Air Quality", X_test, y_test, None, None
    if index < 80:
        X, y = get_airquality("rf")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=max_n, train_size=max_n, shuffle=True
        )
        sm = Slisemap(X_train, y_train, lasso=0.0001)
        return sm, "Air Quality (XAI)", X_test, y_test, None, None
    if index < 90:
        X, y = get_spam()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=max_n, train_size=max_n, shuffle=True, stratify=y[:, 0]
        )
        sm = Slisemap(
            X_train,
            y_train,
            lasso=0.01,
            local_model=logistic_regression,
            local_loss=logistic_regression_loss,
            coefficients=logistic_regression_coefficients,
        )
        return sm, "Spam", X_test, y_test, None, None
    if index < 100:
        X, y = get_spam("rf")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            logit(y[:, 0] * 0.98 + 0.01),
            test_size=max_n,
            train_size=max_n,
            shuffle=True,
            stratify=np.round(y[:, 0]),
        )
        sm = Slisemap(X_train, y_train, lasso=0.001)
        return sm, "Spam (XAI)", X_test, y_test, None, None
    if index < 110:
        X, y = get_higgs()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=max_n, train_size=max_n, shuffle=True, stratify=y[:, 0]
        )
        sm = Slisemap(
            X_train,
            y_train,
            lasso=0.01,
            local_model=logistic_regression,
            local_loss=logistic_regression_loss,
            coefficients=logistic_regression_coefficients,
        )
        return sm, "Higgs", X_test, y_test, None, None
    if index < 120:
        X, y = get_higgs("gb")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            logit(y[:, 0] * 0.98 + 0.01),
            test_size=max_n,
            train_size=max_n,
            shuffle=True,
            stratify=np.round(y[:, 0]),
        )
        sm = Slisemap(X_train, y_train, lasso=0.0001)
        return sm, "Higgs (XAI)", X_test, y_test, None, None
    if index < 130:
        X, y = get_covertype("lb")
        _, y2 = get_covertype()
        mask = y2[:, 0] + y2[:, 1] > 0
        X = X[mask]
        y = y[mask]
        y = (y[:, 0] + 1e-2) / (y[:, 0] + y[:, 1] + 2e-2)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            logit(y),
            test_size=max_n,
            train_size=max_n,
            stratify=np.round(y),
            shuffle=True,
        )
        sm = Slisemap(X_train, y_train, lasso=0.001)
        return sm, "Covertype (XAI)", X_test, y_test, None, None
    if index < 140:
        X, y = get_boston("svm")
        n = (X.shape[0] * 8) // 10
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=n, shuffle=True
        )
        sm = Slisemap(X_train, y_train, lasso=0.0001)
        return sm, "Boston (XAI)", X_test, y_test, None, None
    if index < 150:
        X, y = get_covertype()
        mask = y[:, 0] + y[:, 1] > 0
        X = X[mask]
        y = y[mask, :2]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=max_n, train_size=max_n, shuffle=True, stratify=y[:, 0]
        )
        sm = Slisemap(
            X_train,
            y_train,
            lasso=0.01,
            local_model=logistic_regression,
            local_loss=logistic_regression_loss,
            coefficients=logistic_regression_coefficients,
        )
        return sm, "Covertype", X_test, y_test, None, None

    raise StopIteration(f"Data for index {index} not found")


def evaluate(
    sm: Slisemap,
    time,
    Xnew,
    Ynew,
    c,
    B,
    method_name,
    method_variant,
    data_name,
    job_index,
):
    ev_time = timer()
    n = sm.n
    possible_neighbours = dict(
        nn05=dict(neighbours=euclidean_nearest_neighbours, k=int(n * 0.05)),
        nn10=dict(neighbours=euclidean_nearest_neighbours, k=int(n * 0.1)),
        nn15=dict(neighbours=euclidean_nearest_neighbours, k=int(n * 0.15)),
        nn20=dict(neighbours=euclidean_nearest_neighbours, k=int(n * 0.2)),
        nn25=dict(neighbours=euclidean_nearest_neighbours, k=int(n * 0.25)),
        nn30=dict(neighbours=euclidean_nearest_neighbours, k=int(n * 0.3)),
        nn35=dict(neighbours=euclidean_nearest_neighbours, k=int(n * 0.35)),
        nn40=dict(neighbours=euclidean_nearest_neighbours, k=int(n * 0.4)),
        nn45=dict(neighbours=euclidean_nearest_neighbours, k=int(n * 0.45)),
        nn50=dict(neighbours=euclidean_nearest_neighbours, k=int(n * 0.5)),
        k050=dict(neighbours=kernel_neighbours, epsilon=0.5),
        k075=dict(neighbours=kernel_neighbours, epsilon=0.75),
        k100=dict(neighbours=kernel_neighbours, epsilon=1.0),
        k125=dict(neighbours=kernel_neighbours, epsilon=1.25),
        k150=dict(neighbours=kernel_neighbours, epsilon=1.5),
        rq05=dict(neighbours=radius_neighbours, radius=0.5),
        rq10=dict(neighbours=radius_neighbours, radius=1.0),
        rq15=dict(neighbours=radius_neighbours, radius=1.5),
        rq20=dict(neighbours=radius_neighbours, radius=2.0),
    )
    results = dict(
        job_index=job_index,
        method=method_name,
        method_variant=method_variant,
        data=data_name,
        radius=sm.radius,
        lasso=sm.lasso,
        ridge=sm.ridge,
        n=sm.n,
        m=sm.m,
        task="Regression" if sm.p == 1 else "Classification",
        time=time,
        loss=sm.value(),
        entropy=sm.entropy(),
        heavyweight=heavyweight_diagnostic(sm).mean(),
        lightweight=lightweight_diagnostic(sm).mean(),
        weight_neighbourhood=weight_neighbourhood_diagnostic(sm).mean(),
        loss_neighbourhood=loss_neighbourhood_diagnostic(sm).mean(),
        global_loss=global_loss_diagnostic(sm).mean(),
    )
    results["accuracy"] = accuracy(sm)
    # results["accuracy2"] = accuracy(sm, escape_fn=escape_marginal)
    results["accuracy_new"] = accuracy(sm, Xnew, Ynew)
    # results["accuracy_closest"] = accuracy(sm, Xnew, Ynew, between=False, optimise=False, escape_fn=escape_marginal)

    results["fidelity"] = fidelity(sm)
    results["fidelity_nn00"] = results["fidelity"]
    for key, value in possible_neighbours.items():
        results["fidelity_" + key] = fidelity(sm, **value)

    dummy = sm.local_loss(sm.Y, sm.Y.mean(0, keepdim=True))
    global_loss = sm.local_loss(sm.local_model(sm.X, global_model(sm)), sm.Y)
    for f in (0.2, 0.3, 0.4):
        q = torch.quantile(global_loss, f).cpu().detach().item()
        if np.isnan(q):
            q = torch.quantile(dummy, f).cpu().detach().item()
        name = f"coverage_gl{int(f*10):02d}"
        results[name] = coverage(sm, q)
        if c is not None:
            results[name + "_cl"] = coverage(sm, q, c)
        for key, value in possible_neighbours.items():
            results[name + "_" + key] = coverage(sm, q, **value)

    results["median_loss"] = median_loss(sm)
    for key, value in possible_neighbours.items():
        results["median_loss_" + key] = median_loss(sm, **value)

    results["coherence"] = coherence(sm)
    for key, value in possible_neighbours.items():
        results["coherence_" + key] = coherence(sm, **value)

    results["stability"] = stability(sm)
    for key, value in possible_neighbours.items():
        results["stability_" + key] = stability(sm, **value)

    for e in (0.5, 1.0, 1.5):
        results[f"recall_{int(e*10):02d}"] = recall(sm, e, e)
    for e in (0.5, 1.0, 1.5):
        results[f"precision_{int(e*10):02d}"] = precision(sm, e, e)

    if method_name == "Global":
        results["kmeans_matching"] = np.nan
    else:
        results["kmeans_matching"] = kmeans_matching(sm)

    if c is not None:
        results["fidelity_cl"] = fidelity(sm, c)
        results["median_loss_cl"] = median_loss(sm, c)
        results["coherence_cl"] = coherence(sm, c)
        results["stability_cl"] = stability(sm, c)
        results["cluster_purity"] = cluster_purity(sm, c)
        for e in (0.5, 1.0, 1.5):
            results[f"kernel_purity_{int(e*10):02d}"] = kernel_purity(sm, c, e)

    results["ev_time"] = timer() - ev_time
    return pd.DataFrame([results])


def get_results(filter: bool = False) -> pd.DataFrame:
    files = glob(str(RESULTS_DIR / "*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    data_categories = [
        "Boston",
        "Boston (XAI)",
        "Air Quality",
        "Air Quality (XAI)",
        "Spam",
        "Spam (XAI)",
        "Higgs",
        "Higgs (XAI)",
        "Covertype",
        "Covertype (XAI)",
        "RSynth",
    ]
    method_categories = [
        "Slisemap",
        "Slisemap (no escape)",
        "Slisemap (cheating)",
        "Slisemap (spectral Z0)",
        "Slisemap (random B0)",
        "Slisemap (adjusted B0)",
        "Slisemap (greedy escape)",
        "Slisemap (combined escape)",
        "Slisemap (marginal escape)",
        "Global",
        "PCA",
        "Spectral Embedding",
        "LLE",
        "MLLE",
        "MDS",
        "Non-Metric MDS",
        "Isomap",
        "t-SNE",
        "UMAP",
        "Supervised UMAP",
    ]
    if filter:
        method_categories = [
            m for m in method_categories if "Slisemap " not in m and m != "Global"
        ]
    df["data"] = (
        df["data"]
        .astype("category")
        .cat.set_categories(data_categories)
        .cat.remove_unused_categories()
    )
    df["method"] = (
        df["method"]
        .astype("category")
        .cat.set_categories(method_categories)
        .cat.remove_unused_categories()
    )
    df["task"] = df["task"].astype("category")
    if filter:
        df = df[df["method"].notna() * df["radius"].round(3).isin((0, 3.5))]
    return df


if __name__ == "__main__":
    if len(sys.argv) == 1:
        df = get_results()
        print(df)
    else:
        job = int(sys.argv[1]) - 1
        sm, data, Xnew, Ynew, c, B = get_data(job)
        out_name = f"large_{job:03d}_{data}.parquet"
        out_path = RESULTS_DIR / out_name
        if out_path.is_file():
            df = pd.read_parquet(out_path)
            # Remove outdated results here if necessary!
            # if "Slisemap (escape 2)" in df["method"].cat.categories:
            #     df = df[df["method"] != "Slisemap (escape 2)"]
            # df = df[df["method"].cat.remove_categories(["Slisemap (escape 2)"]).notna()]
        else:
            df = None
        print("Setup:", job, data, flush=True)
        for i, (train_fn, method, variant) in enumerate(method_generator(c=c, B=B)):
            set_seed(job + 42)  # Use a deterministic seed
            if df is not None:
                if (df["method"][df["method_variant"] == variant] == method).any():
                    continue  # Skip already existing results!
            smc = sm.copy()
            gc.collect()
            try:
                time = timer()
                train_fn(smc)
                time = timer() - time
                dfn = evaluate(smc, time, Xnew, Ynew, c, B, method, variant, data, job)
                print("Eval:", job, i, method, variant, data, flush=True)
                if df is None:
                    df = dfn
                    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                else:
                    df = pd.concat([df, dfn], ignore_index=True)
                    df["method"] = df["method"].astype("category")
                    df["data"] = df["data"].astype("category")
                    df["task"] = df["task"].astype("category")
                df.to_parquet(out_path)
            except Exception as e:
                print("Error:", job, i, method, variant, data, "\n", e, flush=True)
                raise e
            del smc
        print("Done:", job, data, flush=True)
