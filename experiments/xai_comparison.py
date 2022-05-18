###############################################################################
#
# This experiment compares Slisemap to local explanation methods.
# This experiment is designed to be run in parallel (e.g. on a computer cluster).
#
# Run this script to perform the experiments, where $index is [1..50]:
#   `python experiments/xai_comparison.py $index`
#
# Run this script again without additional arguments to produce a latex table from the results:
#   `python experiments/xai_comparison.py`
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
import shap
import shap.maskers
import slise
import torch
from lime.lime_tabular import LimeTabularExplainer
from scipy.special import logit
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

sys.path.append(str(Path(__file__).parent.parent))  # Add the project root to the path
from slisemap.slisemap import LBFGS, Slisemap, _tonp
from slisemap.diagnostics import global_model_losses

from experiments.data import get_airquality, get_boston, get_higgs, get_spam
from experiments.large_evaluation import global_model
from experiments.dr_comparison import print_table

RESULTS_DIR = Path(__file__).parent / "results" / "xai"


def slise_L(sm: Slisemap, error_tolerance: float, subset=None, **_) -> torch.Tensor:
    X = sm.get_X(intercept=False)
    Y = sm.get_Y()[:, 0]
    s = slise.SliseExplainer(X, Y, error_tolerance**0.5, lambda1=sm.lasso)
    if subset is None:
        subset = range(sm.n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in subset:
            s.explain(i)
            b = np.concatenate((s.coefficients[1:], s.coefficients[:1]), 0)
            sm._B[i] = torch.as_tensor(b, **sm.tensorargs)
    sm._Z *= 0.0
    # Z = sm.Z[sel].clone().detach().requires_grad_(True)
    # loss_fn = sm.get_loss_fn()
    # LBFGS(lambda: loss_fn(sm.X[sel, :], sm.Y[sel], sm.B, Z), [Z])
    # sm._Z = Z.detach()
    return sm.get_L(numpy=False)


def slisemap_L(sm: Slisemap, **_) -> torch.Tensor:
    sm.optimise()
    return sm.get_L(numpy=False)


def global_L(sm: Slisemap, **_) -> torch.Tensor:
    sm._B = global_model(sm).expand(sm.B.shape)
    sm._Z *= 0.0
    return sm.get_L(numpy=False)


def lime_L(
    sm: Slisemap, pred_fn, num_samples=5000, subset=None, disc=True, **_
) -> torch.Tensor:
    X = sm.get_X(intercept=False)
    explainer = LimeTabularExplainer(
        X, "regression", sm.get_Y(), discretize_continuous=disc
    )
    if subset is None:
        subset = range(sm.n)
    for i in subset:
        exp = explainer.explain_instance(X[i], pred_fn, num_samples=num_samples)
        b = np.zeros(sm.m)
        for j, v in exp.as_map()[1]:
            b[j] = v
        b[-1] = exp.intercept[1]
        sm._B[i] = torch.as_tensor(b, **sm.tensorargs)
    sm._Z *= 0.0
    if disc:
        # Lime works on discretised data, so we need to discretise the data
        # to be able to apply the linear model, and use lime for prediction.
        X = explainer.discretizer.discretize(X)
        X = np.concatenate((X, np.ones((X.shape[0], 1))), 1)
        dX = torch.as_tensor(X, **sm.tensorargs)
        X = dX[:, None] == dX[None, :]
        P = torch.sum(X * sm.B[:, None], -1, keepdim=True)
        return sm.local_loss(P, sm.Y)
    else:
        return sm.get_L(numpy=False)


def shap_L(sm: Slisemap, pred_fn, partition=True, subset=None, **_) -> torch.Tensor:
    if subset is None:
        subset = slice(sm.n)
    X = sm.get_X(intercept=False)
    if partition:
        masker = shap.maskers.Partition(X)
        explainer = shap.explainers.Partition(pred_fn, masker=masker)
        exp = explainer(X[subset], silent=True)
        B = exp.values.reshape((exp.values.shape[0], -1))
        B = np.concatenate((B, exp.base_values.reshape((-1, 1))), 1)
    else:
        explainer = shap.explainers.Sampling(pred_fn, X)
        B = explainer.shap_values(X[subset], silent=True)
        B0 = np.repeat(explainer.expected_value, B.shape[0])[:, None]
        B = np.concatenate((B, B0), 1)
    sm._B[subset] = torch.as_tensor(B, **sm.tensorargs)
    # The SHAP explanation vector is "Shapley values" for the "current" variable values.
    # Thus, we need to know when the value of another data item is "close enough".
    # Here I quickly train simple RBF kernels, which allows us to use SHAP for prediction.
    scale = torch.ones(sm.m, **sm.tensorargs).requires_grad_(True)
    loss_fn = lambda: _shap_get_L(
        sm.X, sm.Y, sm.X[subset], sm.B[subset], scale, sm.local_loss
    ).sum()
    LBFGS(loss_fn, [scale])
    scale = torch.abs(scale.detach())
    sm._Z *= 0.0
    return _shap_get_L(sm.X, sm.Y, sm.X, sm.B, scale, sm.local_loss)


def _shap_get_L(X, Y, shapX, shapB, scale, loss_fn):
    dist = (X[None] - shapX[:, None]) ** 2
    kernel = torch.exp(-(torch.abs(scale) + 1e-6)[None, None] * dist)
    P = torch.sum(kernel * shapB[:, None], -1, keepdim=True)
    return loss_fn(P, Y)


def get_data(index):
    np.random.seed(42 + index)
    random.seed(42 + index)
    torch.manual_seed(42 + index)
    if index < 10:
        X, y = get_boston()
        X_train, X_test, y_train, _ = train_test_split(
            X, y, train_size=(X.shape[0] * 8) // 10, shuffle=True
        )
        svm = SVR().fit(X_train, y_train)
        sm = Slisemap(X_train, svm.predict(X_train), lasso=0.0001, cuda=False)
        return sm, "Boston (XAI)", svm.predict
    elif index < 20:
        X, y = get_airquality()
        rf = RandomForestRegressor(n_jobs=-1).fit(X, y)
        subsample = np.random.choice(X.shape[0], 1000, replace=False)
        sm = Slisemap(X[subsample], y[subsample], lasso=1e-4, cuda=False)
        return sm, "Air Quality (XAI)", rf.predict
    elif index < 30:
        X, y = get_spam()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1000, shuffle=True, stratify=y[:, 0]
        )
        rf = RandomForestClassifier(
            n_jobs=-1,
            criterion="entropy",
            max_features=0.3206,
            min_samples_split=15,
            n_estimators=300,
        ).fit(X_train, y_train)
        pred_fn = lambda X: logit(rf.predict_proba(X)[0][:, :1] * 0.98 + 0.01)
        sm = Slisemap(X_test, pred_fn(X_test), lasso=0.01, cuda=False)
        return sm, "Spam (XAI)", pred_fn
    elif index < 40:
        X, y = get_higgs()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1000, shuffle=True, stratify=y[:, 0]
        )
        gb = GradientBoostingClassifier(
            learning_rate=0.02,
            max_depth=10,
            min_impurity_decrease=0.35,
            min_samples_split=13,
            n_estimators=400,
            n_iter_no_change=400,
            validation_fraction=0.4,
            tol=0.04,
            subsample=0.9,
        ).fit(X_train, y_train[:, 0])
        pred_fn = lambda X: logit(gb.predict_proba(X)[:, :1] * 0.98 + 0.01)
        sm = Slisemap(X_test, pred_fn(X_test), lasso=0.01, cuda=False)
        return sm, "Higgs (XAI)", pred_fn
    raise StopIteration(f"index {index} too large")


def print_results(df: pd.DataFrame):
    data_categories = [
        "Boston (XAI)",
        "Air Quality (XAI)",
        "Spam (XAI)",
        "Higgs (XAI)",
        "Covertype (XAI)",
        "RSynth",
    ]
    df["data"] = (
        df["data"]
        .astype("category")
        .cat.set_categories(data_categories)
        .cat.remove_unused_categories()
    )
    method_categories = ["Slisemap", "SLISE", "SHAP", "LIME", "LIME (nd)", "Global"]
    df["method"] = (
        df["method"]
        .astype("category")
        .cat.set_categories(method_categories)
        .cat.remove_unused_categories()
    )
    metrics = [
        ("fidelity", "Fidelity", False),
        ("coverage", "Coverage", True),
        ("time", "Time (s)", False),
    ]
    print_table(df, metrics, "tab:cmp_xai", False, 1)


def evaluate(sm, fn, error_tolerance, method, data, job, subsample=0):
    sm = sm.copy()
    gc.collect()
    print(f"{job:02d} {data}: {method}", flush=True)
    if subsample > 0:
        subset = range(0, sm.n, (sm.n - 1) // subsample + 1)
        time = timer()
        L = fn(sm, subset=subset)
        time = (timer() - time) * (sm.n / len(subset))
    else:
        subset = slice(sm.n)
        time = timer()
        L = fn(sm)
        time = timer() - time
    return dict(
        job=job,
        method=method,
        data=data,
        n=sm.n,
        m=sm.m,
        time=time,
        fidelity=L.diag()[subset].mean().cpu().item(),
        coverage=_tonp(L[subset] < error_tolerance).mean(),
        max_loss=error_tolerance,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        files = glob(str(RESULTS_DIR / "*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        print_results(df)
    else:
        job_index = int(sys.argv[1]) - 1
        sm, data, pred_fn = get_data(job_index)
        out_path = RESULTS_DIR / f"xai_{job_index:02d}_{data}.parquet"
        if not out_path.exists():
            print(f"{job_index:02d} {data}: Setup", flush=True)
            epsilon = torch.quantile(global_model_losses(sm), 0.3).cpu().item()
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            results = []
            for method, fn in [
                ("SLISE", lambda sm, **kw: slise_L(sm, epsilon, **kw)),
                ("Global", global_L),
                ("Slisemap", slisemap_L),
                ("SHAP", lambda sm, **kw: shap_L(sm, pred_fn, **kw)),
                ("LIME", lambda sm, **kw: lime_L(sm, pred_fn, **kw)),
                ("LIME (nd)", lambda sm, **kw: lime_L(sm, pred_fn, **kw, disc=False)),
            ]:
                try:
                    results.append(
                        evaluate(
                            sm,
                            fn,
                            epsilon,
                            method,
                            data,
                            job_index,
                            200 if method == "LIME" and sm.n > 600 else 0,
                        )
                    )
                    # print(results[-1])
                    df = pd.DataFrame(results)
                    df["data"] = df["data"].astype("category")
                    df["method"] = df["method"].astype("category")
                    df.to_parquet(out_path)
                except Exception as e:
                    print(f"{job_index:02d} {data}: Error\n", e, flush=True)
                    # raise e
            print(f"{job_index:02d} {data}: Done", flush=True)
