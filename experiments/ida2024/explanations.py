###############################################################################
#
# This experiment compares Slipmap to local explanation methods.
# This experiment is designed to be run in parallel (e.g. on a computer cluster).
#
# Run this script to perform the experiments, where $index is [1..10]:
#   `python experiments/ida2024/explanations.py $index`
#
# Run this script again without additional arguments to produce a table from the results:
#   `python experiments/ida2024/explanations.py`
#
###############################################################################

import gc
import sys
import warnings
from functools import partial
from glob import glob
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import shap
import shap.maskers
import slise
import torch
from lime.lime_tabular import LimeTabularExplainer
from hyperparameters import get_bb, get_data, get_slipmap_params, get_slisemap_params
from project_paths import RESULTS_DIR, MANUSCRIPT_DIR
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

from slisemap.utils import LBFGS

OUTPUT_DIR = RESULTS_DIR / "slipmap_xai"


from slisemap.local_models import LinearRegression, LogisticRegression
from slisemap.slipmap import Slipmap
from slisemap.slisemap import Slisemap, tonp


def losses(cls: bool, y: np.ndarray, yhat: np.ndarray):
    y = torch.as_tensor(y, dtype=torch.float32)
    yhat = torch.as_tensor(yhat, dtype=torch.float32)
    if len(yhat.shape) == 1:
        yhat = yhat[:, None]
    y = torch.reshape(y, yhat.shape)
    if cls:
        L = LogisticRegression.loss(yhat, y)
    else:
        L = LinearRegression.loss(yhat, y)
    return L.cpu().numpy()


def get_slise(X, y, classifier, dname, bb, epsilon: float):
    params = get_slipmap_params(dname, bb, squared=False, density=False)
    if classifier:
        y = y[:, 0]
    s = slise.SliseExplainer(
        X,
        y,
        epsilon**0.5,
        logit=classifier,
        lambda1=params["lasso"],
        lambda2=params["ridge"],
    )

    def explain(i):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s.explain(i)
        if classifier:
            return lambda X: np.stack((s.predict(X), 1.0 - s.predict(X)), -1)
        return s.predict

    return explain


def get_global(X, y, classifier, dname, bb):
    lm = LogisticRegression if classifier else LinearRegression
    params = get_slipmap_params(dname, bb, squared=False, density=False)
    sm = Slipmap(X, y, local_model=lm, **params)
    B = sm.get_Bp(False)[:1, :]
    return lambda i: lambda X: tonp(sm.local_model(sm._as_new_X(X), B)[0, ...])


def get_slipmap(
    X, y, classifier, dname, bb, weighted=False, squared=False, density=False
):
    sm = Slipmap(
        X,
        y,
        local_model=LogisticRegression if classifier else LinearRegression,
        **get_slipmap_params(
            dname, bb, weighted=weighted, squared=squared, density=density
        ),
    )
    sm.optimise()
    B = sm.get_B(False)
    return lambda i: lambda X: tonp(sm.local_model(sm._as_new_X(X), B[None, i])[0, ...])


def get_slisemap(X, y, classifier, dname, bb):
    lm = LogisticRegression if classifier else LinearRegression
    sm = Slisemap(
        X, y, local_model=lm, random_state=42, **get_slisemap_params(dname, bb)
    )
    sm.optimise()
    B = sm.get_B(False)
    return lambda i: lambda X: tonp(sm.local_model(sm._as_new_X(X), B[None, i])[0, ...])


def get_lime(X, y, pred_fn, disc=True, classifier=False):
    explainer = LimeTabularExplainer(
        X,
        "classification" if classifier else "regression",
        y,
        discretize_continuous=disc,
    )

    def explain(i):
        exp = explainer.explain_instance(X[i, :], pred_fn, num_samples=5_000)
        b = np.zeros((1, X.shape[1]))
        for j, v in exp.as_map()[1]:
            b[0, j] = v
        inter = exp.intercept[1]
        xi = explainer.discretizer.discretize(X[i : i + 1, :])

        def predict(X):
            if disc:
                # Lime works on discretised data, so we need to discretise the data
                # to be able to apply the linear model, and use lime for prediction.
                X = explainer.discretizer.discretize(X) == xi
            Y = np.sum(X * b, -1, keepdims=True) + inter
            if classifier:
                Y = np.clip(Y, 0.0, 1.0)
                Y = np.concatenate((1.0 - Y, Y), -1)
            return Y

        return predict

    return explain


def get_shap(X, y, pred_fn, partition=True, classifier=False):
    if classifier:
        link = shap.links.logit
        y = y[:, 0]
        old_pred = pred_fn
        pred_fn = lambda X: old_pred(X)[:, 0]
    else:
        link = shap.links.identity
    if partition:
        masker = shap.maskers.Partition(X, max_samples=1_000)
        explainer = shap.explainers.Partition(
            pred_fn, masker=masker, link=link, linearize_link=False
        )

        def explain(i):
            shapX = X[None, i, :]
            exp = explainer(shapX, silent=True)
            b = exp.values.reshape((exp.values.shape[0], -1))
            inter = float(exp.base_values)
            scale = _shap_get_scale(X, y, shapX, b, inter, link)
            return lambda X: _shap_predict(X, shapX, scale, b, inter, classifier)

        return explain
    else:
        explainer = shap.explainers.Sampling(pred_fn, X)

        def explain(i):
            shapX = X[None, i, :]
            b = explainer.shap_values(shapX, silent=True)
            inter = explainer.expected_value
            scale = _shap_get_scale(X, y, shapX, b, inter, link)
            return lambda X: _shap_predict(X, shapX, scale, b, inter, classifier)

        return explain


def _shap_predict(X, shapX, scale, b, intercept, classifier):
    dist = (X - shapX) ** 2
    kernel = np.exp(-(np.abs(scale) + 1e-6) * dist)
    P = np.sum(kernel * b, -1, keepdims=True) + intercept
    if classifier:
        P = shap.links.logit.inverse(P)
        return np.stack((P, 1.0 - P), -1)
    return P


def _shap_get_scale(X, y, shapX, b, intercept, link):
    dist = torch.as_tensor((X - shapX) ** 2)
    scale = torch.ones_like(dist[:1, :], requires_grad=True)
    Y = torch.as_tensor(link(y), dtype=dist.dtype)
    if len(Y.shape) == 1:
        Y = Y[:, None]
    b = torch.as_tensor(b, dtype=dist.dtype)

    def loss():
        kernel = torch.exp(-(torch.abs(scale) + 1e-6) * dist)
        P = torch.sum(kernel * b, -1, keepdim=True) + intercept
        return torch.mean(torch.abs(Y - P))

    LBFGS(loss, [scale])
    return tonp(scale)


def get_methods(dname, bb, cls, epsilon, pred_fn):
    return {
        "Global": partial(get_global, dname=dname, bb=bb, classifier=cls),
        "SLISE": partial(
            get_slise, epsilon=epsilon, dname=dname, bb=bb, classifier=cls
        ),
        "LIME (nd)": partial(get_lime, pred_fn=pred_fn, disc=False, classifier=cls),
        "LIME": partial(get_lime, pred_fn=pred_fn, classifier=cls),
        "SHAP": partial(get_shap, pred_fn=pred_fn, classifier=cls),
        # "Slipmap": partial(get_slipmap, dname=dname, bb=bb, classifier=cls),
        # "SlipmapWS": partial(get_slipmap, dname=dname, bb=bb, classifier=cls, weighted=True, squared=True),
        "SlipmapWSD": partial(
            get_slipmap,
            dname=dname,
            bb=bb,
            classifier=cls,
            weighted=True,
            squared=True,
            density=True,
        ),
        "Slisemap": partial(get_slisemap, dname=dname, bb=bb, classifier=cls),
    }


def evaluate(job):
    print("Job:", job)
    np.random.seed(42 + job)
    torch.manual_seed(42 + job)
    file = OUTPUT_DIR / f"xai_{job:02d}.parquet"
    file.parent.mkdir(parents=True, exist_ok=True)
    if file.exists():
        results = pd.read_parquet(file)
        # results = results[(~results["data"].isin(["Higgs"]))]
        # results = results[(~results["method"].isin(["SlipmapWSD"]))]
    else:
        results = pd.DataFrame()
    for dname, dfn in get_data().items():
        print(f"Loading:    {dname}", flush=True)
        X, y, bb, cls = dfn()
        n = min(5_000, X.shape[0] * 3 // 4)
        X, _, y, _ = train_test_split(
            X, y, train_size=n, random_state=142 + job, shuffle=True
        )
        if cls and y.shape[1] > 2:
            continue
        n_expl = 100
        D = cdist(X[:n_expl, :], X)
        D += np.eye(n_expl, X.shape[0]) * np.max(D)
        nn = np.argsort(D, 1)[:, :5]
        pred_fn = get_bb(bb, cls, X, y, dname)
        y = pred_fn(X)
        epsilon = np.quantile(
            losses(cls, y, get_global(X, y, cls, dname, bb)(0)(X)), 0.3
        )
        for mname, fn in get_methods(dname, bb, cls, epsilon, pred_fn).items():
            if (
                not results.empty
                and ((results["method"] == mname) & (results["data"] == dname)).any()
            ):
                continue
            try:
                print(f"Preparing:  {dname} - {mname}", flush=True)
                gc.collect()
                time = timer()
                explainer = fn(X, y)
                time1 = timer() - time
                print(f"Explaining: {dname} - {mname}", flush=True)
                time2 = 0.0
                L = []
                for i in range(n_expl):
                    time = timer()
                    pred = explainer(i)
                    time2 += timer() - time
                    pi = pred(X)
                    L.append(losses(cls, y, pi))
                L = np.stack(L, 0)
                time2 /= n_expl
                print(f"Evaluating: {dname} - {mname}", flush=True)
                res = dict(
                    job=job,
                    method=mname,
                    data=dname,
                    time_setup=time1,
                    time_explain=time2,
                    time_one=time1 / X.shape[0] + time2,
                    time_all=time1 + time2 * X.shape[0],
                    local_loss=L.diagonal().mean(),
                    coverage=(L < epsilon).mean(),
                    stability=L[np.arange(L.shape[0])[:, None], nn].mean(),
                    epsilon=epsilon,
                    BB=bb,
                )
                results = pd.concat((results, pd.DataFrame([res])), ignore_index=True)
                results.to_parquet(file)
            except Exception as e:
                print(f"{job:02d} {dname}: Error\n", e, flush=True)
                continue
    print("Done", flush=True)


def table_results(df: pd.DataFrame):
    rename_cols = {
        "data": "Data",
        "method": "Method",
        "time_one": "Time (s) $\\downarrow$",
        "local_loss": "Local loss $\\downarrow$",
        "stability": "Stability $\\downarrow$",
        "coverage": "\\bfseries Coverage $\\uparrow$",
    }
    rename_methods = {
        "LIME": "{\\sc lime}",
        "LIME (nd)": "{\\sc lime} (nd)",
        "SHAP": "{\\sc shap}",
        "SLISE": "{\\sc slise}",
        # "Slipmap": "{\\sc slipmap}",
        # "SlipmapWS": "{\\sc slipmap}${}^2$",
        "SlipmapWSD": "{\\sc slipmap}",
        "Slisemap": "{\\sc slisemap}",
    }
    pm = lambda x: (np.mean(x), np.std(x))

    def bold(df: pd.DataFrame):
        for col in rename_cols.values():
            fmt = lambda x: f"{x[0]:.3f} $\\pm$ {x[1]:.2f}"
            if "uparrow" in col:
                th = df[col].apply(lambda x: x[0] - x[1]).max() - 1e-6
                df[col] = df[col].apply(
                    lambda x: fmt(x) if x[0] < th else f"\\textbf{{{fmt(x)}}}"
                )
            elif "downarrow" in col:
                th = df[col].apply(lambda x: x[0] + x[1]).min() + 1e-6
                df[col] = df[col].apply(
                    lambda x: fmt(x) if x[0] > th else f"\\textbf{{{fmt(x)}}}"
                )
        return df

    df["data"] = pd.Categorical(df["data"], list(get_data().keys()) + [""], True)

    df2 = (
        df[list(rename_cols.keys())][df["method"].isin(list(rename_methods.keys()))]
        .rename(columns=rename_cols)
        .groupby(["Data", "Method"], group_keys=False, observed=True)
        .aggregate(pm)
        .groupby("Data", group_keys=False)
        .apply(bold)
        .reset_index()
    )
    df2["Data"].where(~df2["Data"].duplicated(), "", inplace=True)
    df2["Method"].replace(rename_methods, inplace=True)
    df2["Data"] = df2["Data"].cat.rename_categories(
        {c: "\\rule{0pt}{2.8ex}" + c for c in df2["Data"].cat.categories[1:-1]}
    )
    (
        df2.style.format(precision=3)
        .hide(axis=0)
        .applymap_index(lambda v: "font-weight: bold;", axis="columns")
        .to_latex(
            MANUSCRIPT_DIR / "xai_table.tex",
            column_format="l l " + "r@{\\hspace{3mm}}" * 3 + "r",
            hrules=True,
            convert_css=True,
        )
    )
    print("Results exported to", str(MANUSCRIPT_DIR / "xai_table.tex"))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        files = glob(str(OUTPUT_DIR / "*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        table_results(df)
    else:
        evaluate(int(sys.argv[1]))
