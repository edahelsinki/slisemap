###############################################################################
#
# This experiment tunes hyperparameters for Slipmap etc.
# This experiment is designed to be run in parallel (e.g. on a computer cluster).
#
# Run this script to perform the experiments, where $index is [1..10]:
#   `python experiments/ida2024/hyperparameters.py $index`
#
# Run this script again without additional arguments to produce a table from the results:
#   `python experiments/ida2024/hyperparameters.py`
#
###############################################################################

import functools
import gc
import sys
from glob import glob
from timeit import default_timer as timer
from itertools import chain

import numpy as np
import pandas as pd
import torch
from project_paths import RESULTS_DIR
from scipy.stats import trim_mean
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC

from slisemap.utils import softmax_column_kernel, softmax_row_kernel, squared_distance

OUTPUT_DIR = RESULTS_DIR / "slipmap_hyper"

from experiments.data import (
    get_airquality,
    get_covertype3,
    get_gas_turbine,
    get_higgs,
    get_jets,
    get_qm9,
)
from slisemap import Slisemap
from slisemap.local_models import LinearRegression, LogisticRegression
from slisemap.slipmap import Slipmap
from slisemap.tuning import hyperparameter_tune


def discretise(y):
    return np.eye(y.shape[1], dtype=y.dtype)[np.argmax(y, 1), :]


def get_bb(bb, cls, X, y, dname=None):
    if bb == "AdaBoost":
        if cls:
            m = AdaBoostClassifier(random_state=42).fit(X, y)
            return m.predict_proba
        else:
            if dname == "Gas Turbine":
                m = AdaBoostRegressor(random_state=42, learning_rate=0.05).fit(X, y)
            else:
                m = AdaBoostRegressor(random_state=42).fit(X, y)
            return m.predict
    if bb == "Random Forest":
        if cls:
            if dname == "Jets":
                m = RandomForestClassifier(
                    n_jobs=-1, max_leaf_nodes=100, max_features=3, random_state=42
                ).fit(X, y[:, 1])
            else:
                m = RandomForestClassifier(
                    n_jobs=-1, max_leaf_nodes=100, random_state=42
                ).fit(X, y[:, 1])
            return m.predict_proba
        else:
            m = RandomForestRegressor(n_jobs=-1, max_leaf_nodes=100, random_state=42)
            m.fit(X, y)
            return m.predict
    if bb == "Gradient Boosting":
        if cls:
            if dname == "Higgs":
                m = GradientBoostingClassifier(
                    random_state=42, learning_rate=0.1, max_depth=5
                ).fit(X, y[:, 1])
            else:
                m = GradientBoostingClassifier(random_state=42).fit(X, y[:, 1])
            return m.predict_proba
        else:
            m = GradientBoostingRegressor(random_state=42).fit(X, y)
            return m.predict
    if bb == "SVM":
        if cls:
            m = SVC(random_state=42, probability=True).fit(X, y[:, 1])
            return m.predict_proba
        else:
            pass
    if bb == "Neural Network":
        if cls:
            m = MLPClassifier((64, 32, 16), random_state=42, early_stopping=True)
            m.fit(X, y)
            return m.predict_proba
        else:
            m = MLPRegressor((128, 64, 32, 16), random_state=42, early_stopping=True)
            m.fit(X, y)
            return m.predict
    raise NotImplementedError(f"[BB: {bb}] not implemented for [classifier: {cls}]")


def get_data():
    """Get raw data and the name of a black box model from `get_bb`

    Returns:
        Dictionary from `data_name` to `lambda: (X, Y, bb_name, classification)`
    """
    return {
        "Air Quality": lambda: (*get_airquality(), "Random Forest", False),
        "Gas Turbine": lambda: (*get_gas_turbine(), "AdaBoost", False),
        "QM9": lambda: (*get_qm9(), "Neural Network", False),
        "Covertype": lambda: (*get_covertype3(), "Neural Network", True),
        "Higgs": lambda: (*get_higgs(), "Gradient Boosting", True),
        "Jets": lambda: (*get_jets(), "Random Forest", True),
    }


def get_pretrained_data():
    """Get data with predictions and the name of the black box model

    Returns:
        Dictionary from `data_name` to `lambda: (X, P, bb_name, classification)`
    """
    return {
        "Air Quality": lambda: (*get_airquality("rf"), "rf", False),
        "Gas Turbine": lambda: (*get_gas_turbine("ada"), "ada", False),
        "QM9": lambda: (*get_qm9("nn"), "nn", False),
        "Covertype": lambda: (*get_covertype3("lb"), "lb", True),
        "Higgs": lambda: (*get_higgs("gb"), "gb", True),
        "Jets": lambda: (*get_jets("rf"), "rf", True),
    }


def hyperopt_slipmap(
    X_train,
    y_train,
    X_test,
    y_test,
    classifier,
    weighted=False,
    row_kernel=False,
    squared=True,
    density=False,
):
    params = hyperparameter_tune(
        Slipmap,
        X_train,
        y_train,
        X_test,
        y_test,
        model=False,
        local_model=LogisticRegression if classifier else LinearRegression,
        kernel=softmax_row_kernel if row_kernel else softmax_column_kernel,
        distance=squared_distance if squared else torch.cdist,
        prototypes=1.0 if density else 52,
        predict_kws=dict(weighted=weighted),
    )
    return params


def hyperopt_slisemap(X_train, y_train, X_test, y_test, classifier):
    params = hyperparameter_tune(
        Slisemap,
        X_train,
        y_train,
        X_test,
        y_test,
        model=False,
        radius=3.5,
        n_calls=10,
        local_model=LogisticRegression if classifier else LinearRegression,
    )
    return params


def hyperopt_knn(X_train, y_train, X_test, y_test, classifier):
    if classifier:
        mod = KNeighborsClassifier(1, n_jobs=-1)
        y_train = discretise(y_train)
    else:
        mod = KNeighborsRegressor(1, n_jobs=-1)
    grid = {"n_neighbors": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]}
    opt = GridSearchCV(mod, grid, cv=5, refit=False).fit(X_train, y_train)
    k = opt.best_params_["n_neighbors"]
    del opt
    return dict(n_neighbors=k)


def get_hyperopts():
    return {
        # "Slipmap": functools.partial(hyperopt_slipmap, weighted=False, squared=False),
        # "SlipmapW": functools.partial(hyperopt_slipmap, weighted=True, squared=False),
        # "SlipmapWS": functools.partial(hyperopt_slipmap, weighted=True, squared=True),
        "SlipmapWSD": functools.partial(
            hyperopt_slipmap, weighted=True, squared=True, density=True
        ),
        # "SlipmapS": functools.partial(hyperopt_slipmap, weighted=False, squared=True),
        # "SlipmapWT": functools.partial(hyperopt_slipmap, row_kernel=True, weighted=True, squared=False),
        "Slisemap": hyperopt_slisemap,
        # "K Nearest Neighbours": hyperopt_knn,
    }


def evaluate(job):
    print("Job:", job)
    np.random.seed(42 + job)
    torch.manual_seed(42 + job)
    file = OUTPUT_DIR / f"hyper_{job:02d}.parquet"
    file.parent.mkdir(parents=True, exist_ok=True)
    if file.exists():
        results = pd.read_parquet(file)
        # results = results[
        #     (~results["data"].isin(["Higgs"]))
        #     + (~results["method"].isin(["SlipmapWSD"]))
        # ]
    else:
        results = pd.DataFrame()
    nobb = lambda X, y, bb, cls: (X, y, "", cls)
    datasets = chain(
        ((n, f, False) for n, f in get_data().items()),
        ((n, lambda: nobb(*f()), True) for n, f in get_data().items()),
        ((n, f, True) for n, f in get_pretrained_data().items()),
    )
    for dname, dfn, pretrained in datasets:
        print(f"Loading:    {dname}", flush=True)
        X, y, bb, cls = dfn()
        ntrain = min(10_000, X.shape[0] * 3 // 4)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=ntrain, random_state=424242 + job, shuffle=True
        )
        if pretrained:
            p_train = y_train
            p_test = y_test
        else:
            print(f"Training:   {dname} - {bb}", flush=True)
            bb_pred = get_bb(bb, cls, X_train, y_train, dname)
            p_train = bb_pred(X_train)
            p_test = bb_pred(X_test)
        for mname, hyperopt in get_hyperopts().items():
            results = _eval(
                dname,
                mname,
                bb,
                hyperopt,
                X_train,
                p_train,
                X_test,
                p_test,
                cls,
                job,
                results,
                file,
            )
    print("Done", flush=True)


def _eval(
    dname,
    mname,
    bb,
    hyperopt,
    X_train,
    y_train,
    X_test,
    y_test,
    cls,
    job,
    results,
    file,
):
    if not results.empty:
        mask = results["data"] == dname
        mask &= results["method"] == mname
        mask &= results["BB"] == bb
        if mask.any():
            return results
    print(f"Tuning:     {dname} - {mname} - {bb}", flush=True)
    gc.collect()
    try:
        time = timer()
        params = hyperopt(X_train, y_train, X_test, y_test, cls)
        time = timer() - time
        res = {
            "job": job,
            "data": dname,
            "BB": bb,
            "method": mname,
            "time": time,
            **params,
        }
        results = pd.concat((results, pd.DataFrame([res])), ignore_index=True)
        results.to_parquet(file)
    except Exception as e:
        print(f"Failed:     {dname} - {mname} - {bb}\n", e, flush=True)
    return results


### Cached results for use in other experiments


def get_slipmap_params(
    dname, bb, weighted=True, row_kernel=False, squared=True, density=True
):
    variant = "Slipmap"
    if weighted:
        variant += "W"
    if row_kernel:
        variant += "T"
    if squared:
        variant += "S"
    if density:
        variant += "D"
    params = get_params(variant, dname, bb)
    if row_kernel:
        params["kernel"] = softmax_row_kernel
    if squared:
        params["distance"] = squared_distance
    if density:
        params["prototypes"] = 1.0
    return params


def get_knn_params(dname, bb):
    params = get_params("K Nearest Neighbours", dname, bb)
    params["n_neighbors"] = max(2, int(np.round(params["n_neighbors"])))
    return params


def get_slisemap_params(dname, bb):
    return get_params("Slisemap", dname, bb)


def get_params(method, data, black_box):
    if black_box is None:
        black_box = ""
    try:
        return PARAM_CACHE[(method, data, black_box)]
    except:
        raise NotImplementedError(
            f"No cached parameters for {method} & {data} & {black_box}"
        )


PARAM_CACHE = {
    ("K Nearest Neighbours", "Air Quality", ""): {"n_neighbors": 6.66667},
    ("K Nearest Neighbours", "Air Quality", "Random Forest"): {"n_neighbors": 4.66667},
    ("K Nearest Neighbours", "Air Quality", "rf"): {"n_neighbors": 5.33333},
    ("K Nearest Neighbours", "Gas Turbine", ""): {"n_neighbors": 3.0},
    ("K Nearest Neighbours", "Gas Turbine", "AdaBoost"): {"n_neighbors": 3.83333},
    ("K Nearest Neighbours", "Gas Turbine", "ada"): {"n_neighbors": 4.0},
    ("K Nearest Neighbours", "Higgs", ""): {"n_neighbors": 3.0},
    ("K Nearest Neighbours", "Higgs", "Gradient Boosting"): {"n_neighbors": 32.0},
    ("K Nearest Neighbours", "Higgs", "gb"): {"n_neighbors": 32.0},
    ("K Nearest Neighbours", "Jets", ""): {"n_neighbors": 32.0},
    ("K Nearest Neighbours", "Jets", "Random Forest"): {"n_neighbors": 3.0},
    ("K Nearest Neighbours", "Jets", "rf"): {"n_neighbors": 32.0},
    ("K Nearest Neighbours", "QM9", ""): {"n_neighbors": 6.0},
    ("K Nearest Neighbours", "QM9", "Neural Network"): {"n_neighbors": 2.0},
    ("K Nearest Neighbours", "QM9", "nn"): {"n_neighbors": 2.5},
    ("Slipmap", "Air Quality", ""): {"lasso": 7.0941, "ridge": 0.31346},
    ("Slipmap", "Air Quality", "Random Forest"): {"lasso": 0.51339, "ridge": 0.10523},
    ("Slipmap", "Air Quality", "rf"): {"lasso": 3.12182, "ridge": 0.12288},
    ("Slipmap", "Gas Turbine", ""): {"lasso": 0.05617, "ridge": 0.00071},
    ("Slipmap", "Gas Turbine", "AdaBoost"): {"lasso": 0.20794, "ridge": 0.14495},
    ("Slipmap", "Gas Turbine", "ada"): {"lasso": 0.23252, "ridge": 0.22475},
    ("Slipmap", "Higgs", ""): {"lasso": 1.53522, "ridge": 0.00054},
    ("Slipmap", "Higgs", "Gradient Boosting"): {"lasso": 0.00259, "ridge": 0.00272},
    ("Slipmap", "Higgs", "gb"): {"lasso": 0.00268, "ridge": 0.00015},
    ("Slipmap", "Jets", ""): {"lasso": 9.99805, "ridge": 0.89564},
    ("Slipmap", "Jets", "Random Forest"): {"lasso": 0.00325, "ridge": 0.00056},
    ("Slipmap", "Jets", "rf"): {"lasso": 0.00162, "ridge": 0.00013},
    ("Slipmap", "QM9", ""): {"lasso": 0.00404, "ridge": 0.00011},
    ("Slipmap", "QM9", "Neural Network"): {"lasso": 0.00259, "ridge": 0.00017},
    ("Slipmap", "QM9", "nn"): {"lasso": 0.02303, "ridge": 0.00034},
    ("SlipmapS", "Air Quality", "Random Forest"): {
        "lasso": 0.0661,
        "ridge": 0.00293,
        "radius": 1.97353,
    },
    ("SlipmapS", "Air Quality", "rf"): {
        "lasso": 1.75835,
        "ridge": 0.337,
        "radius": 1.76512,
    },
    ("SlipmapS", "Gas Turbine", "AdaBoost"): {
        "lasso": 0.01294,
        "ridge": 0.03071,
        "radius": 6.15951,
    },
    ("SlipmapS", "Gas Turbine", "ada"): {
        "lasso": 0.31336,
        "ridge": 0.01252,
        "radius": 5.81604,
    },
    ("SlipmapS", "Higgs", "Gradient Boosting"): {
        "lasso": 0.00265,
        "ridge": 0.00364,
        "radius": 1.75,
    },
    ("SlipmapS", "Higgs", "gb"): {"lasso": 0.00236, "ridge": 0.00457, "radius": 1.75},
    ("SlipmapS", "Jets", "Random Forest"): {
        "lasso": 0.00216,
        "ridge": 0.00249,
        "radius": 2.22414,
    },
    ("SlipmapS", "Jets", "rf"): {"lasso": 0.001, "ridge": 0.00455, "radius": 2.01111},
    ("SlipmapS", "QM9", "Neural Network"): {
        "lasso": 0.00749,
        "ridge": 0.00071,
        "radius": 1.76379,
    },
    ("SlipmapS", "QM9", "nn"): {"lasso": 0.00191, "ridge": 0.00053, "radius": 1.78047},
    ("SlipmapW", "Air Quality", ""): {"lasso": 5.52191, "ridge": 0.22231},
    ("SlipmapW", "Air Quality", "Random Forest"): {"lasso": 0.09699, "ridge": 0.19023},
    ("SlipmapW", "Air Quality", "rf"): {"lasso": 0.67284, "ridge": 0.01959},
    ("SlipmapW", "Gas Turbine", ""): {"lasso": 0.0687, "ridge": 0.00016},
    ("SlipmapW", "Gas Turbine", "AdaBoost"): {"lasso": 0.14169, "ridge": 0.01774},
    ("SlipmapW", "Gas Turbine", "ada"): {"lasso": 0.18701, "ridge": 0.09055},
    ("SlipmapW", "Higgs", ""): {"lasso": 1.53522, "ridge": 0.00054},
    ("SlipmapW", "Higgs", "Gradient Boosting"): {"lasso": 0.00345, "ridge": 0.1146},
    ("SlipmapW", "Higgs", "gb"): {"lasso": 0.00348, "ridge": 0.0029},
    ("SlipmapW", "Jets", ""): {"lasso": 9.99747, "ridge": 0.87175},
    ("SlipmapW", "Jets", "Random Forest"): {"lasso": 0.00359, "ridge": 0.00012},
    ("SlipmapW", "Jets", "rf"): {"lasso": 0.00375, "ridge": 0.00013},
    ("SlipmapW", "QM9", ""): {"lasso": 0.00116, "ridge": 0.00011},
    ("SlipmapW", "QM9", "Neural Network"): {"lasso": 0.00213, "ridge": 0.00054},
    ("SlipmapW", "QM9", "nn"): {"lasso": 0.00112, "ridge": 0.00017},
    ("SlipmapWS", "Air Quality", ""): {
        "lasso": 9.063,
        "ridge": 0.52392,
        "radius": 1.77337,
    },
    ("SlipmapWS", "Air Quality", "Random Forest"): {
        "lasso": 0.06616,
        "ridge": 0.22716,
        "radius": 1.90794,
    },
    ("SlipmapWS", "Air Quality", "rf"): {
        "lasso": 2.64702,
        "ridge": 0.00924,
        "radius": 2.11927,
    },
    ("SlipmapWS", "Gas Turbine", ""): {
        "lasso": 0.00612,
        "ridge": 0.0022,
        "radius": 1.75,
    },
    ("SlipmapWS", "Gas Turbine", "AdaBoost"): {
        "lasso": 0.09824,
        "ridge": 0.00494,
        "radius": 3.33148,
    },
    ("SlipmapWS", "Gas Turbine", "ada"): {
        "lasso": 0.01723,
        "ridge": 0.42192,
        "radius": 3.90681,
    },
    ("SlipmapWS", "Higgs", ""): {"lasso": 1.6675, "ridge": 0.3334, "radius": 1.75},
    ("SlipmapWS", "Higgs", "Gradient Boosting"): {
        "lasso": 1.28542,
        "ridge": 0.33349,
        "radius": 1.75056,
    },
    ("SlipmapWS", "Higgs", "gb"): {"lasso": 1.69236, "ridge": 0.66676, "radius": 1.75},
    ("SlipmapWS", "Jets", ""): {"lasso": 10.0, "ridge": 1.0, "radius": 1.75},
    ("SlipmapWS", "Jets", "Random Forest"): {
        "lasso": 0.00281,
        "ridge": 0.019,
        "radius": 1.81002,
    },
    ("SlipmapWS", "Jets", "rf"): {
        "lasso": 0.00479,
        "ridge": 0.00074,
        "radius": 1.85935,
    },
    ("SlipmapWS", "QM9", ""): {"lasso": 0.00127, "ridge": 0.0001, "radius": 1.75},
    ("SlipmapWS", "QM9", "Neural Network"): {
        "lasso": 0.00261,
        "ridge": 0.00011,
        "radius": 1.75,
    },
    ("SlipmapWS", "QM9", "nn"): {"lasso": 0.00577, "ridge": 0.00012, "radius": 1.75},
    ("SlipmapWSD", "Air Quality", ""): {
        "lasso": 8.25944,
        "ridge": 0.35008,
        "radius": 1.56704,
    },
    ("SlipmapWSD", "Air Quality", "Random Forest"): {
        "lasso": 7.00995,
        "ridge": 0.00417,
        "radius": 1.54459,
    },
    ("SlipmapWSD", "Air Quality", "rf"): {
        "lasso": 2.67087,
        "ridge": 0.23633,
        "radius": 1.59245,
    },
    ("SlipmapWSD", "Covertype", ""): {
        "lasso": 0.04772,
        "ridge": 0.00124,
        "radius": 1.63856,
    },
    ("SlipmapWSD", "Covertype", "Neural Network"): {
        "lasso": 0.00569,
        "ridge": 0.00011,
        "radius": 1.97278,
    },
    ("SlipmapWSD", "Covertype", "lb"): {
        "lasso": 0.00908,
        "ridge": 0.00011,
        "radius": 1.60191,
    },
    ("SlipmapWSD", "Gas Turbine", ""): {
        "lasso": 0.0889,
        "ridge": 0.00408,
        "radius": 1.58614,
    },
    ("SlipmapWSD", "Gas Turbine", "AdaBoost"): {
        "lasso": 0.27016,
        "ridge": 0.02581,
        "radius": 1.65055,
    },
    ("SlipmapWSD", "Gas Turbine", "ada"): {
        "lasso": 0.07564,
        "ridge": 0.00963,
        "radius": 1.93317,
    },
    ("SlipmapWSD", "Higgs", ""): {
        "lasso": 6.59376,
        "ridge": 0.00629,
        "radius": 2.15053,
    },
    ("SlipmapWSD", "Higgs", "Gradient Boosting"): {
        "lasso": 9.84084,
        "ridge": 0.22028,
        "radius": 2.49878,
    },
    ("SlipmapWSD", "Higgs", "gb"): {
        "lasso": 4.78263,
        "ridge": 0.01411,
        "radius": 2.0861,
    },
    ("SlipmapWSD", "Jets", ""): {"lasso": 10.0, "ridge": 0.98526, "radius": 2.50349},
    ("SlipmapWSD", "Jets", "Random Forest"): {
        "lasso": 0.00207,
        "ridge": 0.00264,
        "radius": 1.60605,
    },
    ("SlipmapWSD", "Jets", "rf"): {
        "lasso": 0.00266,
        "ridge": 0.00016,
        "radius": 2.30815,
    },
    ("SlipmapWSD", "QM9", ""): {"lasso": 0.00143, "ridge": 0.0001, "radius": 1.5},
    ("SlipmapWSD", "QM9", "Neural Network"): {
        "lasso": 0.05423,
        "ridge": 0.00418,
        "radius": 1.54166,
    },
    ("SlipmapWSD", "QM9", "nn"): {"lasso": 0.04422, "ridge": 0.00138, "radius": 1.5},
    ("Slisemap", "Air Quality", ""): {"lasso": 0.00404, "ridge": 0.0001},
    ("Slisemap", "Air Quality", "Random Forest"): {"lasso": 0.00201, "ridge": 0.00015},
    ("Slisemap", "Air Quality", "rf"): {"lasso": 0.00309, "ridge": 0.00013},
    ("Slisemap", "Covertype", ""): {"lasso": 0.00294, "ridge": 0.00019},
    ("Slisemap", "Covertype", "Neural Network"): {"lasso": 0.001, "ridge": 0.0001},
    ("Slisemap", "Covertype", "lb"): {"lasso": 0.00116, "ridge": 0.0001},
    ("Slisemap", "Gas Turbine", ""): {"lasso": 0.00118, "ridge": 0.00011},
    ("Slisemap", "Gas Turbine", "AdaBoost"): {"lasso": 0.00102, "ridge": 0.0002},
    ("Slisemap", "Gas Turbine", "ada"): {"lasso": 0.00171, "ridge": 0.00023},
    ("Slisemap", "Higgs", ""): {"lasso": 0.26733, "ridge": 0.16757},
    ("Slisemap", "Higgs", "Gradient Boosting"): {"lasso": 0.00544, "ridge": 0.32972},
    ("Slisemap", "Higgs", "gb"): {"lasso": 0.01427, "ridge": 0.02019},
    ("Slisemap", "Jets", ""): {"lasso": 0.07081, "ridge": 0.00153},
    ("Slisemap", "Jets", "Random Forest"): {"lasso": 0.001, "ridge": 0.00026},
    ("Slisemap", "Jets", "rf"): {"lasso": 0.001, "ridge": 0.00035},
    ("Slisemap", "QM9", ""): {"lasso": 0.001, "ridge": 0.0001},
    ("Slisemap", "QM9", "Neural Network"): {"lasso": 0.00107, "ridge": 0.00013},
    ("Slisemap", "QM9", "nn"): {"lasso": 0.00112, "ridge": 0.00013},
}


if __name__ == "__main__":
    if len(sys.argv) == 1:
        files = glob(str(OUTPUT_DIR / "*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df_cache = {
            k1: {k2: np.round(v2, 5) for k2, v2 in v1.items() if np.isfinite(v2)}
            for k1, v1 in df.drop(["job", "time"], axis=1)
            .groupby(["method", "data", "BB"])
            .aggregate(trim_mean, 0.2)
            .to_dict("index")
            .items()
        }
        print("PARAM_CACHE = ", str(df_cache).replace(", 'radius': 3.5", ""))
    else:
        evaluate(int(sys.argv[1]))
