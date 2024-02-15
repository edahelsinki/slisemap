import sys
import torch

from glob import glob
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
from scipy.special import logit
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

root = str(Path(__file__).parent.parent.parent.absolute())
if root not in sys.path:
    sys.path.insert(0, root)

from experiments.data import (
    get_airquality,
    get_higgs,
    get_qm9,
    get_jets,
    get_gas_turbine,
    get_covertype3,
    get_rsynth,
)
from slisemap import Slisemap
from slisemap.slipmap import Slipmap
from slisemap.utils import to_tensor, tonp
from slisemap.local_models import LogisticRegression
from hyperparameters import get_slipmap_params, get_slisemap_params
import time
import gc


def get_data():
    # dict of data acq function and bb model
    return {
        "synthetic": ("Synthetic", get_rsynth, None),
        "gas_turbine": ("Gas Turbine", get_gas_turbine, "ada"),
        "qm9": ("QM9", get_qm9, "nn"),
        "air_quality": ("Air Quality", get_airquality, "rf"),
        "jets": ("Jets", get_jets, "rf"),
        "higgs": ("Higgs", get_higgs, "gb"),
        "covertype": ("Covertype", get_covertype3, "lb"),
    }


def get_models():
    def sm_return(sm: Slisemap):
        def pred(X):
            X = to_tensor(X, **sm.tensorargs)[0]
            D = torch.cdist(X, sm.get_X(False, False))
            B = sm.get_B(False)[D.argmin(1), :]
            return sm.predict(X, B)[:, 0]

        return tonp(sm.local_model(sm._X, sm._B)[:, :, 0]), sm.get_D(), pred


def sample_data(dataset_name, n_samples):
    # load dataset
    _, data_fun, bb_model = get_data()[dataset_name]
    X, Y = data_fun(blackbox=bb_model)
    if type(X) == np.ndarray:
        X = torch.tensor(X, device=device)
    if type(Y) == np.ndarray:
        Y = torch.tensor(Y, device=device)
    # ensure data is in GPU (if available)
    if torch.cuda.is_available():
        Y = Y.to(device)
        X = X.to(device)
    # scramble data for subsampling
    random_indices = rng.permutation(len(X))
    X = X[random_indices, :]
    Y = Y[random_indices]
    # get subset of data
    start_idx = (job_id * max(sample_sizes)) % len(X)
    if (start_idx + n_samples) < len(X):
        X_s = X[start_idx : start_idx + n_samples, :]
        Y_s = Y[start_idx : start_idx + n_samples]
    else:
        X_s = torch.vstack(
            (X[start_idx:, :], X[0 : n_samples - (len(X) - start_idx), :])
        )
        if Y.dim() > 1:
            Y_s = torch.vstack(
                (Y[start_idx:, :], Y[0 : n_samples - (len(X) - start_idx), :])
            )
        else:
            Y_s = torch.hstack((Y[start_idx:], Y[0 : n_samples - (len(X) - start_idx)]))

    # get another subset of data, half overlapping with the previous
    start_idx = (job_id * max(sample_sizes) + (n_samples // 2)) % len(X)
    if (start_idx + n_samples) < len(X):
        X_h = X[start_idx : start_idx + n_samples, :]
        Y_h = Y[start_idx : start_idx + n_samples]
    else:
        X_h = torch.vstack(
            (X[start_idx:, :], X[0 : n_samples - (len(X) - start_idx), :])
        )
        if Y.dim() > 1:
            Y_h = torch.vstack(
                (Y[start_idx:, :], Y[0 : n_samples - (len(X) - start_idx), :])
            )
        else:
            Y_h = torch.hstack((Y[start_idx:], Y[0 : n_samples - (len(X) - start_idx)]))
    return X_s, Y_s, X_h, Y_h


def profile_optimisation(sm):
    print("\tProfiling", flush=True)
    # empty caches
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    # time optimisation
    t1 = time.perf_counter(), time.process_time()
    sm.optimise()
    t2 = time.perf_counter(), time.process_time()
    sm.metadata["training_wall_time"] = t2[0] - t1[0]
    sm.metadata["training_cpu_time"] = t2[1] - t1[1]
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated(device=device)
        torch.cuda.reset_peak_memory_stats()
        sm.metadata["peak_cuda_memory"] = mem
        return t2[0] - t1[0], t2[1] - t1[1], mem
    return t2[0] - t1[0], t2[1] - t1[1]


def train_and_save_sm(X, y, fname, hyperparams, profile=True):
    if "slipmap" in str(fname):
        sm = Slipmap(X, y, **hyperparams)
    else:
        sm = Slisemap(X, y, **hyperparams)
    if profile:
        profile_optimisation(sm)
    else:
        sm.optimise()
    sm.save(fname)


def slipmap_hyperparams(dname, bb, squared=True, density=False):
    out = {}
    out["device"] = device
    out = out | get_slipmap_params(dname, bb, squared=squared, density=density)
    if dname in ["Higgs", "Covertype", "Jets"]:
        out["local_model"] = LogisticRegression
    return out


def slisemap_hyperparams(dname, bb):
    out = {}
    out["device"] = device
    out["random_state"] = random_seed
    out = out | get_slisemap_params(dname, bb)
    if dname in ["Higgs", "Covertype", "Jets"]:
        out["local_model"] = LogisticRegression
    return out


# load datasets
job_id = int(sys.argv[1]) - 1
dataset_name = sys.argv[2]
model_dir = Path(sys.argv[3])
model_dir.mkdir(parents=True, exist_ok=True)
slipmap_dir = model_dir / Path("slipmap/")
slipmap_dir.mkdir(parents=True, exist_ok=True)
slisemap_dir = model_dir / Path("slisemap/")
slisemap_dir.mkdir(parents=True, exist_ok=True)
random_seed = 1618033
rng = np.random.default_rng(random_seed)
sample_sizes = np.round(np.logspace(np.log10(100), np.log10(10000), 10)).astype(int)
slipmap_squared = True
slipmap_density = True
profile = True

# set up device
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    print("GPU not available, reverting to CPU!", flush=True)
    device = torch.device("cpu")


# train cohort of models (or use pretrained if they exist)
for n_samples in sample_sizes:
    print(
        f"Job ID {job_id+1} starting on {dataset_name} with "
        + f"n_samples={n_samples}",
        flush=True,
    )
    start = time.time()
    Xs, ys, Xh, yh = sample_data(dataset_name, n_samples)
    for model_type in ["slipmap", "slisemap"]:
        for test_type in ["normal", "half", "permuted"]:
            model_name = model_type + "_" + test_type
            full_dname, _, bb = get_data()[dataset_name]
            hyperparams = (
                slipmap_hyperparams(
                    full_dname, bb, squared=slipmap_squared, density=slipmap_density
                )
                if model_type == "slipmap"
                else slisemap_hyperparams(full_dname, bb)
            )
            fname = (
                model_dir
                / Path(model_type)
                / Path(
                    f'{dataset_name}_{n_samples}_{job_id}_{model_name}_lasso_{hyperparams["lasso"]}_ridge_{hyperparams["ridge"]}.sm'
                )
            )
            if fname.exists():
                print(f"\tFound {fname}, not retraining.", flush=True)
                continue
            train_start = time.time()
            print(f"\tTrain {model_name}.", flush=True)
            match test_type:
                case "normal":
                    X, y = Xs, ys
                case "half":
                    X, y = Xh, yh
                case "permuted":
                    X, y = Xs, ys[rng.permutation(len(ys))]
            train_and_save_sm(
                X,
                y,
                fname,
                hyperparams,
                profile=(profile if test_type == "normal" else False),
            )
            print(f"\tDone. Took {time.time() - train_start:.1f} s.")
    print(
        f"In total {dataset_name} {n_samples} samples run {job_id+1} duration: {time.time() - start:.1f} s.",
        flush=True,
    )
