import sys
import torch

from glob import glob
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch

root = str(Path(__file__).parent.parent.parent.absolute())
if root not in sys.path:
    sys.path.insert(0, root)

from slisemap import Slisemap
from slisemap.slipmap import Slipmap
import time

curr_path = Path(__file__)
sys.path.append(str(curr_path.parent))
from stability_tools import *
import os
from collections import defaultdict
import pickle

# process args
models = sys.argv[2]  # input model directory
results = sys.argv[3]  # results directory
results = Path(results)
results.mkdir(parents=True, exist_ok=True)
load_fun = (
    lambda fname, map_location: Slisemap.load(fname, map_location=map_location)
    if "slisemap" in models
    else Slipmap.load(fname, device=map_location)
)
test_name = sys.argv[4]
assert test_name in [
    "local_model_distance",
    "local_model_min_distance",
    "permutation_loss",
    "set_double_distance",
    "neighbourhood_distance",
], f"{test_name} not implemented!"
comparison_style = sys.argv[5]  # which comparison style should be used
assert comparison_style in [
    "versus",
    "permutation",
    "half",
    "half_perm",
    "perm_versus",
], f"{comparison_style} comparison style not implemented!"
n_runs = 10
job_id = int(sys.argv[1]) - 1
array_size = 10

# set up device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    print("GPU not available, reverting to CPU!")
    device = torch.device("cpu")


def load_models(model_dir: Path, comparison_style: str):
    """Load models for test depending on the comparison style.

    Args:
        model_dir: Path
        comparison_style: string
            How the models should be compared. Possible values are:
            'versus': compare each model against one another in a round-robin fashion
            'permutation': compare each model against a permuted counterpart with the
                same index in the filename.
            'half': compare each model against one with 50% shared points with the same
                index in the filename.
            'half_perm': compare each model against one with permuted labels and 50%
                shared points."""
    norm_files = sorted([x for x in os.listdir(model_dir) if "_normal" in x])
    perm_files = sorted([x for x in os.listdir(model_dir) if "_perm" in x])
    half_files = sorted([x for x in os.listdir(model_dir) if "_half" in x])
    if comparison_style == "permutation":
        normal_models = defaultdict(list)
        perm_models = defaultdict(list)
        for f in norm_files:
            size_idx = [x.isnumeric() for x in f.split("_")].index(True)
            size = int(f.split("_")[size_idx])
            normal_models[size].append(
                load_fun(f"{model_dir}/{f}", map_location=device)
            )
        for f in perm_files:
            size_idx = [x.isnumeric() for x in f.split("_")].index(True)
            size = int(f.split("_")[size_idx])
            perm_models[size].append(load_fun(f"{model_dir}/{f}", map_location=device))
        normal_models = {k: v for k, v in normal_models.items()}
        sample_sizes = sorted(normal_models.keys())
        perm_models = {k: v for k, v in perm_models.items()}
        cohorts = {"normal": normal_models, "permuted": perm_models}
    elif comparison_style == "versus":
        cohorts = defaultdict(list)
        for f in norm_files:
            size_idx = [x.isnumeric() for x in f.split("_")].index(True)
            size = int(f.split("_")[size_idx])
            cohorts[size].append(load_fun(f"{model_dir}/{f}", map_location=device))
        sample_sizes = sorted(cohorts.keys())
        cohorts = {k: v for k, v in cohorts.items()}
    elif comparison_style == "perm_versus":
        perm_models = defaultdict(list)
        normal_models = defaultdict(list)
        for f in norm_files:
            size_idx = [x.isnumeric() for x in f.split("_")].index(True)
            size = int(f.split("_")[size_idx])
            normal_models[size].append(load_fun(f"{model_dir}/{f}", map_location=device))
        for f in perm_files:
            size_idx = [x.isnumeric() for x in f.split("_")].index(True)
            size = int(f.split("_")[size_idx])
            perm_models[size].append(load_fun(f"{model_dir}/{f}", map_location=device))
        sample_sizes = sorted(normal_models.keys())
        normal_models = {k: v for k, v in normal_models.items()}
        perm_models = {k: v for k, v in perm_models.items()}
        cohorts = {"normal": normal_models, "permuted": perm_models}
    elif comparison_style == "half":
        normal_models = defaultdict(list)
        half_models = defaultdict(list)
        for f in norm_files:
            size_idx = [x.isnumeric() for x in f.split("_")].index(True)
            size = int(f.split("_")[size_idx])
            normal_models[size].append(
                load_fun(f"{model_dir}/{f}", map_location=device)
            )
        for f in half_files:
            size_idx = [x.isnumeric() for x in f.split("_")].index(True)
            size = int(f.split("_")[size_idx])
            half_models[size].append(load_fun(f"{model_dir}/{f}", map_location=device))
        normal_models = {k: v for k, v in normal_models.items()}
        half_models = {k: v for k, v in half_models.items()}
        sample_sizes = sorted(normal_models.keys())
        cohorts = {"normal": normal_models, "half": half_models}
    elif comparison_style == "half_perm":
        perm_models = defaultdict(list)
        half_models = defaultdict(list)
        for f in perm_files:
            size_idx = [x.isnumeric() for x in f.split("_")].index(True)
            size = int(f.split("_")[size_idx])
            perm_models[size].append(load_fun(f"{model_dir}/{f}", map_location=device))
        for f in half_files:
            size_idx = [x.isnumeric() for x in f.split("_")].index(True)
            size = int(f.split("_")[size_idx])
            half_models[size].append(load_fun(f"{model_dir}/{f}", map_location=device))
        perm_models = {k: v for k, v in perm_models.items()}
        half_models = {k: v for k, v in half_models.items()}
        sample_sizes = sorted(perm_models.keys())
        cohorts = {"permuted": perm_models, "half": half_models}

    return cohorts, sample_sizes


def run_test(test_function, comparison_style, cohorts, sample_size, **test_kwargs):
    """Run test provided by test_function. Test function must have take in two Slisemap
    models as the first two arguments."""
    if comparison_style == "versus":
        sms = cohorts[sample_size]
        res = torch.zeros((n_runs, n_runs))
        for i in range(n_runs):
            for j in range(i, n_runs):
                check = (
                    (n_runs * (n_runs - 1) / 2)
                    - (n_runs - i) * ((n_runs - i) - 1) / 2
                    + j
                    - i
                    - 1
                )
                if check % array_size != job_id:
                    continue
                res[i, j] = test_function(sms[i], sms[j], **test_kwargs)
    elif comparison_style == "perm_versus":
        sms_normal = cohorts["normal"][sample_size]
        sms_perm = cohorts["permuted"][sample_size]
        res = torch.zeros((n_runs, n_runs))
        for i in range(n_runs):
            for j in range(i, n_runs):
                check = (
                    (n_runs * (n_runs - 1) / 2)
                    - (n_runs - i) * ((n_runs - i) - 1) / 2
                    + j
                    - i
                    - 1
                )
                if check % array_size != job_id:
                    continue
                res[i, j] = test_function(sms_normal[i], sms_perm[j], **test_kwargs)
    elif comparison_style == "half":
        sms_normal = cohorts["normal"][sample_size]
        sms_half = cohorts["half"][sample_size]
        res = torch.zeros(n_runs)
        for i in range(n_runs):
            if i % array_size != job_id:
                continue
            res[i] = test_function(sms_normal[i], sms_half[i], **test_kwargs)
    elif comparison_style == "permutation":
        sms_normal = cohorts["normal"][sample_size]
        sms_perm = cohorts["permuted"][sample_size]
        res = torch.zeros(n_runs)
        for i in range(n_runs):
            if i % array_size != job_id:
                continue
            res[i] = test_function(sms_normal[i], sms_perm[i], **test_kwargs)
    elif comparison_style == "half_perm":
        sms_perm = cohorts["permuted"][sample_size]
        sms_half = cohorts["half"][sample_size]
        res = torch.zeros(n_runs)
        for i in range(n_runs):
            if i % array_size != job_id:
                continue
            res[i] = test_function(sms_perm[i], sms_half[i], **test_kwargs)
    return res


def set_double_distance_wrapper(sm1: Slisemap, sm2: Slisemap):
    """Wrapper for set_double_distance."""
    L1 = sm1.get_L(numpy=False)
    L2 = sm2.get_L(numpy=False)
    D1 = sm1.get_D(numpy=False)
    D2 = sm2.get_D(numpy=False)
    Dnew1 = set_double_distance(L1, L2, D1)
    Dnew2 = set_double_distance(L1, L2, D2)
    out = (Dnew1.mean() + Dnew2.mean()).item() / 2.0
    return out


def set_half_distance_wrapper(sm1: Slisemap, sm2: Slisemap):
    """Wrapper for set_half_distance."""
    L1 = sm1.get_L(numpy=False)
    L2 = sm2.get_L(numpy=False)
    D1 = sm1.get_D(numpy=False)
    D2 = sm2.get_D(numpy=False)
    Dnew1 = set_half_distance(L1, D2)
    Dnew2 = set_half_distance(L2, D1)
    out = (Dnew1.mean() + Dnew2.mean()).item() / 2.0
    return out


def permutation_loss(sm_normal: Slisemap, sm_perm: Slisemap):
    """Calculate permutation loss."""
    Y = sm_normal.get_Y(numpy=False)
    Yhat = sm_normal.predict(numpy=False)
    L = sm_normal.local_loss(Yhat, Y)
    L0 = sm_normal.local_loss(sm_perm.predict(numpy=False), Y)
    out = (L / L0).median().item()
    return out


print(f"Load models from {models}.", flush=True)
cohorts, sample_sizes = load_models(models, comparison_style=comparison_style)
for k in sample_sizes:
    start = time.time()
    if k > 17000:
        continue
    print(f"Calculate {test_name} for {k} samples.", flush=True)
    match test_name:
        case "local_model_distance":
            fname = (
                results / f"local_model_distance_{comparison_style}_{k}_{job_id}.pkl"
            )
            if fname.exists():
                print("\tFound existing result, not recomputing!", flush=True)
                continue
            B_d_m = run_test(
                B_distance,
                comparison_style=comparison_style,
                cohorts=cohorts,
                sample_size=k,
                include_y=True,
                match_by_model=True,
            )
            print(f"Mean: {B_d_m[B_d_m != 0.].mean().item()}")
            with open(fname, "wb") as f:
                torch.save(B_d_m, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\tSaved to {fname}")
        case "set_half_distance":
            fname = results / f"set_half_distance_{comparison_style}_{k}_{job_id}.pkl"
            if fname.exists():
                continue
            s_h_d = run_test(
                set_half_distance_wrapper,
                comparison_style=comparison_style,
                cohorts=cohorts,
                sample_size=k,
            )

            with open(fname, "wb") as f:
                torch.save(s_h_d, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\tSaved to {fname}")
        case "set_double_distance":
            fname = results / f"set_double_distance_{comparison_style}_{k}_{job_id}.pkl"
            if fname.exists():
                print("\tFound existing result, not recomputing!", flush=True)
                continue
            s_d_d = run_test(
                set_double_distance_wrapper,
                comparison_style=comparison_style,
                cohorts=cohorts,
                sample_size=k,
            )

            with open(fname, "wb") as f:
                torch.save(s_d_d, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\tSaved to {fname}")
        case "neighbourhood_distance":
            fname = (
                results / f"neighbourhood_distance_{comparison_style}_{k}_{job_id}.pkl"
            )
            if fname.exists():
                print("\tFound existing result, not recomputing!", flush=True)
                continue
            E_d = run_test(
                faster_epsilon_ball,
                comparison_style=comparison_style,
                cohorts=cohorts,
                sample_size=k,
                debug=True,
                include_y=(comparison_style != "half_perm"),
            )
            print(f"Mean: {E_d[E_d != 0.].mean().item()}")
            with open(fname, "wb") as f:
                torch.save(E_d, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\tSaved to {fname}")
        case "permutation_loss":
            fname = results / f"permutation_loss_{comparison_style}_{k}_{job_id}.pkl"
            if fname.exists():
                print("\tFound existing result, not recomputing!", flush=True)
                continue
            Q_t = run_test(
                permutation_loss,
                comparison_style=comparison_style,
                cohorts=cohorts,
                sample_size=k,
            )
            with open(fname, "wb") as f:
                torch.save(Q_t, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\tSaved to {fname}")
        case _:
            raise ValueError(f"{test_name} not implemented!")
    print(f"Calculating took {time.time() - start:.3f} s.")
