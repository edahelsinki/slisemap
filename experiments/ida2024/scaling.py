import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import time

root = str(Path(__file__).parent.parent.parent.absolute())
if root not in sys.path:
    sys.path.insert(0, root)
from slisemap import Slisemap
from slisemap.slipmap import Slipmap
from slisemap.local_models import LogisticRegression
from hyperparameters import get_slipmap_params, get_slisemap_params
from experiments.data import get_rsynth
from experiments.data import (
    get_airquality,
    get_higgs,
    get_qm9,
    get_jets,
    get_gas_turbine,
    get_covertype3,
)
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


def sample_data(dataset_name, N):
    _, data_fun, bb_model = get_data()[dataset_name]
    if dataset_name == "synthetic":
        _, X, Y, _ = data_fun(N=N, seed=random_seed)
    else:
        X, Y = data_fun(blackbox=bb_model)
    random_indices = rng.permutation(len(X))
    X = X[random_indices, :]
    X = X[:N, :]
    Y = Y[random_indices]
    Y = Y[:N]
    if type(X) == np.ndarray:
        X = torch.tensor(X, device=device)
    if type(Y) == np.ndarray:
        Y = torch.tensor(Y, device=device)
    return X, Y


def profile(function):
    gc.collect()
    t1 = time.perf_counter(), time.process_time()
    function()
    t2 = time.perf_counter(), time.process_time()
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated(device=device)
        torch.cuda.reset_peak_memory_stats()
        return t2[0] - t1[0], t2[1] - t1[1], mem
    return t2[0] - t1[0], t2[1] - t1[1]


def nonzero_mean(arr):
    return np.true_divide(arr.sum(1), (arr != 0.0).sum(1))


if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    print("GPU not available, reverting to CPU!", flush=True)
    device = torch.device("cpu")

job_id = int(sys.argv[3]) - 1 if len(sys.argv) > 3 else None
dataset_name = sys.argv[1]
result_dir = root / Path(sys.argv[2])
result_dir.mkdir(parents=True, exist_ok=True)
sample_sizes = np.logspace(np.log10(500), np.log10(5000), 10, dtype=int)
random_seed = 1618033 + job_id
rng = np.random.default_rng(random_seed)
n_runs = 10


# set hyperparameters
def slipmap_hyperparams(dname, bb):
    out = {}
    out["device"] = device
    out = out | get_slipmap_params(dname, bb, squared=True, density=True)
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


wall_times_sm = np.zeros((len(sample_sizes), n_runs))
wall_times_sp = np.zeros((len(sample_sizes), n_runs))
cpu_times_sm = np.zeros((len(sample_sizes), n_runs))
cpu_times_sp = np.zeros((len(sample_sizes), n_runs))
cuda_max_mem_sm = np.zeros((len(sample_sizes), n_runs))
cuda_max_mem_sp = np.zeros((len(sample_sizes), n_runs))
if not torch.cuda.is_available():
    cuda_max_mem_sm -= 1
    cuda_max_mem_sp -= 1
n_loops = [job_id] if job_id is not None else range(n_runs)
for si, s in enumerate(sample_sizes):
    print(f"Begin runs with sample size {s} (dataset {dataset_name}).", flush=True)
    for ni in n_loops:
        X, y = sample_data(dataset_name, N=s)
        if torch.cuda.is_available():
            X = torch.tensor(X, device=device)
            y = torch.tensor(y, device=device)
        full_dname, _, bb = get_data()[dataset_name]
        hyperparams_sp = slipmap_hyperparams(full_dname, bb)
        hyperparams_sm = slisemap_hyperparams(full_dname, bb)
        sm = Slisemap(X, y, **hyperparams_sm)
        sp = Slipmap(X, y, **hyperparams_sp)
        print(sp.value(), flush=True)
        if torch.cuda.is_available():
            sm_wall, sm_cpu, sm_mem = profile(sm.optimise)
            sp_wall, sp_cpu, sp_mem = profile(sp.optimise)
            wall_times_sm[si, ni] = sm_wall
            cpu_times_sm[si, ni] = sm_cpu
            cuda_max_mem_sm[si, ni] = sm_mem
            wall_times_sp[si, ni] = sp_wall
            cpu_times_sp[si, ni] = sp_cpu
            cuda_max_mem_sp[si, ni] = sp_mem
        else:
            sm_wall, sm_cpu = profile(sm.optimise)
            # sm_wall = sm_cpu = -1
            sp_wall, sp_cpu = profile(lambda: sp.optimise(verbose=0))
            wall_times_sm[si, ni] = sm_wall
            cpu_times_sm[si, ni] = sm_cpu
            wall_times_sp[si, ni] = sp_wall
            cpu_times_sp[si, ni] = sp_cpu
        print(f"Slisemap training took {sm_wall:.1f}s.")
        print(f"Slipmap training took {sp_wall:.1f}s.")


results = pd.DataFrame(
    {
        "sample sizes": sample_sizes,
        "SLISEMAP wall time": nonzero_mean(wall_times_sm),
        "SLISEMAP cpu time": nonzero_mean(cpu_times_sm),
        "SLIPMAP wall time": nonzero_mean(wall_times_sp),
        "SLIPMAP cpu time": nonzero_mean(cpu_times_sp),
        "SLISEMAP CUDA memory": nonzero_mean(cuda_max_mem_sm),
        "SLIPMAP CUDA memory": nonzero_mean(cuda_max_mem_sp),
    }
)
print(results)
if job_id is not None:
    fname = f"slipmap_scaling_{job_id}.pkl.gz"
else:
    fname = "slipmap_scaling.pkl.gz"
results.to_pickle(result_dir / fname)
