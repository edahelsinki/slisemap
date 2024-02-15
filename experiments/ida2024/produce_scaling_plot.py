from pathlib import Path
import sys
from collections import defaultdict

root = str(Path().resolve().absolute())
print(root)
curr_path = Path(__file__)
if root not in sys.path:
    sys.path.insert(0, root)
from slisemap.slipmap import Slipmap
from slisemap import Slisemap
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from experiments.utils import paper_theme
from project_paths import MANUSCRIPT_DIR


def fetch_metadata(fname):
    if "slisemap" in str(fname):
        metadata = Slisemap.load(fname, device="cpu").metadata
    else:
        metadata = Slipmap.load(fname, device="cpu").metadata
    size_idx = [x.isnumeric() for x in fname.name.split("_")].index(True)
    size = int(fname.name.split("_")[size_idx])
    return size, metadata


def result_df(res_dict):
    results_sm = []
    for key, val in res_dict.items():
        size_df = pd.DataFrame()
        size_df["training_wall_time"] = [x[0] for x in val]
        size_df["training_cpu_time"] = [x[1] for x in val]
        size_df["peak_cuda_memory"] = [x[2] for x in val]
        size_df["sample_size"] = key
        results_sm.append(size_df)
    results_sm = pd.concat(results_sm, ignore_index=True)
    results_sm = results_sm.sort_values("sample_size")
    return results_sm


# get results from SLISEMAP models
def results_from_models(model_dir):
    sm_res = defaultdict(list)
    sm_dir = Path(model_dir) / "slisemap/"
    for f in os.listdir(sm_dir):
        if "normal" not in f:
            continue
        size, metadata = fetch_metadata(sm_dir / f)
        sm_res[size].append(
            (
                metadata["training_wall_time"],
                metadata["training_cpu_time"],
                metadata["peak_cuda_memory"],
            )
        )
    sp_res = defaultdict(list)
    sp_dir = Path(model_dir) / "slipmap/"
    for f in os.listdir(sp_dir):
        if "normal" not in f:
            continue
        size, metadata = fetch_metadata(sp_dir / f)
        sp_res[size].append(
            (
                metadata["training_wall_time"],
                metadata["training_cpu_time"],
                metadata["peak_cuda_memory"],
            )
        )
    sm_df = result_df(sm_res)
    sm_df.loc[:, "method"] = "SLISEMAP"
    sp_df = result_df(sp_res)
    sp_df.loc[:, "method"] = "SLIPMAP"
    return pd.concat([sm_df, sp_df], ignore_index=True)


def results_from_chunks(res_dir):
    res_files = [x for x in os.listdir(res_dir) if "scaling" in x]
    res_dfs = []
    for f in res_files:
        res_dfs.append(pd.read_pickle(res_dir / f))
    results = pd.concat(res_dfs)
    results_sm = results[
        [
            "sample sizes",
            "SLISEMAP wall time",
            "SLISEMAP cpu time",
            "SLISEMAP CUDA memory",
        ]
    ]
    results_sm = results_sm.rename(
        columns={
            "sample sizes": "sample_size",
            "SLISEMAP wall time": "training_wall_time",
            "SLISEMAP cpu time": "training_cpu_time",
            "SLISEMAP CUDA memory": "peak_cuda_memory",
        }
    )
    results_sm["method"] = "SLISEMAP"
    results_sp = results[
        [
            "sample sizes",
            "SLIPMAP wall time",
            "SLIPMAP cpu time",
            "SLIPMAP CUDA memory",
        ]
    ]
    results_sp = results_sp.rename(
        columns={
            "sample sizes": "sample_size",
            "SLIPMAP wall time": "training_wall_time",
            "SLIPMAP cpu time": "training_cpu_time",
            "SLIPMAP CUDA memory": "peak_cuda_memory",
        }
    )
    results_sp["method"] = "SLIPMAP"
    return results_sm, results_sp


def collect_cpu_stats(dataset_name, result_dir):
    results_sm, results_sp = results_from_chunks(result_dir)
    results_sm["dataset"] = dataset_name
    results_sp["dataset"] = dataset_name
    res = pd.concat([results_sm, results_sp], ignore_index=True)
    targets = ["training_wall_time", "training_cpu_time", "peak_cuda_memory"]
    res = pd.melt(res, id_vars=[c for c in res.columns if c not in targets])
    # use only wall time
    res = res.loc[~(res["variable"].isin(["peak_cuda_memory", "training_cpu_time"]))]
    res = res.replace("training_wall_time", "Wall time")
    res = res.replace("training_cpu_time", "CPU time")
    return res


def collect_gpu_stats(dataset_name, model_dir):
    print(f"GPU stats for {dataset_name}: {model_dir}")
    res_cuda = results_from_models(model_dir)
    res_cuda["dataset"] = dataset_name
    targets = ["training_wall_time", "training_cpu_time", "peak_cuda_memory"]
    res_cuda = pd.melt(
        res_cuda, id_vars=[c for c in res_cuda.columns if c not in targets]
    )
    res_cuda = res_cuda.loc[
        ~(res_cuda["variable"].isin(["training_wall_time", "training_cpu_time"]))
    ]
    res_cuda = res_cuda.replace("peak_cuda_memory", "Peak CUDA memory")
    return res_cuda


# model_folder = curr_path.parent / f"models/{dataset_name}/{date}/"
# sm_dir = model_folder / "slisemap/"
# sp_dir = model_folder / "slipmap/"
# cpu stats
pdf = True
print("Collecting CPU stats...")
res_dir_aq = curr_path.parent / f"results/air_quality/240207/"
res_aq = collect_cpu_stats("Air Quality", res_dir_aq)
res_dir_jets = curr_path.parent / f"results/jets/240207/"
res_jets = collect_cpu_stats("Jets", res_dir_jets)
res_dir_higgs = curr_path.parent / f"results/higgs/240207/"
res_higgs = collect_cpu_stats("Higgs", res_dir_higgs)
res_dir_qm9 = curr_path.parent / f"results/qm9/240207/"
res_qm9 = collect_cpu_stats("QM9", res_dir_qm9)
res_dir_gt = curr_path.parent / f"results/gas_turbine/240207/"
res_gt = collect_cpu_stats("Gas Turbine", res_dir_gt)
res_dir_ct = curr_path.parent / f"results/covertype/240207/"
res_ct = collect_cpu_stats("Covertype", res_dir_ct)
res = pd.concat(
    # [res_aq, res_jets, res_higgs, res_qm9, res_gt, res_ct], ignore_index=True
    [res_aq, res_jets, res_qm9, res_gt, res_ct],
    ignore_index=True,
)
print("Done.")

# gpu stats
cuda_cache = curr_path.parent / "results/cuda_cache.pkl.gz"
if cuda_cache.exists():
    print("Reading GPU stats...")
    res_cuda = pd.read_pickle(cuda_cache)
else:
    print("Collecting GPU stats...")
    model_dir_aq = curr_path.parent / "models/air_quality/final/"
    cuda_aq = collect_gpu_stats("Air Quality", model_dir_aq)
    model_dir_jets = curr_path.parent / "models/jets/final/"
    cuda_jets = collect_gpu_stats("Jets", model_dir_jets)
    model_dir_higgs = curr_path.parent / "models/higgs/final/"
    cuda_higgs = collect_gpu_stats("Higgs", model_dir_higgs)
    model_dir_qm9 = curr_path.parent / "models/qm9/final/"
    cuda_qm9 = collect_gpu_stats("QM9", model_dir_qm9)
    model_dir_gt = curr_path.parent / "models/gas_turbine/final/"
    cuda_gt = collect_gpu_stats("Gas Turbine", model_dir_gt)
    model_dir_ct = curr_path.parent / "models/covertype/final/"
    cuda_ct = collect_gpu_stats("Covertype", model_dir_ct)
    res_cuda = pd.concat(
        [cuda_aq, cuda_jets, cuda_higgs, cuda_qm9, cuda_gt, cuda_ct], ignore_index=True
    )
    res_cuda.to_pickle(cuda_cache)
    print(f"Done. Cached results to {cuda_cache}.")

res_cuda = res_cuda.loc[res_cuda["dataset"] != "Higgs"]
res = pd.concat([res, res_cuda], ignore_index=True)
res = res.rename(columns={"dataset": "Dataset", "method": "\nMethod"})

g = sns.relplot(
    res,
    x="sample_size",
    col="variable",
    y="value",
    kind="line",
    hue="Dataset",
    style="\nMethod",
    style_order=["SLIPMAP", "SLISEMAP"],
    errorbar=None,
    facet_kws={"sharex": False, "sharey": False},
    **paper_theme(0.8, cols=2),
)
g.axes.flatten()[0].set(yscale="log")
g.axes.flatten()[1].set(yscale="log")
g.axes.flatten()[0].set(xscale="log")
g.axes.flatten()[1].set(xscale="log")
g.axes.flatten()[0].set(xticks=[500, 1000, 2500, 5000])
g.axes.flatten()[0].set(xticklabels=[500, 1000, 2500, 5000])
g.axes.flatten()[1].set(xticks=[100, 500, 1000, 2500, 5000, 10000])
g.axes.flatten()[1].set(xticklabels=[100, 500, 1000, 2500, 5000, 10000])
g.set(xlabel="Sample size")
g.axes.flatten()[0].set_ylabel("Time (s)")
g.axes.flatten()[1].set_ylabel("Bytes")
g.set_titles("{col_name}")
if pdf:
    plt.savefig(MANUSCRIPT_DIR / "scaling_time.pdf")
    plt.close()
else:
    plt.show()
