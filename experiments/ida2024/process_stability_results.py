import matplotlib.pyplot as plt
from pathlib import Path
from os import listdir
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
import io
import pickle
import matplotlib.ticker as mticker
from project_paths import MANUSCRIPT_DIR
from collections.abc import Iterable
import seaborn as sns

curr_path = Path(__file__)
# input directories
res_dir = curr_path.parent / "results/"
air_quality_model_d = res_dir / "air_quality/231113_2/slipmap"
air_quality_model_d_E = res_dir / "air_quality/231113/slipmap"
air_quality_model_d_sm = res_dir / "air_quality/231113/slisemap"
jets_model_d = res_dir / "jets/231113/slipmap"
jets_model_d_sm = res_dir / "jets/231113/slisemap"
plot_dir = curr_path.parent / "figures/"
plot_dir.mkdir(parents=True, exist_ok=True)


def load_results(result_dir, comparison_style, test_name):
    """Helper function to load datasets."""
    res_files = [
        x
        for x in listdir(result_dir)
        if (test_name in x) and (f"_{comparison_style}_" in x)
    ]
    if comparison_style in ["half", "versus"]:
        res_files = [x for x in res_files if "_perm_" not in x]
    results = defaultdict(list)
    for file in res_files:
        sample_size = int(file.split("_")[-2])
        with open(result_dir / file, "rb") as f:
            results[sample_size].append(torch.load(f, map_location="cpu"))

    res = {}
    for k, v in results.items():
        res[k] = torch.sum(torch.stack(v), axis=-1)
    return res


def plot(dataset, ax, comparison_style, **plot_kwargs):
    """Helper function for plots."""
    sample_sizes = [x for x in sorted(dataset.keys()) if x < 17000]
    to_plot = []
    for s in sample_sizes:
        p = dataset[s]
        if comparison_style == "versus":
            p = p[p != 0].flatten()
        p = p[p != np.inf]
        to_plot.append(p.mean())
    ax.plot(sample_sizes, to_plot, **plot_kwargs)


def nanstd(tensor):
    return torch.std(tensor[~torch.isnan(tensor)])


def result_dataframe(
    dataset_name, result_dir, comparison_style, test_name, baseline_dir=None
):
    result_dir_sm = Path(result_dir) / "slisemap"
    result_dir_sp = Path(result_dir) / "slipmap"
    if baseline_dir is None:
        baseline_dir_sm = result_dir_sm
        baseline_dir_sp = result_dir_sp
    res_dict_sm = load_results(result_dir_sm, comparison_style, test_name)
    res_dict_sp = load_results(result_dir_sp, comparison_style, test_name)
    baseline_comparison_style = (
        "perm_versus" if test_name == "local_model_distance" else "half_perm"
    )
    bl_dict_sm = load_results(baseline_dir_sm, baseline_comparison_style, test_name)
    bl_dict_sp = load_results(baseline_dir_sp, baseline_comparison_style, test_name)
    means_sm = []
    means_sp = []
    baselines_sm = []
    baselines_sp = []
    normalised_sm = []
    normalised_sp = []
    stds_sm = []
    stds_sp = []
    sample_sizes = [x for x in sorted(res_dict_sm.keys())]
    for s in sample_sizes:
        raw_sm = res_dict_sm[s]
        raw_sp = res_dict_sp[s]
        bl_sm = bl_dict_sm[s]
        bl_sp = bl_dict_sp[s]
        norm_sm = raw_sm / bl_sm
        norm_sp = raw_sp / bl_sp
        if comparison_style == "versus":
            raw_sm = raw_sm[raw_sm != 0].flatten()
            raw_sp = raw_sp[raw_sp != 0].flatten()
        raw_sm = raw_sm[raw_sm != np.inf]
        raw_sp = raw_sp[raw_sp != np.inf]
        norm_sm = norm_sm[norm_sm != np.inf]
        norm_sp = norm_sp[norm_sp != np.inf]
        means_sm.append(raw_sm.nanmean().item())
        means_sp.append(raw_sp.nanmean().item())
        normalised_sm.append(norm_sm.nanmean().item())
        normalised_sp.append(norm_sp.nanmean().item())
        baselines_sm.append(bl_sm.nanmean().item())
        baselines_sp.append(bl_sp.nanmean().item())
        stds_sm.append(nanstd(raw_sm).item())
        stds_sp.append(nanstd(raw_sp).item())
    test_string = comparison_style if "perm" not in comparison_style else "baseline"
    res_sm = pd.DataFrame(
        {
            "sample_size": sample_sizes,
            test_name: means_sm,
            test_name + "_norm": normalised_sm,
            test_name + "_std": stds_sm,
            test_name + "_baseline": baselines_sm,
            "method": "SLISEMAP",
            "kind": test_string,
            "dataset": dataset_name,
        }
    )
    res_sp = pd.DataFrame(
        {
            "sample_size": sample_sizes,
            test_name: means_sp,
            test_name + "_norm": normalised_sp,
            test_name + "_std": stds_sp,
            test_name + "_baseline": baselines_sp,
            "method": "SLIPMAP",
            "kind": test_string,
            "dataset": dataset_name,
        }
    )
    return pd.concat([res_sm, res_sp], ignore_index=True)


def process_pivot_for_latex(pt):
    # prepare number columns for final table
    round_col = lambda f: f"{f:.3f}"
    pt.loc[:, "Neighbourhood stability"] = pt["neighbourhood_distance"].apply(round_col)
    pt.loc[:, "Local model consistency"] = pt["local_model_distance"].apply(round_col)
    pt.loc[:, "Neighbourhood stability"] = (
        pt["Neighbourhood stability"]
        + " \pm "
        + pt["neighbourhood_distance_std"].apply(round_col)
    )
    pt.loc[:, "Local model consistency"] = (
        pt["Local model consistency"]
        + " \pm "
        + pt["local_model_distance_std"].apply(round_col)
    )
    # bold best value
    for ds_name in pt.index.levels[0]:
        for col in ["local_model_distance", "neighbourhood_distance"]:
            idx = pt.loc[ds_name][col].argmax()
            str_col = (
                "Local model consistency"
                if "local" in col
                else "Neighbourhood stability"
            )
            pt.loc[ds_name][str_col].iloc[idx] = (
                "\mathbf{" + pt.loc[ds_name][str_col].iloc[idx] + "}"
            )
    # into latex numbers
    pt.loc[:, "Local model consistency"] = "$" + pt["Local model consistency"] + "$"
    pt.loc[:, "Neighbourhood stability"] = "$" + pt["Neighbourhood stability"] + "$"
    # pick columns
    pt = pt[["Local model consistency", "Neighbourhood stability"]]
    pt.columns = [x + " $\\uparrow$" for x in pt.columns]
    pt.index = pt.index.set_names(["Data", "Method"])
    # mangle table to save vertical space
    pt = pt.stack(level=0).unstack(level=0).T.swaplevel(0, 1, axis=1)
    # reoreder columns such that measures are paired
    multi_tuples = [
        ("Local model consistency $\\uparrow$", "{\sc slipmap}"),
        ("Local model consistency $\\uparrow$", "{\sc slisemap}"),
        ("Neighbourhood stability $\\uparrow$", "{\sc slipmap}"),
        ("Neighbourhood stability $\\uparrow$", "{\sc slisemap}"),
    ]
    multi_cols = pd.MultiIndex.from_tuples(multi_tuples, names=[None, None])
    pt = pd.DataFrame(pt, columns=multi_cols).reset_index()
    return pt


# load results
res_dfs = []
results_to_read = [
    # ("Jets", res_dir / "jets/231113", "local_model_distance", "versus"),
    ("Jets", res_dir / "jets/final_d", "local_model_distance", "versus"),
    # ("Air Quality", res_dir / "air_quality/231113_2", "local_model_distance", "versus"),
    ("Air Quality", res_dir / "air_quality/final_d", "local_model_distance", "versus"),
    # ("QM9", res_dir / "qm9/231113", "local_model_distance", "versus"),
    ("QM9", res_dir / "qm9/final_d", "local_model_distance", "versus"),
    # ("Higgs", res_dir / "higgs/231120", "local_model_distance", "versus"),
    ("Higgs", res_dir / "higgs/final_d", "local_model_distance", "versus"),
    # ("Gas Turbine", res_dir / "gas_turbine/231120", "local_model_distance", "versus"),
    ("Gas Turbine", res_dir / "gas_turbine/final_d", "local_model_distance", "versus"),
    # ("Covertype", res_dir / "covertype/231120", "local_model_distance", "versus"),
    ("Covertype", res_dir / "covertype/240209", "local_model_distance", "versus"),
    # ("Jets", res_dir / "jets/231120", "neighbourhood_distance", "half"),
    ("Jets", res_dir / "jets/final_d", "neighbourhood_distance", "half"),
    # ("Air Quality", res_dir / "air_quality/231121", "neighbourhood_distance", "half"),
    ("Air Quality", res_dir / "air_quality/final_d", "neighbourhood_distance", "half"),
    # ("QM9", res_dir / "qm9/231121", "neighbourhood_distance", "half"),
    ("QM9", res_dir / "qm9/final_d", "neighbourhood_distance", "half"),
    # ("Higgs", res_dir / "higgs/231121", "neighbourhood_distance", "half"),
    ("Higgs", res_dir / "higgs/final_d", "neighbourhood_distance", "half"),
    # ("Gas Turbine", res_dir / "gas_turbine/231121", "neighbourhood_distance", "half"),
    ("Gas Turbine", res_dir / "gas_turbine/final_d", "neighbourhood_distance", "half"),
    # ("Covertype", res_dir / "covertype/231120", "neighbourhood_distance", "half"),
    ("Covertype", res_dir / "covertype/240209", "neighbourhood_distance", "half"),
]
for ds, rdir, test, comp in results_to_read:
    rdf = result_dataframe(ds, rdir, comp, test)
    res_dfs.append(rdf)
res = pd.concat(res_dfs)
res_for_pivot = res.copy()
res_for_pivot.loc[:, "method"] = "{\sc " + res_for_pivot["method"].str.lower() + "}"
pt = pd.pivot_table(
    res_for_pivot.loc[(res_for_pivot["sample_size"] > 9000)],
    values=[
        "local_model_distance",
        "neighbourhood_distance",
        "local_model_distance_std",
        "neighbourhood_distance_std",
    ],
    index=["dataset", "method"],
)
pt.loc[:, "local_model_distance"] = 1 - pt["local_model_distance"]
pt.loc[:, "neighbourhood_distance"] = 1 - pt["neighbourhood_distance"]
pt = process_pivot_for_latex(pt)
(
    pt.style.hide(axis=0)
    .applymap_index(lambda v: "font-weight: bold;", axis="columns")
    .to_latex(
        MANUSCRIPT_DIR / "local_exp_stability_table.tex",
        column_format="l@{\\hspace{3mm}} " + "r@{\\hspace{3mm}}" * 3 + "r",
        hrules=True,
        convert_css=True,
    )
)
# targets = ["Local model stability", "Neighbourhood stability"]
# res = pd.melt(
#     res[
#         [
#             "sample_size",
#             "method",
#             "dataset",
#             "Local model stability",
#             "Neighbourhood stability",
#         ]
#     ],
#     id_vars=["sample_size", "method", "dataset"],
# )
# g = sns.relplot(
#     res,
#     x="sample_size",
#     col="dataset",
#     row="variable",
#     y="value",
#     kind="line",
#     hue="method",
#     style="method",
# )
# g.tight_layout()
# plt.show()
