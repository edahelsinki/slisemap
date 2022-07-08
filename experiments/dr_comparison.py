###############################################################################
#
# This experiment compares different dimensionality reduction methods.
# This experiment uses data from `large_evaluation.py` (run that first!).
# Run this script to produce latex tables from the results:
#   `python experiments/dr_comparison.py`
#
###############################################################################

import sys
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent))
from experiments.large_evaluation import get_results


def summarise_step1(x):
    x = x.values[np.isfinite(x.values)]
    if len(x) == 0:
        return np.nan, np.nan
    return x.mean(), x.std()


def summarise_step2(x, maximise=True):
    if maximise:
        return max(m - s for m, s in x.values) - 1e-8
    else:
        return min(m + s for m, s in x.values) + 1e-8


def summarise_step3(row, minmax, maximise=True):
    d = row.index[0][0]
    m, s = row.values[0]
    if np.isnan(m):
        return ""
    if maximise:
        if m >= minmax[d]:
            return f"${{\\bf{m:.2f} \\pm {s:.2f}}}$"
        else:
            return f"${m:.2f} \\pm {s:.2f}$"
    else:
        if m <= minmax[d]:
            return f"${{\\bf{m:.2f} \\pm {s:.2f}}}$"
        else:
            return f"${m:.2f} \\pm {s:.2f}$"


def print_dr_comparison(df: pd.DataFrame, variant=0):
    metrics = [
        ("loss", "Loss", False),
        ("fidelity", "Fidelity", False),
        ("fidelity_nn20", "Fidelity NN", False),
        ("coverage_gl03_nn20", "Coverage NN", True),
        ("cluster_purity", "Cluster Purity", True),
        ("time", "Time (s)", False),
    ]
    if variant == 0:
        mask = df["data"].isin(("Air Quality", "Spam (XAI)"))
        mask = mask + ((df["data"] == "RSynth") * (df["n"] == 400))
        metrics.remove(("cluster_purity", "Cluster Purity", True))
    elif variant == 1:
        mask = df["data"] == "RSynth"
    elif variant == 2:
        mask = ("Air Quality", "Air Quality (XAI)", "Boston", "Boston (XAI)", "Spam")
        mask = df["data"].isin(mask)
        metrics.remove(("cluster_purity", "Cluster Purity", True))
    elif variant == 3:
        mask = ("Spam (XAI)", "Higgs", "Higgs (XAI)", "Covertype", "Covertype (XAI)")
        mask = df["data"].isin(mask)
        metrics.remove(("cluster_purity", "Cluster Purity", True))
    else:
        mask = slice(df.shape[0])
    print_table(df[mask], metrics, f"tab:cmp_dr{variant if variant else ''}", variant)


def print_table(
    df: pd.DataFrame,
    metrics: List[Tuple[str, str, bool]],
    label: str,
    ht: bool = True,
    dense_data: Union[bool, int] = 2,
):
    inv_met = list(zip(*metrics))
    df2 = df[["data", "n", "m", "method"] + list(inv_met[0])].copy()
    df2.sort_values(["data", "n", "method"], inplace=True)
    if dense_data == 2:
        rename = lambda x: (
            f"\\multicolumn{{3}}{{l}}{{{{\\sc {x[0].lower()}}}: ${x[1]:4d}\\times {x[2]-1:2d}$}}\\\\"
        )
    elif dense_data:
        rename = lambda x: f"{{\\sc {x[0].lower()}}}\\\\${x[1]:4d}\\times {x[2]-1:2d}$"
    else:
        rename = lambda x: f"\\\\{{\\sc {x[0].lower()}}} ${x[1]:4d}\\times {x[2]-1:2d}$"
    df2["dataset"] = df2.apply(rename, 1).astype("category")
    df2.rename(columns=dict(dataset="Dataset", method="Method"), inplace=True)
    agg = {m: summarise_step1 for m, _, _ in metrics}
    df2 = df2.groupby(["Dataset", "Method"], sort=False, observed=True).agg(agg)
    agg = {m: lambda x, s=s: summarise_step2(x, s) for m, _, s in metrics}
    minmax = df2.groupby("Dataset", sort=False).agg(agg).to_dict()
    agg = {
        m: lambda x, m=m, s=s: summarise_step3(x, minmax[m], s) for m, _, s in metrics
    }
    df2 = df2.groupby(["Dataset", "Method"], sort=False, observed=True).agg(agg)
    print(
        df2.to_latex(
            header=inv_met[1],
            float_format="%.3f",
            sparsify=True,
            label=label,
            caption="TODO",
            escape=False,
            na_rep="",
            column_format="ll" + "r" * len(metrics),
            position="ht" if ht else None,
        )
        .replace("\\\\$", "\\\\\n$")
        .replace("\\\\ &", "\\\\\n &")
        .replace("\\\\{", "\\\\\n{")
        .replace("\\midrule\n\\\\\n", "\\midrule\n")
        .replace("Dataset", "")
        .replace("\\toprule\n", "\\toprule\n\multicolumn{2}{l}{Dataset}")
        .replace("&", "", 1)
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Print one table at a time
        index = (int(sys.argv[1]),)
    else:
        index = range(4)
    df = get_results(True)
    for i in index:
        print_dr_comparison(df, i)
