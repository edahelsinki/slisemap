###############################################################################
#
# This experiment evaluates the escape heuristic.
# This experiment uses data from `large_evaluation.py` (run that first!).
# Run this script again to produce a latex from the results:
#   `python experiments/escape.py`
#
###############################################################################

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from experiments.large_evaluation import get_results
from experiments.dr_comparison import print_table


def print_escape(df: pd.DataFrame, slim=False):
    metrics = [
        ("loss", "Loss", False),
        # ("fidelity", "Fidelity", False),
        ("fidelity_nn20", "Fidelity NN", False),
        ("coverage_gl03_nn20", "Coverage NN", True),
        ("cluster_purity", "Cluster Purity", True),
        ("time", "Time (s)", False),
    ]
    mask = (df["data"] != "RSynth") + (df["n"].isin((400,)))
    mask *= df["radius"].round(3) == 3.5
    mask *= df["method"].isin(("Slisemap", "Slisemap (no escape)"))
    if slim:
        slim_data = ["RSynth", "Boston", "Air Quality (XAI)", "Spam", "Higgs (XAI)"]
        mask *= df["data"].isin(slim_data)
    df = df[mask].copy()
    df["method"] = (
        df["method"]
        .cat.remove_unused_categories()
        .cat.rename_categories({"Slisemap (no escape)": "No escape"})
    )
    if slim:
        df["data"] = (
            df["data"].cat.set_categories(slim_data).cat.remove_unused_categories()
        )
    print_table(df, metrics, "tab:escape" if slim else "tab:escape_full", False)


if __name__ == "__main__":
    df = get_results(False)
    print_escape(df, True)
    print_escape(df, False)
