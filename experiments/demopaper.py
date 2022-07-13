###############################################################################
#
# This experiment creates visualisations for the demo paper.
# Run this script to produce plots: `python experiments/demopaper.py`.
#
###############################################################################

import sys

import numpy as np
import pandas as pd

from pathlib import Path
from urllib.request import urlretrieve

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))

from slisemap import Slisemap
from experiments.data import get_autompg
from experiments.utils import paper_theme

RESULTS_DIR = Path(__file__).parent / "results"


if __name__ == "__main__":
    # Use the same Slisemap object as in the example notebook
    SM_CACHE_PATH = RESULTS_DIR / "01_regression_example_autompg.sm"
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    if not SM_CACHE_PATH.exists():
        urlretrieve(
            f"https://raw.githubusercontent.com/edahelsinki/slisemap/data/examples/cache/{SM_CACHE_PATH.name}",
            SM_CACHE_PATH,
        )

    X, y, names = get_autompg(names=True, blackbox="rf")
    if not SM_CACHE_PATH.exists():
        sm = Slisemap(X, y, lasso=0.01, random_state=42)
        sm.optimise()
        sm.save(SM_CACHE_PATH)
    else:
        sm = Slisemap.load(SM_CACHE_PATH, device="cpu")

    sm.plot(
        clusters=5,
        bars=5,
        jitter=0.1,
        variables=names,
        figsize=paper_theme(0.9, 1, 2, figsize=True),
        show=False,
    )
    plt.savefig(RESULTS_DIR / f"autompg_clusters.pdf")
    plt.close()
