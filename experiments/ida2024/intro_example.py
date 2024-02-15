###############################################################################
#
# This experiment creates a Slipmap plot for the intro-
#
# Run this script to produce a Slipmap plot:
#   `python experiments/ida2024/intro_example.py`
#
###############################################################################

from hyperparameters import get_slipmap_params
from matplotlib import pyplot as plt
from project_paths import MANUSCRIPT_DIR
from sklearn.model_selection import train_test_split

from experiments.data import get_jets
from experiments.utils import paper_theme
from slisemap.local_models import LogisticRegression
from slisemap.slipmap import Slipmap

if __name__ == "__main__":
    X, Y, variables = get_jets("rf", True)
    X, _, Y, _ = train_test_split(X, Y, train_size=2_000, random_state=42)
    params = get_slipmap_params("Jets", "rf", weighted=True, squared=True, density=True)
    sm = Slipmap(X, Y, local_model=LogisticRegression, **params)
    sm.optimise()
    sm.metadata.set_variables(variables)
    sm.plot(clusters=5, figsize=paper_theme(0.8, cols=2, figsize=True), show=False)
    plt.savefig(MANUSCRIPT_DIR / "slipmap_jets.pdf")
    plt.close()
