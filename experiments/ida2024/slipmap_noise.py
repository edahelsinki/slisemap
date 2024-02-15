import sys
import torch

from glob import glob
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.special import logit
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

root = str(Path(__file__).parent.parent.parent.absolute())
if root not in sys.path:
    sys.path.insert(0, root)

from experiments.data import get_rsynth
from slisemap import Slisemap
from slisemap.slipmap import Slipmap
from slisemap.utils import to_tensor, tonp
import time
from experiments.data import (
    get_airquality,
    get_covertype3,
    get_gas_turbine,
    get_higgs,
    get_jets,
    get_qm9,
)
from slisemap.local_models import LogisticRegression

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    print("GPU not available, reverting to CPU!", flush=True)
    device = torch.device("cpu")


def pred(sm, X):
    X = to_tensor(X, **sm.tensorargs)[0]
    D = torch.cdist(X, sm.get_X(False, False))
    B = sm.get_B(False)[D.argmin(1), :]
    return sm.predict(X, B)[:, 0]


def get_data():
    # dict of data acq function and bb model
    return {
        "synthetic": (get_rsynth, None),
        "gas_turbine": (get_gas_turbine, "ada"),
        "qm9": (get_qm9, "nn"),
        "air_quality": (get_airquality, "rf"),
        "jets": (get_jets, "rf"),
        "higgs": (get_higgs, "gb"),
        "covertype": (get_covertype3, "lb"),
    }


def sample_data(dataset_name, N):
    data_fun, bb_model = get_data()[dataset_name]
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


# set hyperparameters
def get_slipmap_params(dname):
    out = {}
    out["device"] = device
    if dname == "air_quality":
        out["lasso"] = 0.52
        out["ridge"] = 0.18
    elif dname == "covertype":
        out["lasso"] = 0.0080
        out["ridge"] = 0.0003
        out["local_model"] = LogisticRegression
    elif dname == "gas_turbine":
        out["lasso"] = 0.033
        out["ridge"] = 0.306
    elif dname == "higgs":
        out["lasso"] = 0.0016
        out["ridge"] = 0.0006
        out["local_model"] = LogisticRegression
    elif dname == "jets":
        out["lasso"] = 0.0019
        out["ridge"] = 0.0002
        out["local_model"] = LogisticRegression
    elif dname == "qm9":
        out["lasso"] = 0.0017
        out["ridge"] = 0.0001
    elif dname == "synthetic":
        out["lasso"] = 0
    else:
        raise ValueError(f"No hyperparams for dataset {dname}.")
    return out


def get_slisemap_params(dname):
    out = {}
    out["device"] = device
    out["random_state"] = random_seed
    if dname == "air_quality":
        out["lasso"] = 0.0064
        out["ridge"] = 0.0034
    elif dname == "covertype":
        out["lasso"] = 0.1
        out["ridge"] = 0.01
        out["local_model"] = LogisticRegression
    elif dname == "gas_turbine":
        out["lasso"] = 0.0028
        out["ridge"] = 0.0028
    elif dname == "higgs":
        out["lasso"] = 0.1
        out["ridge"] = 0.01
        out["local_model"] = LogisticRegression
    elif dname == "jets":
        out["lasso"] = 0.001
        out["ridge"] = 0.0001
        out["local_model"] = LogisticRegression
    elif dname == "qm9":
        out["lasso"] = 0.0013
        out["ridge"] = 0.0008
    elif dname == "synthetic":
        out["lasso"] = 0
    else:
        raise ValueError(f"No hyperparams for dataset {dname}.")
    return out


hyperparam_dict = {
    "air_quality": ("Air Quality", "Random Forest"),
    "gas_turbine": ("Gas Turbine", "AdaBoost"),
    "qm9": ("QM9", "Neural Network"),
    "covertype": ("Covertype", "Neural Network"),
    "higgs": ("Higgs", "Gradient Boosting"),
    "jets": ("Jets", "Random Forest"),
}

dataset_name = sys.argv[1]
sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 200
result_dir = root / Path("experiments/ida2024/results/231115/")
result_dir.mkdir(parents=True, exist_ok=True)
scales = np.logspace(-2, np.log10(2), 10)
n_samples = 10
random_seed = 168
rng = np.random.default_rng(random_seed)
slipmap_loss = np.zeros((len(scales), n_samples))
slipmap_pred_loss = np.zeros_like(slipmap_loss)
slisemap_loss = np.zeros((len(scales), n_samples))
slisemap_pred_loss = np.zeros_like(slisemap_loss)

for si, scale in enumerate(scales):
    print(f"Start with scale {scale}.")
    for ni in range(n_samples):
        X, y = sample_data(dataset_name, sample_size)
        # what is noise for logistic regression?
        # think about this, you fool
        hyperparams_sm = get_slisemap_params(dataset_name)
        hyperparams_sp = get_slipmap_params(dataset_name)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )
        y_train += np.random.normal(scale=scale, size=y_train.shape)
        sm = Slisemap(X_train, y_train, **hyperparams_sm)
        sm.optimise()
        Ls = np.median(np.diag(sm.get_L()))
        slisemap_loss[si, ni] = Ls
        yhat = pred(sm, X_test)
        slisemap_pred_loss[si, ni] = np.median(
            np.sqrt(np.mean(np.square(y_test.numpy() - yhat)))
        )
        sp = Slipmap(X_train, y_train, **hyperparams_sp)
        sp.optimise()
        Lp = np.median(np.diag(sp.get_L()))
        slipmap_loss[si, ni] = Lp
        yhatp = pred(sp.into(), X_test)
        slipmap_pred_loss[si, ni] = np.median(
            np.sqrt(np.mean(np.square(y_test.numpy() - yhatp)))
        )

results = pd.DataFrame(
    {
        "scales": scales,
        "SLISEMAP loss": np.mean(slisemap_loss, axis=1),
        "SLISEMAP pred loss": np.mean(slisemap_pred_loss, axis=1),
        "SLIPMAP loss": np.mean(slipmap_loss, axis=1),
        "SLIPMAP pred loss": np.mean(slipmap_pred_loss, axis=1),
    }
)
print(results)
results.to_pickle(result_dir / f"slipmap_noise_{dataset_name}_{sample_size}.pkl.gz")

fig, ax = plt.subplots(ncols=2)
ax[0].plot(scales, np.mean(slisemap_loss, axis=1), label="Slisemap")
ax[0].plot(scales, np.mean(slipmap_loss, axis=1), label="Slipmap")
ax[0].legend()
ax[0].set_title("Training loss")
ax[0].set_ylabel("Loss")
ax[0].set_xlabel("Added noise variance")
ax[1].plot(scales, np.mean(slisemap_pred_loss, axis=1), label="Slisemap")
ax[1].plot(scales, np.mean(slipmap_pred_loss, axis=1), label="Slipmap")
ax[1].legend()
ax[1].set_title("Test loss")
ax[1].set_xlabel("Added noise variance")
plt.tight_layout()
plt.savefig(result_dir / f"{dataset_name}_{sample_size}_noise.pdf", dpi=150)
plt.show()
