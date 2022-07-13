"""
    This script loads datasets (including downloading them if necessary).
    Note that all datasets are also normalised in some way.
"""
from urllib.request import urlretrieve
from typing import Optional, Tuple, Union, List, Sequence
from pathlib import Path
from urllib.request import urlretrieve
from itertools import combinations

from scipy.io import arff
import numpy as np
import pandas as pd
import openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def find_path(dir_name: str = "data") -> Path:
    path = Path(dir_name)
    if not path.is_absolute():
        # Use the path of this file to find the root directory of the project
        path = Path(__file__).parent.parent / dir_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_dataset(id: int, cache_dir: Union[str, Path]) -> openml.OpenMLDataset:
    openml.config.apikey = "c1994bdb7ecb3c6f3c8f3b35f4b47f1f"
    # OpenML can handle the caching of datasets
    openml.config.cache_directory = cache_dir
    return openml.datasets.get_dataset(id, download_data=True)


def _get_predictions(id: int, columns: Sequence[str], cache_dir: Union[str, Path]):
    openml.config.apikey = "c1994bdb7ecb3c6f3c8f3b35f4b47f1f"
    openml.config.cache_directory = cache_dir
    run = openml.runs.get_run(id, False)
    dir = cache_dir / "org" / "openml" / "www" / "runs" / str(id)
    path = dir / "predictions.arff"
    if not path.exists():
        urlretrieve(run.predictions_url, path)
    data, meta = arff.loadarff(path)
    pred = np.stack(tuple(data[c] for c in columns), -1)
    return pred[np.argsort(data["row_id"])]


def get_boston(
    blackbox: Optional[str] = None,
    names: bool = False,
    normalise: bool = True,
    remove_B: bool = False,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    dir = find_path(data_dir)
    dataset = _get_dataset(531, dir)
    X, y, _, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="array"
    )
    if remove_B:
        X = X[:, [n != "B" for n in attribute_names]]
        attribute_names.remove("B")
    if blackbox is None:
        pass
    elif blackbox.lower() in ("svm", "svr"):
        y = _get_predictions(9918403, ("prediction",), dir)[:, 0]
    else:
        raise Exception(f"Unimplemented black box for boston: '{blackbox}'")
    if normalise:
        X = StandardScaler().fit_transform(X)
        y = StandardScaler().fit_transform(y[:, None])[:, 0]
    if names:
        return X, y, attribute_names
    else:
        return X, y


def get_fashon_mnist(
    blackbox: Optional[str] = None, names: bool = False, data_dir: str = "data"
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    dir = find_path(data_dir)
    dataset = _get_dataset(40996, dir)
    X, y, _, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="array"
    )
    X = X / 255.0
    if blackbox is None:
        Y = np.eye(10, dtype=X.dtype)[y]
    elif blackbox == "cnn":
        # This is a convolutional neural network with 94% accuracy
        # The predictions are from a 10-fold crossvalidation
        Y = _get_predictions(9204216, (f"confidence.{c:d}" for c in range(10)), dir)
    else:
        raise Exception(f"Unimplemented black box for fashion mnist: '{blackbox}'")
    if names:
        return X, Y, attribute_names
    else:
        return X, Y


def get_mnist(
    blackbox: Optional[str] = None, names: bool = False, data_dir: str = "data"
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    dir = find_path(data_dir)
    dataset = _get_dataset(554, dir)
    X, y, _, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="array"
    )
    X = X / 255.0
    if blackbox is None:
        Y = np.eye(10, dtype=X.dtype)[y]
    elif blackbox == "cnn":
        Y = _get_predictions(9204129, (f"confidence.{c:d}" for c in range(10)), dir)
    else:
        raise Exception(f"Unimplemented black box for mnist: '{blackbox}'")
    if names:
        return X, Y, attribute_names
    else:
        return X, Y


def get_emnist(
    blackbox: Optional[str] = None, names: bool = False, data_dir: str = "data"
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    dir = find_path(data_dir)
    dataset = _get_dataset(41039, dir)
    X, y, _, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="array"
    )
    mask = y < 10
    X = X[mask]
    y = y[mask]
    X = X / 255.0
    X = np.reshape(np.transpose(np.reshape(X, (-1, 28, 28)), (0, 2, 1)), (-1, 28 * 28))
    if blackbox is None:
        Y = np.eye(10, dtype=X.dtype)[y]
    elif blackbox == "cnn":
        Y = _get_predictions(9204295, (f"confidence.{c:d}" for c in range(10)), dir)
        Y = Y[mask]
    else:
        raise Exception(f"Unimplemented black box for mnist: '{blackbox}'")
    if names:
        return X, Y, attribute_names
    else:
        return X, Y


def get_spam(
    blackbox: Optional[str] = None, names: bool = False, data_dir: str = "data"
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    # https://archive.ics.uci.edu/ml/datasets/spambase
    dir = find_path(data_dir)
    dataset = _get_dataset(44, dir)
    X, y, _, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="array"
    )
    X = X / np.max(X, 0, keepdims=True)
    if blackbox is None:
        Y = np.eye(2, dtype=X.dtype)[y]
    elif blackbox.lower() in ("rf", "random forest", "randomforest"):
        Y = _get_predictions(9132654, (f"confidence.{c:d}" for c in range(2)), dir)
    else:
        raise Exception(f"Unimplemented black box for spam: '{blackbox}'")
    if names:
        return X, Y, attribute_names
    else:
        return X, Y


def get_higgs(
    blackbox: Optional[str] = None, names: bool = False, data_dir: str = "data"
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    # https://archive.ics.uci.edu/ml/datasets/HIGGS
    dir = find_path(data_dir)
    dataset = _get_dataset(23512, dir)
    X, y, _, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="array"
    )
    X = X[:-1]
    y = y[:-1]
    if blackbox is None:
        Y = np.eye(2, dtype=X.dtype)[y]
    elif blackbox.lower() in ("gb", "gradient boosting", "gradientboosting"):
        Y = _get_predictions(9907793, (f"confidence.{c:d}" for c in range(2)), dir)
        Y = Y[:-1]
    else:
        raise Exception(f"Unimplemented black box for higgs: '{blackbox}'")
    X = StandardScaler().fit_transform(X)
    if names:
        return X, Y, attribute_names
    else:
        return X, Y


def get_covertype(
    blackbox: Optional[str] = None, names: bool = False, data_dir: str = "data"
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    # https://archive.ics.uci.edu/ml/datasets/Covertype
    dir = find_path(data_dir)
    dataset = _get_dataset(150, dir)  # This is the already normalised version
    X, y, _, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="array"
    )
    X = X[:-1]
    y = y[:-1]
    if blackbox is None:
        Y = np.eye(7, dtype=X.dtype)[y]
    elif blackbox.lower() in ("lb", "logit boost", "logitboost"):
        Y = _get_predictions(157511, (f"confidence.{c:d}" for c in range(1, 8)), dir)
        Y = Y[:-1]
    else:
        raise Exception(f"Unimplemented black box for covertype: '{blackbox}'")
    if names:
        return X, Y, attribute_names
    else:
        return X, Y


def get_autompg(
    blackbox: Optional[str] = None,
    names: bool = False,
    normalise: bool = True,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    # https://archive-beta.ics.uci.edu/ml/datasets/auto+mpg
    dir = find_path(data_dir)
    dataset = _get_dataset(196, dir)
    X, y, _, anames = dataset.get_data(target=dataset.default_target_attribute)
    X = np.concatenate(
        (X.values[:, :-1].astype(float), np.eye(3)[X["origin"].values.astype(int) - 1]),
        1,
    )
    mask = ~np.isnan(X[:, 2])
    X = X[mask]
    y = y[mask]
    anames = anames[:-2] + ["year", "origin USA", "origin Europe", "origin Japan"]
    if blackbox is None:
        Y = y.values
    elif blackbox.lower() in ("svm", "svr"):
        Y = _get_predictions(9918402, ("prediction",), dir)[mask, 0]
    elif blackbox.lower() in ("rf", "randomforest", "random forest"):
        random_forest = RandomForestRegressor(random_state=42).fit(X, y.ravel())
        Y = random_forest.predict(X)
    else:
        raise Exception(f"Unimplemented black box for Auto MPG: '{blackbox}'")
    if normalise:
        X[:, :-3] = StandardScaler().fit_transform(X[:, :-3])
        Y = StandardScaler().fit_transform(Y[:, None])[:, 0]
    if names:
        return X, Y, anames
    else:
        return X, Y


def get_iris(
    blackbox: Optional[str] = None, names: bool = False, data_dir: str = "data"
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    dataset = _get_dataset(61, find_path(data_dir))
    X, y, _, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="array"
    )
    Y = np.eye(3, dtype=X.dtype)[y]
    # TODO: get blackbox predictions from openml
    X = StandardScaler().fit_transform(X)
    if names:
        return X, Y, attribute_names
    else:
        return X, Y


def get_airquality(
    blackbox: Optional[str] = None,
    names: bool = False,
    normalise: bool = True,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    """Get the Air Quality dataset.

    Cleaned and preprocessed as in:

        Oikarinen E, Tiittanen H, Henelius A, Puolamäki K (2021)
        Detecting virtual concept drift of regressors without ground truth values.
        Data Mining and Knowledge Discovery 35(3):726-747, DOI 10.1007/s10618-021-00739-7

    Args:
        blackbox (Optional[str]): Return predictions from a black box instead of y (currently not implemented). Defaults to None.
        names (bool, optional): Return the names of the columns. Defaults to False.
        data_dir (str, optional): Directory where the data is saved. Defaults to "data".

    Returns:
        Tuple[np.ndarray, np.ndarray, Optional[List[str]]]: X and y (and column names).
    """
    path = find_path(data_dir) / "AQ_cleaned_version.csv"
    if not path.exists():
        url = "https://raw.githubusercontent.com/edahelsinki/drifter/master/TiittanenOHP2019-code/data/AQ_cleaned_version.csv"
        urlretrieve(url, path)
    AQ = pd.read_csv(path)
    columns = [
        "PT08.S1(CO)",
        "C6H6(GT)",
        "PT08.S2(NMHC)",
        "NOx(GT)",
        "PT08.S3(NOx)",
        "NO2(GT)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
        "T",
        "RH",
        "AH",
    ]
    nnames = [
        "CO(sensor)",
        "C6H6(GT)",
        "NMHC(sensor)",
        "NOx(GT)",
        "NOx(sensor)",
        "NO2(GT)",
        "NO2(sensor)",
        "O3(sensor)",
        "Temperature",
        "Relative hum.",
        "Absolute hum.",
    ]

    X = AQ[columns].to_numpy()
    y = AQ["CO(GT)"].to_numpy()
    if normalise:
        X = StandardScaler().fit_transform(X)
        y = StandardScaler().fit_transform(y[:, None])[:, 0]
    if blackbox is None:
        pass
    elif blackbox.lower() in ("rf", "random forest", "randomforest"):
        y = RandomForestRegressor(n_jobs=-1).fit(X, y).predict(X)
    else:
        raise Exception(f"Unimplemented black box for airquality: '{blackbox}'")
    if names:
        return X, y, nnames
    else:
        return X, y


def get_rsynth(
    N: int = 100,
    M: int = 11,
    k: int = 3,
    s: float = 0.25,
    se: float = 0.1,
    seed: Union[None, int, np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic regression data

    Args:
        N (int, optional): Number of rows in X. Defaults to 100.
        M (int, optional): Number of columns in X. Defaults to 11.
        k (int, optional): Number of clusters (with their own true model). Defaults to 3.
        s (float, optional): Scale for the randomisation of the cluster centers. Defaults to 0.25.
        se (float, optional): Scale for the noise of y. Defaults to 0.1.
        seed (Union[None, int, np.random.RandomState], optional): Local random seed. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: cluster_ids[N], X[N,M+1], y[N], B[k,M+1].
    """
    if seed is None:
        npr = np.random
    elif isinstance(seed, np.random.RandomState):
        npr = seed
    else:
        npr = np.random.RandomState(seed)

    B = npr.normal(size=[k, M + 1])  # k x (M+1)
    while not _are_models_different(B):
        B = npr.normal(size=[k, M + 1])
    c = npr.normal(scale=s, size=[k, M])  # k X M
    while not _are_centroids_different(c, s * 0.5):
        c = npr.normal(scale=s, size=[k, M])
    j = npr.randint(k, size=N)  # N
    e = npr.normal(scale=se, size=N)  # N
    X = npr.normal(loc=c[j, :])  # N x M
    X = StandardScaler().fit_transform(X)
    y = (B[j, :-1] * X).sum(axis=1) + e + B[j, -1]
    return j, X, y, B


def get_rsynth2(
    N: int = 100,
    M: int = 11,
    k1: int = 3,
    k2: int = 3,
    s1: float = 2.0,
    s2: float = 5.0,
    se: float = 0.1,
    seed: Union[None, int, np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic regression data 2 (where half of the variables are adversarial)

    Args:
        N (int, optional): Number of rows in X. Defaults to 100.
        M (int, optional): Number of columns in X. Defaults to 11.
        k1 (int, optional): Number of true clusters (with their own true model). Defaults to 3.
        k2 (int, optional): Number of false clusters (not affecting y). Defaults to 3.
        s1 (float, optional): Scale for the randomisation of the true cluster centers. Defaults to 2.0.
        s2 (float, optional): Scale for the randomisation of the false cluster centers. Defaults to 5.0.
        se (float, optional): Scale for the noise of y. Defaults to 0.1.
        seed (Union[None, int, np.random.RandomState], optional): Local random seed. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: cluster_ids[N], X[N,M], y[N], B[k,M+1].
    """
    if seed is None:
        npr = np.random
    elif isinstance(seed, np.random.RandomState):
        npr = seed
    else:
        npr = np.random.RandomState(seed)

    B = npr.normal(size=[k1, M // 2 + 1])  # k1 x (M/2+1)
    while not _are_models_different(B):
        B = npr.normal(size=[k1, M + 1])
    c1 = npr.normal(scale=s1, size=[k1, M // 2])  # k1 X M/2
    while not _are_centroids_different(c1, s1 * 0.5):
        c1 = npr.normal(scale=s1, size=[k1, M // 2])
    c2 = npr.normal(scale=s2, size=[k2, M - M // 2])  # k2 X M/2
    while not _are_centroids_different(c2, s2 * 0.5):
        c1 = npr.normal(scale=s2, size=[k2, M - M // 2])
    j1 = npr.randint(k1, size=N)  # N
    j2 = npr.randint(k2, size=N)  # N
    e = npr.normal(scale=se, size=N)  # N
    X1 = npr.normal(loc=c1[j1])  # N x M/2
    X2 = npr.normal(loc=c2[j2])  # N x M/2
    X = np.concatenate((X1, X2), 1)  # N x M
    y = (B[j1, :-1] * X1).sum(axis=1) + e + B[j1, -1]
    B = np.concatenate((B[:, :-1], np.zeros((k1, M - M // 2)), B[:, -1:]), 1)
    return j1, X, y, B


def _are_models_different(B: np.ndarray, treshold: float = 0.5) -> bool:
    """Check if a set of linear models are different enough (using cosine similarity).

    Args:
        B (np.ndarray): Matrix where the rows are linear models.
        treshold (float, optional): Upper treshold for cosine similarity. Defaults to 0.5.

    Returns:
        bool: True if no pair of models are more similar than the treshold.
    """
    for i, j in combinations(range(B.shape[0]), 2):
        cosine_similarity = B[i] @ B[j] / (B[i] @ B[i] * B[j] @ B[j])
        if cosine_similarity > treshold:
            return False
    return True


def _are_centroids_different(c: np.ndarray, treshold: float = 0.5) -> bool:
    """Check if a set of linear models are different enough (using euclidean distance).

    Args:
        c (np.ndarray): Matrix where the rows are centroids.
        treshold (float, optional): Lower treshold for euclidean distance. Defaults to 0.5.

    Returns:
        bool: True if no pair of models are more similar than the treshold.
    """
    for i, j in combinations(range(c.shape[0]), 2):
        if np.sum((c[i] - c[j]) ** 2) < treshold**2:
            return False
    return True


if __name__ == "__main__":
    print("Downloading all datasets...")
    get_boston("svm")
    get_fashon_mnist("cnn")
    get_iris()
    get_airquality()
    get_mnist("cnn")
    get_emnist("cnn")
    get_spam("rf")
    get_higgs("gb")
    get_covertype("lb")
    get_autompg("svr")
