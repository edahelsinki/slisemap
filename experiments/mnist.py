###############################################################################
#
# This experiment attempts slisemap on EMNIST.
# Run this script (preferably with GPU acceleration) to produce a plot.
#
###############################################################################

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.model_selection import train_test_split
from scipy.special import logit
import torch
import torch.nn.functional
import torch.utils.data


sys.path.append(str(Path(__file__).parent.parent))  # Add the project root to the path
from slisemap import Slisemap
from slisemap.local_models import logistic_regression, logistic_regression_loss
from experiments.data import get_mnist, get_emnist

RESULTS_DIR = Path(__file__).parent / "results" / "emnist"


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc1_drop = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc1_drop(x)
        x = self.fc2(x)
        return x

    def optimise(self, X, Y, epochs=20, smoothing=0.1):
        optimiser = torch.optim.Adam(self.parameters())
        data = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                X, Y * (1 - smoothing) + smoothing / Y.shape[1]
            ),
            batch_size=64,
            shuffle=True,
        )
        print("Epoch: Accuracy  Loss")
        for e in range(epochs):
            self.train(True)
            for x, y in data:
                optimiser.zero_grad()
                output = self(x)
                loss = torch.nn.functional.cross_entropy(output, y)
                loss.backward()
                optimiser.step()
            self.train(False)
            with torch.no_grad():
                Yhat = self(X)
                loss = torch.nn.functional.cross_entropy(Yhat, Y)
                acc = torch.mean((torch.argmax(Yhat, 1) == torch.argmax(Y, 1)).float())
                print(f"{e:5d}:   {acc.cpu().item():.3f}   {loss.cpu().item():.2f}")

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


def plot_scatter_img(
    pos,
    imgs,
    cls,
    num=50,
    zoom=1.0,
    radius=0.05,
    pad=0.1,
    cmap="Greys",
    norm=None,
    title="",
    show=True,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    sns.scatterplot(
        x=pos[:, 0],
        y=pos[:, 1],
        hue=cls,
        style=cls,
        ax=ax,
        palette=["#5e4cb2", "#e87b11"],  # "PuOr"
    )
    counter = 0
    radius = (
        (ax.get_xlim()[0] - ax.get_xlim()[1]) ** 2
        + (ax.get_ylim()[0] + ax.get_ylim()[1]) ** 2
    ) ** 0.5 * radius
    mask = np.ones(pos.shape[0], dtype=bool)
    for i in range(pos.shape[0]):
        if not mask[i]:
            continue
        image = np.reshape(imgs[i], (28, 28))
        im = OffsetImage(image, zoom=zoom, cmap=cmap, norm=norm)
        ab = AnnotationBbox(
            im, (pos[i, 0], pos[i, 1]), xycoords="data", frameon=True, pad=pad
        )
        ax.add_artist(ab)
        counter += 1
        if counter >= num:
            break
        mask[i:] *= (
            np.sum((pos[i:] - pos[None, i]) ** 2, 1) / (1 + (cls[i] == cls[i:]))
            > radius**2
        )
    ax.autoscale()
    ax.set_title(title)
    if show:
        plt.show()
    return ax


def plot_scatter_img2(Z, X, B, classes, show=True, **kwargs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    plot_scatter_img(Z, X, classes, title="Embedding", show=False, ax=ax1, **kwargs)
    plot_scatter_img(
        Z,
        B,
        classes,
        cmap="PuOr",
        norm=TwoSlopeNorm(0),
        title="Local Models",
        show=False,
        ax=ax2,
        **kwargs,
    )
    sns.despine(fig)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    X, Y = get_mnist()
    y = np.argmax(Y, 1)
    yc = pd.Categorical(y)
    # plot_scatter_img(PCA(2).fit_transform(X)[::10], X[::10], yc[::10], 100, title="PCA on MNIST")

    map_path = RESULTS_DIR / "slisemap.pt"
    model_path = RESULTS_DIR / "model.pt"

    if not map_path.exists():
        network = Net()
        Xt = torch.as_tensor(X, dtype=torch.float32)
        Yt = torch.as_tensor(Y, dtype=torch.float32)
        if torch.cuda.is_available():
            network = network.cuda()
            Xt = Xt.cuda()
            Yt = Yt.cuda()

        if model_path.exists():
            print("Loading network")
            network.load(model_path)
        else:
            print("Training network")
            network.optimise(Xt, Yt)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            network.save(model_path)

        Ycnn = torch.softmax(network(Xt), 1).detach().cpu().numpy()
        del network

        class1 = 2
        class2 = 3
        mask = yc.isin([class1, class2])
        Y2 = Ycnn[mask, class1] / (Ycnn[mask, class1] + Ycnn[mask, class2])
        Y2 = logit(Y2 * 0.98 + 0.01)
        X2, _, Y2, _, yc2, _ = train_test_split(
            X[mask],
            Y2,
            yc[mask],
            train_size=5000,
            random_state=42,
            stratify=Y[mask, class1],
        )
        yc2 = yc2.remove_unused_categories()
        # sns.histplot(Y2)
        # plt.show()

    if map_path.exists():
        print("Loading Slisemap")
        sm = Slisemap.load(map_path)
    else:
        print("Optimising Slisemap")
        sm = Slisemap(
            X2,
            Y2,
            lasso=0.01,
            ridge=0.01,
            radius=3,
            # local_model=logistic_regression,
            # local_loss=logistic_regression_loss,
        )
        sm.optimise(verbose=True)
        sm.save(map_path)

    # sm.plot(jitter=0.1, clusters=yc2, show=False)
    # plot_scatter_img2(sm.get_Z(), X2, sm.get_B()[:, :-1], yc2)

    plot_scatter_img2(
        sm.get_Z(), X2, sm.get_B()[:, :-1], yc2, False, num=40, radius=0.115, zoom=2.1
    )
    plt.savefig(RESULTS_DIR / ".." / f"mnist_{class1}vs{class2}.pdf")
    plt.close()
