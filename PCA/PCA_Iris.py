import os
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import mindspore as ms
from mindspore import nn, context
from mindspore.ops import operations as ops


context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_PATH = ROOT_DIR / "data" / "iris.data"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


with open(DATA_PATH) as csv_file:
    data = list(csv.reader(csv_file, delimiter=","))

print(data[0:5], data[50:55], data[100:105])

label_map = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
}

X = np.array([[float(x) for x in s[:-1]] for s in data[:150]], np.float32)
Y = np.array([label_map[s[-1]] for s in data[:150]], np.int32)


class PcaNet(nn.Cell):
    def __init__(self):
        super(PcaNet, self).__init__()
        self.reduce_mean = ops.ReduceMean(keep_dims=True)
        self.matmul_a = ops.MatMul(transpose_a=True)

    def construct(self, x, dim=2):
        """Project samples to the first dim principal components."""
        m = x.shape[0]
        mean = self.reduce_mean(x, axis=0)
        x_new = x - mean
        cov = self.matmul_a(x_new, x_new) / (m - 1)

        e, v = np.linalg.eigh(cov.asnumpy())
        e_index_sort = np.argsort(e)[::-1][:dim]
        v_new = ms.Tensor(v[:, e_index_sort], ms.float32)
        pca = np.matmul(x_new.asnumpy(), v_new.asnumpy())
        return ms.Tensor(pca, ms.float32)


if __name__ == "__main__":
    net = PcaNet()

    pca_input = ms.Tensor(np.reshape(X, (X.shape[0], -1)), ms.float32)
    pca_data = net(pca_input, dim=2)

    color_mapping = {0: "tab:purple", 1: "tab:red", 2: "tab:green"}
    colors = list(map(lambda x: color_mapping[x], Y))

    plt.figure()
    plt.scatter(pca_data[:, 0].asnumpy(), pca_data[:, 1].asnumpy(), c=colors)
    plt.title("PCA of Iris dataset (4D -> 2D)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "pca_iris.png", dpi=150, bbox_inches="tight")
    if os.environ.get("SHOW_PLOTS") == "1":
        plt.show()
    plt.close()
