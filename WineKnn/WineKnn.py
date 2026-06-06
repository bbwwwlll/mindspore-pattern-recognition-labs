import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import mindspore as ms
from mindspore import nn, ops

# 在本地 Windows 环境上使用 CPU
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_PATH = ROOT_DIR / "data" / "wine.data"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ===================== 读取 Wine 数据集 =====================
with open(DATA_PATH) as csv_file:
    data = list(csv.reader(csv_file, delimiter=','))

# 查看部分样本，确认读取成功
print(data[56:62] + data[130:133])

# 取三类样本（共 178 条），13 个属性作为自变量 X，类别作为因变量 Y
X = np.array([[float(x) for x in s[1:]] for s in data[:178]], np.float32)
Y = np.array([int(s[0]) for s in data[:178]], np.int32)

# ===================== 可视化部分属性分布 =====================
attrs = [
    'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
    'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
    'Proanthocyanins', 'Color intensity', 'Hue',
    'OD280/OD315 of diluted wines', 'Proline'
]

plt.figure(figsize=(10, 8))
for i in range(0, 4):
    plt.subplot(2, 2, i + 1)
    a1, a2 = 2 * i, 2 * i + 1
    # 类别 1
    plt.scatter(X[:59, a1], X[:59, a2], label='1')
    # 类别 2
    plt.scatter(X[59:130, a1], X[59:130, a2], label='2')
    # 类别 3
    plt.scatter(X[130:, a1], X[130:, a2], label='3')
    plt.xlabel(attrs[a1])
    plt.ylabel(attrs[a2])
    plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "wine_feature_scatter.png", dpi=150, bbox_inches="tight")
if os.environ.get("SHOW_PLOTS") == "1":
    plt.show()
plt.close()

# ===================== 划分训练集和验证集 =====================
np.random.seed(42)  # 方便复现实验结果
train_idx = np.random.choice(178, 128, replace=False)
test_idx = np.array(list(set(range(178)) - set(train_idx)))

X_train, Y_train = X[train_idx], Y[train_idx]
X_test, Y_test = X[test_idx], Y[test_idx]

# ===================== 模型构建：KNN 距离计算 =====================
class KnnNet(nn.Cell):
    def __init__(self, k):
        super(KnnNet, self).__init__()
        self.k = k

    def construct(self, x, X_train):
        # 将输入 x 平铺为与 X_train 样本数相同的形状 (128, 13)
        x_tile = ops.tile(x, (X_train.shape[0], 1))
        # 逐元素平方差
        square_diff = ops.square(x_tile - X_train)
        # 按特征维度求和，得到平方距离
        square_dist = ops.sum(square_diff, 1)
        # 开根号得到欧氏距离
        dist = ops.sqrt(square_dist)
        # -dist：距离越小，值越大，便于使用 topk 选择最近邻
        values, indices = ops.topk(-dist, self.k)
        return indices


def knn(knn_net, x, X_train, Y_train):
    # 转为 Tensor 以便送入 MindSpore 网络
    x_tensor = ms.Tensor(x)
    X_train_tensor = ms.Tensor(X_train)

    indices = knn_net(x_tensor, X_train_tensor)
    idx_np = indices.asnumpy()

    # 统计 top-k 邻居中各类别出现次数
    topk_cls = [0] * (int(np.max(Y_train)) + 1)
    for idx in idx_np:
        topk_cls[Y_train[idx]] += 1

    cls = np.argmax(topk_cls)
    return cls

# ===================== 在验证集上测试 KNN 分类性能 =====================
acc = 0
k = 5
knn_net = KnnNet(k)

for x, y in zip(X_test, Y_test):
    pred = knn(knn_net, x, X_train, Y_train)
    acc += (pred == y)
    print('label: %d, prediction: %s' % (y, pred))

accuracy = acc / len(Y_test)
print('Validation accuracy is %f' % accuracy)
