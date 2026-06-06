import os
import csv
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

import mindspore as ms
from mindspore import nn, ops, ms_function, dataset, Tensor
from mindspore.train import LossMonitor
from mindspore.ops import SigmoidCrossEntropyWithLogits
import mindspore.ops as P  # 文档里用到 P.ReduceMean，这里补上

# 在普通 Windows 电脑上使用 CPU
ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_PATH = ROOT_DIR / "data" / "iris.data"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ===================== 读取 Iris 数据 =====================
with open(DATA_PATH) as csv_file:
    data = list(csv.reader(csv_file, delimiter=','))

print("Sample data (40~60):")
print(data[40:60])  # 打印部分数据

# ===================== 抽取前两类样本，构造 X, Y =====================
label_map = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
}

# 前 100 条：前两类，每条 4 个特征 + 1 个类别
X = np.array([[float(x) for x in s[:-1]] for s in data[:100]], np.float32)
Y = np.array([[label_map[s[-1]]] for s in data[:100]], np.float32)

# ===================== 样本可视化（前两维特征） =====================
plt.figure()
plt.scatter(X[:50, 0], X[:50, 1], label='Iris-setosa')
plt.scatter(X[50:, 0], X[50:, 1], label='Iris-versicolor')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.title('Iris binary dataset (first two features)')
plt.savefig(OUTPUT_DIR / "iris_binary_scatter.png", dpi=150, bbox_inches="tight")
if os.environ.get("SHOW_PLOTS") == "1":
    plt.show()
plt.close()

# ===================== 划分训练集 / 测试集 =====================
np.random.seed(42)
train_idx = np.random.choice(100, 80, replace=False)
test_idx = np.array(list(set(range(100)) - set(train_idx)))
X_train, Y_train = X[train_idx], Y[train_idx]
X_test, Y_test = X[test_idx], Y[test_idx]

# ===================== 转换为 MindSpore Dataset =====================
XY_train = list(zip(X_train, Y_train))
ds_train = dataset.GeneratorDataset(XY_train, ['x', 'y'], shuffle=True)
ds_train = ds_train.shuffle(buffer_size=80).batch(32, drop_remainder=True)

# ===================== 可视化 Sigmoid 函数 =====================
coor_x = np.arange(-10, 11, dtype=np.float32)
coor_y = nn.Sigmoid()(ms.Tensor(coor_x)).asnumpy()
plt.figure()
plt.plot(coor_x, coor_y)
plt.xlabel('x')
plt.ylabel('p')
plt.title('Sigmoid function')
plt.grid(True)
plt.savefig(OUTPUT_DIR / "sigmoid_function.png", dpi=150, bbox_inches="tight")
if os.environ.get("SHOW_PLOTS") == "1":
    plt.show()
plt.close()

# ===================== 定义模型与损失函数 =====================
class Loss(nn.Cell):
    def __init__(self):
        super(Loss, self).__init__()
        self.sigmoid_cross_entropy_with_logits = SigmoidCrossEntropyWithLogits()
        self.reduce_mean = P.ReduceMean(keep_dims=False)

    def construct(self, x, y):
        loss = self.sigmoid_cross_entropy_with_logits(x, y)
        return self.reduce_mean(loss, -1)


# 线性部分：4 维输入，1 维输出
net = nn.Dense(4, 1)
loss = Loss()
opt = nn.optim.SGD(net.trainable_params(), learning_rate=0.003)

# ===================== 模型训练 =====================
model = ms.train.Model(net, loss, opt)
print("Start training...")
model.train(10,
            ds_train,
            callbacks=[LossMonitor(per_print_times=ds_train.get_dataset_size())],
            dataset_sink_mode=False)

# ===================== 在测试集上评估 =====================
x_logits = model.predict(ms.Tensor(X_test)).asnumpy()
# Sigmoid 将 logit 转为概率，再四舍五入到 {0,1}
pred = np.round(1 / (1 + np.exp(-x_logits)))
correct = np.equal(pred, Y_test)
acc = np.mean(correct)
print('Test accuracy is', acc)
