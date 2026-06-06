import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ===================== 1) 生成二维样本（带重叠，非严格线性可分更常见） =====================
rng = np.random.default_rng(2025)
n = 60

# 你可以微调均值/方差来控制重叠程度（越重叠越“软间隔”特征明显）
X1 = np.column_stack([rng.normal(loc=2.0, scale=1.0, size=n),
                      rng.normal(loc=2.0, scale=1.0, size=n)])
X2 = np.column_stack([rng.normal(loc=-1.0, scale=1.0, size=n),
                      rng.normal(loc=-1.0, scale=1.0, size=n)])

X = np.vstack([X1, X2])
y = np.hstack([np.ones(n), -np.ones(n)])

# ===================== 2) 选择更细的 C 梯度（跨度不要太夸张） =====================
# 这组通常能看到“逐渐变硬”的过程，又不至于 0.1->1000 跳太大
C_list = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]

# 子图布局：自动按数量排
cols = 4
rows = int(np.ceil(len(C_list) / cols))

fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows))
axes = np.array(axes).reshape(-1)

# 画图范围
x1_min, x1_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
x2_min, x2_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
xx = np.linspace(x1_min, x1_max, 300)

# ===================== 3) 循环训练 + 可视化 =====================
for idx, C in enumerate(C_list):
    ax = axes[idx]

    clf = SVC(kernel="linear", C=C)  # 线性软间隔 SVM
    clf.fit(X, y)

    # 取出 w, b
    w = clf.coef_.ravel()
    b = clf.intercept_[0]

    # 支持向量
    sv = clf.support_vectors_
    sv_count = sv.shape[0]

    # 训练精度
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)

    # 几何间隔宽度：2/||w||
    w_norm = np.linalg.norm(w)
    margin_width = 2.0 / w_norm if w_norm > 1e-12 else np.inf

    # 画散点
    ax.scatter(X1[:, 0], X1[:, 1], s=25, marker="o", label="+1")
    ax.scatter(X2[:, 0], X2[:, 1], s=25, marker="o", label="-1")

    # 决策边界与间隔线：w1*x + w2*y + b = 0 / ±1
    # 如果 w2 过小，说明边界接近竖直线，改用 x = 常数 的方式画，避免数值爆炸
    if abs(w[1]) > 1e-8:
        yy0 = (-w[0] * xx - b) / w[1]
        yy1 = (-w[0] * xx - b + 1.0) / w[1]
        yy2 = (-w[0] * xx - b - 1.0) / w[1]
        ax.plot(xx, yy0, linewidth=2, label="w^T x + b = 0")
        ax.plot(xx, yy1, "--", linewidth=1, label="margin ±1")
        ax.plot(xx, yy2, "--", linewidth=1)
    else:
        # 竖直线情况：w1*x + b = 0 => x = -b/w1
        if abs(w[0]) > 1e-12:
            x0 = -b / w[0]
            x1_line = -(b - 1.0) / w[0]
            x2_line = -(b + 1.0) / w[0]
            ax.axvline(x0, linewidth=2, label="w^T x + b = 0")
            ax.axvline(x1_line, linestyle="--", linewidth=1, label="margin ±1")
            ax.axvline(x2_line, linestyle="--", linewidth=1)

    # 标出支持向量
    ax.scatter(sv[:, 0], sv[:, 1], s=90, facecolors="none",
               edgecolors="k", linewidths=1.5, label="SV")

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"C={C:g} | SV={sv_count} | 2/||w||={margin_width:.3f} | acc={acc:.3f}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    # 图例太多会乱，只在第一个子图显示完整图例
    if idx == 0:
        ax.legend(loc="best")
    else:
        ax.legend().remove()

# 多余子图隐藏
for k in range(len(C_list), len(axes)):
    axes[k].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "soft_margin_svm.png", dpi=150, bbox_inches="tight")
if os.environ.get("SHOW_PLOTS") == "1":
    plt.show()
plt.close()
