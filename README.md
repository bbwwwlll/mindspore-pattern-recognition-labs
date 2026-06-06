# 模式识别课程作业

## 项目简介

本项目为模式识别课程作业代码整理，包含 Iris 数据集二分类、Wine 数据集 KNN 分类、Iris 数据集 K-Means 聚类、PCA 降维可视化以及软间隔 SVM 示例实验。代码主要完成数据读取、特征处理、模型训练或计算、结果评估与可视化输出。

## 目录结构

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── iris.data            # Iris 小型测试数据集
│   └── wine.data            # Wine 小型测试数据集
├── IrisSort/
│   └── IrisSort.py          # 基于 MindSpore 的 Iris 二分类实验
├── WineKnn/
│   ├── WineKnn.py           # Wine 数据集 KNN 分类主脚本
│   └── KnnNet.py            # KNN 分类实验的早期脚本版本
├── K-Means/
│   └── K_Means.py           # Iris 数据集 K-Means 聚类实验
├── PCA/
│   └── PCA_Iris.py          # Iris 数据集 PCA 降维可视化
└── 软限定SVM/
    └── soft_margin_SVM.py   # 软间隔 SVM 不同 C 参数对比实验
```

## 环境依赖

建议使用 Python 3.9 或以上版本。安装依赖：

```bash
pip install -r requirements.txt
```

依赖包括：

```text
numpy
matplotlib
mindspore
scikit-learn
```

说明：多个实验入口使用 MindSpore。MindSpore 在不同操作系统和 Python 版本下的安装包支持情况可能不同，如安装失败，请参考 MindSpore 官方安装说明选择匹配的 Python、CPU/GPU/Ascend 版本。

## 数据说明

仓库已保留小型课程测试数据集，无需额外下载数据。数据统一放在项目根目录的 `data/` 下：

```text
data/
├── iris.data
└── wine.data
```

各脚本均使用相对项目根目录推导出的数据路径，不依赖当前命令行所在目录。

## 运行方法

在项目根目录执行以下命令：

```bash
python IrisSort/IrisSort.py
python WineKnn/WineKnn.py
python K-Means/K_Means.py
python PCA/PCA_Iris.py
python 软限定SVM/soft_margin_SVM.py
```

默认情况下，绘图结果会保存到项目根目录的 `output/` 文件夹，不会弹出图形窗口。如需显示图形窗口，可设置环境变量：

```bash
# Windows PowerShell
$env:SHOW_PLOTS="1"

# macOS/Linux
export SHOW_PLOTS=1
```

## 实验流程

1. 读取 `data/` 下的 Iris 或 Wine 数据集。
2. 将字符串标签转换为数值标签。
3. 按实验需要选择特征并划分训练集、测试集。
4. 训练或计算分类、聚类、降维模型。
5. 输出准确率、聚类统计或可视化图像。
6. 将图像结果保存到 `output/` 目录。

## 主要结果

运行脚本后会在 `output/` 目录生成可视化结果，例如：

```text
output/
├── iris_binary_scatter.png
├── sigmoid_function.png
├── wine_feature_scatter.png
├── wine_knnnet_feature_scatter.png
├── kmeans_iris_clusters.png
├── pca_iris.png
└── soft_margin_svm.png
```

分类和聚类指标会打印在终端中。本文档不编造固定实验数值，实际结果以本地运行输出为准。

## 注意事项

- 当前脚本默认使用 CPU 运行 MindSpore。
- 随机划分训练集的脚本已设置随机种子，便于复现实验。
- 如果运行时报缺少 `mindspore`、`numpy`、`matplotlib` 或 `sklearn`，请先安装 `requirements.txt` 中的依赖。
- 如果中文目录在某些终端或系统中导致路径问题，可进入对应目录后运行，或将目录名改为英文后同步修改文档。
- `output/` 为运行生成结果目录，已加入 `.gitignore`，不随仓库提交。

## 课程说明

本仓库为模式识别课程作业代码整理，仅用于学习与交流。
