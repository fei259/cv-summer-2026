# CV Summer 2026

## 项目简介

本项目是一个基于 PyTorch 的可复现计算机视觉实验项目。项目将先通过 FashionMNIST 跑通完整训练流程，再以 CIFAR-10 图像分类为核心，完成基线模型与正则化方法的对照实验。

## 项目目标

- 建立规范的训练、验证和推理流程
- 搭建 CIFAR-10 CNN 基线模型
- 比较数据增强、Dropout 和 L2 权重衰减
- 尝试 MixUp，并记录对照实验结果
- 保存训练曲线、混淆矩阵和实验配置
- 形成可复现的代码仓库与实验报告

## 当前进展

- 已使用 FashionMNIST 跑通完整训练、评估、最佳模型保存和结果绘制流程
- 已完成 CIFAR-10 数据管道和 SimpleCNN 基线模型
- 已实现训练指标统计、最佳模型保存和 CSV 实验记录
- 已完成独立评估，并生成训练曲线和混淆矩阵

## 项目结构

```text
cv-summer-2026/
├── data/          # 数据集创建与 DataLoader
├── models/        # 神经网络模型
├── utils/         # 训练、评估与实验日志工具
├── notes/         # Python 和 PyTorch 学习练习
├── results/       # 实验记录、训练曲线和混淆矩阵
├── train.py       # 训练入口
└── evaluate.py    # 独立评估入口
```

## 运行环境

本项目当前验证环境如下：

- Python 3.12.4
- PyTorch 2.12.1
- torchvision 0.27.1
- NumPy 2.5.1
- Matplotlib 3.11.0
- Windows PowerShell

训练脚本会自动检测 CUDA。存在可用 NVIDIA GPU 时使用 GPU，否则回退到 CPU；使用 CPU 不影响代码正确性，但训练耗时会更长。

## 环境配置

在项目根目录打开 PowerShell，创建并激活虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

安装项目所需的核心依赖：

```powershell
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy matplotlib
```

检查 PyTorch 是否安装成功以及 CUDA 是否可用：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 运行项目

首次运行时执行训练脚本。CIFAR-10 不存在时会自动下载，训练结束后会保存最佳模型、训练曲线，并把本次实验参数和指标追加到统一 CSV：

```powershell
python train.py
```

训练完成后，可以单独加载最佳模型进行测试，并生成混淆矩阵：

```powershell
python evaluate.py
```

模型权重 `*.pth` 属于本地产物，不会提交到 Git。因此，新环境必须先运行 `train.py` 生成 `results/cifar10/best_model.pth`，才能运行 `evaluate.py`。

主要输出位置：

- `results/cifar10/best_model.pth`：本地最佳模型权重
- `results/cifar10/training_curves.png`：当前训练曲线
- `results/experiments.csv`：历次实验配置与结果
- `results/baseline/`：提交到仓库的基线训练曲线和混淆矩阵

修改代码后可先执行快速语法检查：

```powershell
python -m compileall train.py evaluate.py data models utils
```

## CIFAR-10 基线结果

当前基线使用 `SimpleCNN`，在 CIFAR-10 测试集上的实验结果如下：

| 配置项 | 数值 |
| --- | --- |
| 训练轮数 | 10 |
| Batch size | 64 |
| 优化器 | SGD |
| 学习率 | 0.1 |
| 最佳轮次 | 10 |
| 测试损失 | 1.4433 |
| 测试准确率 | 70.37% |

### 训练曲线

![CIFAR-10 基线训练曲线](results/baseline/training_curves.png)

### 混淆矩阵

![CIFAR-10 基线混淆矩阵](results/baseline/confusion_matrix.png)
