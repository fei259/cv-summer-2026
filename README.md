# CV Summer 2026

## 项目简介

本项目是一个基于 PyTorch 的可复现计算机视觉实验项目。项目将先通过 FashionMNIST 跑通完整训练流程，再以 CIFAR-10 图像分类为核心，完成基线模型与正则化方法的对照实验。

## 项目目标

- 建立规范的训练、验证和推理流程
- 搭建 CIFAR-10 CNN 基线模型
- 比较数据增强、Dropout 和 L2 权重衰减
- 使用独立验证集选择模型轮次与实验配置
- 保存训练曲线、混淆矩阵和实验配置
- 形成可复现的代码仓库与实验报告

## 当前进展

- 已使用 FashionMNIST 跑通完整训练、评估、最佳模型保存和结果绘制流程
- 已完成 CIFAR-10 数据管道和 SimpleCNN 基线模型
- 已实现训练指标统计、最佳模型保存和 CSV 实验记录
- 已完成独立评估，并生成训练曲线和混淆矩阵
- 已完成 10%、25%、50% 和 100% 有限训练数据实验
- 已完成三档数据增强对照及两个随机种子的关键配置复验
- 已形成有限样本与数据增强阶段报告
- 已修正测试集参与选模的问题，并完成 25% 数据四组正式验证实验
- 已由验证集选出 `Dropout=0.3`，随后在测试集上完成一次最终评估

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

首次运行时执行训练脚本。CIFAR-10 不存在时会自动下载。训练期间每轮使用验证集评估，结束后保存验证准确率最高的模型和训练曲线，并把候选实验追加到验证实验 CSV：

```powershell
python train.py
```

所有候选配置比较完成后，先按验证准确率锁定唯一配置，再单独加载其最佳模型进行一次最终测试并生成混淆矩阵：

```powershell
python evaluate.py
```

模型权重 `*.pth` 属于本地产物，不会提交到 Git。因此，新环境必须先运行 `train.py` 生成对应候选配置的 checkpoint，才能运行 `evaluate.py`。

主要输出位置：

- `results/formal_validation/`：按数据比例和配置保存的正式曲线与本地 checkpoint
- `results/formal_validation_experiments.csv`：只包含训练与验证指标的候选实验记录
- `results/formal_validation/README.md`：正式流程、25% 选模与 100% 对照总结
- `results/formal_test_results.csv`：配置锁定后的最终测试记录
- `results/baseline/`：提交到仓库的基线训练曲线和混淆矩阵
- `results/sample_fraction/`：有限训练数据实验曲线与汇总图
- `results/augmentation/`：数据增强实验曲线与汇总图
- `notes/report_draft.md`：有限样本与数据增强阶段报告

修改代码后可先执行快速语法检查：

```powershell
python -m compileall train.py evaluate.py data models utils
```

## 25% 数据正式实验结果

固定 `seed=123`、10 个 epoch、SGD 和学习率 0.1，在 25% 训练池上由验证集选出 `augmentation=none`、`Dropout=0.3`、`weight_decay=0`。将该配置扩展至 100% 训练池后，最终测试准确率为 70.47%；同规模无 Dropout 基线为 66.73%。完整方法与对照见 [正式实验总结](results/formal_validation/README.md)。

## 早期探索性 CIFAR-10 基线

以下结果来自引入独立验证集之前的探索性流程，仅作为历史学习记录，不用于正式配置选择：

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
