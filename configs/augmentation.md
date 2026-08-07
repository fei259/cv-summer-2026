# CIFAR-10 数据增强实验计划

## 研究问题

在训练数据有限时，基础增强和较强增强能否提高
SimpleCNN 的测试准确率并缓解过拟合？

## 增强配置

| 配置 | 训练集处理 |
|---|---|
| none | ToTensor + Normalize |
| basic | RandomCrop + RandomHorizontalFlip + ToTensor + Normalize |
| strong | RandomCrop + RandomHorizontalFlip + ColorJitter + ToTensor + Normalize |

测试集始终只使用 `ToTensor + Normalize`，不进行随机增强。

## 控制变量

所有实验保持以下条件一致：

- 数据集：CIFAR-10
- 测试集：完整的 10,000 张图片
- 模型：SimpleCNN
- 训练比例：先测试 10% 和 25%
- Batch size：64
- Epochs：10
- 优化器：SGD
- 学习率：0.1
- 随机种子：42
- Dropout：不使用
- Weight decay：不使用

每组实验只改变数据增强配置。

## 实验组合

| 训练比例 | none | basic | strong |
|---:|---|---|---|
| 10% | 已有基线 | 待运行 | 待运行 |
| 25% | 已有基线 | 待运行 | 待运行 |

需要新增四组完整训练：

1. 10% + basic
2. 10% + strong
3. 25% + basic
4. 25% + strong

## 实验假设

1. 基础增强可以提高测试准确率，并减小模型对训练样本的记忆。
2. 数据比例越低，数据增强可能带来越明显的收益。
3. 较强增强提供更多样本变化，但也可能因图片失真而弱于基础增强。
4. 数据增强不会增加原始图片数量，但会让模型在不同 epoch 中看到随机变化后的样本。

## 记录指标

- 最佳测试准确率
- 最佳轮次
- 最佳轮训练准确率
- 泛化差距
- 测试损失
- 训练耗时

## 样本图

三套配置的可视化结果保存在：

`results/augmentation_samples/`