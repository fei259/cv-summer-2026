import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.0):
        super().__init__()

        self.features = nn.Sequential(      #nn.Sequential 可以把多个网络层按照顺序组合起来
            nn.Conv2d(
                #输入为RGB三通道
                in_channels=3,

                #使用32个卷积核，生成32张特征图
                out_channels=32,

                #每个卷积核包含3个独立的3×3权重切片，覆盖全部RGB通道
                kernel_size=3,

                # 在图片四周填充 1 圈像素，保持宽高不变
                padding=1
            ),
            nn.ReLU(),

            # 最大池化：将特征图的宽和高都缩小为原来的一半，2×2范围取最大值
            nn.MaxPool2d(kernel_size=2),

            # TODO 1：将 32 个通道卷积为 64 个通道
            # 每个卷积核包含32个独立的3×3权重切片
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            # TODO 2：ReLU
            #引入非线性
            nn.ReLU(),

            # TODO 3：再次执行 2×2 最大池化
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            # TODO 4：64×8×8 → 128
            nn.Linear(
                in_features=64 * 8 * 8,
                out_features=128
            ),

            # TODO 5：ReLU
            nn.ReLU(),

            nn.Dropout(p=dropout_rate),

            # TODO 6：128 → num_classes
            #将 128 个特征映射为各类别的得分
            nn.Linear(
                in_features=128,
                out_features=num_classes
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    model = SimpleCNN(dropout_rate=0.5)
    sample_images = torch.randn(4, 3, 32, 32)

    model.train()
    train_logits_1 = model(sample_images)
    train_logits_2 = model(sample_images)

    print(
        "训练模式下两次输出相同：",
        torch.allclose(train_logits_1, train_logits_2),
    )

    model.eval()

    with torch.no_grad():
        eval_logits_1 = model(sample_images)
        eval_logits_2 = model(sample_images)

    print(
        "测试模式下两次输出相同：",
        torch.allclose(eval_logits_1, eval_logits_2),
    )

    print("\n观察 Dropout 的数值变化：")

    dropout = nn.Dropout(p=0.5)

    # 创建一个全为 1 的输入张量，大小为 10,000
    input_values = torch.ones(10_000)

    dropout.train()

    # 把 input_values 输入到 dropout 层中，得到经过 Dropout 处理后的结果，并保存到 train_values
    train_values = dropout(input_values)

    print("训练模式前 20 个输出：", train_values[:20])
    print("训练模式输出平均值：", train_values.mean().item())

    dropout.eval()
    eval_values = dropout(input_values)

    print("测试模式前 20 个输出：", eval_values[:20])
    print("测试模式输出平均值：", eval_values.mean().item())
