import torch
from torch import nn        #nn是PyTorch中用于构建神经网络的模块，包含了网络层、激活函数和损失函数

class MLP(nn.Module):       #定义一个名为 MLP 的类，继承自 nn.Module，MLP通常由多层全连接层组成
    def __init__(self, input_size=28 * 28, hidden_size=128, num_classes=10):        #输入特征数量为 28*28，隐藏层神经元数量为 128，输出类别数量为 10
        super().__init__()      #调用父类 nn.Module 的初始化方法，确保父类的属性和方法被正确初始化

        self.flatten = nn.Flatten()     #nn.Flatten()默认保留batch维度，并不是把整个批次展平成一个一维张量

        self.classifier = nn.Sequential(
            # 输入层到隐藏层的线性变换
            nn.Linear(input_size,hidden_size),

            # 加入 ReLU 激活函数
            nn.ReLU(),      #ReLU定义为f(x) = max(0, x)

            # 隐藏层到输出层的线性变换
            nn.Linear(hidden_size,num_classes)
        )

    #前向传播
    def forward(self, x):
        # 先把图片展平，再送入 classifier
        x=self.flatten(x)       #相当于调用nn.Flatten()层
        x=self.classifier(x)        #相当于调用nn.Sequential()层，将展平后的张量送入分类器

        return x

if __name__ == "__main__":
    model = MLP()       #自动调用 MLP 类的 __init__ 方法，创建一个 MLP 模型实例
    sample_images = torch.randn(4, 1, 28, 28)       #创建一个形状为 (4, 1, 28, 28) 的随机张量，表示 4 张 28x28 的灰度图像，1 表示通道数（灰度图像只有一个通道）
    logits = model(sample_images)       #调用模型的 forward 方法，将 sample_images 作为输入，得到输出 logits，logits 的形状为 (4, 10)，表示 4 张图像对应的 10 个类别的预测分数

    print(model)
    print("输入形状：", sample_images.shape)
    print("输出形状：", logits.shape)
    assert logits.shape == (4, 10)      #断言 logits 的形状是否为 (4, 10)，如果不满足条件则抛出异常，确保模型输出的形状正确，（4, 10）表示 4 张图像对应的 10 个类别的预测分数

    labels=torch.tensor([2,0,7,1],dtype=torch.long)      #创建一个形状为 (4,) 的张量，表示 4 张图像的真实类别标签，数据类型为长整型（int64）
    loss_fn=nn.CrossEntropyLoss()       #创建一个交叉熵损失函数实例，用于计算模型预测结果与真实标签之间的差异
    optimizer=torch.optim.SGD(model.parameters(),lr=0.01)        #创建一个随机梯度下降优化器实例，指定学习率为 0.01，并将模型的参数传入优化器
    model.train()       #将模型设置为训练模式，启用 dropout 和 batch normalization 等训练特性
    print("训练模式：", model.training)       #打印模型的训练模式状态，True 表示模型处于训练模式
    optimizer.zero_grad()      #清除优化器中所有参数的梯度，避免梯度累加
    train_logits=model(sample_images)       #将 sample_images 作为输入，得到模型的预测输出 train_logits
    loss=loss_fn(train_logits,labels)       #计算预测输出 train_logits 与真实标签 labels 之间的交叉熵损失，得到标量 loss
    loss.backward()     #反向传播计算梯度
    optimizer.step()       #根据计算得到的梯度更新模型参数，完成一次训练迭代
    print("本次损失：",loss.item())

    #切换到验证模式
    model.eval()
    print("验证模式：",model.training)

    #关闭梯度计算
    with torch.no_grad():
        eval_logits=model(sample_images)        #验证前向传播

    print("验证输出形状：",eval_logits.shape)
