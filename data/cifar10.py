from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt


DATA_ROOT = Path(__file__).resolve().parent

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def create_datasets():
    transform = transforms.Compose(
        [
            # TODO 1：将 PIL 图片转换为 Tensor(缩放像素值，除以255)
            transforms.ToTensor(),

            # TODO 2：分别对 RGB 三个通道进行标准化
            #CIFAR10_MEAN = (R均值, G均值, B均值)
            #CIFAR10_STD = (R标准差, G标准差, B标准差)
            #新像素值 = (原像素值 - 通道均值) / 通道标准差,让输入数据大致围绕 0 分布，并让三个通道处于相近的数值尺度，使神经网络的梯度更新更稳定
            transforms.Normalize(
                mean=CIFAR10_MEAN,
                std=CIFAR10_STD
            )
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=DATA_ROOT,

        # TODO 3：选择训练集
        train=True,

        # TODO 4：本地不存在时自动下载
        download=True,

        # TODO 5：应用 transform
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_ROOT,

        # TODO 6：选择测试集
        train=False,

        # TODO 7：本地不存在时自动下载
        download=True,

        # TODO 8：应用 transform
        transform=transform
    )

    return train_dataset, test_dataset

def create_dataloaders(train_dataset,test_dataset,batch_size=64):
    train_dataloader=DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_dataloader=DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_dataloader,test_dataloader

#反标准化函数
def denormalize(image):
    mean=torch.tensor(CIFAR10_MEAN).view(3,1,1)     #[通道数，高度，宽度]
    std=torch.tensor(CIFAR10_STD).view(3,1,1)

    #反标准化
    image=image*std+mean

    return image.clamp(0,1)     #clamp(0, 1) 会把所有数值限制在 0～1 之间

#每类收集一张图片
def show_class_samples(dataloader,class_names):
    samples={}

    for images,labels in dataloader:
        for image,label in zip(images,labels):      #zip(images, labels) 会把图片和对应的标签一一配对
            class_index=label.item()

            # TODO 1：
            # 如果该类别尚未收集，就把 image 保存到 samples 中
            if class_index not in samples:
                samples[class_index]=image

        # 如果已经收集到全部 10 个类别，就结束外层循环
        if len(samples) == 10:
            break

    figure, axes = plt.subplots(2, 5, figsize=(12, 5))

    for class_index, ax in enumerate(axes.flat):        #遍历一维的十个子对象
        image = samples[class_index]

        # 撤销标准化
        image = denormalize(image)

        # (通道, 高, 宽) → (高, 宽, 通道)
        image = image.permute(1, 2, 0)

        ax.imshow(image)
        ax.set_title(class_names[class_index])
        ax.axis("off")

    figure.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_dataset, test_dataset = create_datasets()

    train_dataloader, test_dataloader = create_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=64
    )

    images, labels = next(iter(train_dataloader))

    print("训练集 batch 数量：", len(train_dataloader))
    print("测试集 batch 数量：", len(test_dataloader))
    print("图片 batch 形状：", images.shape)
    print("标签 batch 形状：", labels.shape)
    print("前 8 个标签：", labels[:8])

    show_class_samples(
        train_dataloader,
        train_dataset.classes
    )
