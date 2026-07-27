from pathlib import Path

#torchvision 是配合 PyTorch 使用的计算机视觉工具库
#torchvision.datasets：常见计算机视觉数据集
#torchvision.transforms：图片预处理操作
from torchvision import datasets, transforms

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


DATA_ROOT = Path(__file__).resolve().parent     #.resolve()把路径转换为规范的绝对路径

#创建数据集
def create_datasets():
    transform = transforms.Compose(     #用于把多个图片处理步骤组合起来
        [
            # TODO 1：把 PIL 图片转换为 PyTorch Tensor
            transforms.ToTensor()
        ]
    )

    #创建训练数据集
    train_dataset = datasets.FashionMNIST(
        root=DATA_ROOT,

        # TODO 2：使用训练集
        train=True,

        # TODO 3：本地不存在时自动下载
        download=True,

        # TODO 4：应用上面定义的 transform
        transform=transform     #左边的是FashionMNIST构造函数的参数名称，右边的是前面创建的变量
    )

    #创建测试数据集
    test_dataset = datasets.FashionMNIST(
        root=DATA_ROOT,

        # TODO 5：使用测试集
        train=False,

        # TODO 6：本地不存在时自动下载
        download=True,

        # TODO 7：应用上面定义的 transform
        transform=transform
    )

    return train_dataset, test_dataset

def create_dataloaders(train_dataset,test_dataset,batch_size=64):
    train_dataloader=DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0       #不创建额外的子进程，所有数据读取工作都由当前主进程完成
    )

    test_dataloader=DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

    return train_dataloader,test_dataloader

def show_samples(images, labels, class_names):
    figure, axes = plt.subplots(4, 4, figsize=(8, 8))       #plt.subplots(4, 4) 会创建：4 行 × 4 列 = 16 个小画布

    #遍历16个小画布
    for index, ax in enumerate(axes.flat):      #axes.flat会把二维排列展开为一维
        image = images[index].squeeze(0)        #.squeeze(0)表示：如果第 0 维的长度是 1，就删除这一维,即删除通道数这个维度，保留长和宽维度
        label = labels[index].item()

        #在当前小画布中显示图片
        ax.imshow(image, cmap="gray")       #cmap="gray"表示采用灰度色彩映射：较小像素值显示得较黑,较大像素值显示得较白
        ax.set_title(class_names[label])        #设置标题为标签（数字）对应的具体类别

        #关闭坐标轴
        ax.axis("off")

    plt.tight_layout()

    #负责把已经画好的整张图真正显示出来
    plt.show()


if __name__ == "__main__":
    train_dataset, test_dataset = create_datasets()
    image, label = train_dataset[0]

    print("训练集样本数：", len(train_dataset))
    print("测试集样本数：", len(test_dataset))
    print("第一张图片形状：", image.shape)
    print("第一张图片数据类型：", image.dtype)
    print("第一张图片像素范围：", image.min().item(), image.max().item())
    print("第一张图片标签：", label)

    train_dataloader, test_dataloader = create_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=64
    )

    #从train_dataloader中取出一个批次的数据，并分别保存图片和标签
    batch_images, batch_labels = next(iter(train_dataloader))

    print("训练集 batch 数量：", len(train_dataloader))
    print("测试集 batch 数量：", len(test_dataloader))
    print("一个 batch 的图片形状：", batch_images.shape)
    print("一个 batch 的标签形状：", batch_labels.shape)
    print("前 8 个标签：", batch_labels[:8])

    show_samples(
        batch_images[:16],
        batch_labels[:16],
        train_dataset.classes
    )
