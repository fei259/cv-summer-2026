import matplotlib.pyplot as plt
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


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

def create_stratified_subset(
    dataset,
    fraction=1.0,
    seed=42,
):
    allowed_fractions = (0.1, 0.25, 0.5, 1.0)

    if fraction not in allowed_fractions:
        raise ValueError(
            f"fraction 必须是 {allowed_fractions} 之一"
        )

    if fraction == 1.0:
        return dataset

    # 获取所有样本的标签
    targets = torch.tensor(dataset.targets)

    # 创建随机数生成器，并固定随机种子
    generator = torch.Generator().manual_seed(seed)

    # 用来保存最终选中的样本索引
    selected_indices = []

    for class_index in range(len(dataset.classes)):
        # 找到当前类别的所有样本索引
        class_indices = torch.where(
            targets == class_index
        )[0]

        # 计算当前类别需要选择的样本数量
        sample_count = int(
            len(class_indices) * fraction
        )

        # torch.randperm(n) 会生成从 0 到 n-1 的随机排列
        shuffled_order = torch.randperm(
            len(class_indices),
            generator=generator,
        )

        # 选择前 sample_count 个索引
        chosen_indices = class_indices[
            shuffled_order[:sample_count]
        ]

        selected_indices.extend(
            chosen_indices.tolist()
        )

    # 用选中的下标包装原数据集
    return Subset(dataset, selected_indices)


def create_dataloaders(
    train_dataset,
    test_dataset,
    batch_size=64,
    train_fraction=1.0,
    seed=42,
):
    sample_train_dataset = create_stratified_subset(
        train_dataset,
        fraction=train_fraction,
        seed=seed,
    )

    loader_generator = torch.Generator().manual_seed(seed)

    train_dataloader = DataLoader(
        sample_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_dataloader, test_dataloader

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
        batch_size=64,
        train_fraction=0.1,
        seed=42,
    )

    print(
        "实际训练样本数：",
        len(train_dataloader.dataset),
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
