from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DATA_ROOT = Path(__file__).resolve().parent

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

SAMPLE_OUTPUT_DIR = (
    DATA_ROOT.parent
    / "results"
    / "augmentation_samples"
)


# 数据增强函数
def create_transforms(augmentation="none"):
    train_steps = []

    if augmentation == "none":
        pass
    elif augmentation == "basic":
        # TODO 1：添加基本的数据增强方法，包括随机裁剪和随机水平翻转
        train_steps.extend(
            [
                # 先填充到 40×40，再随机裁剪回 32×32
                transforms.RandomCrop(
                    32,
                    padding=4,
                ),

                # 以 50% 概率水平翻转
                transforms.RandomHorizontalFlip(
                    p=0.5,
                ),
            ]
        )
    elif augmentation == "strong":
        train_steps.extend(
            [
                transforms.RandomCrop(
                    32,
                    padding=4,
                ),
                transforms.RandomHorizontalFlip(
                    p=0.5,
                ),

                # 随机改变颜色和光照
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1,
                ),
            ]
        )
    else:
        raise ValueError(
            "augmentation 必须是 none、basic 或 strong"
        )

    train_steps.extend(
        [
            transforms.ToTensor(),

            # 对 RGB 三个通道进行标准化
            transforms.Normalize(
                mean=CIFAR10_MEAN,
                std=CIFAR10_STD,
            ),
        ]
    )

    # TODO 3：将 train_steps 列表中的变换操作组合成一个 transform 对象
    train_transform = transforms.Compose(
        train_steps
    )

    # TODO 4：为测试集创建一个 transform 对象，只包含 ToTensor 和 Normalize
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CIFAR10_MEAN,
                std=CIFAR10_STD,
            ),
        ]
    )

    return train_transform, test_transform


def create_datasets(augmentation="none"):
    train_transform, test_transform = (
        create_transforms(augmentation)
    )

    train_dataset = datasets.CIFAR10(
        root=DATA_ROOT,

        # TODO 3：选择训练集
        train=True,

        # TODO 4：本地不存在时自动下载
        download=True,

        # TODO 5：应用 transform
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_ROOT,

        # TODO 6：选择测试集
        train=False,

        # TODO 7：本地不存在时自动下载
        download=True,

        # TODO 8：应用 transform
        transform=test_transform,
    )

    return train_dataset, test_dataset


# 创建分层子集，保证每个类别的样本比例与原始数据集一致
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

# 每类收集一张图片
def save_class_samples(
    dataloader,
    class_names,
    augmentation,
    save_path,
):
    samples = {}

    for images, labels in dataloader:
        # zip 会把图片和对应的标签一一配对
        for image, label in zip(images, labels):
            class_index = label.item()

            # TODO 1：
            # 如果该类别尚未收集，就把 image 保存到 samples 中
            if class_index not in samples:
                samples[class_index] = image

        # 如果已经收集到全部 10 个类别，就结束外层循环
        if len(samples) == 10:
            break

    figure, axes = plt.subplots(2, 5, figsize=(12, 5))

    # axes.flat 将 2×5 的子图按一维顺序遍历
    for class_index, ax in enumerate(axes.flat):
        image = samples[class_index]

        # 撤销标准化
        image = denormalize(image)

        # (通道, 高, 宽) → (高, 宽, 通道)
        image = image.permute(1, 2, 0)

        ax.imshow(image)
        ax.set_title(class_names[class_index])
        ax.axis("off")

    figure.suptitle(
        f"Augmentation: {augmentation}"
    )
    figure.tight_layout(
        rect=(0, 0, 1, 0.94),
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("增强样本图已保存到：", save_path)


if __name__ == "__main__":
    augmentations = (
        "none",
        "basic",
        "strong",
    )

    for augmentation in augmentations:
        torch.manual_seed(42)

        train_dataset, test_dataset = create_datasets(
            augmentation=augmentation
        )

        train_dataloader, test_dataloader = (
            create_dataloaders(
                train_dataset,
                test_dataset,
                batch_size=64,
                train_fraction=0.1,
                seed=42,
            )
        )

        images, labels = next(
            iter(train_dataloader)
        )

        print(
            f"{augmentation} 图片形状：",
            images.shape,
        )
        print(
            f"{augmentation} 标签形状：",
            labels.shape,
        )

        save_class_samples(
            train_dataloader,
            train_dataset.classes,
            augmentation,
            SAMPLE_OUTPUT_DIR
            / f"{augmentation}.png",
        )
