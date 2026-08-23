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
        # 基本增强：随机裁剪 + 随机水平翻转
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

    # 将训练变换组合成一个 transform 对象
    train_transform = transforms.Compose(
        train_steps
    )

    # 测试/验证变换：只做张量化和标准化
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

        # 官方训练集
        train=True,

        # 本地不存在时自动下载
        download=True,

        # 应用训练增强
        transform=train_transform,
    )

    validation_dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=test_transform,       # 使用测试集的 transform 进行验证集的处理，不进行数据增强
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_ROOT,

        # 官方测试集
        train=False,

        # 本地不存在时自动下载
        download=True,

        # 应用测试变换
        transform=test_transform,
    )

    return train_dataset, validation_dataset, test_dataset


# 从同一批 CIFAR-10 官方训练数据里，先划分验证集，再从剩余数据中按比例取训练集
def create_stratified_train_validation_subsets(
    train_dataset,
    validation_dataset,
    train_fraction=1.0,
    validation_fraction=0.1,
    seed=42,
):
    allowed_train_fractions = (0.1, 0.25, 0.5, 1.0)

    if train_fraction not in allowed_train_fractions:
        raise ValueError(
            f"train_fraction 必须是 "
            f"{allowed_train_fractions} 之一"
        )

    targets = torch.tensor(train_dataset.targets)
    generator = torch.Generator().manual_seed(seed)

    train_indices = []
    validation_indices = []

    for class_index in range(len(train_dataset.classes)):
        class_indices = torch.where(
            targets == class_index
        )[0]

        shuffled_order = torch.randperm(
            len(class_indices),
            generator=generator,
        )
        shuffled_indices = class_indices[shuffled_order]

        validation_count = int(
            len(shuffled_indices) * validation_fraction
        )

        class_validation_indices = (
            shuffled_indices[:validation_count]
        )
        remaining_indices = (
            shuffled_indices[validation_count:]
        )

        train_count = int(
            len(remaining_indices) * train_fraction
        )
        class_train_indices = remaining_indices[:train_count]

        validation_indices.extend(
            class_validation_indices.tolist()
        )
        train_indices.extend(
            class_train_indices.tolist()
        )

    train_subset = Subset(
        train_dataset,
        train_indices,
    )
    validation_subset = Subset(
        validation_dataset,
        validation_indices,
    )

    return train_subset, validation_subset


def create_dataloaders(
    train_dataset,
    validation_dataset,
    test_dataset,
    batch_size=64,
    train_fraction=1.0,
    validation_fraction=0.1,
    seed=42,
):
    # 创建分层训练集和验证集子集
    train_subset, validation_subset = (
        create_stratified_train_validation_subsets(
            train_dataset,
            validation_dataset,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            seed=seed,
        )
    )

    loader_generator = torch.Generator().manual_seed(seed)

    train_dataloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )

    validation_dataloader = DataLoader(
        validation_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return (
        train_dataloader,
        validation_dataloader,
        test_dataloader,
    )

#反标准化函数
def denormalize(image):
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)  # (通道数, 高度, 宽度)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    # 反标准化
    image = image * std + mean

    return image.clamp(0, 1)  # 把数值限制在 0～1 之间

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

            # 该类别尚未收集时，保存一张样本
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

        train_dataset, validation_dataset, test_dataset = (
            create_datasets(
                augmentation=augmentation
            )
        )

        (
            train_dataloader,
            validation_dataloader,
            test_dataloader,
        ) = create_dataloaders(
            train_dataset,
            validation_dataset,
            test_dataset,
            batch_size=64,
            train_fraction=0.1,
            validation_fraction=0.1,
            seed=42,
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
