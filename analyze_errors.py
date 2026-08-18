from pathlib import Path

import torch
import matplotlib.pyplot as plt

from data.cifar10 import (
    create_dataloaders,
    create_datasets,
    denormalize,
)
from models.simple_cnn import SimpleCNN


RANDOM_SEED = 123
BATCH_SIZE = 64
TRAIN_FRACTION = 1.0
VALIDATION_FRACTION = 0.1
AUGMENTATION = "none"
DROPOUT_RATE = 0.3
WEIGHT_DECAY = 0.0

PROJECT_ROOT = Path(__file__).resolve().parent

FRACTION_NAME = f"{int(TRAIN_FRACTION * 100)}pct"
DROPOUT_NAME = str(DROPOUT_RATE).replace(".", "p")
WEIGHT_DECAY_NAME = str(WEIGHT_DECAY).replace(".", "p")

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "formal_validation"
    / FRACTION_NAME
    / AUGMENTATION
    / f"dropout_{DROPOUT_NAME}"
    / f"weight_decay_{WEIGHT_DECAY_NAME}"
    / f"seed_{RANDOM_SEED}"
)


def load_model_and_test_dataloader(device):
    train_dataset, validation_dataset, test_dataset = (
        create_datasets(augmentation=AUGMENTATION)
    )

    _, _, test_dataloader = create_dataloaders(
        train_dataset,
        validation_dataset,
        test_dataset,
        batch_size=BATCH_SIZE,
        train_fraction=TRAIN_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        seed=RANDOM_SEED,
    )

    model = SimpleCNN(
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    checkpoint_path = RESULTS_DIR / "best_model.pth"

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, test_dataloader, test_dataset.classes


# 跑一遍测试集，统计模型哪里容易错（混淆矩阵），并且保存每一种错误类型的一张图片用于分析
def collect_error_examples(model, test_dataloader, device, num_classes):
    # 创建一个：类别数 × 类别数的矩阵
    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
    )

    # 创建一个字典，用于存储每种错误类型的一张图片
    error_examples = {}

    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)

            images = images.cpu()
            labels = labels.cpu()
            predictions = predictions.cpu()

            for image, true_label, predicted_label in zip(
                images,
                labels,
                predictions,
            ):
                true_label = true_label.item()
                predicted_label = predicted_label.item()

                confusion_matrix[true_label, predicted_label] += 1

                if true_label != predicted_label:
                    # 生成错误类型的键值对，键为 (真实标签, 预测标签)，值为图片
                    key = (true_label, predicted_label)

                    # 如果该错误类型还没有保存过图片，就保存当前图片
                    if key not in error_examples:
                        error_examples[key] = image.clone()

    return confusion_matrix, error_examples


# 把前面统计出的“每个类别最常见错误”可视化保存成一张图片
def save_typical_error_figure(
    confusion_matrix,
    error_examples,
    class_names,
    save_path,
):
    figure, axes = plt.subplots(2, 5, figsize=(15, 6))

    for true_label, axis in enumerate(axes.flat):
        row = confusion_matrix[true_label].clone()
        row[true_label] = -1

        predicted_label = row.argmax().item()
        count = row[predicted_label].item()

        image = error_examples[(true_label, predicted_label)]
        image = denormalize(image)

        axis.imshow(image.permute(1, 2, 0).numpy())
        axis.set_title(
            f"{class_names[true_label]} → "
            f"{class_names[predicted_label]}\n"
            f"{count} samples"
        )
        axis.axis("off")

    figure.suptitle("CIFAR-10 Most Frequent Error Examples")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(save_path, dpi=200)
    plt.close(figure)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, test_dataloader, class_names = (
        load_model_and_test_dataloader(device)
    )

    confusion_matrix, error_examples = collect_error_examples(
        model,
        test_dataloader,
        device,
        len(class_names),
    )

    print("混淆矩阵总样本数：", confusion_matrix.sum().item())

    for true_label, class_name in enumerate(class_names):
        row = confusion_matrix[true_label].clone()

        # 将真实标签所在的行置为 -1，这样在找最大值时就不会选中正确分类的数量
        row[true_label] = -1

        # 找到该类别最常被错分的类别及其数量
        predicted_label = row.argmax().item()
        count = row[predicted_label].item()

        print(
            f"{class_name} 最常错分为 "
            f"{class_names[predicted_label]}：{count} 张"
        )

    figure_path = RESULTS_DIR / "typical_errors.png"

    save_typical_error_figure(
        confusion_matrix,
        error_examples,
        class_names,
        figure_path,
    )

    print("典型错分样本图已保存到：", figure_path)

    print("使用设备：", device)
    print("加载的模型：", RESULTS_DIR / "best_model.pth")
    print("测试类别：", class_names)
    print("测试批次数：", len(test_dataloader))
    print("模型处于测试模式：", not model.training)


if __name__ == "__main__":
    main()
