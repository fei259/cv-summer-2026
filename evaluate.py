from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from data.cifar10 import create_dataloaders, create_datasets
from models.simple_cnn import SimpleCNN
from utils.engine import evaluate


# 统计混淆矩阵
@torch.no_grad()  # 禁用梯度计算
def build_confusion_matrix(
    model,
    dataloader,
    device,
    num_classes,
):
    # 切换模型状态
    model.eval()

    # 初始化计数矩阵
    matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
    )

    for images, labels in dataloader:
        images = images.to(device)

        logits = model(images)
        # 找每张照片最大分数的下标，即预测类别
        predictions = logits.argmax(dim=1).cpu()

        for true_label, predicted_label in zip(
            labels,
            predictions,
        ):
            matrix[true_label, predicted_label] += 1

    return matrix


# 绘制混淆矩阵
def save_confusion_matrix(
    matrix,
    class_names,
    output_path,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, ax = plt.subplots(figsize=(10, 8))

    # 数值越大，蓝色越深
    image = ax.imshow(
        matrix.numpy(),
        cmap="Blues",
    )

    figure.colorbar(image, ax=ax)

    positions = range(len(class_names))

    ax.set_xticks(positions)
    ax.set_yticks(positions)

    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
    )
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("CIFAR-10 Confusion Matrix")

    # 取最大值的一半作为文字颜色分界线
    threshold = matrix.max().item() / 2

    for row in range(len(class_names)):
        for column in range(len(class_names)):
            value = matrix[row, column].item()

            # 根据阈值设置文本颜色，确保在深色背景上仍然可读
            text_color = (
                "white"
                if value > threshold
                else "black"
            )

            # 在每个单元格中添加文本，显示预测数量
            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=150,
    )
    plt.close(figure)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("使用设备：", device)

    # 这里只需要测试集，但沿用现有数据创建接口
    train_dataset, validation_dataset, test_dataset = (
        create_datasets(augmentation="none")
    )

    _, _, test_dataloader = create_dataloaders(     # 这一处的两个 _ 表示训练和验证 DataLoader 在独立测试脚本中不需要使用，只保留第三个测试 DataLoader
        train_dataset,
        validation_dataset,
        test_dataset,
        batch_size=64,
        train_fraction=0.25,
        validation_fraction=0.1,
        seed=123,
    )

    model = SimpleCNN(
        dropout_rate=0.3,
    ).to(device)

    results_dir = (
        Path(__file__).resolve().parent
        / "results"
        / "formal_validation"
        / "25pct"
        / "none"
        / "dropout_0p3"
        / "weight_decay_0p0"
        / "seed_123"
    )

    checkpoint_path = results_dir / "best_model.pth"

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    loss_fn = nn.CrossEntropyLoss()

    test_loss, test_accuracy = evaluate(
        model,
        test_dataloader,
        loss_fn,
        device,
    )

    print(
        f"最佳模型测试结果 | "
        f"loss: {test_loss:.4f} | "
        f"accuracy: {test_accuracy:.2%}"
    )

    confusion_matrix = build_confusion_matrix(
        model,
        test_dataloader,
        device,
        num_classes=len(test_dataset.classes),
    )

    # 模型在这次验证中预测正确的样本总数
    correct_predictions = (
        # diagonal() 提取主对角线（左上到右下）的元素
        confusion_matrix.diagonal().sum().item()
    )

    total_predictions = confusion_matrix.sum().item()

    print("混淆矩阵形状：", confusion_matrix.shape)
    print("测试图片总数：", total_predictions)
    print("预测正确数量：", correct_predictions)
    print(
        "由混淆矩阵计算的准确率：",
        f"{correct_predictions / total_predictions:.2%}",
    )

    print(confusion_matrix)

    confusion_matrix_path = (
        results_dir / "confusion_matrix.png"
    )

    save_confusion_matrix(
        confusion_matrix,
        test_dataset.classes,
        confusion_matrix_path,
    )

    print(
        "混淆矩阵已保存到：",
        confusion_matrix_path,
    )


if __name__ == "__main__":
    main()
