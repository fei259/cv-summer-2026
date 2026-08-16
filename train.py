import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from data.cifar10 import (
    create_dataloaders as create_cifar10_dataloaders,
    create_datasets as create_cifar10_datasets,
)
from data.fashion_mnist import (
    create_dataloaders as create_fashion_dataloaders,
    create_datasets as create_fashion_datasets,
)
from models.mlp import MLP
from models.simple_cnn import SimpleCNN
from utils.engine import evaluate, train_one_epoch
from utils.logger import log_experiment


def main():
    # TODO 1：固定随机种子
    random_seed = 123
    print("随机种子：", random_seed)

    torch.manual_seed(random_seed)  # 固定 CPU 的随机数生成器种子

    #当电脑可以使用 CUDA 显卡时，给所有 GPU 的随机数生成器设置固定种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # TODO 2：选择 CPU 或 GPU
    device=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("使用设备：",device)

    # TODO 3：创建 Dataset 和 DataLoader
    # TODO 4：创建模型、损失函数和优化器
    experiment_name = "cifar10"
    batch_size = 64
    learning_rate = 0.1
    num_epochs = 10
    train_fraction = 0.25
    validation_fraction = 0.1
    augmentation = "none"
    dropout_rate = 0.3
    weight_decay = 0.0

    if experiment_name == "fashion":
        train_dataset, test_dataset = create_fashion_datasets()

        train_dataloader, test_dataloader = create_fashion_dataloaders(
            train_dataset,
            test_dataset,
            batch_size=batch_size,
        )

        model = MLP().to(device)
        model_name = "MLP"
    elif experiment_name == "cifar10":
        train_dataset, validation_dataset, test_dataset = (
            create_cifar10_datasets(
                augmentation=augmentation
            )
        )

        (
            train_dataloader,
            validation_dataloader,
            _,
        ) = create_cifar10_dataloaders(
            train_dataset,
            validation_dataset,
            test_dataset,
            batch_size=batch_size,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            seed=random_seed,
        )

        model = SimpleCNN(
            dropout_rate=dropout_rate,
        ).to(device)

        model_name = "SimpleCNN"
    else:
        raise ValueError(f"不支持的实验：{experiment_name}")

    print(
        "实际训练样本数：",
        len(train_dataloader.dataset),
    )

    if experiment_name == "cifar10":
        print(
            "验证样本数：",
            len(validation_dataloader.dataset),
        )

        print("数据增强配置：", augmentation)
        print("Dropout 概率：", dropout_rate)
        print("Weight Decay：", weight_decay)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # TODO 5：训练若干个 epoch，并在每轮后进行验证和打印指标
    project_root = Path(__file__).resolve().parent

    if experiment_name == "cifar10":
        fraction_name = (
            f"{int(train_fraction * 100)}pct"
        )

        #把 dropout_rate 转换为字符串，并把小数点替换为字母 p，方便在文件夹名中使用
        dropout_name = str(dropout_rate).replace(".", "p")

        weight_decay_name = str(weight_decay).replace(".", "p")

        results_dir = (
            project_root
            / "results"
            / "formal_validation"
            / fraction_name
            / augmentation
            / f"dropout_{dropout_name}"
            / f"weight_decay_{weight_decay_name}"
            / f"seed_{random_seed}"
        )
    else:
        results_dir = (
            project_root
            / "results"
            / experiment_name
        )

    results_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = results_dir / "best_model.pth"
    curve_path = results_dir / "training_curves.png"

    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []

    best_train_accuracy = 0.0
    best_validation_accuracy = 0.0
    best_validation_loss = float("inf")
    best_epoch = 0

    # 记录训练开始时的计时器数值，方便后面计算整个训练过程花了多长时间
    training_start_time = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            device
        )

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_dataloader,
            loss_fn,
            device,
        )

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_validation_loss = validation_loss
            best_train_accuracy = train_accuracy
            best_epoch = epoch

            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_accuracy:.2%} | "
            f"val loss: {validation_loss:.4f} | "
            f"val acc: {validation_accuracy:.2%}"
        )

    # 完整训练耗时
    training_seconds = time.perf_counter() - training_start_time

    #绘制曲线
    epochs = range(1, num_epochs + 1)

    #创建一张画布，在画布中横向放置两个子图
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    #axes[0]表示左边的第一个子图对象
    axes[0].plot(epochs, train_losses, marker="o", label="Train")
    axes[0].plot(epochs, validation_losses, marker="o", label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    #axes[1]表示右边的第二个子图对象
    axes[1].plot(epochs, train_accuracies, marker="o", label="Train")

    axes[1].plot(
        epochs,
        validation_accuracies,
        marker="o",
        label="Validation",
    )

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    figure.tight_layout()
    figure.savefig(curve_path, dpi=150)
    plt.close(figure)

    print("最佳模型已保存到：", best_model_path)
    print("训练曲线已保存到：", curve_path)

    print(
        f"最佳验证结果 | "
        f"epoch: {best_epoch} | "
        f"val loss: {best_validation_loss:.4f} | "
        f"val acc: {best_validation_accuracy:.2%}"
    )

    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiments_csv_path = (
        Path(__file__).resolve().parent
        / "results"
        / "formal_validation_experiments.csv"
    )

    record = {
        "experiment_id": experiment_id,
        "dataset": experiment_name,
        "model": model_name,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "dropout_rate": (
            dropout_rate
            if experiment_name == "cifar10"
            else 0.0
        ),
        "weight_decay": weight_decay,
        "optimizer": type(optimizer).__name__,
        "learning_rate": learning_rate,
        "device": str(device),
        "duration_seconds": round(training_seconds, 2),
        "best_epoch": best_epoch,
        "best_validation_loss": round(best_validation_loss, 4),
        "best_validation_accuracy": round(best_validation_accuracy, 4),
        "train_fraction": (
            train_fraction
            if experiment_name == "cifar10"
            else 1.0
        ),
        "train_samples": len(train_dataloader.dataset),
        "validation_fraction": validation_fraction,
        "validation_samples": (
            len(validation_dataloader.dataset)
            if experiment_name == "cifar10"
            else 0
        ),

        "best_train_accuracy": round(
            best_train_accuracy,
            4
        ),

        "augmentation": (
            augmentation
            if experiment_name == "cifar10"
            else "none"
        ),
        "random_seed": random_seed,

        "validation_gap": round(
            best_train_accuracy - best_validation_accuracy,
            4,
        ),
    }

    log_experiment(
        experiments_csv_path,
        record,
    )

    print("实验记录已追加到：", experiments_csv_path)


if __name__ == "__main__":
    main()
