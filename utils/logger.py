import csv
from pathlib import Path

EXPERIMENT_FIELDS = [
    "experiment_id",
    "dataset",
    "model",
    "epochs",
    "batch_size",
    "train_fraction",
    "validation_fraction",
    "train_samples",
    "validation_samples",
    "random_seed",
    "augmentation",
    "dropout_rate",
    "weight_decay",
    "optimizer",
    "learning_rate",
    "device",
    "duration_seconds",
    "best_epoch",
    "best_train_accuracy",
    "best_validation_loss",     # 准确率接近时辅助判断
    "best_validation_accuracy",     # 选择候选配置的主要依据
    "validation_gap",       # 最佳轮次的训练准确率减去验证准确率，用来观察过拟合程度
]


def log_experiment(csv_path, record):
    csv_path = Path(csv_path)

    # 如果父目录不存在，就自动创建
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 写入前判断文件是否已经存在
    file_exists = csv_path.exists()

    with csv_path.open(
        mode="a",  # 表示追加写入
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=EXPERIMENT_FIELDS,
        )

        # CSV 第一次创建时写入表头，之后追加时不重复写
        if not file_exists:
            writer.writeheader()  # 会把 EXPERIMENT_FIELDS 写成 CSV 的第一行

        # 将一条记录写入 CSV
        writer.writerow(record)
