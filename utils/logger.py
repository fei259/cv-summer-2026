import csv
from pathlib import Path

EXPERIMENT_FIELDS = [
    "experiment_id",
    "dataset",
    "model",
    "epochs",
    "batch_size",
    "train_fraction",
    "train_samples",  # 实际训练样本数
    "optimizer",
    "learning_rate",
    "device",
    "duration_seconds",
    "best_epoch",
    "best_train_accuracy",
    "best_test_loss",
    "best_test_accuracy",
    "generalization_gap",  # 泛化差距 = best_train_accuracy - best_test_accuracy
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
