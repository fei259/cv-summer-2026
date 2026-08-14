# 分析 Dropout 和 Weight Decay 对模型性能的影响
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "results" / "experiments.csv"

TARGET_WEIGHT_DECAYS = (0.0, 0.0001, 0.001)
TARGET_DROPOUT_RATES = (0.0, 0.3, 0.5)
TARGET_ABLATION_FRACTIONS = (0.1, 0.25)
TARGET_TRANSFER_FRACTIONS = (0.1, 0.25, 0.5, 1.0)
TARGET_TRANSFER_DROPOUT_RATES = (0.0, 0.3)

TARGET_ABLATIONS = (
    (0.0, 0.0),
    (0.0, 0.001),
    (0.3, 0.0),
    (0.3, 0.001),
)       # 每一个元组表示 (dropout_rate, weight_decay) 的组合


def load_weight_decay_results(csv_path):
    results = {}

    with csv_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["dataset"] != "cifar10":
                continue
            if int(row["epochs"]) != 10:
                continue
            if float(row["train_fraction"]) != 0.25:
                continue
            if int(row["random_seed"]) != 123:
                continue
            if row["augmentation"] != "none":
                continue
            if float(row["dropout_rate"]) != 0.0:
                continue

            weight_decay = float(row["weight_decay"])

            if weight_decay not in TARGET_WEIGHT_DECAYS:
                continue

            results[weight_decay] = {
                "best_epoch": int(row["best_epoch"]),
                "train_accuracy": float(
                    row["best_train_accuracy"]
                ),
                "test_loss": float(row["best_test_loss"]),
                "test_accuracy": float(
                    row["best_test_accuracy"]
                ),
                "generalization_gap": float(
                    row["generalization_gap"]
                ),
            }

    return results


def load_dropout_results(csv_path):
    results = {}

    with csv_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["dataset"] != "cifar10":
                continue
            if int(row["epochs"]) != 10:
                continue
            if float(row["train_fraction"]) != 0.25:
                continue
            if int(row["random_seed"]) != 123:
                continue
            if row["augmentation"] != "none":
                continue
            if float(row["weight_decay"]) != 0.0:
                continue

            dropout_rate = float(row["dropout_rate"])

            if dropout_rate not in TARGET_DROPOUT_RATES:
                continue

            results[dropout_rate] = {
                "best_epoch": int(row["best_epoch"]),
                "train_accuracy": float(
                    row["best_train_accuracy"]
                ),
                "test_loss": float(row["best_test_loss"]),
                "test_accuracy": float(
                    row["best_test_accuracy"]
                ),
                "generalization_gap": float(
                    row["generalization_gap"]
                ),
            }

    return results


def load_ablation_results(csv_path, train_fraction):
    results = {}

    with csv_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["dataset"] != "cifar10":
                continue
            if int(row["epochs"]) != 10:
                continue
            if float(row["train_fraction"]) != train_fraction:
                continue
            if int(row["random_seed"]) != 123:
                continue
            if row["augmentation"] != "none":
                continue

            configuration = (
                float(row["dropout_rate"]),
                float(row["weight_decay"]),
            )

            if configuration not in TARGET_ABLATIONS:
                continue

            results[configuration] = {
                "best_epoch": int(row["best_epoch"]),
                "train_accuracy": float(
                    row["best_train_accuracy"]
                ),
                "test_loss": float(row["best_test_loss"]),
                "test_accuracy": float(
                    row["best_test_accuracy"]
                ),
                "generalization_gap": float(
                    row["generalization_gap"]
                ),
            }

    return results


def load_dropout_transfer_results(csv_path):
    results = {}

    with csv_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["dataset"] != "cifar10":
                continue
            if int(row["epochs"]) != 10:
                continue
            if int(row["random_seed"]) != 123:
                continue
            if row["augmentation"] != "none":
                continue
            if float(row["weight_decay"]) != 0.0:
                continue

            train_fraction = float(row["train_fraction"])
            dropout_rate = float(row["dropout_rate"])

            if train_fraction not in TARGET_TRANSFER_FRACTIONS:
                continue
            if dropout_rate not in TARGET_TRANSFER_DROPOUT_RATES:
                continue

            results[(train_fraction, dropout_rate)] = {
                "test_accuracy": float(
                    row["best_test_accuracy"]
                ),
                "test_loss": float(row["best_test_loss"]),
                "generalization_gap": float(
                    row["generalization_gap"]
                ),
            }

    return results


def print_results(title, parameter_name, parameter_values, results):
    print(title)
    print(
        f"{parameter_name}\t"
        "最佳轮次\t"
        "训练准确率\t"
        "测试准确率\t"
        "测试损失\t"
        "泛化差距"
    )

    for parameter_value in parameter_values:
        result = results[parameter_value]

        print(
            f"{parameter_value:g}\t\t"
            f"{result['best_epoch']}\t\t"
            f"{result['train_accuracy']:.2%}\t\t"
            f"{result['test_accuracy']:.2%}\t\t"
            f"{result['test_loss']:.4f}\t\t"
            f"{result['generalization_gap']:.2%}"
        )


def print_ablation_results(train_fraction, results):
    print(f"{train_fraction:.0%} 数据正则化组合消融")
    print(
        "Dropout\t"
        "Weight Decay\t"
        "训练准确率\t"
        "测试准确率\t"
        "测试损失\t"
        "泛化差距"
    )

    for dropout_rate, weight_decay in TARGET_ABLATIONS:
        result = results[(dropout_rate, weight_decay)]

        print(
            f"{dropout_rate:g}\t\t"
            f"{weight_decay:g}\t\t"
            f"{result['train_accuracy']:.2%}\t\t"
            f"{result['test_accuracy']:.2%}\t\t"
            f"{result['test_loss']:.4f}\t\t"
            f"{result['generalization_gap']:.2%}"
        )


def print_dropout_transfer_results(results):
    print("Dropout 跨数据比例迁移")
    print(
        "数据比例\t"
        "基线准确率\t"
        "Dropout准确率\t"
        "准确率变化\t"
        "无Dropout泛化差距\t"
        "Dropout=0.3泛化差距\t"
        "差距变化"
    )

    for train_fraction in TARGET_TRANSFER_FRACTIONS:
        baseline = results[(train_fraction, 0.0)]
        dropout = results[(train_fraction, 0.3)]

        accuracy_gain = (
            dropout["test_accuracy"]
            - baseline["test_accuracy"]
        )
        gap_change = (
            dropout["generalization_gap"]
            - baseline["generalization_gap"]
        )

        print(
            f"{train_fraction:.0%}\t\t"
            f"{baseline['test_accuracy']:.2%}\t\t"
            f"{dropout['test_accuracy']:.2%}\t\t"
            f"{accuracy_gain:+.2%}\t\t"
            f"{baseline['generalization_gap']:.2%}\t\t"
            f"{dropout['generalization_gap']:.2%}\t\t"
            f"{gap_change:+.2%}"
        )


def main():
    weight_decay_results = load_weight_decay_results(
        CSV_PATH
    )
    dropout_results = load_dropout_results(CSV_PATH)

    dropout_transfer_results = (
        load_dropout_transfer_results(CSV_PATH)
    )

    print_results(
        "Weight Decay 对照实验",
        "Weight Decay",
        TARGET_WEIGHT_DECAYS,
        weight_decay_results,
    )

    print()

    print_results(
        "Dropout 对照实验",
        "Dropout",
        TARGET_DROPOUT_RATES,
        dropout_results,
    )

    print()

    for train_fraction in TARGET_ABLATION_FRACTIONS:
        ablation_results = load_ablation_results(
            CSV_PATH,
            train_fraction,
        )

        print()
        print_ablation_results(
            train_fraction,
            ablation_results,
        )

    print()

    print_dropout_transfer_results(
        dropout_transfer_results
    )


if __name__ == "__main__":
    main()
