import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_test_results.csv"
)
BASELINE_DROPOUT = 0.0
TARGET_DROPOUT = 0.3


def load_results(csv_path):
    results = defaultdict(dict)

    with csv_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if (
                float(row["train_fraction"]) != 1.0
                or row["augmentation"] != "none"
                or float(row["weight_decay"]) != 0.0
            ):
                continue

            dropout = float(row["dropout_rate"])
            seed = int(row["random_seed"])

            if seed in results[dropout]:
                raise ValueError(
                    f"配置 dropout={dropout}、seed={seed} 存在重复记录"
                )

            results[dropout][seed] = {
                "accuracy": float(row["final_test_accuracy"]),
                "loss": float(row["final_test_loss"]),
            }

    return results


def summarize_accuracies(accuracies):
    return mean(accuracies), stdev(accuracies)


def main():
    results = load_results(RESULTS_PATH)
    baseline_results = results[BASELINE_DROPOUT]
    dropout_results = results[TARGET_DROPOUT]
    common_seeds = sorted(
        set(baseline_results) & set(dropout_results)
    )

    if len(common_seeds) < 2:
        raise ValueError(
            "至少需要两个共同随机种子才能计算样本标准差"
        )

    baseline_accuracies = []
    dropout_accuracies = []
    gains = []

    print("随机种子  无 Dropout  Dropout=0.3  配对提升")

    for seed in common_seeds:
        baseline_accuracy = baseline_results[seed]["accuracy"]
        dropout_accuracy = dropout_results[seed]["accuracy"]
        gain = dropout_accuracy - baseline_accuracy

        baseline_accuracies.append(baseline_accuracy)
        dropout_accuracies.append(dropout_accuracy)
        gains.append(gain)

        print(
            f"{seed:<9} "
            f"{baseline_accuracy:>9.2%}  "
            f"{dropout_accuracy:>11.2%}  "
            f"{gain * 100:>+7.2f} 个百分点"
        )

    baseline_mean, baseline_std = summarize_accuracies(
        baseline_accuracies
    )
    dropout_mean, dropout_std = summarize_accuracies(
        dropout_accuracies
    )
    gain_mean, gain_std = summarize_accuracies(gains)

    print()
    print(
        "无 Dropout 测试准确率："
        f"{baseline_mean:.2%} ± {baseline_std:.2%}"
    )
    print(
        "Dropout=0.3 测试准确率："
        f"{dropout_mean:.2%} ± {dropout_std:.2%}"
    )
    print(
        "配对提升："
        f"{gain_mean * 100:+.2f} ± "
        f"{gain_std * 100:.2f} 个百分点"
    )


if __name__ == "__main__":
    main()
