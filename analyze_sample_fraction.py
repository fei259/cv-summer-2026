import csv
from pathlib import Path
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "results" / "experiments.csv"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "sample_fraction"
    / "sample_fraction_comparison.png"
)

TARGET_FRACTIONS = (0.1, 0.25, 0.5, 1.0)


def load_latest_results(csv_path):
    latest_results = {}

    with csv_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["dataset"] != "cifar10":
                continue

            # 只有当训练比例在目标范围内时才记录结果
            train_fraction = float(row["train_fraction"])

            if train_fraction not in TARGET_FRACTIONS:
                continue

            # 记录最新的结果，同一个比例的结果会覆盖之前的记录
            latest_results[train_fraction] = {
                "train_fraction": train_fraction,
                "train_samples": int(row["train_samples"]),
                "best_test_accuracy": float(
                    row["best_test_accuracy"]
                ),
                "generalization_gap": float(
                    row["generalization_gap"]
                ),
                "duration_seconds": float(
                    row["duration_seconds"]
                ),
            }

    # 按照 TARGET_FRACTIONS 的顺序返回结果
    return [
        latest_results[fraction]
        for fraction in TARGET_FRACTIONS
    ]


def plot_results(results, output_path):
    fractions = [
        result["train_fraction"] * 100  # 转换为百分比
        for result in results
    ]
    accuracies = [
        result["best_test_accuracy"] * 100
        for result in results
    ]
    gaps = [
        result["generalization_gap"] * 100
        for result in results
    ]
    durations = [
        result["duration_seconds"]
        for result in results
    ]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
    )

    axes[0].plot(
        fractions,
        accuracies,
        marker="o",
        linewidth=2,
    )
    axes[0].set_title("Test Accuracy")
    axes[0].set_ylabel("Accuracy (%)")

    axes[1].plot(
        fractions,
        gaps,
        marker="o",
        linewidth=2,
        color="orange",
    )
    axes[1].set_title("Generalization Gap")
    axes[1].set_ylabel("Gap (percentage points)")

    axes[2].plot(
        fractions,
        durations,
        marker="o",
        linewidth=2,
        color="green",
    )
    axes[2].set_title("Training Duration")
    axes[2].set_ylabel("Duration (seconds)")

    for axis in axes:
        axis.set_xlabel("Training Data (%)")
        axis.set_xticks(fractions)
        axis.grid(alpha=0.3)

    figure.suptitle(
        "CIFAR-10 Sample Fraction Experiment"
    )
    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("对比图已保存到：", output_path)


def main():
    results = load_latest_results(CSV_PATH)

    print(
        "训练比例\t样本数\t测试准确率\t泛化差距\t耗时"
    )

    for result in results:
        print(
            f"{result['train_fraction']:.0%}\t"
            f"{result['train_samples']}\t"
            f"{result['best_test_accuracy']:.2%}\t"
            f"{result['generalization_gap']:.2%}\t"
            f"{result['duration_seconds']:.2f}s"
        )

    plot_results(results, OUTPUT_PATH)


if __name__ == "__main__":
    main()
