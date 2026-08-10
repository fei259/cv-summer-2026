# 汇总和可视化数据增强实验结果
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "results" / "experiments.csv"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "augmentation"
    / "augmentation_comparison.png"
)

TARGET_FRACTIONS = (0.1, 0.25, 0.5)
TARGET_AUGMENTATIONS = ("none", "basic", "strong")
TARGET_EPOCHS = 10
TARGET_RANDOM_SEED = 42


def load_latest_results(csv_path):
    latest_results = {}

    with csv_path.open(mode="r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["dataset"] != "cifar10":
                continue
            if int(row["epochs"]) != TARGET_EPOCHS:
                continue
            if int(row["random_seed"]) != TARGET_RANDOM_SEED:
                continue

            fraction = float(row["train_fraction"])
            augmentation = row["augmentation"]

            if fraction not in TARGET_FRACTIONS:
                continue
            if augmentation not in TARGET_AUGMENTATIONS:
                continue

            latest_results[(fraction, augmentation)] = {
                "best_test_accuracy": float(row["best_test_accuracy"]),
                "duration_seconds": float(row["duration_seconds"]),
            }

    return latest_results


def get_value(results, fraction, augmentation, field):
    result = results.get((fraction, augmentation))
    if result is None:
        return float("nan")
    return result[field]


def add_value_labels(axis, bars, suffix="", rotation=0):
    for bar in bars:
        height = bar.get_height()
        if math.isnan(height):
            continue
        axis.annotate(
            f"{height:.2f}{suffix}",
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=rotation,
        )


def plot_results(results, output_path):
    fraction_labels = [f"{fraction:.0%}" for fraction in TARGET_FRACTIONS]
    positions = list(range(len(TARGET_FRACTIONS)))
    width = 0.24
    colors = {
        "none": "steelblue",
        "basic": "seagreen",
        "strong": "orange",
    }

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for index, augmentation in enumerate(TARGET_AUGMENTATIONS):
        accuracies = [
            get_value(
                results,
                fraction,
                augmentation,
                "best_test_accuracy",
            )
            * 100
            for fraction in TARGET_FRACTIONS
        ]
        bar_positions = [
            position + (index - 1) * width
            for position in positions
        ]
        bars = axes[0].bar(
            bar_positions,
            accuracies,
            width=width,
            label=augmentation,
            color=colors[augmentation],
        )
        add_value_labels(axes[0], bars, "%", rotation=90)

    axes[0].set_title("Test Accuracy")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xticks(positions, fraction_labels)
    axes[0].set_xlabel("Training Data")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].margins(y=0.18)

    for index, augmentation in enumerate(("basic", "strong")):
        gains = []
        for fraction in TARGET_FRACTIONS:
            baseline = get_value(
                results,
                fraction,
                "none",
                "best_test_accuracy",
            )
            accuracy = get_value(
                results,
                fraction,
                augmentation,
                "best_test_accuracy",
            )
            gains.append((accuracy - baseline) * 100)

        bar_positions = [
            position + (index - 0.5) * width
            for position in positions
        ]
        bars = axes[1].bar(
            bar_positions,
            gains,
            width=width,
            label=augmentation,
            color=colors[augmentation],
        )
        add_value_labels(axes[1], bars)

    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Accuracy Gain vs. None")
    axes[1].set_ylabel("Gain (percentage points)")
    axes[1].set_xticks(positions, fraction_labels)
    axes[1].set_xlabel("Training Data")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    for index, augmentation in enumerate(TARGET_AUGMENTATIONS):
        durations = [
            get_value(
                results,
                fraction,
                augmentation,
                "duration_seconds",
            )
            for fraction in TARGET_FRACTIONS
        ]
        bar_positions = [
            position + (index - 1) * width
            for position in positions
        ]
        axes[2].bar(
            bar_positions,
            durations,
            width=width,
            label=augmentation,
            color=colors[augmentation],
        )

    axes[2].set_title("Training Duration")
    axes[2].set_ylabel("Duration (seconds)")
    axes[2].set_xticks(positions, fraction_labels)
    axes[2].set_xlabel("Training Data")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.3)

    figure.suptitle("CIFAR-10 Augmentation Experiments (seed 42)")
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print("对比图已保存到：", output_path)


def print_results(results):
    print("训练比例\t无增强\t基础增强\t强增强")

    for fraction in TARGET_FRACTIONS:
        values = []
        for augmentation in TARGET_AUGMENTATIONS:
            accuracy = get_value(
                results,
                fraction,
                augmentation,
                "best_test_accuracy",
            )
            values.append(
                "--" if math.isnan(accuracy) else f"{accuracy:.2%}"
            )

        print(f"{fraction:.0%}\t" + "\t".join(values))


def main():
    results = load_latest_results(CSV_PATH)
    print_results(results)
    plot_results(results, OUTPUT_PATH)


if __name__ == "__main__":
    main()
