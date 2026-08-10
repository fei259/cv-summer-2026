import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "results" / "experiments.csv"

TARGET_FRACTION = 0.5
TARGET_AUGMENTATIONS = ("none", "basic")


# 从 CSV 实验结果文件里，筛选出 CIFAR-10、指定训练集比例、指定数据增强方法的实验结果，
# 然后按照「数据增强方法 → 随机种子 → 准确率」的结构保存
def load_results(csv_path):
    # 读取 CSV 文件，返回一个字典，按数据增强方法和随机种子组织实验结果
    # 首次使用 results["none"] 时会自动创建一个空字典
    results = defaultdict(dict)

    with csv_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["dataset"] != "cifar10":
                continue
            if float(row["train_fraction"]) != TARGET_FRACTION:
                continue
            if row["augmentation"] not in TARGET_AUGMENTATIONS:
                continue

            augmentation = row["augmentation"]
            seed = int(row["random_seed"])
            accuracy = float(row["best_test_accuracy"])

            results[augmentation][seed] = accuracy

    return results


def main():
    results = load_results(CSV_PATH)

    # 取出两种数据增强方法的随机种子集合
    # 把字典传给 set() 时，默认取得字典的键
    none_seeds = set(results["none"])
    basic_seeds = set(results["basic"])

    # & 表示集合交集，只比较两种配置都运行过的种子
    common_seeds = sorted(none_seeds & basic_seeds)

    gains = []

    print("随机种子\t无增强\t基础增强\t增强收益")

    for seed in common_seeds:
        none_accuracy = results["none"][seed]
        basic_accuracy = results["basic"][seed]

        # 小数准确率之差乘 100，转换为百分点
        gain_points = (
            basic_accuracy - none_accuracy
        ) * 100

        gains.append(gain_points)

        print(
            f"{seed}\t"
            f"{none_accuracy:.2%}\t"
            f"{basic_accuracy:.2%}\t"
            f"{gain_points:+.2f} 个百分点"
        )

    print(
        "平均增强收益：",
        f"{mean(gains):+.2f} 个百分点",
    )


if __name__ == "__main__":
    main()
