
import csv
import random
import math
from pathlib import Path

import matplotlib.pyplot as plt

FILES = {
    "training": "data/processed/training_candidates.csv",
    "student_testing": "data/processed/student_testing_candidates.csv"
}

FEATURES = [
    "candidate_rank",
    "longitude",
    "latitude",
    "time_gap",
    "distance_m",
    "speed_mps",
    "speed_kmh",
    "bearing_deg",
    "gps_to_road_distance_m",
    "bearing_difference_deg"
]

SAMPLE_LIMIT = 10000
OUTPUT = Path("experiments/results")
PLOTS = Path("experiments/plots")

OUTPUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

def analyse_file(name, filename):
    print(f"\nAnalysing {name}...")

    stats = {
        feature: {
            "count": 0,
            "missing": 0,
            "mean": 0.0,
            "m2": 0.0,
            "min": float("inf"),
            "max": float("-inf"),
            "sample": []
        }
        for feature in FEATURES
    }

    rng = random.Random(42)
    total_rows = 0

    with open(filename, "r", encoding="utf-8-sig",
              newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            total_rows += 1

            for feature in FEATURES:
                value = row[feature].strip()

                if not value:
                    stats[feature]["missing"] += 1
                    continue

                try:
                    number = float(value)
                except ValueError:
                    stats[feature]["missing"] += 1
                    continue

                if not math.isfinite(number):
                    stats[feature]["missing"] += 1
                    continue

                item = stats[feature]
                item["count"] += 1
                count = item["count"]

                delta = number - item["mean"]
                item["mean"] += delta / count
                item["m2"] += delta * (number - item["mean"])

                item["min"] = min(item["min"], number)
                item["max"] = max(item["max"], number)

                sample = item["sample"]

                if len(sample) < SAMPLE_LIMIT:
                    sample.append(number)
                else:
                    index = rng.randrange(count)
                    if index < SAMPLE_LIMIT:
                        sample[index] = number

    print("Rows:", total_rows)

    report_path = OUTPUT / f"{name}_feature_report.csv"

    with open(report_path, "w", newline="",
              encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "feature", "valid_values", "missing_values",
            "mean", "standard_deviation", "minimum", "maximum"
        ])

        for feature, item in stats.items():
            count = item["count"]

            if count:
                std = math.sqrt(item["m2"] / count)
                writer.writerow([
                    feature, count, item["missing"],
                    item["mean"], std, item["min"], item["max"]
                ])
            else:
                writer.writerow([
                    feature, 0, item["missing"],
                    "", "", "", ""
                ])

    print("Report saved:", report_path)

    # Plot sampled feature distributions
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()

    for axis, feature in zip(axes, FEATURES):
        values = stats[feature]["sample"]

        if values:
            axis.hist(values, bins=30)
            axis.set_title(feature)
            axis.set_ylabel("Frequency")
        else:
            axis.set_title(feature + " (no valid values)")

        axis.tick_params(axis="x", labelrotation=25)

    fig.suptitle(f"{name.upper()} Feature Distributions")
    fig.tight_layout()

    plot_path = PLOTS / f"{name}_feature_distributions.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print("Plot saved:", plot_path)


if __name__ == "__main__":
    for name, filename in FILES.items():
        if Path(filename).exists():
            analyse_file(name, filename)
        else:
            print("Missing file:", filename)

    print("\nFEATURE ANALYSIS COMPLETE")
