
"""
PathSnap - Model Comparison Framework

Stores and compares measured results from five ML models.
Does not generate or fabricate model metrics.
"""

import csv
from pathlib import Path

MODEL_NAMES = [
    "LogisticRegression",
    "KNeighbors",
    "DecisionTree",
    "RandomForest",
    "HistGradientBoosting"
]

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "log_loss",
    "top1_accuracy",
    "candidate_miss_rate",
    "training_time_seconds",
    "inference_time_seconds"
]

OUTPUT_FILE = Path("experiments/results/model_comparison.csv")


def save_comparison(results):
    """
    Save measured model results.

    Each result must contain a model name and measured metrics.
    Missing metrics remain blank; no values are invented.
    """
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fields = ["model", "status"] + METRICS

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for result in results:
            model = result.get("model")

            if model not in MODEL_NAMES:
                raise ValueError(f"Unknown model: {model}")

            row = {
                "model": model,
                "status": result.get("status", "pending")
            }

            for metric in METRICS:
                row[metric] = result.get(metric, "")

            writer.writerow(row)

    print("Comparison file saved:", OUTPUT_FILE)


def display_comparison():
    if not OUTPUT_FILE.exists():
        print("No comparison results available yet.")
        return

    with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        print("\nPATHSNAP MODEL COMPARISON")
        print("-" * 65)

        for row in reader:
            print(
                f"{row['model']:<25} "
                f"Status: {row['status']:<12} "
                f"F1: {row['f1_score'] or 'Not measured'}"
            )


if __name__ == "__main__":
    initial_results = [
        {"model": model, "status": "pending"}
        for model in MODEL_NAMES
    ]

    save_comparison(initial_results)
    display_comparison()

    print("\nComparison framework test completed.")
    print("No model training or performance evaluation was performed.")
