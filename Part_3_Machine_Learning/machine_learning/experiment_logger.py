
import csv
import json
from pathlib import Path
from datetime import datetime

LOG_FILE = Path("experiments/logs/experiment_log.csv")

FIELDS = [
    "experiment_id",
    "model_name",
    "model_type",
    "parameters",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "log_loss",
    "training_time_seconds",
    "inference_time_seconds",
    "random_seed",
    "notes"
]


def log_experiment(
    model_name,
    model_type,
    parameters=None,
    metrics=None,
    training_time=None,
    inference_time=None,
    random_seed=42,
    notes=""
):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    metrics = metrics or {}
    parameters = parameters or {}

    file_exists = LOG_FILE.exists()
    needs_header = (
        not file_exists or LOG_FILE.stat().st_size == 0
    )

    record = {
        "experiment_id": datetime.now().strftime(
            "EXP_%Y%m%d_%H%M%S_%f"
        ),
        "model_name": model_name,
        "model_type": model_type,
        "parameters": json.dumps(parameters),
        "accuracy": metrics.get("accuracy", ""),
        "precision": metrics.get("precision", ""),
        "recall": metrics.get("recall", ""),
        "f1_score": metrics.get("f1_score", ""),
        "log_loss": metrics.get("log_loss", ""),
        "training_time_seconds": training_time or "",
        "inference_time_seconds": inference_time or "",
        "random_seed": random_seed,
        "notes": notes
    }

    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)

        if needs_header:
            writer.writeheader()

        writer.writerow(record)

    print("Experiment logged:", record["experiment_id"])


if __name__ == "__main__":
    # Test using a separate file, not the official experiment log.
    original_log = LOG_FILE
    LOG_FILE = Path("experiments/logs/logger_test.csv")

    log_experiment(
        model_name="Test Model",
        model_type="Test",
        parameters={"test_parameter": 10},
        metrics={},
        training_time=0.5,
        inference_time=0.1,
        notes="Logger functionality test only; not a real ML result"
    )

    print("LOGGER TEST COMPLETE")
    print("Test file:", LOG_FILE)

    LOG_FILE = original_log
