
import math
from collections import defaultdict


def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate binary classification metrics.
    Positive class = correct candidate (1).
    """

    if not (len(y_true) == len(y_pred) == len(y_prob)):
        raise ValueError("Input lengths must match.")

    if not y_true:
        raise ValueError("Evaluation data cannot be empty.")

    tp = sum(
        1 for actual, predicted in zip(y_true, y_pred)
        if actual == 1 and predicted == 1
    )

    tn = sum(
        1 for actual, predicted in zip(y_true, y_pred)
        if actual == 0 and predicted == 0
    )

    fp = sum(
        1 for actual, predicted in zip(y_true, y_pred)
        if actual == 0 and predicted == 1
    )

    fn = sum(
        1 for actual, predicted in zip(y_true, y_pred)
        if actual == 1 and predicted == 0
    )

    accuracy = (tp + tn) / len(y_true)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    epsilon = 1e-15
    loss = 0.0

    for actual, probability in zip(y_true, y_prob):
        probability = min(max(probability, epsilon), 1 - epsilon)

        loss -= (
            actual * math.log(probability)
            + (1 - actual) * math.log(1 - probability)
        )

    loss /= len(y_true)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "log_loss": loss,
        "confusion_matrix": [
            [tn, fp],
            [fn, tp]
        ]
    }


def calculate_candidate_coverage(observations):
    """
    observations format:
    {
        (trip_id, observation_id): {
            "reference_arc_id": "123",
            "candidate_arc_ids": ["123", "124"]
        }
    }

    Coverage measures whether the reference arc
    exists among generated candidates.
    """

    total = 0
    covered = 0

    for item in observations.values():
        reference = item["reference_arc_id"]

        if reference is None or reference == "":
            continue

        total += 1

        if reference in item["candidate_arc_ids"]:
            covered += 1

    return {
        "total_observations": total,
        "covered_observations": covered,
        "candidate_misses": total - covered,
        "coverage": covered / total if total else None,
        "candidate_miss_rate": (
            (total - covered) / total if total else None
        )
    }


def calculate_observation_top1(observations):
    """
    Each observation must contain:
    reference_arc_id
    candidates: [{"candidate_arc_id": ..., "score": ...}]

    Select the candidate with the highest model score.
    """

    total = 0
    correct = 0

    for item in observations.values():
        reference = item["reference_arc_id"]
        candidates = item["candidates"]

        if not reference or not candidates:
            continue

        total += 1

        best = max(candidates, key=lambda x: x["score"])

        if best["candidate_arc_id"] == reference:
            correct += 1

    return {
        "total_observations": total,
        "correct_observations": correct,
        "top1_accuracy": correct / total if total else None
    }


if __name__ == "__main__":

    # Synthetic test data only.
    # These are NOT PathSnap model results.

    actual = [1, 0, 1, 0]
    predicted = [1, 0, 0, 0]
    probabilities = [0.9, 0.2, 0.4, 0.1]

    metrics = calculate_metrics(
        actual, predicted, probabilities
    )

    print("SYNTHETIC METRICS TEST")
    for name, value in metrics.items():
        print(name, ":", value)

    print("\nEVALUATION UTILITY TEST PASSED")
    print("No real PathSnap results were generated.")
