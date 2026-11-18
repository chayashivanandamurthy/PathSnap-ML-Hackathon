
import csv
from pathlib import Path


OUTPUT_COLUMNS = [
    "trip_id",
    "observation_id",
    "candidate_arc_id",
    "prediction_score"
]


def export_predictions(rows, output_path):
    """
    Export candidate-level model scores for HMM integration.

    Each row must contain:
    trip_id
    observation_id
    candidate_arc_id
    prediction_score

    prediction_score must be a probability between 0 and 1.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    seen = set()

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_COLUMNS
        )
        writer.writeheader()

        for row in rows:
            missing = [
                column for column in OUTPUT_COLUMNS
                if column not in row
            ]

            if missing:
                raise ValueError(
                    f"Missing required fields: {missing}"
                )

            score = float(row["prediction_score"])

            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Invalid prediction score: {score}"
                )

            key = (
                str(row["trip_id"]),
                str(row["observation_id"]),
                str(row["candidate_arc_id"])
            )

            if key in seen:
                raise ValueError(
                    f"Duplicate candidate prediction: {key}"
                )

            seen.add(key)

            writer.writerow({
                "trip_id": row["trip_id"],
                "observation_id": row["observation_id"],
                "candidate_arc_id": row["candidate_arc_id"],
                "prediction_score": score
            })

            count += 1

    print("Prediction export successful.")
    print("Total candidate predictions:", count)
    print("Output file:", output_path)

    return count


if __name__ == "__main__":

    # Synthetic test records only.
    # These are NOT actual PathSnap model predictions.

    test_predictions = [
        {
            "trip_id": "TEST_TRIP",
            "observation_id": "0",
            "candidate_arc_id": "101",
            "prediction_score": 0.85
        },
        {
            "trip_id": "TEST_TRIP",
            "observation_id": "0",
            "candidate_arc_id": "102",
            "prediction_score": 0.10
        },
        {
            "trip_id": "TEST_TRIP",
            "observation_id": "0",
            "candidate_arc_id": "103",
            "prediction_score": 0.05
        }
    ]

    export_predictions(
        test_predictions,
        "experiments/results/prediction_export_test.csv"
    )

    print("SYNTHETIC EXPORT TEST COMPLETE")
