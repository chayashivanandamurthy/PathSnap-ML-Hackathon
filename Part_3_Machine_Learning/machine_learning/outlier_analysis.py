
import csv
from pathlib import Path

FILES = {
    "training": "data/processed/training_candidates.csv",
    "student_testing": "data/processed/student_testing_candidates.csv"
}

OUTPUT = Path("experiments/results")
OUTPUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = {
    "speed_mps": 55,
    "time_gap": 300,
    "distance_m": 500,
    "gps_to_road_distance_m": 500
}


def analyse_dataset(name, filename):
    seen = set()
    flagged = []
    totals = {feature: 0 for feature in THRESHOLDS}
    observations = 0

    with open(filename, "r", encoding="utf-8-sig",
              newline="") as file:

        reader = csv.DictReader(file)

        for row in reader:
            key = (row["trip_id"], row["observation_id"])

            if key in seen:
                continue

            seen.add(key)
            observations += 1
            reasons = []

            for feature, threshold in THRESHOLDS.items():
                value = row[feature].strip()

                if not value:
                    continue

                number = float(value)

                if number > threshold:
                    totals[feature] += 1
                    reasons.append(
                        f"{feature}={number:.3f} > {threshold}"
                    )

            if reasons:
                flagged.append({
                    "trip_id": row["trip_id"],
                    "observation_id": row["observation_id"],
                    "reasons": "; ".join(reasons)
                })

    report = OUTPUT / f"{name}_outlier_report.csv"

    with open(report, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["trip_id", "observation_id", "reasons"]
        )
        writer.writeheader()
        writer.writerows(flagged)

    print("\nDATASET:", name.upper())
    print("Unique observations:", observations)
    print("Flagged observations:", len(flagged))

    for feature, count in totals.items():
        print(f"{feature} above threshold: {count}")

    print("Report saved:", report)


if __name__ == "__main__":
    for name, filename in FILES.items():
        analyse_dataset(name, filename)

    print("\nOUTLIER INVESTIGATION COMPLETE")
    print("Flagged values require review; they are not automatically errors.")
