
import csv
from collections import Counter, defaultdict
from pathlib import Path

FILES = {
    "TRAINING": "data/processed/training_candidates.csv",
    "STUDENT TESTING": "data/processed/student_testing_candidates.csv"
}

for name, filename in FILES.items():
    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    observation_counts = Counter()
    trip_rows = Counter()
    missing_by_observation = Counter()
    total_rows = 0
    missing_totals = Counter()

    with open(filename, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            total_rows += 1

            trip_id = row["trip_id"]
            observation_id = row["observation_id"]
            key = (trip_id, observation_id)

            observation_counts[key] += 1
            trip_rows[trip_id] += 1

            for column in [
                "time_gap", "distance_m", "speed_mps",
                "speed_kmh", "bearing_deg",
                "bearing_difference_deg",
                "reference_arc_id", "is_correct_candidate"
            ]:
                if not row[column].strip():
                    missing_totals[column] += 1
                    if column not in (
                        "reference_arc_id", "is_correct_candidate"
                    ):
                        missing_by_observation[key] += 1

    print("Total candidate rows:", total_rows)
    print("Unique trips:", len(trip_rows))
    print("Unique observations (trip + observation):",
          len(observation_counts))

    distribution = Counter(observation_counts.values())

    print("\nCANDIDATES PER OBSERVATION:")
    for candidate_count, number_of_observations in sorted(distribution.items()):
        print(f"{candidate_count} candidates: "
              f"{number_of_observations} observations")

    print("\nOBSERVATIONS WITH MISSING DERIVED FEATURES:",
          len(missing_by_observation))

    print("\nMISSING VALUES:")
    for column, count in sorted(missing_totals.items()):
        print(f"{column}: {count}")

    print("\nROWS PER TRIP:")
    for trip, count in sorted(trip_rows.items()):
        print(f"{trip}: {count}")

print("\nVERIFICATION COMPLETE")
