
"""
PathSnap - Machine Learning Preprocessing
Prepares candidate features for supervised model training.

This module does not train models.
"""

import numpy as np

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

IDENTIFIER_COLUMNS = [
    "trip_id",
    "observation_id",
    "candidate_arc_id",
    "reference_arc_id"
]

TARGET_COLUMN = "is_correct_candidate"


def prepare_features(rows):
    """
    Prepare numeric feature matrix using NumPy.

    Missing feature values are replaced with the
    median of the corresponding feature.

    Returns:
        X: numeric feature matrix
        feature_names: ordered feature names
    """
    matrix = []

    for row in rows:
        values = []

        for feature in FEATURES:
            raw_value = row.get(feature, "")

            try:
                value = float(raw_value)
                if not np.isfinite(value):
                    value = np.nan
            except (ValueError, TypeError):
                value = np.nan

            values.append(value)

        matrix.append(values)

    X = np.array(matrix, dtype=float)

    if X.size == 0:
        return X.reshape(0, len(FEATURES)), FEATURES

    for column in range(X.shape[1]):
        valid = X[:, column][np.isfinite(X[:, column])]

        median = np.median(valid) if len(valid) else 0.0

        X[~np.isfinite(X[:, column]), column] = median

    return X, FEATURES


def extract_groups(rows):
    """Return trip IDs for group-aware cross-validation."""
    return np.array([row["trip_id"] for row in rows])


def extract_targets(rows):
    """Extract binary labels, rejecting missing labels."""
    targets = []

    for row in rows:
        value = row.get(TARGET_COLUMN, "").strip()

        if value == "":
            raise ValueError(
                "Missing target labels. "
                "Validated labels are required before training."
            )

        targets.append(int(value))

    return np.array(targets, dtype=int)


if __name__ == "__main__":
    print("PATHSNAP PREPROCESSING MODULE")
    print("Features:", FEATURES)
    print("Identifier columns excluded:", IDENTIFIER_COLUMNS)
    print("Target:", TARGET_COLUMN)
    print("Group variable: trip_id")
    print("Missing values: median imputation")
    print("Preprocessing module ready.")
