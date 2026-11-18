
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix
)

# Features used for candidate-level classification.
# IDs and ground-truth columns are deliberately excluded.
FEATURE_COLUMNS = [
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

# Features with different scales benefit from standardisation.
def make_scaled_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            keep_empty_features=True
        )),
        ("scaler", StandardScaler()),
        ("model", model)
    ])


def make_tree_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            keep_empty_features=True
        )),
        ("model", model)
    ])


def get_models():
    return {
        "Logistic Regression": make_scaled_pipeline(
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        ),

        "K-Nearest Neighbors": make_scaled_pipeline(
            KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                n_jobs=-1
            )
        ),

        "Decision Tree": make_tree_pipeline(
            DecisionTreeClassifier(
                max_depth=15,
                class_weight="balanced",
                random_state=42
            )
        ),

        "Random Forest": make_tree_pipeline(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        ),

        "HistGradientBoosting": make_tree_pipeline(
            HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.1,
                max_leaf_nodes=31,
                random_state=42
            )
        )
    }


def get_group_cv():
    # Keep observations from the same trip together.
    return GroupKFold(n_splits=5)


def calculate_binary_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "log_loss": log_loss(y_true, y_prob),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred
        ).tolist()
    }


if __name__ == "__main__":
    models = get_models()

    print("FEATURES:")
    for feature in FEATURE_COLUMNS:
        print("-", feature)

    print("\nBASELINE MODELS:")
    for name in models:
        print("-", name)

    print("\nCross-validation: 5-fold GroupKFold")
    print("Grouping variable: trip_id")
    print("\nMODEL SETUP SUCCESSFUL")
    print("No models have been trained yet.")
