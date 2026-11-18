
# PathSnap - Bayesian Optimisation Configuration
# Configuration only: does not train models.

import optuna
from pathlib import Path

# Reproducibility and optimisation settings
RANDOM_SEED = 42
N_TRIALS = 30
N_SPLITS = 5
OPTIMISATION_METRIC = "f1"
STUDY_DIRECTION = "maximize"

RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = "sqlite:///experiments/results/optuna_studies.db"

# Five distinct candidate-classification models
MODEL_NAMES = [
    "LogisticRegression",
    "KNeighbors",
    "DecisionTree",
    "RandomForest",
    "HistGradientBoosting"
]


def suggest_parameters(trial, model_name):

    if model_name == "LogisticRegression":
        return {
            "C": trial.suggest_float("C", 0.001, 100, log=True),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            )
        }

    elif model_name == "KNeighbors":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 25),
            "weights": trial.suggest_categorical(
                "weights", ["uniform", "distance"]
            ),
            "p": trial.suggest_int("p", 1, 2)
        }

    elif model_name == "DecisionTree":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 30
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 20
            ),
            "criterion": trial.suggest_categorical(
                "criterion", ["gini", "entropy", "log_loss"]
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            )
        }

    elif model_name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 100, 500, step=100
            ),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 20
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 10
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2"]
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            )
        }

    elif model_name == "HistGradientBoosting":
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "max_iter": trial.suggest_int("max_iter", 100, 400),
            "max_leaf_nodes": trial.suggest_int(
                "max_leaf_nodes", 15, 63
            ),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-5, 10, log=True
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 10, 50
            )
        }

    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_study(model_name):
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)

    return optuna.create_study(
        study_name=f"pathsnap_{model_name}",
        direction=STUDY_DIRECTION,
        sampler=sampler,
        storage=DATABASE_URL,
        load_if_exists=True
    )


if __name__ == "__main__":
    print("PATHSNAP BAYESIAN OPTIMISATION CONFIGURATION")
    print("Models:", MODEL_NAMES)
    print("Trials per model:", N_TRIALS)
    print("Cross-validation folds:", N_SPLITS)
    print("CV grouping: trip_id")
    print("Optimisation metric:", OPTIMISATION_METRIC)
    print("Sampler: Optuna TPE")
    print("Random seed:", RANDOM_SEED)
    print("\nConfiguration loaded successfully.")
    print("No models have been trained or tuned.")
