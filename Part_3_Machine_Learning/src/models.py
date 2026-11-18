"""
PathSnap Machine Learning Models Module
Defines models from 5 distinct algorithm families for comparative evaluation.
"""

import time
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def get_model_catalog(random_state=42):
    """
    Returns a dictionary of models representing 5 distinct algorithm families:
    1. Linear and Probabilistic: Logistic Regression
    2. Instance-based: Distance-Weighted K-Nearest Neighbors
    3. Tree-based Ensemble: Random Forest Classifier
    4. Boosting: Histogram-based Gradient Boosting (LightGBM style)
    5. Neural Networks: Multi-Layer Perceptron (MLP)
    """
    models = {
        "Logistic_Regression": {
            "name": "Logistic Regression",
            "family": "Linear & Probabilistic",
            "model": LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state
            ),
            "key_config": "C=1.0, penalty=l2, solver=lbfgs, class_weight=balanced"
        },
        "KNN": {
            "name": "Distance-Weighted KNN",
            "family": "Instance-based",
            "model": KNeighborsClassifier(
                n_neighbors=15,
                weights="distance",
                metric="minkowski",
                p=2,
                n_jobs=-1
            ),
            "key_config": "k=15, weights=distance, metric=euclidean"
        },
        "Random_Forest": {
            "name": "Random Forest",
            "family": "Ensemble (Bagging)",
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            ),
            "key_config": "n_estimators=100, max_depth=12, class_weight=balanced"
        },
        "Gradient_Boosting": {
            "name": "HistGradientBoosting",
            "family": "Boosting",
            "model": HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.08,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=random_state
            ),
            "key_config": "max_iter=120, lr=0.08, max_leaf_nodes=31, class_weight=balanced"
        },
        "MLP_Neural_Network": {
            "name": "Multi-Layer Perceptron",
            "family": "Neural Networks",
            "model": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                batch_size=256,
                learning_rate_init=0.003,
                max_iter=100,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=random_state
            ),
            "key_config": "layers=(64, 32), act=relu, solver=adam, alpha=0.001"
        }
    }
    return models


def train_and_time_model(model_obj, X_train, y_train, X_test=None):
    """
    Fits the model, records training duration, and benchmarks inference duration.
    """
    t0 = time.time()
    model_obj.fit(X_train, y_train)
    train_time = time.time() - t0

    inf_time = 0.0
    if X_test is not None:
        t1 = time.time()
        _ = model_obj.predict(X_test)
        inf_time = time.time() - t1

    return train_time, inf_time
