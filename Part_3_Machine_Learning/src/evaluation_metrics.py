"""
PathSnap Evaluation Metrics Module
Computes candidate-level classification metrics, observation-level Top-1 accuracy,
candidate miss rates, route continuity, and sequence-level alignment metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    log_loss
)


def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Compute candidate-level classification metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    prec_pos = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_pos = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_pos = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    cm = confusion_matrix(y_true, y_pred).tolist()

    loss = None
    if y_prob is not None:
        try:
            # Handle probability bounds
            eps = 1e-7
            y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
            if y_prob_clipped.ndim == 1:
                y_prob_clipped = np.column_stack([1.0 - y_prob_clipped, y_prob_clipped])
            loss = float(log_loss(y_true, y_prob_clipped))
        except Exception:
            loss = None

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_pos": prec_pos,
        "recall_pos": rec_pos,
        "f1_pos": f1_pos,
        "confusion_matrix": cm,
        "log_loss": loss
    }


def compute_top1_observation_accuracy(obs_indices, candidate_arc_ids, candidate_scores, true_arcs):
    """
    Observation-level Top-1 Accuracy:
    For each GPS observation t, select the candidate with the highest predicted score
    and compare it with the reference road segment true_arcs[t].
    """
    T = len(true_arcs)
    best_cands = {}
    best_scores = {}

    for obs_idx, arc_id, score in zip(obs_indices, candidate_arc_ids, candidate_scores):
        if obs_idx not in best_scores or score > best_scores[obs_idx]:
            best_scores[obs_idx] = score
            best_cands[obs_idx] = arc_id

    correct = 0
    total = 0
    pred_sequence = []

    for t in range(T):
        pred_arc = best_cands.get(t, -1)
        pred_sequence.append(pred_arc)
        if pred_arc == true_arcs[t]:
            correct += 1
        total += 1

    top1_acc = (correct / total) if total > 0 else 0.0
    return top1_acc, pred_sequence


def compute_route_continuity(pred_arc_sequence, arcs_array):
    """
    Route Continuity:
    Fraction of consecutive transitions in predicted arc sequence that are either
    the identical arc (vehicle staying on road) or topologically adjacent in the network
    (arcs_array[arc_prev, 1] == arcs_array[arc_curr, 0]).
    """
    N = len(pred_arc_sequence)
    if N <= 1:
        return 1.0

    valid_transitions = 0
    total_transitions = N - 1

    for i in range(total_transitions):
        a1 = pred_arc_sequence[i]
        a2 = pred_arc_sequence[i + 1]

        if a1 == -1 or a2 == -1:
            continue

        if a1 == a2:
            valid_transitions += 1
        elif 0 <= a1 < len(arcs_array) and 0 <= a2 < len(arcs_array):
            to_node_a1 = arcs_array[a1, 1]
            from_node_a2 = arcs_array[a2, 0]
            if to_node_a1 == from_node_a2:
                valid_transitions += 1

    return valid_transitions / total_transitions


def compute_sequence_alignment(pred_arc_sequence, ground_truth_route):
    """
    Sequence-level comparison:
    Compresses consecutive duplicate arcs and calculates Jaccard similarity and Recall
    against the ground truth route arcs.
    """
    # Compress consecutive duplicates
    compressed_pred = []
    for a in pred_arc_sequence:
        if a != -1 and (not compressed_pred or compressed_pred[-1] != a):
            compressed_pred.append(a)

    pred_set = set(compressed_pred)
    gt_set = set(ground_truth_route)

    intersection = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    jaccard = (intersection / union) if union > 0 else 0.0
    recall = (intersection / len(gt_set)) if len(gt_set) > 0 else 0.0

    return {
        "compressed_pred_len": len(compressed_pred),
        "ground_truth_route_len": len(ground_truth_route),
        "route_jaccard_similarity": jaccard,
        "route_arc_recall": recall
    }
