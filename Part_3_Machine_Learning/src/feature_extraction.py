"""
PathSnap Feature Extraction Module
Extracts engineered features for candidate road segments to enable robust supervised classification.
"""

import math
import numpy as np
from .data_loader import compute_track_kinematics


FEATURE_NAMES = [
    "dist_to_segment",       # Perpendicular distance in meters
    "heading_diff",          # Absolute angular difference [0, 180] deg
    "heading_cos",           # Cosine of heading difference [-1, 1]
    "t_proj",                # Projection ratio on arc [0, 1]
    "is_endpoint",           # Binary flag if projection is clamped at endpoint
    "arc_length",            # Length of candidate arc in meters
    "dist_to_u",             # Distance from GPS observation to arc from_node
    "dist_to_v",             # Distance from GPS observation to arc to_node
    "speed_mps",             # Instantaneous GPS speed (m/s)
    "time_gap",              # Time gap from previous GPS point (s)
    "relative_dist_rank",    # Distance rank among candidates (0, 1, 2, ...)
    "dist_ratio",            # Ratio to closest candidate distance
    "gaussian_emission_score"# Spatial likelihood exp(-d^2 / (2*sigma^2))
]


def angular_difference(bearing1, bearing2):
    """
    Compute minimal angular difference in degrees between two bearings in [0, 180].
    """
    diff = abs(bearing1 - bearing2) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    return diff


def extract_candidate_features(trip_data, trip_candidates_res, sigma=20.0):
    """
    Extract feature matrix X, label vector y, and metadata arrays from trip candidate results.
    Returns:
        X: np.ndarray of shape (N_candidates, num_features)
        y: np.ndarray of shape (N_candidates,)
        groups: np.ndarray of trip_id hashes or string for group-aware CV
        obs_indices: list/array tracking which observation each candidate belongs to
        candidate_meta: list of candidate dicts
    """
    track = trip_data["track"]
    kinematics = compute_track_kinematics(track)
    candidates_by_obs = trip_candidates_res["candidates"]
    net_index = trip_candidates_res["net_index"]

    X_list = []
    y_list = []
    obs_idx_list = []
    meta_list = []

    T = len(track)
    for t in range(T):
        obs_cands = candidates_by_obs[t]
        if not obs_cands:
            continue

        min_d = min(c["dist"] for c in obs_cands)
        traj_bearing = kinematics["bearings_deg"][t]
        speed_mps = kinematics["speeds_mps"][t]
        time_gap = kinematics["time_gaps"][t]
        px, py = kinematics["xy_m"][t, 0], kinematics["xy_m"][t, 1]

        for rank, c in enumerate(obs_cands):
            dist = c["dist"]
            t_proj = c["t_proj"]
            arc_len = c["arc_length"]
            arc_bearing = c["arc_bearing"]

            u_coord = net_index.u_coords[c["arc_id"]]
            v_coord = net_index.v_coords[c["arc_id"]]
            dist_to_u = math.hypot(px - u_coord[0], py - u_coord[1])
            dist_to_v = math.hypot(px - v_coord[0], py - v_coord[1])

            head_diff = angular_difference(traj_bearing, arc_bearing)
            head_cos = math.cos(math.radians(head_diff))
            is_end = 1.0 if (t_proj <= 0.02 or t_proj >= 0.98) else 0.0
            dist_ratio = (dist + 1e-3) / (min_d + 1e-3)
            gauss_score = math.exp(-0.5 * (dist / sigma) ** 2)

            feat = [
                dist,
                head_diff,
                head_cos,
                t_proj,
                is_end,
                arc_len,
                dist_to_u,
                dist_to_v,
                speed_mps,
                time_gap,
                float(rank),
                dist_ratio,
                gauss_score
            ]

            X_list.append(feat)
            y_list.append(c.get("label", 0))
            obs_idx_list.append(t)
            meta_list.append(c)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    obs_indices = np.array(obs_idx_list, dtype=np.int32)

    return X, y, obs_indices, meta_list
