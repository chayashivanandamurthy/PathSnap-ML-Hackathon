"""
PathSnap Candidate Generator Module
Generates spatial candidate road segments (arcs) for GPS observations using cKDTree spatial indexing.
Establishes ground truth labels for training data and computes candidate miss rates.
"""

import math
import numpy as np
from scipy.spatial import cKDTree
from .data_loader import project_to_meters


def point_to_segment_distance_and_proj(px, py, x1, y1, x2, y2):
    """
    Compute perpendicular distance from point (px, py) to segment [(x1,y1), (x2,y2)],
    clamped projection parameter t in [0, 1], and segment bearing.
    """
    dx = x2 - x1
    dy = y2 - y1
    L2 = dx * dx + dy * dy
    if L2 <= 1e-8:
        dist = math.hypot(px - x1, py - y1)
        return dist, 0.0, 0.0, 0.0

    t = ((px - x1) * dx + (py - y1) * dy) / L2
    t_clamped = max(0.0, min(1.0, t))
    proj_x = x1 + t_clamped * dx
    proj_y = y1 + t_clamped * dy
    dist = math.hypot(px - proj_x, py - proj_y)
    length = math.sqrt(L2)

    # Bearing of arc: 0=North (dy>0), 90=East (dx>0)
    arc_bearing = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

    return dist, t_clamped, length, arc_bearing


class NetworkSpatialIndex:
    """
    Spatial index for road network arcs using cKDTree on arc midpoints.
    """
    def __init__(self, nodes, arcs, ref_lat=None):
        self.nodes = nodes
        self.arcs = arcs
        self.num_nodes = len(nodes)
        self.num_arcs = len(arcs)

        if ref_lat is None:
            ref_lat = np.mean(nodes[:, 1])
        self.ref_lat = ref_lat

        # Project nodes to Cartesian coordinates (meters)
        self.nodes_m, self.meters_per_deg = project_to_meters(nodes, ref_lat=ref_lat)

        # Arc endpoints and geometry
        u_idx = arcs[:, 0]
        v_idx = arcs[:, 1]
        self.u_coords = self.nodes_m[u_idx]
        self.v_coords = self.nodes_m[v_idx]

        self.midpoints = (self.u_coords + self.v_coords) / 2.0
        dx = self.v_coords[:, 0] - self.u_coords[:, 0]
        dy = self.v_coords[:, 1] - self.u_coords[:, 1]
        self.arc_lengths = np.hypot(dx, dy)
        self.arc_bearings = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

        # Build KDTree on midpoints
        self.tree = cKDTree(self.midpoints)

    def get_candidates_for_point(self, px, py, radius=60.0, top_k=10):
        """
        Find candidate arcs within spatial search radius around (px, py).
        Guarantees at least min(top_k, num_arcs) candidates are evaluated.
        """
        # Query tree with buffer to account for segment length
        query_r = max(radius + 30.0, 75.0)
        indices = self.tree.query_ball_point([px, py], r=query_r)

        if len(indices) < top_k:
            # Fallback to k-nearest midpoints to guarantee candidate set is never empty
            _, indices = self.tree.query([px, py], k=min(top_k * 2, self.num_arcs))
            if not isinstance(indices, (list, np.ndarray)):
                indices = [indices]
            else:
                indices = list(indices)

        candidates = []
        for arc_id in indices:
            u_pt = self.u_coords[arc_id]
            v_pt = self.v_coords[arc_id]
            dist, t_proj, length, arc_bearing = point_to_segment_distance_and_proj(
                px, py, u_pt[0], u_pt[1], v_pt[0], v_pt[1]
            )
            candidates.append({
                "arc_id": int(arc_id),
                "dist": dist,
                "t_proj": t_proj,
                "arc_length": length,
                "arc_bearing": arc_bearing,
                "u": int(self.arcs[arc_id, 0]),
                "v": int(self.arcs[arc_id, 1])
            })

        # Sort by distance
        candidates.sort(key=lambda c: c["dist"])

        # Filter: keep all within radius, capped at top_k, but ensure at least min(top_k, 5)
        filtered = [c for c in candidates if c["dist"] <= radius]
        if len(filtered) < min(top_k, 5):
            filtered = candidates[:min(top_k, len(candidates))]
        else:
            filtered = filtered[:top_k]

        return filtered


def align_ground_truth_route(track, network_index, route_arcs):
    """
    Monotonically align GPS track points to ground truth route arcs.
    Returns:
        np.ndarray of true_arc_ids corresponding to each observation in track.
    """
    T = len(track)
    ref_lat = network_index.ref_lat
    track_m, _ = project_to_meters(track[:, :2], ref_lat=ref_lat)

    route_len = len(route_arcs)
    assigned_arcs = np.zeros(T, dtype=np.int32)
    assigned_dists = np.zeros(T, dtype=np.float64)

    curr_idx = 0
    for t in range(T):
        px, py = track_m[t, 0], track_m[t, 1]

        # Search window along route
        win_start = max(0, curr_idx - 2)
        win_end = min(route_len, curr_idx + 15)

        best_d = 1e12
        best_r_idx = curr_idx

        for r_idx in range(win_start, win_end):
            arc_id = route_arcs[r_idx]
            u_pt = network_index.u_coords[arc_id]
            v_pt = network_index.v_coords[arc_id]
            d, _, _, _ = point_to_segment_distance_and_proj(
                px, py, u_pt[0], u_pt[1], v_pt[0], v_pt[1]
            )
            if d < best_d:
                best_d = d
                best_r_idx = r_idx

        curr_idx = best_r_idx
        assigned_arcs[t] = route_arcs[best_r_idx]
        assigned_dists[t] = best_d

    return assigned_arcs, assigned_dists


def generate_trip_candidates(trip_data, radius=60.0, top_k=10):
    """
    Generate candidates for every observation in a trip.
    Returns:
        trip_candidates: list of length T, each element is a list of candidate dicts
        candidate_misses: int (number of observations where true arc was not in candidate set)
        true_arcs: np.ndarray of length T (if route is provided) or None
    """
    track = trip_data["track"]
    nodes = trip_data["nodes"]
    arcs = trip_data["arcs"]
    route = trip_data["route"]

    net_index = NetworkSpatialIndex(nodes, arcs)
    track_m, _ = project_to_meters(track[:, :2], ref_lat=net_index.ref_lat)

    T = len(track)
    true_arcs = None
    if route is not None:
        true_arcs, _ = align_ground_truth_route(track, net_index, route)

    all_candidates = []
    candidate_misses = 0

    for t in range(T):
        px, py = track_m[t, 0], track_m[t, 1]
        cands = net_index.get_candidates_for_point(px, py, radius=radius, top_k=top_k)

        cand_arc_ids = set(c["arc_id"] for c in cands)
        if true_arcs is not None:
            target_arc = true_arcs[t]
            is_miss = target_arc not in cand_arc_ids
            if is_miss:
                candidate_misses += 1
            for c in cands:
                c["label"] = 1 if c["arc_id"] == target_arc else 0
        else:
            for c in cands:
                c["label"] = 0

        all_candidates.append(cands)

    miss_rate = (candidate_misses / T) if T > 0 else 0.0

    return {
        "candidates": all_candidates,
        "true_arcs": true_arcs,
        "candidate_misses": candidate_misses,
        "candidate_miss_rate": miss_rate,
        "net_index": net_index
    }
