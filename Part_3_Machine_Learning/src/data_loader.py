"""
PathSnap Data Loader Module
Handles loading, validation, and preprocessing of GPS tracks, road network nodes, arcs, and reference routes.
"""

import os
import math
import numpy as np


def load_track(filepath):
    """
    Load a .track file.
    Columns: longitude, latitude, timestamp (seconds)
    Returns:
        np.ndarray of shape (N, 3): [lon, lat, timestamp]
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Track file not found: {filepath}")
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                data.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return np.array(data, dtype=np.float64)


def load_nodes(filepath):
    """
    Load a .nodes file.
    Columns: longitude, latitude
    Row index (0-based) represents node_id.
    Returns:
        np.ndarray of shape (V, 2): [lon, lat]
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Nodes file not found: {filepath}")
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                data.append((float(parts[0]), float(parts[1])))
    return np.array(data, dtype=np.float64)


def load_arcs(filepath):
    """
    Load an .arcs file.
    Columns: from_node_id, to_node_id
    Row index (0-based) represents arc_id.
    Returns:
        np.ndarray of shape (E, 2): [from_node, to_node]
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arcs file not found: {filepath}")
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                data.append((int(parts[0]), int(parts[1])))
    return np.array(data, dtype=np.int32)


def load_route(filepath):
    """
    Load a .route file.
    Columns: arc_id traversed in chronological order.
    Returns:
        list of int or np.ndarray of arc_ids
    """
    if not os.path.exists(filepath):
        return None
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text:
                data.append(int(text))
    return np.array(data, dtype=np.int32)


def load_trip_data(trip_dir, trip_id):
    """
    Load all files for a single trip.
    Returns dict with keys: 'track', 'nodes', 'arcs', 'route' (optional).
    """
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")

    track = load_track(track_path)
    nodes = load_nodes(nodes_path)
    arcs = load_arcs(arcs_path)
    route = load_route(route_path) if os.path.exists(route_path) else None

    return {
        "trip_id": trip_id,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route
    }


def project_to_meters(lon_lat_array, ref_lat=None):
    """
    Convert spherical (lon, lat) coordinates into local Cartesian meter coordinates
    using conformal Equirectangular projection centered at ref_lat.
    Returns:
        (xy_meters, (meters_per_lon, meters_per_lat))
    """
    if ref_lat is None:
        ref_lat = np.mean(lon_lat_array[:, 1])

    lat_rad = math.radians(ref_lat)
    meters_per_deg_lon = 111320.0 * math.cos(lat_rad)
    meters_per_deg_lat = 110540.0

    xy_m = np.empty_like(lon_lat_array, dtype=np.float64)
    xy_m[:, 0] = lon_lat_array[:, 0] * meters_per_deg_lon
    xy_m[:, 1] = lon_lat_array[:, 1] * meters_per_deg_lat

    return xy_m, (meters_per_deg_lon, meters_per_deg_lat)


def compute_track_kinematics(track):
    """
    Compute time_gap, distance_m, speed_mps, speed_kmh, and bearing_deg for a track.
    track: np.ndarray shape (N, 3): [lon, lat, timestamp]
    Returns:
        dict of kinematic arrays of length N
    """
    N = len(track)
    time_gaps = np.zeros(N, dtype=np.float64)
    distances_m = np.zeros(N, dtype=np.float64)
    speeds_mps = np.zeros(N, dtype=np.float64)
    bearings_deg = np.zeros(N, dtype=np.float64)

    ref_lat = np.mean(track[:, 1])
    xy_m, (m_lon, m_lat) = project_to_meters(track[:, :2], ref_lat=ref_lat)

    for i in range(1, N):
        dt = track[i, 2] - track[i - 1, 2]
        dx = xy_m[i, 0] - xy_m[i - 1, 0]
        dy = xy_m[i, 1] - xy_m[i - 1, 1]
        dist = math.hypot(dx, dy)
        speed = dist / dt if dt > 1e-4 else 0.0

        # Bearing in degrees (0 = North, 90 = East, 180 = South, 270 = West)
        # In cartesian: dy is North, dx is East
        deg = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

        time_gaps[i] = dt
        distances_m[i] = dist
        speeds_mps[i] = speed
        bearings_deg[i] = deg

    # First point inherits from second point if available
    if N > 1:
        time_gaps[0] = time_gaps[1]
        distances_m[0] = distances_m[1]
        speeds_mps[0] = speeds_mps[1]
        bearings_deg[0] = bearings_deg[1]

    return {
        "time_gaps": time_gaps,
        "distances_m": distances_m,
        "speeds_mps": speeds_mps,
        "speeds_kmh": speeds_mps * 3.6,
        "bearings_deg": bearings_deg,
        "xy_m": xy_m
    }
