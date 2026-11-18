from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pyproj import CRS, Transformer

DATA = Path("data/raw/Student folder")
OUTPUT = Path("data/processed")

TOP_K = 10
NEAREST_NODES = 30

def gps_bearings(lon, lat):
    lon1 = np.radians(lon)
    lat1 = np.radians(lat)
    lon2 = np.radians(pd.Series(lon).shift(-1).to_numpy())
    lat2 = np.radians(pd.Series(lat).shift(-1).to_numpy())

    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = (np.cos(lat1) * np.sin(lat2)
         - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))

    result = (np.degrees(np.arctan2(y, x)) + 360) % 360
    return result

def process_trip(trip_dir, partition):
    trip_id = trip_dir.name

    track = pd.read_csv(
        trip_dir / f"{trip_id}.track",
        sep=r"\s+", header=None,
        names=["longitude", "latitude", "timestamp"]
    )
    nodes = pd.read_csv(
        trip_dir / f"{trip_id}.nodes",
        sep=r"\s+", header=None,
        names=["longitude", "latitude"]
    )
    arcs = pd.read_csv(
        trip_dir / f"{trip_id}.arcs",
        sep=r"\s+", header=None,
        names=["from_node", "to_node"]
    )

    # Project this trip into local metres.
    lon0 = float(track["longitude"].median())
    lat0 = float(track["latitude"].median())

    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+datum=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs(
        "EPSG:4326", local_crs, always_xy=True
    )

    node_x, node_y = transformer.transform(
        nodes["longitude"].to_numpy(),
        nodes["latitude"].to_numpy()
    )
    gps_x, gps_y = transformer.transform(
        track["longitude"].to_numpy(),
        track["latitude"].to_numpy()
    )

    node_xy = np.column_stack([node_x, node_y])
    gps_xy = np.column_stack([gps_x, gps_y])

    # Build node-to-arc lookup. Arc ID = original zero-based row index.
    adjacency = [[] for _ in range(len(nodes))]
    for arc_id, (u, v) in enumerate(
        arcs[["from_node", "to_node"]].to_numpy()
    ):
        adjacency[int(u)].append(arc_id)
        adjacency[int(v)].append(arc_id)

    tree = cKDTree(node_xy)

    u = arcs["from_node"].to_numpy(dtype=int)
    v = arcs["to_node"].to_numpy(dtype=int)

    ax, ay = node_x[u], node_y[u]
    bx, by = node_x[v], node_y[v]

    # Directed road-arc bearing: from_node -> to_node.
    arc_bearing = (
        np.degrees(np.arctan2(bx - ax, by - ay)) + 360
    ) % 360

    track["time_gap"] = track["timestamp"].diff()
    dx = np.diff(gps_x, prepend=np.nan)
    dy = np.diff(gps_y, prepend=np.nan)
    track["distance_m"] = np.hypot(dx, dy)
    track.loc[track.index[0], "distance_m"] = np.nan

    track["speed_mps"] = (
        track["distance_m"] / track["time_gap"]
    )
    track["speed_kmh"] = track["speed_mps"] * 3.6
    track["bearing_deg"] = gps_bearings(
        track["longitude"].to_numpy(),
        track["latitude"].to_numpy()
    )
    # Use previous-to-current direction for each observation.
    track["bearing_deg"] = track["bearing_deg"].shift(1)

    rows = []

    for obs_id, point in enumerate(gps_xy):
        _, nearest = tree.query(
            point, k=min(NEAREST_NODES, len(nodes))
        )
        nearest = np.atleast_1d(nearest)

        candidate_ids = sorted({
            arc_id
            for node_id in nearest
            for arc_id in adjacency[int(node_id)]
        })

        # Expand search if too few connected arcs were found.
        if len(candidate_ids) < TOP_K:
            _, nearest = tree.query(
                point, k=min(100, len(nodes))
            )
            nearest = np.atleast_1d(nearest)
            candidate_ids = sorted({
                arc_id
                for node_id in nearest
                for arc_id in adjacency[int(node_id)]
            })

        if not candidate_ids:
            continue

        ids = np.asarray(candidate_ids, dtype=int)

        # Exact point-to-segment distance in local metres.
        vx = bx[ids] - ax[ids]
        vy = by[ids] - ay[ids]
        denom = vx * vx + vy * vy

        t = np.zeros(len(ids))
        valid = denom > 0
        t[valid] = (
            (point[0] - ax[ids[valid]]) * vx[valid]
            + (point[1] - ay[ids[valid]]) * vy[valid]
        ) / denom[valid]
        t = np.clip(t, 0, 1)

        closest_x = ax[ids] + t * vx
        closest_y = ay[ids] + t * vy
        distances = np.hypot(
            point[0] - closest_x,
            point[1] - closest_y
        )

        order = np.argsort(distances)[:TOP_K]
        gps_bearing = track.iloc[obs_id]["bearing_deg"]

        for rank, pos in enumerate(order, start=1):
            arc_id = int(ids[pos])
            difference = np.nan

            if pd.notna(gps_bearing):
                raw = abs(float(gps_bearing) - arc_bearing[arc_id])
                difference = min(raw, 360 - raw)

            row = {
                "trip_id": trip_id,
                "observation_id": obs_id,
                "candidate_arc_id": arc_id,
                "candidate_rank": rank,
                "timestamp": track.iloc[obs_id]["timestamp"],
                "longitude": track.iloc[obs_id]["longitude"],
                "latitude": track.iloc[obs_id]["latitude"],
                "time_gap": track.iloc[obs_id]["time_gap"],
                "distance_m": track.iloc[obs_id]["distance_m"],
                "speed_mps": track.iloc[obs_id]["speed_mps"],
                "speed_kmh": track.iloc[obs_id]["speed_kmh"],
                "bearing_deg": gps_bearing,
                "gps_to_road_distance_m": float(distances[pos]),
                "bearing_difference_deg": difference,
                # Labels intentionally left unknown until alignment is verified.
                "reference_arc_id": np.nan,
                "is_correct_candidate": np.nan
            }
            rows.append(row)

    result = pd.DataFrame(rows)
    print(
        f"{partition} {trip_id}: "
        f"{len(track)} GPS observations -> {len(result)} candidate rows"
    )
    return result


def process_partition(partition):
    folder = DATA / partition
    all_results = []

    for trip_dir in sorted(folder.iterdir()):
        if not trip_dir.is_dir():
            continue
        all_results.append(process_trip(trip_dir, partition))

    if not all_results:
        raise RuntimeError(f"No trips found in {folder}")

    combined = pd.concat(all_results, ignore_index=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    filename = (
        "training_candidates.csv"
        if partition == "train"
        else "student_testing_candidates.csv"
    )
    destination = OUTPUT / filename
    combined.to_csv(destination, index=False)

    print("\nSaved:", destination)
    print("Rows:", len(combined))
    print("Columns:", combined.columns.tolist())
    return combined


if __name__ == "__main__":
    process_partition("train")
    process_partition("student_test")
    print("\nCandidate CSV generation finished.")


