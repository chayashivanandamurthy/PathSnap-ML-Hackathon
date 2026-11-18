from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

TOP_STATES = 15
DISTANCE_SIGMA = 12.0
TRANSITION_SIGMA = 40.0

def run_alignment(folder):
    trip_id = folder.name

    track = pd.read_csv(
        folder / f"{trip_id}.track",
        sep=r"\s+", header=None,
        names=["longitude", "latitude", "timestamp"]
    )
    nodes = pd.read_csv(
        folder / f"{trip_id}.nodes",
        sep=r"\s+", header=None,
        names=["longitude", "latitude"]
    )
    arcs = pd.read_csv(
        folder / f"{trip_id}.arcs",
        sep=r"\s+", header=None,
        names=["from_node", "to_node"]
    )
    route = np.loadtxt(folder / f"{trip_id}.route", dtype=int, ndmin=1)

    if np.any(route < 0) or np.any(route >= len(arcs)):
        raise ValueError("Invalid arc ID in route file")

    u = arcs["from_node"].to_numpy(dtype=int)[route]
    v = arcs["to_node"].to_numpy(dtype=int)[route]

    if np.any(u[1:] != v[:-1]):
        raise ValueError("Reference route has disconnected arcs")

    crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={track.latitude.median()} "
        f"+lon_0={track.longitude.median()} "
        "+datum=WGS84 +units=m +no_defs"
    )
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    nx, ny = tf.transform(nodes.longitude.to_numpy(),
                          nodes.latitude.to_numpy())
    gx, gy = tf.transform(track.longitude.to_numpy(),
                          track.latitude.to_numpy())

    ax, ay = nx[u], ny[u]
    bx, by = nx[v], ny[v]

    vx, vy = bx - ax, by - ay
    lengths = np.hypot(vx, vy)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths[:-1])))

    arc_bearing = (np.degrees(np.arctan2(vx, vy)) + 360) % 360

    gps_dx = np.diff(gx, prepend=gx[0])
    gps_dy = np.diff(gy, prepend=gy[0])
    gps_step = np.hypot(gps_dx, gps_dy)
    gps_bearing = (np.degrees(np.arctan2(gps_dx, gps_dy)) + 360) % 360

    positions = []
    chainages = []
    distances = []
    bearings = []
    emissions = []

    for i, (px, py) in enumerate(zip(gx, gy)):
        denom = vx * vx + vy * vy
        t = np.zeros(len(route))
        valid = denom > 0

        t[valid] = (
            (px - ax[valid]) * vx[valid]
            + (py - ay[valid]) * vy[valid]
        ) / denom[valid]
        t = np.clip(t, 0, 1)

        cx = ax + t * vx
        cy = ay + t * vy
        dist = np.hypot(px - cx, py - cy)
        chainage = cumulative + t * lengths

        bearing_diff = np.abs(gps_bearing[i] - arc_bearing)
        bearing_diff = np.minimum(bearing_diff, 360 - bearing_diff)

        # Keep the closest route positions as possible Viterbi states.
        selected = np.argsort(dist)[:min(TOP_STATES, len(route))]
        selected = selected[np.argsort(selected)]

        state_dist = dist[selected]
        state_chain = chainage[selected]

        emission = 0.5 * (state_dist / DISTANCE_SIGMA) ** 2

        # Ignore heading for the first or nearly stationary observation.
        if i > 0 and gps_step[i] > 2:
            emission += 0.25 * (bearing_diff[selected] / 60.0) ** 2

        positions.append(selected)
        chainages.append(state_chain)
        distances.append(state_dist)
        bearings.append(bearing_diff[selected])
        emissions.append(emission)

    # Viterbi dynamic programming with non-decreasing route positions.
    costs = [emissions[0].copy()]
    back = [np.full(len(positions[0]), -1, dtype=int)]

    for i in range(1, len(track)):
        prev_pos = positions[i - 1]
        curr_pos = positions[i]
        prev_chain = chainages[i - 1]
        curr_chain = chainages[i]

        current_cost = np.full(len(curr_pos), np.inf)
        current_back = np.full(len(curr_pos), -1, dtype=int)

        for j in range(len(curr_pos)):
            allowed = np.where(prev_pos <= curr_pos[j])[0]

            if len(allowed) == 0:
                continue

            route_delta = np.maximum(
                0.0, curr_chain[j] - prev_chain[allowed]
            )
            gps_delta = gps_step[i]

            transition = (
                0.5 * ((route_delta - gps_delta) / TRANSITION_SIGMA) ** 2
            )

            options = costs[-1][allowed] + transition
            best_local = int(np.argmin(options))
            best_previous = allowed[best_local]

            current_cost[j] = (
                emissions[i][j] + options[best_local]
            )
            current_back[j] = best_previous

        if not np.isfinite(current_cost).any():
            raise RuntimeError(
                f"No valid sequential alignment at observation {i}"
            )

        costs.append(current_cost)
        back.append(current_back)

    state = int(np.argmin(costs[-1]))
    chosen = [0] * len(track)

    for i in range(len(track) - 1, -1, -1):
        chosen[i] = state
        if i > 0:
            state = int(back[i][state])

    chosen_positions = np.array([
        positions[i][chosen[i]] for i in range(len(track))
    ])
    chosen_distances = np.array([
        distances[i][chosen[i]] for i in range(len(track))
    ])
    chosen_bearings = np.array([
        bearings[i][chosen[i]] for i in range(len(track))
    ])

    result = pd.DataFrame({
        "trip_id": trip_id,
        "observation_id": np.arange(len(track)),
        "reference_route_position": chosen_positions,
        "reference_arc_id": route[chosen_positions],
        "gps_to_reference_distance_m": chosen_distances,
        "bearing_difference_deg": chosen_bearings
    })

    output = Path("reports")
    output.mkdir(exist_ok=True)
    destination = output / f"{trip_id}_alignment_preview.csv"
    result.to_csv(destination, index=False)

    print("Trip:", trip_id)
    print("GPS observations:", len(track))
    print("Route arcs:", len(route))
    print("Backward route-position jumps:",
          int(np.sum(np.diff(chosen_positions) < 0)))
    print("Median distance (m):", round(float(np.median(chosen_distances)), 2))
    print("95th percentile distance (m):",
          round(float(np.percentile(chosen_distances, 95)), 2))
    print("Maximum distance (m):", round(float(np.max(chosen_distances)), 2))
    print("First route position:", int(chosen_positions[0]))
    print("Last route position:", int(chosen_positions[-1]))
    print("Preview saved:", destination)

if __name__ == "__main__":
    run_alignment(Path("data/raw/Student folder/train/00000001"))
