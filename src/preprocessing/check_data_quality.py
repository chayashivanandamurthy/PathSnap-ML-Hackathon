from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/Student folder")
trip_id = "00000001"
trip_dir = DATA_DIR / "train" / trip_id

track = pd.read_csv(
    trip_dir / f"{trip_id}.track",
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude", "timestamp"]
)

nodes = pd.read_csv(
    trip_dir / f"{trip_id}.nodes",
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude"]
)

arcs = pd.read_csv(
    trip_dir / f"{trip_id}.arcs",
    sep=r"\s+",
    header=None,
    names=["from_node_id", "to_node_id"]
)

route = pd.read_csv(
    trip_dir / f"{trip_id}.route",
    sep=r"\s+",
    header=None,
    names=["arc_id"]
)

print("\n========== DATA QUALITY CHECKS ==========")

print("\n1. GPS COORDINATE RANGE")
print("Longitude range:", track["longitude"].min(), "to", track["longitude"].max())
print("Latitude range:", track["latitude"].min(), "to", track["latitude"].max())

print("\n2. DUPLICATE RECORDS")
print("Duplicate GPS rows:", track.duplicated().sum())
print("Duplicate node rows:", nodes.duplicated().sum())
print("Duplicate arc rows:", arcs.duplicated().sum())

print("\n3. TIMESTAMP CHECK")
time_diff = track["timestamp"].diff().dropna()
print("Minimum time gap:", time_diff.min(), "seconds")
print("Maximum time gap:", time_diff.max(), "seconds")
print("Intervals equal to 1 second:", (time_diff == 1).sum())
print("Intervals not equal to 1 second:", (time_diff != 1).sum())

print("\n4. ROAD NETWORK ID CHECK")
invalid_from = (~arcs["from_node_id"].between(0, len(nodes) - 1)).sum()
invalid_to = (~arcs["to_node_id"].between(0, len(nodes) - 1)).sum()
print("Invalid from_node IDs:", invalid_from)
print("Invalid to_node IDs:", invalid_to)

print("\n5. ROUTE ARC ID CHECK")
invalid_route = (~route["arc_id"].between(0, len(arcs) - 1)).sum()
print("Invalid route arc IDs:", invalid_route)

print("\n========== QUALITY CHECKS COMPLETED ==========")
