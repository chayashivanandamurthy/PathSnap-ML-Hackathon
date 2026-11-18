from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/Student folder")
trip_id = "00000001"
trip_dir = DATA_DIR / "train" / trip_id

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
    names=["from_node_id", "to_node_id"]
)

route = pd.read_csv(
    trip_dir / f"{trip_id}.route",
    sep=r"\s+", header=None,
    names=["arc_id"]
)

time_diff = track["timestamp"].diff().dropna()

report = f"""
PATHSNAP - PART 1
DATA PREPROCESSING REPORT
================================

Sample Trip ID: {trip_id}

1. DATASET SIZES
GPS observations: {len(track)}
Road network nodes: {len(nodes)}
Road network arcs: {len(arcs)}
Ground-truth route arcs: {len(route)}

2. MISSING VALUES
GPS: {track.isnull().sum().sum()}
Nodes: {nodes.isnull().sum().sum()}
Arcs: {arcs.isnull().sum().sum()}
Route: {route.isnull().sum().sum()}

3. DUPLICATES
GPS rows: {track.duplicated().sum()}
Node rows: {nodes.duplicated().sum()}
Arc rows: {arcs.duplicated().sum()}

4. TIMESTAMP ANALYSIS
Minimum gap: {time_diff.min()} seconds
Maximum gap: {time_diff.max()} seconds
One-second intervals: {(time_diff == 1).sum()}
Other intervals: {(time_diff != 1).sum()}

5. COORDINATE RANGE
Longitude: {track.longitude.min()} to {track.longitude.max()}
Latitude: {track.latitude.min()} to {track.latitude.max()}

6. ID VALIDATION
Invalid node IDs: {
    ((~arcs.from_node_id.between(0, len(nodes)-1)).sum()
    + (~arcs.to_node_id.between(0, len(nodes)-1)).sum())
}
Invalid route arc IDs: {
    (~route.arc_id.between(0, len(arcs)-1)).sum()
}

================================
END OF REPORT
"""

Path("reports").mkdir(exist_ok=True)
Path("reports/data_quality_report.txt").write_text(report, encoding="utf-8")

print(report)
print("Report saved successfully!")
