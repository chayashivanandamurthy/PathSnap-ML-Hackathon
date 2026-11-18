from pathlib import Path
import pandas as pd

# Dataset location
DATA_DIR = Path("data/raw/Student folder")

# Select one training trip
trip_id = "00000001"
trip_dir = DATA_DIR / "train" / trip_id

# Load all four files
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
    header=None,
    names=["arc_id"]
)

# Display results
print("\n========== PATHSNAP DATA INSPECTION ==========")

print("\n1. TRACK DATA")
print(track.head())
print("Shape:", track.shape)
print("Missing values:\n", track.isnull().sum())

print("\n2. NODES DATA")
print(nodes.head())
print("Shape:", nodes.shape)
print("Missing values:\n", nodes.isnull().sum())

print("\n3. ARCS DATA")
print(arcs.head())
print("Shape:", arcs.shape)
print("Missing values:\n", arcs.isnull().sum())

print("\n4. ROUTE DATA")
print(route.head())
print("Shape:", route.shape)
print("Missing values:\n", route.isnull().sum())

print("\n========== INSPECTION COMPLETED ==========")
