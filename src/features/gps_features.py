from pathlib import Path
import numpy as np
import pandas as pd

# Load GPS data
DATA_DIR = Path("data/raw/Student folder")
trip_id = "00000001"
trip_dir = DATA_DIR / "train" / trip_id

track = pd.read_csv(
    trip_dir / f"{trip_id}.track",
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude", "timestamp"]
)

# Time gap
track["time_gap"] = track["timestamp"].diff()

# Convert coordinates to radians
lat1 = np.radians(track["latitude"].shift(1))
lat2 = np.radians(track["latitude"])
lon1 = np.radians(track["longitude"].shift(1))
lon2 = np.radians(track["longitude"])

dlat = lat2 - lat1
dlon = lon2 - lon1

# Haversine distance in metres
a = (
    np.sin(dlat / 2) ** 2
    + np.cos(lat1) * np.cos(lat2)
    * np.sin(dlon / 2) ** 2
)

a = a.clip(0, 1)
earth_radius = 6371000

track["distance_m"] = (
    2 * earth_radius * np.arctan2(
        np.sqrt(a), np.sqrt(1 - a)
    )
)

# Speed
track["speed_mps"] = track["distance_m"] / track["time_gap"]
track["speed_kmh"] = track["speed_mps"] * 3.6

# Direction / bearing
y = np.sin(dlon) * np.cos(lat2)

x = (
    np.cos(lat1) * np.sin(lat2)
    - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
)

bearing = np.degrees(np.arctan2(y, x))
track["bearing_deg"] = (bearing + 360) % 360

# Display results
print("\n========== GPS FEATURE ENGINEERING ==========")

print("\nFirst 10 rows:")
print(track.head(10).round(3).to_string(index=False))

print("\nFeature columns:")
print(track.columns.tolist())

print("\nMissing values:")
print(track.isnull().sum())

print("\nFeature summary:")
print(track[[
    "time_gap", "distance_m",
    "speed_mps", "speed_kmh", "bearing_deg"
]].describe().round(3))

# Save output
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f"{trip_id}_features.csv"
track.to_csv(output_file, index=False)

print("\nSaved file:", output_file)
print("GPS feature engineering completed!")
