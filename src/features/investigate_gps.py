from pathlib import Path
import pandas as pd

trip_id = "00000001"
trip_dir = Path("data/raw/Student folder/train") / trip_id

# Load original GPS data
track = pd.read_csv(
    trip_dir / f"{trip_id}.track",
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude", "timestamp"]
)

# Load engineered features
features = pd.read_csv(
    f"data/processed/{trip_id}_features.csv"
)

# Add previous GPS coordinates for comparison
track["previous_longitude"] = track["longitude"].shift(1)
track["previous_latitude"] = track["latitude"].shift(1)

# Add calculated speed
track["distance_m"] = features["distance_m"]
track["speed_kmh"] = features["speed_kmh"]
track["time_gap"] = features["time_gap"]

print("\n========== HIGH SPEED GPS INVESTIGATION ==========")

high_speed = track.nlargest(10, "speed_kmh")

print(
    high_speed[
        [
            "timestamp",
            "previous_longitude",
            "previous_latitude",
            "longitude",
            "latitude",
            "time_gap",
            "distance_m",
            "speed_kmh"
        ]
    ].round(6).to_string(index=False)
)

print("\nDistance statistics (metres):")
print(track["distance_m"].describe().round(2))

print("\nSpeed by timestamp gap:")
print(
    track.groupby("time_gap")["speed_kmh"]
    .agg(["count", "mean", "max"])
    .round(2)
)

print("\n========== INVESTIGATION COMPLETED ==========")
