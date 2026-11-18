import pandas as pd

file = "data/processed/00000001_features.csv"
df = pd.read_csv(file)

speed = df["speed_kmh"].dropna()

print("\n========== SPEED ANALYSIS ==========")

print("\nSpeed statistics:")
print(speed.describe(percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]).round(2))

print("\nSpeed categories:")
print("Below 10 km/h:", (speed < 10).sum())
print("10–60 km/h:", ((speed >= 10) & (speed <= 60)).sum())
print("60–100 km/h:", ((speed > 60) & (speed <= 100)).sum())
print("Above 100 km/h:", (speed > 100).sum())

print("\nTop 10 highest speed observations:")
print(
    df.nlargest(10, "speed_kmh")[
        ["timestamp", "time_gap", "distance_m", "speed_kmh"]
    ].round(2).to_string(index=False)
)

print("\n========== ANALYSIS COMPLETED ==========")
