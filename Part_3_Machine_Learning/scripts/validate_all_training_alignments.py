from pathlib import Path
import pandas as pd
from align_route_sequence import run_alignment

TRAIN_DIR = Path("data/raw/Student folder/train")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

folders = sorted(
    [p for p in TRAIN_DIR.iterdir() if p.is_dir()],
    key=lambda p: p.name
)

print("Training trip folders found:", len(folders))

summary = []

for index, folder in enumerate(folders, start=1):
    trip_id = folder.name
    print(f"\n[{index}/{len(folders)}] Processing trip {trip_id}")

    try:
        run_alignment(folder)

        result_path = REPORT_DIR / f"{trip_id}_alignment_preview.csv"
        result = pd.read_csv(result_path)

        positions = result["reference_route_position"].to_numpy()
        distances = result["gps_to_reference_distance_m"].to_numpy()

        summary.append({
            "trip_id": trip_id,
            "status": "SUCCESS",
            "observations": len(result),
            "backward_jumps": int((positions[1:] < positions[:-1]).sum()),
            "median_distance_m": round(float(pd.Series(distances).median()), 2),
            "p95_distance_m": round(float(pd.Series(distances).quantile(0.95)), 2),
            "max_distance_m": round(float(pd.Series(distances).max()), 2),
            "error": ""
        })

    except Exception as exc:
        print("FAILED:", str(exc))

        summary.append({
            "trip_id": trip_id,
            "status": "FAILED",
            "observations": 0,
            "backward_jumps": None,
            "median_distance_m": None,
            "p95_distance_m": None,
            "max_distance_m": None,
            "error": str(exc)
        })

summary_df = pd.DataFrame(summary)
summary_path = REPORT_DIR / "training_alignment_validation_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\n========== BATCH VALIDATION SUMMARY ==========")
print("Total trips:", len(summary_df))
print("Successful:", int((summary_df.status == "SUCCESS").sum()))
print("Failed:", int((summary_df.status == "FAILED").sum()))

if len(summary_df):
    print("\nFailure reasons:")
    print(
        summary_df[summary_df.status == "FAILED"][
            ["trip_id", "error"]
        ].to_string(index=False)
    )

print("\nSummary saved:", summary_path)
