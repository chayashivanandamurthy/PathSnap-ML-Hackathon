import csv
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Part_4_Route_Reconstruction.pipeline import reconstruct_trip


class PipelineTests(unittest.TestCase):
    def test_builds_prediction_side_payload_without_reading_reference_route(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trip = root / "student_test" / "00000000"
            trip.mkdir(parents=True)
            (trip / "00000000.track").write_text("77.0 12.0 0\n77.001 12.0 1\n", encoding="utf-8")
            (trip / "00000000.nodes").write_text("77.0 12.0\n77.001 12.0\n77.002 12.0\n", encoding="utf-8")
            (trip / "00000000.arcs").write_text("0 1\n1 2\n", encoding="utf-8")
            # A malformed route proves inference does not consume ground-truth labels.
            (trip / "00000000.route").write_text("not-used\n", encoding="utf-8")

            scored_candidates = root / "scored_candidates.csv"
            with scored_candidates.open("w", newline="", encoding="utf-8") as destination:
                writer = csv.DictWriter(destination, fieldnames=["trip_id", "observation_id", "candidate_arc_id", "timestamp", "longitude", "latitude", "prediction_score"])
                writer.writeheader()
                writer.writerows([
                    {"trip_id": "00000000", "observation_id": 0, "candidate_arc_id": 0, "timestamp": 0, "longitude": 77.0, "latitude": 12.0, "prediction_score": 0.9},
                    {"trip_id": "00000000", "observation_id": 1, "candidate_arc_id": 1, "timestamp": 1, "longitude": 77.001, "latitude": 12.0, "prediction_score": 0.9},
                ])

            payload = reconstruct_trip(root, "student_test", "00000000", scored_candidates)
            self.assertEqual(payload["trip_id"], "00000000")
            self.assertEqual(payload["predicted_arc_ids"], [0, 1])
            self.assertEqual(payload["route_continuity"], 1.0)
            self.assertNotIn("reference_route", payload)


if __name__ == "__main__":
    unittest.main()
