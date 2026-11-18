import csv
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Part_4_Route_Reconstruction.candidate_adapter import load_decoding_input


class CandidateAdapterTests(unittest.TestCase):
    def test_preserves_raw_observations_without_candidates(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            track = base / "00000000.track"
            candidates = base / "scored.csv"
            track.write_text("77.0 12.0 0\n77.001 12.0 1\n77.002 12.0 2\n", encoding="utf-8")
            with candidates.open("w", newline="", encoding="utf-8") as destination:
                writer = csv.DictWriter(destination, fieldnames=["trip_id", "observation_id", "candidate_arc_id", "timestamp", "longitude", "latitude", "prediction_score"])
                writer.writeheader()
                writer.writerow({"trip_id": "00000000", "observation_id": 0, "candidate_arc_id": 4, "timestamp": 0, "longitude": 77.0, "latitude": 12.0, "prediction_score": 8.0})
                writer.writerow({"trip_id": "00000000", "observation_id": 0, "candidate_arc_id": 6, "timestamp": 0, "longitude": 77.0, "latitude": 12.0, "prediction_score": 2.0})
                writer.writerow({"trip_id": "00000000", "observation_id": 2, "candidate_arc_id": 5, "timestamp": 2, "longitude": 77.002, "latitude": 12.0, "prediction_score": 0.6})

            result = load_decoding_input(candidates, track, "00000000")
            self.assertEqual(len(result.observations), 3)
            self.assertEqual([len(scores) for scores in result.candidate_sets], [2, 0, 1])
            self.assertAlmostEqual(result.candidate_sets[0][0].probability, 0.8)
            self.assertAlmostEqual(result.candidate_sets[0][1].probability, 0.2)
            self.assertEqual(result.candidate_sets[2][0].arc_id, 5)

    def test_requires_prediction_score(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            track = base / "00000000.track"
            candidates = base / "unscored.csv"
            track.write_text("77.0 12.0 0\n", encoding="utf-8")
            candidates.write_text("trip_id,observation_id,candidate_arc_id,timestamp,longitude,latitude\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prediction_score"):
                load_decoding_input(candidates, track, "00000000")


if __name__ == "__main__":
    unittest.main()
