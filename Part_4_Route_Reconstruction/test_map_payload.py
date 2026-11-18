from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Part_4_Route_Reconstruction.hmm_viterbi import DecodingResult
from Part_4_Route_Reconstruction.map_payload import build_ui_payload, continuity_metrics, route_segments_for_map
from Part_4_Route_Reconstruction.road_network import RoadNetwork


class MapPayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.network = RoadNetwork(
            nodes=[(77.0, 12.0), (77.001, 12.0), (77.002, 12.0), (77.003, 12.0), (77.004, 12.0)],
            arc_endpoints=[(0, 1), (1, 2), (3, 4)],
        )

    def test_disconnected_arcs_are_not_joined_on_the_map(self) -> None:
        segments = route_segments_for_map(self.network, [0, 1, None, 2])
        self.assertEqual(len(segments), 2)
        self.assertEqual(len(segments[0]), 3)
        self.assertEqual(len(segments[1]), 2)

    def test_continuity_metrics_follow_directed_arc_endpoints(self) -> None:
        metrics = continuity_metrics(self.network, [0, 1, 2])
        self.assertEqual(metrics.evaluated_transitions, 2)
        self.assertEqual(metrics.directly_connected_transitions, 1)
        self.assertEqual(metrics.disconnected_transitions, 1)
        self.assertEqual(metrics.direct_continuity, 0.5)

    def test_payload_uses_lat_lng_and_keeps_decoder_diagnostics(self) -> None:
        decoding = DecodingResult([0, 1], 1, [], [])
        payload = build_ui_payload(self.network, [(77.0, 12.0, 0.0), (77.001, 12.0, 1.0)], decoding, [0, 1])
        self.assertEqual(payload["gps_points"][0], {"lat": 12.0, "lng": 77.0})
        self.assertEqual(payload["predicted_arc_ids"], [0, 1])
        self.assertEqual(payload["route_continuity"], 1.0)


if __name__ == "__main__":
    unittest.main()
