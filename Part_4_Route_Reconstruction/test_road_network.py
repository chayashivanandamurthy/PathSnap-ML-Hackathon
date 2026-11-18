from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Part_4_Route_Reconstruction.road_network import RoadNetwork, haversine_meters


class RoadNetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        # 0 -> 1 -> 2 is connected; arc 2 begins at 3 and is disconnected.
        self.network = RoadNetwork(
            nodes=[(77.0, 12.0), (77.001, 12.0), (77.002, 12.0), (77.003, 12.0), (77.004, 12.0)],
            arc_endpoints=[(0, 1), (1, 2), (3, 4)],
        )

    def test_directed_connectivity_and_shortest_path(self) -> None:
        self.assertTrue(self.network.directly_connected(0, 1))
        self.assertFalse(self.network.directly_connected(1, 2))
        self.assertIsNotNone(self.network.transition_network_distance_meters(0, 1))
        self.assertIsNone(self.network.transition_network_distance_meters(1, 2))

    def test_map_geometry_uses_lat_lng(self) -> None:
        self.assertEqual(
            self.network.arcs_to_map_geometry([0]),
            [[{"lat": 12.0, "lng": 77.0}, {"lat": 12.0, "lng": 77.001}]],
        )

    def test_haversine_is_positive_for_distinct_points(self) -> None:
        self.assertGreater(haversine_meters((77.0, 12.0), (77.001, 12.0)), 0)


if __name__ == "__main__":
    unittest.main()
