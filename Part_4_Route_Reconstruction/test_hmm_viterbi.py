from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Part_4_Route_Reconstruction.hmm_viterbi import CandidateScore, HMMConfig, ViterbiDecoder
from Part_4_Route_Reconstruction.road_network import RoadNetwork


class ViterbiDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.network = RoadNetwork(
            nodes=[(77.0, 12.0), (77.001, 12.0), (77.002, 12.0), (77.003, 12.0), (77.004, 12.0)],
            # Arc 0 -> arc 1 is connected; arc 2 is a disconnected high-score distractor.
            arc_endpoints=[(0, 1), (1, 2), (3, 4)],
        )
        self.decoder = ViterbiDecoder(self.network, HMMConfig(transition_scale_meters=20.0))

    def test_decoder_prefers_a_connected_sequence(self) -> None:
        observations = [(77.0, 12.0, 0.0), (77.001, 12.0, 1.0)]
        candidates = [
            [CandidateScore(0, 0.6), CandidateScore(2, 0.4)],
            [CandidateScore(1, 0.49), CandidateScore(2, 0.51)],
        ]
        result = self.decoder.decode(observations, candidates)
        self.assertEqual(result.arc_ids, [0, 1])
        self.assertEqual(result.segment_count, 1)

    def test_large_gaps_and_missing_candidates_split_sequences(self) -> None:
        observations = [(77.0, 12.0, 0.0), (77.001, 12.0, 10.0), (77.002, 12.0, 11.0), (77.003, 12.0, 12.0)]
        candidates = [[CandidateScore(0, 1.0)], [CandidateScore(1, 1.0)], [], [CandidateScore(2, 1.0)]]
        result = self.decoder.decode(observations, candidates)
        self.assertEqual(result.arc_ids, [0, 1, None, 2])
        self.assertEqual(result.gap_break_indices, [1])
        self.assertEqual(result.candidate_miss_indices, [2])
        self.assertEqual(result.segment_count, 3)


if __name__ == "__main__":
    unittest.main()
