"""HMM/Viterbi route decoding over classifier-produced road-arc candidates."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Sequence

from .road_network import RoadNetwork, haversine_meters


@dataclass(frozen=True)
class CandidateScore:
    """One model-scored candidate road arc for a GPS observation."""

    arc_id: int
    probability: float


@dataclass(frozen=True)
class HMMConfig:
    """Decoder parameters, to be tuned on permitted development data only."""

    gap_split_seconds: float = 5.0
    transition_scale_meters: float = 100.0
    invalid_transition_log_penalty: float = -30.0
    probability_floor: float = 1e-12


@dataclass(frozen=True)
class DecodingResult:
    """Decoded arcs aligned to observations; ``None`` marks an undecodable gap."""

    arc_ids: list[int | None]
    segment_count: int
    gap_break_indices: list[int]
    candidate_miss_indices: list[int]

    @property
    def predicted_count(self) -> int:
        return sum(arc_id is not None for arc_id in self.arc_ids)


class ViterbiDecoder:
    """Decode the most likely graph-consistent sequence of candidate road arcs."""

    def __init__(self, network: RoadNetwork, config: HMMConfig | None = None) -> None:
        self.network = network
        self.config = config or HMMConfig()

    def decode(
        self,
        observations: Sequence[tuple[float, float, float]],
        candidate_sets: Sequence[Sequence[CandidateScore]],
    ) -> DecodingResult:
        """Decode candidates while splitting across large GPS gaps or empty sets.

        ``observations`` use raw PathSnap order: ``(longitude, latitude,
        timestamp_seconds)``. Candidate arrays must have one entry per
        observation and preserve the classifier's arc-ID correspondence.
        """
        if len(observations) != len(candidate_sets):
            raise ValueError("observations and candidate_sets must have identical lengths.")

        result: list[int | None] = [None] * len(observations)
        breaks: list[int] = []
        misses: list[int] = []
        segment_start: int | None = None
        segment_count = 0

        def close_segment(end_index: int) -> None:
            nonlocal segment_start, segment_count
            if segment_start is None:
                return
            decoded = self._decode_contiguous_segment(observations[segment_start:end_index], candidate_sets[segment_start:end_index])
            result[segment_start:end_index] = decoded
            segment_start = None
            segment_count += 1

        for index, candidates in enumerate(candidate_sets):
            if not candidates:
                close_segment(index)
                misses.append(index)
                continue
            if index > 0 and segment_start is not None:
                elapsed = observations[index][2] - observations[index - 1][2]
                if elapsed > self.config.gap_split_seconds:
                    close_segment(index)
                    breaks.append(index)
            if segment_start is None:
                segment_start = index
        close_segment(len(observations))
        return DecodingResult(result, segment_count, breaks, misses)

    def _decode_contiguous_segment(
        self,
        observations: Sequence[tuple[float, float, float]],
        candidate_sets: Sequence[Sequence[CandidateScore]],
    ) -> list[int]:
        candidates = [_normalise_candidates(scores, self.config.probability_floor) for scores in candidate_sets]
        previous_scores = {candidate.arc_id: log(candidate.probability) for candidate in candidates[0]}
        back_pointers: list[dict[int, int]] = []

        for index in range(1, len(candidates)):
            observed_distance = haversine_meters(observations[index - 1][:2], observations[index][:2])
            current_scores: dict[int, float] = {}
            pointers: dict[int, int] = {}
            for current in candidates[index]:
                best_previous_arc, best_score = max(
                    (
                        (previous_arc, previous_score + self._transition_log_probability(previous_arc, current.arc_id, observed_distance))
                        for previous_arc, previous_score in previous_scores.items()
                    ),
                    key=lambda item: item[1],
                )
                current_scores[current.arc_id] = best_score + log(current.probability)
                pointers[current.arc_id] = best_previous_arc
            previous_scores = current_scores
            back_pointers.append(pointers)

        final_arc = max(previous_scores, key=previous_scores.get)
        decoded = [final_arc]
        for pointers in reversed(back_pointers):
            decoded.append(pointers[decoded[-1]])
        return list(reversed(decoded))

    def _transition_log_probability(self, previous_arc_id: int, next_arc_id: int, observed_distance: float) -> float:
        network_distance = self.network.transition_network_distance_meters(previous_arc_id, next_arc_id)
        if network_distance is None:
            return self.config.invalid_transition_log_penalty
        mismatch = abs(network_distance - observed_distance)
        return -mismatch / self.config.transition_scale_meters


def _normalise_candidates(scores: Sequence[CandidateScore], floor: float) -> list[CandidateScore]:
    """Validate probabilities and retain the strongest score for duplicate arc IDs."""
    best_by_arc: dict[int, float] = {}
    for score in scores:
        if not 0.0 <= score.probability <= 1.0:
            raise ValueError(f"Candidate probability for arc {score.arc_id} is outside [0, 1].")
        best_by_arc[score.arc_id] = max(best_by_arc.get(score.arc_id, 0.0), score.probability)
    if not best_by_arc:
        raise ValueError("A contiguous Viterbi segment cannot contain an empty candidate set.")
    return [CandidateScore(arc_id, max(probability, floor)) for arc_id, probability in best_by_arc.items()]
