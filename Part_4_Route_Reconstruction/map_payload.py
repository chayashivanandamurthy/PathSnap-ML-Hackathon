"""Convert HMM decoding output into gap-safe geometry for the evaluator UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .hmm_viterbi import DecodingResult
from .road_network import RoadNetwork


@dataclass(frozen=True)
class ContinuityMetrics:
    evaluated_transitions: int
    directly_connected_transitions: int
    disconnected_transitions: int

    @property
    def direct_continuity(self) -> float | None:
        if not self.evaluated_transitions:
            return None
        return self.directly_connected_transitions / self.evaluated_transitions


def route_segments_for_map(network: RoadNetwork, decoded_arc_ids: Sequence[int | None]) -> list[list[dict[str, float]]]:
    """Return polyline segments without drawing false joins across discontinuities.

    A segment extends only when adjacent selected arcs share the same endpoint.
    Missing candidates and non-directly-connected arcs always begin a new visual
    segment. This lets the UI reveal route breaks rather than concealing them.
    """
    segments: list[list[dict[str, float]]] = []
    active_segment: list[dict[str, float]] = []
    previous_arc_id: int | None = None

    for arc_id in decoded_arc_ids:
        if arc_id is None:
            if active_segment:
                segments.append(active_segment)
            active_segment = []
            previous_arc_id = None
            continue

        arc = network.get_arc(arc_id)
        if previous_arc_id is None or not network.directly_connected(previous_arc_id, arc_id):
            if active_segment:
                segments.append(active_segment)
            active_segment = arc.as_map_coordinates()
        else:
            active_segment.append(arc.as_map_coordinates()[1])
        previous_arc_id = arc_id

    if active_segment:
        segments.append(active_segment)
    return segments


def road_context_for_map(network: RoadNetwork, arc_ids: Iterable[int], maximum_arcs: int = 400) -> list[list[dict[str, float]]]:
    """Return de-duplicated nearby arc geometry while limiting browser payload size."""
    unique_arc_ids = list(dict.fromkeys(arc_ids))[:maximum_arcs]
    return network.arcs_to_map_geometry(unique_arc_ids)


def continuity_metrics(network: RoadNetwork, decoded_arc_ids: Sequence[int | None]) -> ContinuityMetrics:
    """Measure direct arc-end-to-arc-start continuity, matching the EDA definition."""
    evaluated = connected = disconnected = 0
    previous_arc_id: int | None = None
    for arc_id in decoded_arc_ids:
        if arc_id is None:
            previous_arc_id = None
            continue
        if previous_arc_id is not None:
            evaluated += 1
            if network.directly_connected(previous_arc_id, arc_id):
                connected += 1
            else:
                disconnected += 1
        previous_arc_id = arc_id
    return ContinuityMetrics(evaluated, connected, disconnected)


def build_ui_payload(
    network: RoadNetwork,
    observations: Sequence[tuple[float, float, float]],
    decoding: DecodingResult,
    candidate_arc_ids: Iterable[int],
) -> dict:
    """Build the route/map portion of the API response using real decoder output only."""
    metrics = continuity_metrics(network, decoding.arc_ids)
    return {
        "gps_points": [{"lat": latitude, "lng": longitude} for longitude, latitude, _ in observations],
        "predicted_route_segments": route_segments_for_map(network, decoding.arc_ids),
        "road_network": road_context_for_map(network, candidate_arc_ids),
        "predicted_arc_ids": decoding.arc_ids,
        "route_continuity": metrics.direct_continuity,
        "decoder_diagnostics": {
            "segment_count": decoding.segment_count,
            "gap_break_indices": decoding.gap_break_indices,
            "candidate_miss_indices": decoding.candidate_miss_indices,
            "disconnected_transitions": metrics.disconnected_transitions,
        },
    }
