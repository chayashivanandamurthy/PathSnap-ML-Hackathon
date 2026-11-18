"""One-call reconstruction pipeline for FastAPI integration.

The API layer supplies a model-scored candidate CSV and selects a trip. This
module loads the corresponding raw trip network and track, runs HMM/Viterbi,
and returns only prediction-side map data. Reference labels are never returned.
"""

from __future__ import annotations

from pathlib import Path

from .candidate_adapter import decode_scored_trip
from .hmm_viterbi import HMMConfig, ViterbiDecoder
from .map_payload import build_ui_payload
from .road_network import RoadNetwork


def reconstruct_trip(
    data_root: str | Path,
    partition: str,
    trip_id: str,
    scored_candidates_csv: str | Path,
    config: HMMConfig | None = None,
    probability_column: str = "prediction_score",
) -> dict:
    """Decode one trip and return the prediction-side response for the UI.

    ``data_root`` contains the raw ``train`` and ``student_test`` folders.
    ``scored_candidates_csv`` is Member 1's candidate CSV after Member 3 adds
    the requested score column. Scores are normalised candidate-by-candidate
    within an observation. No ``.route`` file is read here, keeping
    reference labels outside the inference response.
    """
    if partition not in {"train", "student_test"}:
        raise ValueError("partition must be 'train' or 'student_test'.")

    trip_directory = Path(data_root) / partition / trip_id
    track_path = trip_directory / f"{trip_id}.track"
    network = RoadNetwork.from_trip_directory(trip_directory, trip_id)
    decoder = ViterbiDecoder(network, config)
    decoding_input, decoding_result = decode_scored_trip(
        decoder,
        scored_candidates_csv=scored_candidates_csv,
        track_path=track_path,
        trip_id=trip_id,
        probability_column=probability_column,
    )
    context_arc_ids = [candidate.arc_id for candidate_set in decoding_input.candidate_sets for candidate in candidate_set]
    payload = build_ui_payload(network, decoding_input.observations, decoding_result, context_arc_ids)
    payload.update(
        {
            "trip_id": trip_id,
            "decoder_note": "Route decoded from model-scored candidates using directed HMM/Viterbi transitions.",
        }
    )
    return payload
