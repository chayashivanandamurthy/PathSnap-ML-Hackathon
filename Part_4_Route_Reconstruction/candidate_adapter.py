"""Adapter between candidate generation/model scoring and HMM decoding.

Member 1's candidate CSV supplies one row per (GPS observation, candidate arc).
Member 3 must add a calibrated ``model_probability`` column without changing
``trip_id``, ``observation_id`` or ``candidate_arc_id``. This adapter groups
those rows back into one candidate set per raw GPS observation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import isclose, isfinite
from pathlib import Path
from typing import Sequence

from .hmm_viterbi import CandidateScore, DecodingResult, ViterbiDecoder

BASE_CANDIDATE_COLUMNS = {
    "trip_id",
    "observation_id",
    "candidate_arc_id",
    "timestamp",
    "longitude",
    "latitude",
}


@dataclass(frozen=True)
class DecodingInput:
    """A complete trip's raw observations and model-scored candidate sets."""

    trip_id: str
    observations: list[tuple[float, float, float]]
    candidate_sets: list[list[CandidateScore]]


def load_decoding_input(
    scored_candidates_csv: str | Path,
    track_path: str | Path,
    trip_id: str,
    probability_column: str = "model_probability",
) -> DecodingInput:
    """Load one trip from model-scored candidates and its original ``.track``.

    The raw track is authoritative for observation count and order. This means
    an observation omitted by candidate generation remains visible to the HMM
    as an empty candidate set instead of shifting every later prediction.
    """
    grouped = _read_scored_candidates(scored_candidates_csv, trip_id, probability_column)
    observations = _read_track(track_path)
    candidate_sets: list[list[CandidateScore]] = [[] for _ in observations]

    for observation_id, (row_observation, candidates) in grouped.items():
        if not 0 <= observation_id < len(observations):
            raise ValueError(f"Candidate observation_id {observation_id} is outside trip {trip_id}'s track range.")
        _validate_observation_alignment(observations[observation_id], row_observation, observation_id)
        candidate_sets[observation_id] = candidates

    return DecodingInput(trip_id=trip_id, observations=observations, candidate_sets=candidate_sets)


def decode_scored_trip(
    decoder: ViterbiDecoder,
    scored_candidates_csv: str | Path,
    track_path: str | Path,
    trip_id: str,
    probability_column: str = "model_probability",
) -> tuple[DecodingInput, DecodingResult]:
    """Load and decode one trip using the exact candidate IDs scored by a model."""
    decoding_input = load_decoding_input(scored_candidates_csv, track_path, trip_id, probability_column)
    return decoding_input, decoder.decode(decoding_input.observations, decoding_input.candidate_sets)


def _read_scored_candidates(
    csv_path: str | Path,
    trip_id: str,
    probability_column: str,
) -> dict[int, tuple[tuple[float, float, float], list[CandidateScore]]]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = set(reader.fieldnames or [])
        missing = (BASE_CANDIDATE_COLUMNS | {probability_column}) - fieldnames
        if missing:
            raise ValueError(f"Scored candidate CSV is missing columns: {', '.join(sorted(missing))}.")

        grouped: dict[int, tuple[tuple[float, float, float], list[CandidateScore]]] = {}
        for row_number, row in enumerate(reader, start=2):
            if row["trip_id"] != trip_id:
                continue
            try:
                observation_id = int(row["observation_id"])
                observation = (float(row["longitude"]), float(row["latitude"]), float(row["timestamp"]))
                candidate = CandidateScore(int(row["candidate_arc_id"]), float(row[probability_column]))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Could not parse scored candidate CSV row {row_number}.") from error
            if not isfinite(candidate.probability) or not 0.0 <= candidate.probability <= 1.0:
                raise ValueError(f"Row {row_number} has an invalid {probability_column} value.")

            known = grouped.get(observation_id)
            if known is None:
                grouped[observation_id] = (observation, [candidate])
            else:
                known_observation, candidates = known
                _validate_observation_alignment(known_observation, observation, observation_id)
                candidates.append(candidate)
    return grouped


def _read_track(track_path: str | Path) -> list[tuple[float, float, float]]:
    observations: list[tuple[float, float, float]] = []
    with Path(track_path).open("r", encoding="utf-8") as source:
        for row_number, line in enumerate(source, start=1):
            values = line.split()
            if len(values) != 3:
                raise ValueError(f"{Path(track_path).name}, row {row_number}: expected longitude, latitude and timestamp.")
            try:
                observations.append((float(values[0]), float(values[1]), float(values[2])))
            except ValueError as error:
                raise ValueError(f"{Path(track_path).name}, row {row_number}: could not parse values.") from error
    return observations


def _validate_observation_alignment(
    expected: Sequence[float], actual: Sequence[float], observation_id: int,
) -> None:
    if not all(isclose(left, right, rel_tol=0.0, abs_tol=1e-8) for left, right in zip(expected, actual)):
        raise ValueError(f"Candidate metadata does not match raw track observation {observation_id}.")
