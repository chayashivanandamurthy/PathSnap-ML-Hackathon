"""Trip-local directed road-network loading and routing utilities.

The raw PathSnap IDs are row positions: an arc ID indexes its trip's ``.arcs``
file and node IDs index the corresponding ``.nodes`` file. IDs must therefore
never be reused across trips.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Iterable

EARTH_RADIUS_METERS = 6_371_008.8


def haversine_meters(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Return geodesic distance for ``(longitude, latitude)`` coordinate pairs."""
    longitude_1, latitude_1 = map(radians, first)
    longitude_2, latitude_2 = map(radians, second)
    delta_longitude = longitude_2 - longitude_1
    delta_latitude = latitude_2 - latitude_1
    value = sin(delta_latitude / 2) ** 2 + cos(latitude_1) * cos(latitude_2) * sin(delta_longitude / 2) ** 2
    return 2 * EARTH_RADIUS_METERS * asin(sqrt(value))


@dataclass(frozen=True)
class ArcGeometry:
    """A directed road segment with its endpoint coordinates."""

    arc_id: int
    from_node_id: int
    to_node_id: int
    start: tuple[float, float]
    end: tuple[float, float]
    length_meters: float

    def as_map_coordinates(self) -> list[dict[str, float]]:
        """Return endpoint coordinates in the frontend's ``{lat, lng}`` format."""
        return [
            {"lat": self.start[1], "lng": self.start[0]},
            {"lat": self.end[1], "lng": self.end[0]},
        ]


class RoadNetwork:
    """A directed, trip-local graph constructed from PathSnap ``.nodes`` and ``.arcs`` files."""

    def __init__(self, nodes: list[tuple[float, float]], arc_endpoints: list[tuple[int, int]]) -> None:
        self.nodes = nodes
        self.arcs: list[ArcGeometry] = []
        self.outgoing: dict[int, list[ArcGeometry]] = {}
        self._shortest_path_cache: dict[tuple[int, int], float | None] = {}

        for arc_id, (from_node_id, to_node_id) in enumerate(arc_endpoints):
            if not 0 <= from_node_id < len(nodes) or not 0 <= to_node_id < len(nodes):
                raise ValueError(f"Arc {arc_id} references node IDs outside the nodes table.")
            start, end = nodes[from_node_id], nodes[to_node_id]
            arc = ArcGeometry(arc_id, from_node_id, to_node_id, start, end, haversine_meters(start, end))
            self.arcs.append(arc)
            self.outgoing.setdefault(from_node_id, []).append(arc)

    @classmethod
    def from_trip_directory(cls, trip_directory: str | Path, trip_id: str | None = None) -> "RoadNetwork":
        """Load a network from a directory containing one trip's raw files."""
        directory = Path(trip_directory)
        prefix = trip_id or directory.name
        return cls.from_files(directory / f"{prefix}.nodes", directory / f"{prefix}.arcs")

    @classmethod
    def from_files(cls, nodes_path: str | Path, arcs_path: str | Path) -> "RoadNetwork":
        """Load headerless, whitespace-separated PathSnap node and arc files."""
        nodes = _read_rows(nodes_path, expected_columns=2, cast=float)
        arc_values = _read_rows(arcs_path, expected_columns=2, cast=int)
        return cls(nodes=[(row[0], row[1]) for row in nodes], arc_endpoints=[(row[0], row[1]) for row in arc_values])

    def get_arc(self, arc_id: int) -> ArcGeometry:
        """Fetch an arc while producing a useful error for invalid classifier output."""
        if not 0 <= arc_id < len(self.arcs):
            raise KeyError(f"Arc ID {arc_id} is invalid for this trip; valid range is 0..{len(self.arcs) - 1}.")
        return self.arcs[arc_id]

    def arcs_to_map_geometry(self, arc_ids: Iterable[int]) -> list[list[dict[str, float]]]:
        """Convert selected directed arcs to the geometry expected by the evaluator UI."""
        return [self.get_arc(arc_id).as_map_coordinates() for arc_id in arc_ids]

    def directly_connected(self, previous_arc_id: int, next_arc_id: int) -> bool:
        """Return whether the end of one arc is exactly the start of the next."""
        return self.get_arc(previous_arc_id).to_node_id == self.get_arc(next_arc_id).from_node_id

    def shortest_path_distance_meters(self, from_node_id: int, to_node_id: int) -> float | None:
        """Return directed road distance, or ``None`` when the nodes are disconnected.

        This is intentionally directed because the EDA confirms that the raw arc
        data represents directed road segments. Results are cached per trip.
        """
        if not 0 <= from_node_id < len(self.nodes) or not 0 <= to_node_id < len(self.nodes):
            raise KeyError("Node ID is outside this trip's node table.")
        cache_key = (from_node_id, to_node_id)
        if cache_key in self._shortest_path_cache:
            return self._shortest_path_cache[cache_key]
        if from_node_id == to_node_id:
            self._shortest_path_cache[cache_key] = 0.0
            return 0.0

        frontier: list[tuple[float, int]] = [(0.0, from_node_id)]
        distances = {from_node_id: 0.0}
        while frontier:
            distance, node_id = heappop(frontier)
            if distance != distances[node_id]:
                continue
            if node_id == to_node_id:
                self._shortest_path_cache[cache_key] = distance
                return distance
            for arc in self.outgoing.get(node_id, []):
                candidate = distance + arc.length_meters
                if candidate < distances.get(arc.to_node_id, float("inf")):
                    distances[arc.to_node_id] = candidate
                    heappush(frontier, (candidate, arc.to_node_id))

        self._shortest_path_cache[cache_key] = None
        return None

    def transition_network_distance_meters(self, previous_arc_id: int, next_arc_id: int) -> float | None:
        """Road distance between two candidate arcs, including their endpoints.

        The HMM will compare this value with the observed GPS displacement. A
        ``None`` result represents an invalid directed transition and should be
        assigned a strong penalty rather than silently joined on the map.
        """
        previous = self.get_arc(previous_arc_id)
        following = self.get_arc(next_arc_id)
        return self.shortest_path_distance_meters(previous.to_node_id, following.from_node_id)


def _read_rows(path: str | Path, expected_columns: int, cast: type[float] | type[int]) -> list[tuple]:
    file_path = Path(path)
    rows: list[tuple] = []
    with file_path.open("r", encoding="utf-8") as source:
        for row_number, line in enumerate(source, start=1):
            values = line.split()
            if len(values) != expected_columns:
                raise ValueError(f"{file_path.name}, row {row_number}: expected {expected_columns} values, found {len(values)}.")
            try:
                rows.append(tuple(cast(value) for value in values))
            except ValueError as error:
                raise ValueError(f"{file_path.name}, row {row_number}: could not parse values.") from error
    return rows
