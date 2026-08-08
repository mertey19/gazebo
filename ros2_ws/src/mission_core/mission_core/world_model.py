"""The mission's internal world model - the "digital twin" data layer.

Holds everything the system has *learned* rather than everything that exists:
targets are only present once a camera decoded their QR code, obstacles only
once the mapper saw them.  Nothing in here is seeded from simulation ground
truth.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .occupancy import OCCUPIED, OccupancyGrid


class TargetStatus(str, Enum):
    """Confidence state of a discovered target."""

    #: Seen at least once, still below ``min_observations``.
    TENTATIVE = "TENTATIVE"
    #: Seen enough times from a consistent position - usable for planning.
    CONFIRMED = "CONFIRMED"
    #: The same payload was observed at two incompatible places.  Unusable:
    #: acting on it could send the rover to the wrong physical station.
    AMBIGUOUS = "AMBIGUOUS"


@dataclass
class TargetObservation:
    """One camera sighting of a QR-coded station, already lifted into ``map``."""

    qr_id: str
    position_map: np.ndarray
    normal_map: np.ndarray
    confidence: float
    stamp: float
    #: Pose of the observing camera, kept so a bad observation can be replayed.
    observer_position_map: np.ndarray
    source: str = "unknown"


@dataclass
class TargetRecord:
    """Fused estimate of one physical station."""

    qr_id: str
    position: np.ndarray
    normal: np.ndarray
    observation_count: int = 0
    weight_sum: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    status: TargetStatus = TargetStatus.TENTATIVE
    #: RMS spread of the contributing observations about the fused position.
    position_spread_m: float = 0.0
    _samples: List[np.ndarray] = field(default_factory=list, repr=False)

    @property
    def confidence(self) -> float:
        """Fused confidence in ``[0, 1]``.

        Rises with the number of consistent observations and falls when they
        disagree spatially.  An ambiguous record is worth nothing.
        """
        if self.status is TargetStatus.AMBIGUOUS:
            return 0.0
        count_term = 1.0 - math.exp(-self.observation_count / 3.0)
        # 0.5 m of scatter across sightings roughly halves the confidence.
        spread_term = 1.0 / (1.0 + 2.0 * self.position_spread_m)
        return float(np.clip(count_term * spread_term, 0.0, 1.0))

    @property
    def is_usable(self) -> bool:
        return self.status is TargetStatus.CONFIRMED


@dataclass
class ObstacleRecord:
    """An axis-aligned obstacle footprint extracted from the occupancy grid."""

    obstacle_id: str
    centre: np.ndarray
    size_xy: np.ndarray
    height_m: float
    cell_count: int

    @property
    def radius(self) -> float:
        """Circumscribed radius of the footprint - handy for quick clearance checks."""
        return float(0.5 * np.linalg.norm(self.size_xy))

    def contains(self, point: Sequence[float], margin: float = 0.0) -> bool:
        delta = np.abs(np.asarray(point, dtype=float)[:2] - self.centre)
        return bool(np.all(delta <= self.size_xy / 2.0 + margin))


class WorldModel:
    """Accumulates target sightings and obstacle geometry.

    Thread-safety is the caller's responsibility; in the ROS deployment a
    single node owns the instance and all access happens on its executor.
    """

    def __init__(
        self,
        *,
        association_radius_m: float = 1.5,
        min_observations: int = 3,
        min_confidence: float = 0.35,
    ) -> None:
        if association_radius_m <= 0.0:
            raise ValueError("association_radius_m must be positive")
        self.association_radius_m = float(association_radius_m)
        self.min_observations = int(min_observations)
        self.min_confidence = float(min_confidence)
        # payload -> list of spatial clusters carrying that payload.  More than
        # one cluster is the duplicate-QR condition.
        self._clusters: Dict[str, List[TargetRecord]] = {}
        self._obstacles: Dict[str, ObstacleRecord] = {}
        self._grid: Optional[OccupancyGrid] = None
        self._rejected_observations = 0

    # -- targets ---------------------------------------------------------
    def add_observation(self, observation: TargetObservation) -> TargetRecord:
        """Fuse one sighting; returns the record it was merged into."""
        position = np.asarray(observation.position_map, dtype=float)[:3]
        if not np.isfinite(position).all():
            self._rejected_observations += 1
            raise ValueError(f"non-finite observation position for {observation.qr_id!r}")
        weight = float(np.clip(observation.confidence, 1e-3, 1.0))

        clusters = self._clusters.setdefault(observation.qr_id, [])
        record = self._nearest_cluster(clusters, position)
        if record is None:
            record = TargetRecord(
                qr_id=observation.qr_id,
                position=position.copy(),
                normal=np.asarray(observation.normal_map, dtype=float).copy(),
                first_seen=observation.stamp,
            )
            clusters.append(record)
        else:
            # Confidence-weighted running mean: a crisp close-up sighting
            # outweighs a blurry one from scan altitude without needing a
            # full covariance filter for an MVP.
            total = record.weight_sum + weight
            record.position = (record.position * record.weight_sum + position * weight) / total
            blended_normal = (
                record.normal * record.weight_sum
                + np.asarray(observation.normal_map, dtype=float) * weight
            )
            norm = float(np.linalg.norm(blended_normal))
            if norm > 1e-9:
                record.normal = blended_normal / norm

        record.weight_sum += weight
        record.observation_count += 1
        record.last_seen = observation.stamp
        record._samples.append(position.copy())
        if len(record._samples) > 1:
            deltas = np.asarray(record._samples) - record.position
            record.position_spread_m = float(np.sqrt(np.mean(np.sum(deltas**2, axis=1))))

        self._refresh_status(observation.qr_id)
        return record

    def _nearest_cluster(
        self, clusters: List[TargetRecord], position: np.ndarray
    ) -> Optional[TargetRecord]:
        best: Optional[TargetRecord] = None
        best_distance = self.association_radius_m
        for cluster in clusters:
            distance = float(np.linalg.norm(cluster.position - position))
            if distance <= best_distance:
                best, best_distance = cluster, distance
        return best

    def _refresh_status(self, qr_id: str) -> None:
        clusters = self._clusters.get(qr_id, [])
        # Only clusters with real support count towards the duplicate test; a
        # single stray reprojection must not veto an otherwise good target.
        supported = [c for c in clusters if c.observation_count >= max(2, self.min_observations - 1)]
        duplicate = len(supported) > 1
        for cluster in clusters:
            if duplicate:
                cluster.status = TargetStatus.AMBIGUOUS
            elif (
                cluster.observation_count >= self.min_observations
                and cluster.confidence >= self.min_confidence
            ):
                cluster.status = TargetStatus.CONFIRMED
            else:
                cluster.status = TargetStatus.TENTATIVE

    def replace_targets(self, records: Iterable[TargetRecord]) -> None:
        """Overwrite all target records with an externally fused set.

        Used by nodes that *consume* the world model over topics instead of
        building it (the mission manager).  Replacing wholesale rather than
        merging is deliberate: the publishing node is the single authority on
        fusion, and a subscriber that also merged would drift away from it.
        """
        self._clusters = {}
        for record in records:
            self._clusters.setdefault(record.qr_id, []).append(record)

    def get_target(self, qr_id: str) -> Optional[TargetRecord]:
        """Best record for ``qr_id``; ``None`` if never seen.

        An ambiguous payload deliberately returns its record so the caller can
        distinguish "unknown" from "known but untrustworthy" and log the right
        failure reason.
        """
        clusters = self._clusters.get(qr_id)
        if not clusters:
            return None
        return max(clusters, key=lambda c: (c.observation_count, c.weight_sum))

    def has_confirmed(self, qr_id: str) -> bool:
        record = self.get_target(qr_id)
        return record is not None and record.is_usable

    def targets(self) -> List[TargetRecord]:
        return [record for clusters in self._clusters.values() for record in clusters]

    def confirmed_targets(self) -> List[TargetRecord]:
        return sorted(
            (t for t in self.targets() if t.is_usable), key=lambda t: t.qr_id
        )

    def __iter__(self) -> Iterator[TargetRecord]:
        return iter(self.targets())

    @property
    def rejected_observations(self) -> int:
        return self._rejected_observations

    # -- obstacles -------------------------------------------------------
    def set_occupancy(
        self,
        grid: OccupancyGrid,
        height_map: Optional[np.ndarray] = None,
        *,
        min_cells: int = 4,
    ) -> List[ObstacleRecord]:
        """Store the latest grid and re-extract discrete obstacle footprints."""
        self._grid = grid
        self._obstacles = {}
        for index, (cells, bounds) in enumerate(_connected_components(grid.data >= OCCUPIED)):
            if len(cells) < min_cells:
                continue
            (min_row, min_col), (max_row, max_col) = bounds
            lower = grid.cell_to_world((min_row, min_col)) - 0.5 * grid.metadata.resolution
            upper = grid.cell_to_world((max_row, max_col)) + 0.5 * grid.metadata.resolution
            centre = 0.5 * (lower + upper)
            size = upper - lower
            height = 0.0
            if height_map is not None:
                rows = np.array([c[0] for c in cells])
                cols = np.array([c[1] for c in cells])
                height = float(np.max(height_map[rows, cols]))
            record = ObstacleRecord(
                obstacle_id=f"obstacle_{index:02d}",
                centre=centre,
                size_xy=size,
                height_m=height,
                cell_count=len(cells),
            )
            self._obstacles[record.obstacle_id] = record
        return self.obstacles()

    def obstacles(self) -> List[ObstacleRecord]:
        return sorted(self._obstacles.values(), key=lambda o: o.obstacle_id)

    @property
    def occupancy(self) -> Optional[OccupancyGrid]:
        return self._grid

    # -- summary ---------------------------------------------------------
    def summary(self) -> Dict[str, object]:
        """Machine-readable snapshot, mirrored onto ``/world_model/status``."""
        return {
            "targets_total": len(self.targets()),
            "targets_confirmed": len(self.confirmed_targets()),
            "targets_ambiguous": sum(
                1 for t in self.targets() if t.status is TargetStatus.AMBIGUOUS
            ),
            "obstacles": len(self._obstacles),
            "map_known_fraction": self._grid.known_fraction if self._grid else 0.0,
            "rejected_observations": self._rejected_observations,
            "qr_ids": sorted({t.qr_id for t in self.targets()}),
        }


def _connected_components(
    mask: np.ndarray,
) -> Iterator[Tuple[List[Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """Label 8-connected ``True`` regions.

    Hand-rolled rather than pulled from SciPy so ``mission_core`` stays
    dependency-light on the robot (numpy + OpenCV only).
    """
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue
            queue = deque([(row, col)])
            visited[row, col] = True
            cells: List[Tuple[int, int]] = []
            min_row = max_row = row
            min_col = max_col = col
            while queue:
                current_row, current_col = queue.popleft()
                cells.append((current_row, current_col))
                min_row, max_row = min(min_row, current_row), max(max_row, current_row)
                min_col, max_col = min(min_col, current_col), max(max_col, current_col)
                for d_row, d_col in neighbours:
                    next_row, next_col = current_row + d_row, current_col + d_col
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and mask[next_row, next_col]
                        and not visited[next_row, next_col]
                    ):
                        visited[next_row, next_col] = True
                        queue.append((next_row, next_col))
            yield cells, ((min_row, min_col), (max_row, max_col))
