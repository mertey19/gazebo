"""2D occupancy representation and the mapper that builds it from camera evidence.

The grid is byte-for-byte compatible with ``nav_msgs/msg/OccupancyGrid``:
``-1`` unknown, ``0`` free, ``100`` occupied, row-major with row ``0`` at the
grid origin.  Keeping the same convention here means the ROS node never has to
re-interpret the data, only copy it.

The evidence itself comes from :mod:`mission_core.vision_mapping`, which
recovers obstacle bases and free floor from single RGB frames.  This module is
sensor-agnostic: it consumes ground-plane points, not rays or ranges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

UNKNOWN = -1
FREE = 0
OCCUPIED = 100


@dataclass(frozen=True)
class GridMetadata:
    """Geometry of an occupancy grid in the ``map`` frame."""

    resolution: float
    width: int
    height: int
    origin_x: float
    origin_y: float

    def __post_init__(self) -> None:
        if self.resolution <= 0.0:
            raise ValueError("grid resolution must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("grid dimensions must be positive")


class OccupancyGrid:
    """A 2D occupancy grid with world <-> cell conversion and collision queries."""

    def __init__(self, metadata: GridMetadata, data: np.ndarray | None = None) -> None:
        self.metadata = metadata
        if data is None:
            self.data = np.full((metadata.height, metadata.width), UNKNOWN, dtype=np.int8)
        else:
            data = np.asarray(data, dtype=np.int8)
            if data.shape != (metadata.height, metadata.width):
                raise ValueError(
                    f"data shape {data.shape} does not match metadata "
                    f"{(metadata.height, metadata.width)}"
                )
            self.data = data

    # -- construction ----------------------------------------------------
    @classmethod
    def covering(
        cls, min_xy: Sequence[float], max_xy: Sequence[float], resolution: float
    ) -> "OccupancyGrid":
        """Allocate an all-unknown grid covering the given world rectangle."""
        min_x, min_y = float(min_xy[0]), float(min_xy[1])
        max_x, max_y = float(max_xy[0]), float(max_xy[1])
        if max_x <= min_x or max_y <= min_y:
            raise ValueError("grid extent must be positive in both axes")
        width = int(np.ceil((max_x - min_x) / resolution))
        height = int(np.ceil((max_y - min_y) / resolution))
        return cls(GridMetadata(resolution, width, height, min_x, min_y))

    def copy(self) -> "OccupancyGrid":
        return OccupancyGrid(self.metadata, self.data.copy())

    # -- coordinate conversion -------------------------------------------
    def world_to_cell(self, point: Sequence[float]) -> Tuple[int, int]:
        """Map a world ``(x, y)`` to ``(row, col)``; may fall outside the grid."""
        meta = self.metadata
        col = int(np.floor((float(point[0]) - meta.origin_x) / meta.resolution))
        row = int(np.floor((float(point[1]) - meta.origin_y) / meta.resolution))
        return row, col

    def cell_to_world(self, cell: Sequence[int]) -> np.ndarray:
        """Centre of ``(row, col)`` in world coordinates."""
        meta = self.metadata
        x = meta.origin_x + (float(cell[1]) + 0.5) * meta.resolution
        y = meta.origin_y + (float(cell[0]) + 0.5) * meta.resolution
        return np.array([x, y], dtype=float)

    def contains_cell(self, cell: Sequence[int]) -> bool:
        row, col = int(cell[0]), int(cell[1])
        return 0 <= row < self.metadata.height and 0 <= col < self.metadata.width

    def contains_point(self, point: Sequence[float]) -> bool:
        return self.contains_cell(self.world_to_cell(point))

    # -- queries ---------------------------------------------------------
    def value_at(self, point: Sequence[float]) -> int:
        cell = self.world_to_cell(point)
        if not self.contains_cell(cell):
            return UNKNOWN
        return int(self.data[cell[0], cell[1]])

    def is_occupied_cell(self, cell: Sequence[int], *, unknown_is_occupied: bool) -> bool:
        if not self.contains_cell(cell):
            return True  # outside the mapped world is never traversable
        value = int(self.data[int(cell[0]), int(cell[1])])
        if value == UNKNOWN:
            return unknown_is_occupied
        return value >= OCCUPIED

    def is_free_point(self, point: Sequence[float], *, unknown_is_occupied: bool = False) -> bool:
        return not self.is_occupied_cell(
            self.world_to_cell(point), unknown_is_occupied=unknown_is_occupied
        )

    # -- editing ---------------------------------------------------------
    def set_cell(self, cell: Sequence[int], value: int) -> None:
        if self.contains_cell(cell):
            self.data[int(cell[0]), int(cell[1])] = np.int8(value)

    def mark_box(
        self,
        centre: Sequence[float],
        size_xy: Sequence[float],
        yaw: float = 0.0,
        value: int = OCCUPIED,
    ) -> None:
        """Stamp an oriented rectangle into the grid (used by tests and priors)."""
        half_x, half_y = float(size_xy[0]) / 2.0, float(size_xy[1]) / 2.0
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        rows, cols = np.mgrid[0 : self.metadata.height, 0 : self.metadata.width]
        world_x = self.metadata.origin_x + (cols + 0.5) * self.metadata.resolution
        world_y = self.metadata.origin_y + (rows + 0.5) * self.metadata.resolution
        dx = world_x - float(centre[0])
        dy = world_y - float(centre[1])
        local_x = cos_y * dx + sin_y * dy
        local_y = -sin_y * dx + cos_y * dy
        mask = (np.abs(local_x) <= half_x) & (np.abs(local_y) <= half_y)
        self.data[mask] = np.int8(value)

    # -- derived grids ---------------------------------------------------
    def inflate(self, radius_m: float, *, unknown_is_obstacle: bool = False) -> "OccupancyGrid":
        """Grow every obstacle by ``radius_m`` so the robot can be a point.

        The inflation radius must already include the robot radius *and* the
        desired safety margin; this function does not add either implicitly.
        """
        if radius_m < 0.0:
            raise ValueError("inflation radius must not be negative")
        inflated = self.copy()
        resolution = self.metadata.resolution
        # An occupied cell is a square of side `resolution`, not a point. A
        # free cell whose *centre* is `d` away can have obstacle material as
        # close as `d - resolution*sqrt(2)/2`, so the kernel must reach that
        # much further than the nominal radius. Measuring centre-to-centre
        # instead silently delivers less clearance than was configured - with
        # 0.20 m cells and a 0.55 m radius it applies only 0.40 m.
        reach_m = radius_m + resolution * math.sqrt(2.0) / 2.0
        cells = int(np.ceil(reach_m / resolution))
        if cells == 0:
            return inflated

        obstacle_mask = self.data >= OCCUPIED
        if unknown_is_obstacle:
            obstacle_mask |= self.data == UNKNOWN
        if not obstacle_mask.any():
            return inflated

        # Circular structuring element: a square kernel would over-inflate
        # diagonals by sqrt(2) and can wall off legitimately open corridors.
        offsets = [
            (dr, dc)
            for dr in range(-cells, cells + 1)
            for dc in range(-cells, cells + 1)
            if (dr * dr + dc * dc) * resolution**2 <= reach_m**2 + 1e-12
        ]
        grown = np.zeros_like(obstacle_mask)
        height, width = obstacle_mask.shape
        for dr, dc in offsets:
            src_rows = slice(max(0, -dr), height - max(0, dr))
            dst_rows = slice(max(0, dr), height - max(0, -dr))
            src_cols = slice(max(0, -dc), width - max(0, dc))
            dst_cols = slice(max(0, dc), width - max(0, -dc))
            grown[dst_rows, dst_cols] |= obstacle_mask[src_rows, src_cols]
        inflated.data[grown] = np.int8(OCCUPIED)
        return inflated

    # -- ray casting -----------------------------------------------------
    def cells_on_segment(
        self, start: Sequence[float], end: Sequence[float]
    ) -> List[Tuple[int, int]]:
        """Every cell touched by the world-space segment ``start`` -> ``end``.

        This is a *supercover* traversal (Amanatides-Woo): unlike Bresenham it
        never squeezes diagonally between two occupied cells, which is exactly
        the failure mode that lets a "collision-free" path clip an obstacle
        corner.
        """
        meta = self.metadata
        start = np.asarray(start, dtype=float)[:2]
        end = np.asarray(end, dtype=float)[:2]
        row, col = self.world_to_cell(start)
        end_row, end_col = self.world_to_cell(end)
        cells = [(row, col)]
        if (row, col) == (end_row, end_col):
            return cells

        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < 1e-12:
            return cells
        direction = delta / length

        step_col = 1 if direction[0] > 0 else (-1 if direction[0] < 0 else 0)
        step_row = 1 if direction[1] > 0 else (-1 if direction[1] < 0 else 0)

        def _boundary(index: int, step: int, origin: float) -> float:
            edge = origin + (index + (1 if step > 0 else 0)) * meta.resolution
            return edge

        t_max_x = (
            (_boundary(col, step_col, meta.origin_x) - start[0]) / direction[0]
            if step_col != 0
            else np.inf
        )
        t_max_y = (
            (_boundary(row, step_row, meta.origin_y) - start[1]) / direction[1]
            if step_row != 0
            else np.inf
        )
        t_delta_x = abs(meta.resolution / direction[0]) if step_col != 0 else np.inf
        t_delta_y = abs(meta.resolution / direction[1]) if step_row != 0 else np.inf

        # Bounded by the Manhattan cell distance plus slack: guarantees
        # termination even if floating point nudges us off the exact end cell.
        max_steps = abs(end_row - row) + abs(end_col - col) + 4
        for _ in range(max_steps):
            if t_max_x < t_max_y:
                col += step_col
                t_max_x += t_delta_x
            else:
                row += step_row
                t_max_y += t_delta_y
            cells.append((row, col))
            if (row, col) == (end_row, end_col):
                break
        return cells

    def segment_is_free(
        self,
        start: Sequence[float],
        end: Sequence[float],
        *,
        unknown_is_occupied: bool = False,
    ) -> bool:
        return all(
            not self.is_occupied_cell(cell, unknown_is_occupied=unknown_is_occupied)
            for cell in self.cells_on_segment(start, end)
        )

    def path_is_free(
        self,
        path: Iterable[Sequence[float]],
        *,
        unknown_is_occupied: bool = False,
    ) -> bool:
        points = [np.asarray(p, dtype=float)[:2] for p in path]
        if len(points) < 2:
            return bool(points) and self.is_free_point(
                points[0], unknown_is_occupied=unknown_is_occupied
            )
        return all(
            self.segment_is_free(a, b, unknown_is_occupied=unknown_is_occupied)
            for a, b in zip(points, points[1:])
        )

    # -- ROS interop -----------------------------------------------------
    def to_row_major_list(self) -> List[int]:
        return [int(v) for v in self.data.reshape(-1)]

    @property
    def occupied_fraction(self) -> float:
        return float(np.count_nonzero(self.data >= OCCUPIED)) / float(self.data.size)

    @property
    def known_fraction(self) -> float:
        return float(np.count_nonzero(self.data != UNKNOWN)) / float(self.data.size)


class OccupancyMapper:
    """Builds an :class:`OccupancyGrid` from monocular ground evidence.

    Each frame contributes two kinds of ``map``-frame point: the floor
    positions where an obstacle was seen to *stand* (hits, carrying the height
    measured for that obstacle), and the floor positions that were seen to be
    empty (misses).  Counting the two separately rather than overwriting keeps
    one mis-segmented frame from carving a hole in an obstacle, and makes a
    contact that only ever appears from a single viewpoint fail the ratio test.

    Cells that were never projected into stay ``UNKNOWN``.  That matters more
    with a camera than it did with a lidar: what a camera cannot see behind an
    obstacle is exactly the obstacle's own interior, and ``UNKNOWN`` is what
    stops the planner routing through it.
    """

    def __init__(
        self,
        metadata: GridMetadata,
        *,
        min_hits: int = 2,
        hit_ratio_threshold: float = 0.35,
    ) -> None:
        if min_hits < 1:
            raise ValueError("min_hits must be at least 1")
        if not 0.0 < hit_ratio_threshold <= 1.0:
            raise ValueError("hit_ratio_threshold must lie in (0, 1]")
        self.metadata = metadata
        self.min_hits = int(min_hits)
        self.hit_ratio_threshold = float(hit_ratio_threshold)
        self._hits = np.zeros((metadata.height, metadata.width), dtype=np.int32)
        self._misses = np.zeros((metadata.height, metadata.width), dtype=np.int32)
        # The rover's own camera is direct evidence that a route changed after
        # the drone mapped it. Keep that evidence separate from historical
        # ground misses so a new obstacle is not diluted by the old free ratio.
        self._runtime_obstacle_hits = np.zeros(
            (metadata.height, metadata.width), dtype=np.int32
        )
        self._max_height = np.zeros((metadata.height, metadata.width), dtype=np.float32)

    def _cells(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(rows, cols, inside_mask)`` for ``(N, 3)`` map-frame points."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if pts.size == 0:
            empty = np.zeros(0, dtype=int)
            return empty, empty, np.zeros(0, dtype=bool)
        if pts.shape[1] != 3:
            raise ValueError(f"expected (N, 3) points, got {pts.shape}")
        meta = self.metadata
        finite = np.isfinite(pts).all(axis=1)
        # Non-finite coordinates are parked outside the grid rather than left
        # to propagate NaN through the floor division.
        safe = np.where(finite[:, None], pts[:, :2], np.array([meta.origin_x - 1.0, 0.0]))
        cols = np.floor((safe[:, 0] - meta.origin_x) / meta.resolution)
        rows = np.floor((safe[:, 1] - meta.origin_y) / meta.resolution)
        inside = (
            finite
            & (rows >= 0)
            & (rows < meta.height)
            & (cols >= 0)
            & (cols < meta.width)
        )
        rows = np.where(inside, rows, 0).astype(int)
        cols = np.where(inside, cols, 0).astype(int)
        return rows, cols, inside

    def integrate_ground_observation(
        self, contacts_map: np.ndarray, free_map: np.ndarray
    ) -> int:
        """Fold one frame of monocular evidence in; returns the cells touched.

        ``contacts_map`` is ``(N, 3)`` with the measured obstacle height in
        ``z``; ``free_map`` is ``(M, 3)`` of floor samples.  Free evidence that
        lands in a cell this same frame saw an obstacle base in is dropped: at
        the foot of an obstacle the two are separated by centimetres, well
        inside one cell, and letting the far denser floor samples vote there
        would erase every boundary the camera just measured.
        """
        hit_rows, hit_cols, hit_inside = self._cells(contacts_map)
        hit_rows, hit_cols = hit_rows[hit_inside], hit_cols[hit_inside]
        if hit_rows.size:
            heights = np.atleast_2d(np.asarray(contacts_map, dtype=float))[hit_inside, 2]
            np.add.at(self._hits, (hit_rows, hit_cols), 1)
            np.maximum.at(
                self._max_height, (hit_rows, hit_cols), heights.astype(np.float32)
            )

        free_rows, free_cols, free_inside = self._cells(free_map)
        free_rows, free_cols = free_rows[free_inside], free_cols[free_inside]
        if free_rows.size:
            occupied_now = np.zeros(self._hits.shape, dtype=bool)
            occupied_now[hit_rows, hit_cols] = True
            keep = ~occupied_now[free_rows, free_cols]
            np.add.at(self._misses, (free_rows[keep], free_cols[keep]), 1)
            return int(hit_rows.size + np.count_nonzero(keep))
        return int(hit_rows.size)

    def integrate_runtime_obstacle_hits(self, points_map: np.ndarray) -> int:
        """Add obstacle contacts seen by a vehicle's own local camera.

        Two consistent hits (or the configured ``min_hits``) override earlier
        ground/free observations. Evidence is intentionally sticky; removing
        moving obstacles requires the decay model documented as future work.
        """
        rows, cols, inside = self._cells(points_map)
        rows, cols = rows[inside], cols[inside]
        if rows.size == 0:
            return 0
        heights = np.atleast_2d(np.asarray(points_map, dtype=float))[inside, 2]
        np.add.at(self._runtime_obstacle_hits, (rows, cols), 1)
        np.maximum.at(self._max_height, (rows, cols), heights.astype(np.float32))
        return int(rows.size)

    def build(self) -> OccupancyGrid:
        """Threshold the accumulated evidence into an occupancy grid."""
        grid = OccupancyGrid(self.metadata)
        total = self._hits + self._misses
        observed = total > 0
        ratio = np.zeros_like(self._hits, dtype=float)
        np.divide(self._hits, np.maximum(total, 1), out=ratio)
        occupied = observed & (self._hits >= self.min_hits) & (ratio >= self.hit_ratio_threshold)
        runtime_occupied = self._runtime_obstacle_hits >= self.min_hits
        observed |= self._runtime_obstacle_hits > 0
        occupied |= runtime_occupied
        grid.data[observed] = np.int8(FREE)
        grid.data[occupied] = np.int8(OCCUPIED)
        return grid

    @property
    def height_map(self) -> np.ndarray:
        """Maximum observed obstacle height per cell (0 where none seen)."""
        return self._max_height.copy()

    @property
    def observation_count(self) -> int:
        return int(self._hits.sum() + self._misses.sum())
