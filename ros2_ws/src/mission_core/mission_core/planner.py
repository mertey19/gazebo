"""Obstacle-aware 2D path planning for the rover.

A* on the inflated occupancy grid, followed by line-of-sight shortcutting.  A*
was chosen over a sampling planner because the mission area is small, the grid
is coarse, and a deterministic, complete, and easily-verified planner is worth
far more here than asymptotic optimality.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .errors import FailureReason, PlanningError
from .occupancy import OccupancyGrid

_SQRT2 = math.sqrt(2.0)
#: 8-connected neighbourhood with its step costs.
_NEIGHBOURS: Tuple[Tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, _SQRT2),
    (-1, 1, _SQRT2),
    (1, -1, _SQRT2),
    (1, 1, _SQRT2),
)


@dataclass(frozen=True)
class PlannedPath:
    """An ordered, collision-free sequence of ``(x, y, yaw)`` poses."""

    poses: np.ndarray
    length_m: float
    expanded_nodes: int
    inflation_radius_m: float

    def __len__(self) -> int:
        return int(self.poses.shape[0])

    @property
    def xy(self) -> np.ndarray:
        return self.poses[:, :2]

    @property
    def start(self) -> np.ndarray:
        return self.poses[0]

    @property
    def goal(self) -> np.ndarray:
        return self.poses[-1]


class AStarPlanner:
    """Grid A* with configurable robot footprint and safety margin."""

    def __init__(
        self,
        *,
        rover_radius_m: float = 0.30,
        safety_margin_m: float = 0.25,
        allow_unknown: bool = True,
        heuristic_weight: float = 1.0,
        max_start_snap_m: float = 1.5,
        shortcut: bool = True,
    ) -> None:
        if rover_radius_m < 0.0 or safety_margin_m < 0.0:
            raise ValueError("rover radius and safety margin must not be negative")
        if heuristic_weight < 1.0:
            raise ValueError("heuristic_weight below 1.0 makes A* slower, not better")
        self.rover_radius_m = float(rover_radius_m)
        self.safety_margin_m = float(safety_margin_m)
        #: Treating unmapped space as drivable lets the mission start before the
        #: drone has covered every cell; obstacles are still respected because
        #: they are *observed* occupied, not merely unknown.
        self.allow_unknown = bool(allow_unknown)
        self.heuristic_weight = float(heuristic_weight)
        self.max_start_snap_m = float(max_start_snap_m)
        self.shortcut = bool(shortcut)

    @property
    def inflation_radius_m(self) -> float:
        return self.rover_radius_m + self.safety_margin_m

    def inflate(self, grid: OccupancyGrid) -> OccupancyGrid:
        return grid.inflate(self.inflation_radius_m)

    # -- main entry point -------------------------------------------------
    def plan(
        self,
        grid: OccupancyGrid,
        start_xy: Sequence[float],
        goal_xy: Sequence[float],
        goal_yaw: Optional[float] = None,
        *,
        pre_inflated: bool = False,
    ) -> PlannedPath:
        """Plan from ``start_xy`` to ``goal_xy``; raises :class:`PlanningError`."""
        inflated = grid if pre_inflated else self.inflate(grid)
        unknown_blocks = not self.allow_unknown

        start_cell, start_snapped = self._snap_to_free(inflated, start_xy, unknown_blocks, "start")
        goal_cell, goal_snapped = self._snap_to_free(inflated, goal_xy, unknown_blocks, "goal")

        if start_cell == goal_cell:
            pose = np.array(
                [
                    [*inflated.cell_to_world(start_cell), float(goal_yaw or 0.0)],
                ]
            )
            return PlannedPath(pose, 0.0, 0, self.inflation_radius_m)

        cells, expanded = self._search(inflated, start_cell, goal_cell, unknown_blocks)
        if cells is None:
            raise PlanningError(
                f"A* exhausted {expanded} nodes without reaching the goal cell "
                f"{goal_cell} from {start_cell}"
            )

        points = [inflated.cell_to_world(cell) for cell in cells]
        # Use the exact requested coordinates rather than a cell centre - but
        # only for an endpoint that was already in free space. An endpoint that
        # had to be snapped is, by definition, inside the inflation zone;
        # writing it back would hand out a path whose first or last segment
        # fails the very collision check this planner exists to guarantee. The
        # rover simply drives from its true pose to the first free waypoint.
        if not start_snapped:
            points[0] = np.asarray(start_xy, dtype=float)[:2]
        if not goal_snapped:
            points[-1] = np.asarray(goal_xy, dtype=float)[:2]
        if self.shortcut:
            points = self._shortcut(inflated, points, unknown_blocks)

        poses = self._assign_headings(points, goal_yaw)
        length = float(np.sum(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)))

        # Defence in depth: never hand out a path we have not re-checked
        # against the same grid the planner used.
        if not inflated.path_is_free(poses[:, :2], unknown_is_occupied=unknown_blocks):
            raise PlanningError(
                "internal error: planned path failed post-hoc collision check",
                FailureReason.PATH_INTERSECTS_OBSTACLE,
            )
        return PlannedPath(poses, length, expanded, self.inflation_radius_m)

    # -- approach pose ----------------------------------------------------
    def _has_line_of_sight(
        self,
        raw_grid: OccupancyGrid,
        observer: np.ndarray,
        target: np.ndarray,
        target_clearance_m: float,
    ) -> bool:
        """Can a camera at ``observer`` see the station at ``target``?

        Checked on the *raw* grid, because inflation models where the wheels
        may go, not what the lens can see.  Occupied cells within
        ``target_clearance_m`` of the target are skipped: those cells *are* the
        station, and a station always occludes its own centre.
        """
        for cell in raw_grid.cells_on_segment(observer, target):
            if not raw_grid.is_occupied_cell(cell, unknown_is_occupied=False):
                continue
            if float(np.linalg.norm(raw_grid.cell_to_world(cell) - target)) <= target_clearance_m:
                continue
            return False
        return True

    def select_approach_pose(
        self,
        raw_grid: OccupancyGrid,
        target_xy: Sequence[float],
        rover_xy: Sequence[float],
        *,
        approach_distance_m: float,
        target_clearance_m: float,
        inflated_grid: Optional[OccupancyGrid] = None,
        distance_tolerance_m: float = 0.6,
        samples: int = 72,
    ) -> np.ndarray:
        """Pick a collision-free standoff pose facing the target.

        The rover must stop *near* the station, not on top of it: the station
        is itself a mapped obstacle.  Candidates are sampled on rings around
        the target and scored by distance from the rover, so the chosen side is
        the one the rover is already coming from - which is also the QR face it
        will end up looking at.  Rings slightly nearer and further than the
        nominal standoff are tried too, so a partly-blocked station degrades
        the approach instead of failing the mission.
        """
        if approach_distance_m <= 0.0:
            raise ValueError("approach_distance_m must be positive")
        if target_clearance_m <= 0.0:
            raise ValueError("target_clearance_m must be positive")
        inflated = inflated_grid if inflated_grid is not None else self.inflate(raw_grid)
        unknown_blocks = not self.allow_unknown
        target = np.asarray(target_xy, dtype=float)[:2]
        rover = np.asarray(rover_xy, dtype=float)[:2]

        step = max(self.rover_radius_m, raw_grid.metadata.resolution)
        radii = [approach_distance_m]
        offset = step
        while offset <= distance_tolerance_m + 1e-9:
            radii.extend([approach_distance_m + offset, approach_distance_m - offset])
            offset += step
        radii = [r for r in radii if r > target_clearance_m]

        best_pose: Optional[np.ndarray] = None
        best_cost = math.inf
        for radius in radii:
            for index in range(samples):
                angle = 2.0 * math.pi * index / samples
                candidate = target + radius * np.array([math.cos(angle), math.sin(angle)])
                if not inflated.is_free_point(candidate, unknown_is_occupied=unknown_blocks):
                    continue
                if not self._has_line_of_sight(raw_grid, candidate, target, target_clearance_m):
                    continue
                # Prefer the nominal standoff: deviating from it changes how
                # large the code appears to the verification camera.
                cost = float(np.linalg.norm(candidate - rover)) + 2.0 * abs(
                    radius - approach_distance_m
                )
                if cost < best_cost:
                    best_cost = cost
                    # Face the station so the forward camera frames the QR code.
                    yaw = math.atan2(target[1] - candidate[1], target[0] - candidate[0])
                    best_pose = np.array([candidate[0], candidate[1], yaw])
            if best_pose is not None and abs(radius - approach_distance_m) < 1e-9:
                break  # the nominal ring worked; no need to consider the others
        if best_pose is None:
            raise PlanningError(
                f"no collision-free approach pose within "
                f"{approach_distance_m:.2f}+/-{distance_tolerance_m:.2f} m of target "
                f"({target[0]:.2f}, {target[1]:.2f}) that also has line of sight to it"
            )
        return best_pose

    # -- internals --------------------------------------------------------
    def _snap_to_free(
        self,
        grid: OccupancyGrid,
        point: Sequence[float],
        unknown_blocks: bool,
        label: str,
    ) -> Tuple[Tuple[int, int], bool]:
        """Nearest free cell to ``point``; returns ``(cell, was_snapped)``.

        Inflation can swallow a pose that is physically fine (the rover parked
        close to a wall), so a bounded search outwards is more useful than an
        immediate failure - but the bound keeps it from silently relocating the
        goal to somewhere unrelated, and the flag lets the caller know the
        returned cell is not exactly what was asked for.
        """
        cell = grid.world_to_cell(point)
        if not grid.contains_cell(cell):
            raise PlanningError(
                f"{label} {np.asarray(point, dtype=float)[:2].tolist()} lies outside the map"
            )
        if not grid.is_occupied_cell(cell, unknown_is_occupied=unknown_blocks):
            return cell, False

        max_cells = int(math.ceil(self.max_start_snap_m / grid.metadata.resolution))
        for radius in range(1, max_cells + 1):
            best: Optional[Tuple[float, Tuple[int, int]]] = None
            for d_row in range(-radius, radius + 1):
                for d_col in range(-radius, radius + 1):
                    if max(abs(d_row), abs(d_col)) != radius:
                        continue
                    candidate = (cell[0] + d_row, cell[1] + d_col)
                    if grid.is_occupied_cell(candidate, unknown_is_occupied=unknown_blocks):
                        continue
                    distance = math.hypot(d_row, d_col)
                    if best is None or distance < best[0]:
                        best = (distance, candidate)
            if best is not None:
                return best[1], True
        raise PlanningError(
            f"{label} {np.asarray(point, dtype=float)[:2].tolist()} is inside an obstacle and no "
            f"free cell was found within {self.max_start_snap_m:.2f} m"
        )

    @staticmethod
    def _octile(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        d_row = abs(a[0] - b[0])
        d_col = abs(a[1] - b[1])
        return float(max(d_row, d_col) + (_SQRT2 - 1.0) * min(d_row, d_col))

    def _search(
        self,
        grid: OccupancyGrid,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        unknown_blocks: bool,
    ) -> Tuple[Optional[List[Tuple[int, int]]], int]:
        open_heap: List[Tuple[float, int, Tuple[int, int]]] = []
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        closed: set[Tuple[int, int]] = set()
        counter = 0
        heapq.heappush(open_heap, (self._octile(start, goal), counter, start))
        expanded = 0

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            expanded += 1
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path, expanded

            current_g = g_score[current]
            for d_row, d_col, step in _NEIGHBOURS:
                neighbour = (current[0] + d_row, current[1] + d_col)
                if neighbour in closed:
                    continue
                if grid.is_occupied_cell(neighbour, unknown_is_occupied=unknown_blocks):
                    continue
                if d_row != 0 and d_col != 0:
                    # Forbid cutting a corner between two blocked orthogonals -
                    # geometrically the robot would clip the obstacle.
                    if grid.is_occupied_cell(
                        (current[0] + d_row, current[1]), unknown_is_occupied=unknown_blocks
                    ) or grid.is_occupied_cell(
                        (current[0], current[1] + d_col), unknown_is_occupied=unknown_blocks
                    ):
                        continue
                tentative = current_g + step
                if tentative < g_score.get(neighbour, math.inf):
                    g_score[neighbour] = tentative
                    came_from[neighbour] = current
                    counter += 1
                    priority = tentative + self.heuristic_weight * self._octile(neighbour, goal)
                    heapq.heappush(open_heap, (priority, counter, neighbour))
        return None, expanded

    @staticmethod
    def _shortcut(
        grid: OccupancyGrid, points: List[np.ndarray], unknown_blocks: bool
    ) -> List[np.ndarray]:
        """Greedy string-pulling: keep the furthest visible waypoint each time.

        Turns the 8-connected staircase into a handful of straight legs, which
        the pure-pursuit controller tracks far more smoothly.
        """
        if len(points) <= 2:
            return points
        result = [points[0]]
        index = 0
        while index < len(points) - 1:
            furthest = index + 1
            for candidate in range(len(points) - 1, index, -1):
                if grid.segment_is_free(
                    points[index], points[candidate], unknown_is_occupied=unknown_blocks
                ):
                    furthest = candidate
                    break
            result.append(points[furthest])
            index = furthest
        return result

    @staticmethod
    def _assign_headings(points: List[np.ndarray], goal_yaw: Optional[float]) -> np.ndarray:
        poses = np.zeros((len(points), 3), dtype=float)
        poses[:, :2] = np.asarray(points, dtype=float)[:, :2]
        for index in range(len(points) - 1):
            delta = poses[index + 1, :2] - poses[index, :2]
            poses[index, 2] = math.atan2(delta[1], delta[0])
        poses[-1, 2] = float(goal_yaw) if goal_yaw is not None else poses[-2, 2]
        return poses


def path_intersects_obstacles(
    path_xy: np.ndarray,
    grid: OccupancyGrid,
    *,
    clearance_m: float = 0.0,
    unknown_is_occupied: bool = False,
) -> bool:
    """Independent checker used by the validator and the tests.

    Deliberately re-derives the inflated grid instead of trusting the planner's
    copy: this is the check that decides whether a mission may be called a
    success, so it must not share state with the code it is auditing.
    """
    audit_grid = grid.inflate(clearance_m) if clearance_m > 0.0 else grid
    return not audit_grid.path_is_free(
        np.asarray(path_xy, dtype=float)[:, :2], unknown_is_occupied=unknown_is_occupied
    )
