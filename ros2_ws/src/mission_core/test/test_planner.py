"""TEST 3 - the planner never produces a path that intersects an obstacle."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mission_core.errors import FailureReason, PlanningError
from mission_core.occupancy import FREE, OCCUPIED, GridMetadata, OccupancyGrid
from mission_core.planner import AStarPlanner, path_intersects_obstacles


def make_planner(**overrides) -> AStarPlanner:
    kwargs = dict(rover_radius_m=0.30, safety_margin_m=0.25, allow_unknown=True)
    kwargs.update(overrides)
    return AStarPlanner(**kwargs)


def sample_path_densely(path_xy: np.ndarray, step: float = 0.05) -> np.ndarray:
    """Resample a polyline so collision checks cannot slip between vertices."""
    points = []
    for start, end in zip(path_xy[:-1], path_xy[1:]):
        distance = float(np.linalg.norm(end - start))
        count = max(2, int(distance / step))
        for t in np.linspace(0.0, 1.0, count):
            points.append(start + t * (end - start))
    points.append(path_xy[-1])
    return np.asarray(points)


def test_straight_line_when_nothing_is_in_the_way(empty_grid) -> None:
    path = make_planner().plan(empty_grid, (-8.0, -8.0), (8.0, 8.0))
    assert len(path) >= 2
    assert np.allclose(path.start[:2], (-8.0, -8.0))
    assert np.allclose(path.goal[:2], (8.0, 8.0))
    straight = float(np.linalg.norm(np.array([16.0, 16.0])))
    # Shortcutting should recover very nearly the straight line.
    assert path.length_m == pytest.approx(straight, rel=0.02)


def test_path_detours_around_a_wall_and_never_touches_it(walled_grid) -> None:
    """The core anti-cheat check: a path through an obstacle is a failure."""
    planner = make_planner()
    path = planner.plan(walled_grid, (-6.0, 0.0), (6.0, 0.0))

    dense = sample_path_densely(path.xy)
    assert not path_intersects_obstacles(
        dense, walled_grid, clearance_m=planner.inflation_radius_m
    ), "the planned path passes through the inflated obstacle"

    # And it must actually be a detour, not a straight line that got lucky.
    assert path.length_m > 12.0
    assert float(np.max(np.abs(path.xy[:, 1]))) > 2.0, "path did not route around the wall"


def test_every_waypoint_clears_the_obstacle_by_the_full_clearance(walled_grid) -> None:
    planner = make_planner()
    path = planner.plan(walled_grid, (-6.0, -3.0), (6.0, 3.0))
    inflated = planner.inflate(walled_grid)
    for point in sample_path_densely(path.xy):
        assert inflated.is_free_point(point), (
            f"waypoint {point} lies inside the inflated obstacle"
        )


def test_larger_safety_margin_pushes_the_path_further_out(walled_grid) -> None:
    narrow = make_planner(safety_margin_m=0.05).plan(walled_grid, (-6.0, 0.0), (6.0, 0.0))
    wide = make_planner(safety_margin_m=0.60).plan(walled_grid, (-6.0, 0.0), (6.0, 0.0))
    assert wide.length_m > narrow.length_m


def test_impossible_path_is_rejected_rather_than_approximated(empty_grid) -> None:
    """A fully enclosed goal must raise, not return a best-effort path."""
    grid = empty_grid
    # Box the goal in on all four sides.
    grid.mark_box((5.0, 0.0), (0.6, 6.0))
    grid.mark_box((9.0, 0.0), (0.6, 6.0))
    grid.mark_box((7.0, 3.0), (4.6, 0.6))
    grid.mark_box((7.0, -3.0), (4.6, 0.6))

    with pytest.raises(PlanningError) as excinfo:
        make_planner().plan(grid, (-8.0, 0.0), (7.0, 0.0))
    assert excinfo.value.reason is FailureReason.NO_VALID_PATH


def test_goal_outside_the_map_is_rejected(empty_grid) -> None:
    with pytest.raises(PlanningError, match="outside the map"):
        make_planner().plan(empty_grid, (0.0, 0.0), (50.0, 0.0))


def test_start_inside_inflation_is_snapped_not_failed(empty_grid) -> None:
    """Parking close to a wall must not make the rover unplannable.

    The path then begins at the nearest *free* point rather than at the rover
    itself: emitting the true start would produce a first segment inside the
    inflation zone, which is exactly what the planner guarantees never to do.
    """
    grid = empty_grid
    grid.mark_box((0.0, 0.0), (2.0, 2.0))
    planner = make_planner()
    # 1.15 m from the box centre: outside the box, inside its inflation.
    path = planner.plan(grid, (1.15, 0.0), (6.0, 0.0))

    assert len(path) >= 2
    assert float(np.linalg.norm(path.start[:2] - np.array([1.15, 0.0]))) <= (
        planner.max_start_snap_m
    )
    assert planner.inflate(grid).is_free_point(path.start[:2])
    for point in sample_path_densely(path.xy):
        assert planner.inflate(grid).is_free_point(point)


def test_unknown_space_is_traversable_only_when_configured() -> None:
    grid = OccupancyGrid(GridMetadata(0.2, 100, 100, -10.0, -10.0))  # all unknown
    assert make_planner(allow_unknown=True).plan(grid, (-8.0, 0.0), (8.0, 0.0)).length_m > 0
    with pytest.raises(PlanningError):
        make_planner(allow_unknown=False).plan(grid, (-8.0, 0.0), (8.0, 0.0))


def test_diagonal_moves_cannot_cut_an_obstacle_corner() -> None:
    """8-connected search must not squeeze between two blocked orthogonals."""
    grid = OccupancyGrid(GridMetadata(1.0, 5, 5, 0.0, 0.0))
    grid.data[:] = FREE
    grid.data[1, 2] = OCCUPIED
    grid.data[2, 1] = OCCUPIED
    planner = make_planner(rover_radius_m=0.0, safety_margin_m=0.0)
    path = planner.plan(grid, (1.5, 1.5), (2.5, 2.5), pre_inflated=True)
    for point in sample_path_densely(path.xy):
        assert grid.is_free_point(point), f"{point} clipped an obstacle corner"


# ---------------------------------------------------------------------------
# Approach pose selection
# ---------------------------------------------------------------------------

def test_approach_pose_stands_off_and_faces_the_target(empty_grid) -> None:
    grid = empty_grid
    grid.mark_box((7.0, -5.0), (0.8, 0.8))  # the station is itself an obstacle
    planner = make_planner()
    pose = planner.select_approach_pose(
        grid, (7.0, -5.0), (-8.0, -8.0), approach_distance_m=1.8, target_clearance_m=0.7
    )

    distance = float(np.linalg.norm(pose[:2] - np.array([7.0, -5.0])))
    assert distance == pytest.approx(1.8, abs=0.05)
    # It must face the station, or the camera cannot read the code.
    bearing = math.atan2(-5.0 - pose[1], 7.0 - pose[0])
    assert abs(math.atan2(math.sin(bearing - pose[2]), math.cos(bearing - pose[2]))) < 1e-6
    # And it should be on the side the rover approaches from.
    assert pose[0] < 7.0


def test_approach_pose_avoids_a_blocked_side(empty_grid) -> None:
    grid = empty_grid
    grid.mark_box((7.0, -5.0), (0.8, 0.8))
    # Wall the station off from the rover's natural approach direction.
    grid.mark_box((4.5, -5.0), (0.6, 8.0))
    planner = make_planner()
    pose = planner.select_approach_pose(
        grid, (7.0, -5.0), (-8.0, -8.0), approach_distance_m=1.8, target_clearance_m=0.7
    )
    assert grid.inflate(planner.inflation_radius_m).is_free_point(pose[:2])
    assert pose[0] > 5.2, "the chosen standoff is on the walled-off side"


def test_approach_pose_fails_when_the_station_is_fully_enclosed(empty_grid) -> None:
    """A station in a box too tight to stand off from must abort the mission.

    The walls sit 1.5 m from the station centre; once inflated by the rover
    clearance no point on any candidate ring (1.2 m to 2.4 m) is free, so there
    is nowhere the rover could park and still read the code.
    """
    grid = empty_grid
    grid.mark_box((7.0, -5.0), (0.8, 0.8))
    for centre, size in [
        ((5.5, -5.0), (0.4, 3.4)),
        ((8.5, -5.0), (0.4, 3.4)),
        ((7.0, -3.5), (3.4, 0.4)),
        ((7.0, -6.5), (3.4, 0.4)),
    ]:
        grid.mark_box(centre, size)

    with pytest.raises(PlanningError, match="no collision-free approach pose"):
        make_planner().select_approach_pose(
            grid, (7.0, -5.0), (-8.0, -8.0), approach_distance_m=1.8, target_clearance_m=0.7
        )


def test_path_intersects_obstacles_detects_a_deliberately_bad_path(walled_grid) -> None:
    """The independent auditor must catch a path that a planner did not."""
    straight = np.array([[-6.0, 0.0], [6.0, 0.0]])
    assert path_intersects_obstacles(straight, walled_grid, clearance_m=0.55)
