"""Pytest fixtures shared by the mission_core suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# The harness modules live beside the tests rather than inside the installed
# package: they are development tooling, not part of the robot runtime.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mission_core.camera import PinholeCamera  # noqa: E402
from mission_core.config import MissionConfig  # noqa: E402
from mission_core.occupancy import GridMetadata, OccupancyGrid  # noqa: E402

from offline_mission import default_world  # noqa: E402
from sim_harness import OrientedBox, Station, SyntheticWorld  # noqa: E402


@pytest.fixture(scope="session")
def config() -> MissionConfig:
    return MissionConfig()


@pytest.fixture(scope="session")
def drone_camera(config: MissionConfig) -> PinholeCamera:
    return PinholeCamera.from_hfov(
        config.drone.camera.width,
        config.drone.camera.height,
        config.drone.camera.horizontal_fov_rad,
    )


@pytest.fixture(scope="session")
def rover_camera(config: MissionConfig) -> PinholeCamera:
    return PinholeCamera.from_hfov(
        config.rover.camera.width,
        config.rover.camera.height,
        config.rover.camera.horizontal_fov_rad,
    )


@pytest.fixture(scope="session")
def arena() -> SyntheticWorld:
    """The standard three-station, two-obstacle arena."""
    return default_world()


@pytest.fixture
def empty_grid() -> OccupancyGrid:
    """A 20x20 m all-free grid at the mission's planning resolution."""
    grid = OccupancyGrid(GridMetadata(0.2, 100, 100, -10.0, -10.0))
    grid.data[:] = 0
    return grid


@pytest.fixture
def walled_grid(empty_grid: OccupancyGrid) -> OccupancyGrid:
    """Free space split by a wall with a single gap, forcing a detour."""
    grid = empty_grid
    grid.mark_box(centre=(0.0, 1.5), size_xy=(1.0, 13.0))
    return grid


def make_world(stations, obstacles=()) -> SyntheticWorld:
    """Build an ad-hoc synthetic world from ``(payload, x, y)`` tuples."""
    return SyntheticWorld(
        stations=[Station(payload, (float(x), float(y))) for payload, x, y in stations],
        obstacles=[
            OrientedBox(np.asarray(centre, dtype=float), np.asarray(size, dtype=float), name=name)
            for name, centre, size in obstacles
        ],
    )
