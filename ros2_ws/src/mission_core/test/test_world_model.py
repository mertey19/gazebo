"""TEST 2 - the world model stores, fuses and retrieves targets correctly."""

from __future__ import annotations

import numpy as np
import pytest

from mission_core.occupancy import (
    FREE,
    OCCUPIED,
    GridMetadata,
    OccupancyGrid,
    OccupancyMapper,
    planar_scan_hit_points,
)
from mission_core.world_model import TargetObservation, TargetStatus, WorldModel


def observation(qr_id: str, xy, confidence: float = 0.8, stamp: float = 0.0):
    return TargetObservation(
        qr_id=qr_id,
        position_map=np.array([xy[0], xy[1], 1.0]),
        normal_map=np.array([0.0, 0.0, 1.0]),
        confidence=confidence,
        stamp=stamp,
        observer_position_map=np.array([0.0, 0.0, 6.0]),
        source="test",
    )


def test_planar_scan_converts_only_real_hits_to_points() -> None:
    points = planar_scan_hit_points(
        [1.0, np.inf, 3.0, 4.0],
        angle_min=0.0,
        angle_increment=np.pi / 2.0,
        range_min=0.2,
        range_max=4.0,
    )
    assert points.shape == (2, 3)
    assert np.allclose(points[0], [1.0, 0.0, 0.0])
    assert np.allclose(points[1], [-3.0, 0.0, 0.0], atol=1e-12)


def test_runtime_lidar_hits_override_old_ground_evidence() -> None:
    mapper = OccupancyMapper(
        GridMetadata(0.2, 20, 20, -2.0, -2.0),
        min_hits=2,
        hit_ratio_threshold=0.35,
    )
    # The drone previously classified this cell as ground many times.
    mapper.integrate(np.repeat([[0.1, 0.1, 0.0]], 20, axis=0))
    assert mapper.build().value_at((0.1, 0.1)) == FREE

    # Two consistent rover-height endpoints mean a newly appeared obstacle;
    # historical ground returns must not dilute that direct evidence.
    mapper.integrate_runtime_obstacle_hits(np.array([[0.1, 0.1, 0.3]]))
    assert mapper.build().value_at((0.1, 0.1)) == FREE
    mapper.integrate_runtime_obstacle_hits(np.array([[0.1, 0.1, 0.3]]))
    assert mapper.build().value_at((0.1, 0.1)) == OCCUPIED


def test_single_observation_is_stored_but_not_yet_usable() -> None:
    model = WorldModel(min_observations=3)
    record = model.add_observation(observation("TARGET_1", (4.0, 5.0)))

    assert record.qr_id == "TARGET_1"
    assert record.observation_count == 1
    assert record.status is TargetStatus.TENTATIVE
    assert not model.has_confirmed("TARGET_1")
    # A single sighting is retrievable - the mission needs to distinguish
    # "never seen" from "seen once".
    assert model.get_target("TARGET_1") is record


def test_repeated_observations_confirm_and_average() -> None:
    model = WorldModel(min_observations=3)
    for index, offset in enumerate([(0.0, 0.0), (0.10, -0.05), (-0.08, 0.06), (0.02, 0.01)]):
        model.add_observation(
            observation("TARGET_2", (7.0 + offset[0], -5.0 + offset[1]), stamp=float(index))
        )

    record = model.get_target("TARGET_2")
    assert record is not None
    assert record.observation_count == 4
    assert record.status is TargetStatus.CONFIRMED
    assert model.has_confirmed("TARGET_2")
    assert record.position[0] == pytest.approx(7.0, abs=0.10)
    assert record.position[1] == pytest.approx(-5.0, abs=0.10)
    assert record.position_spread_m < 0.15
    assert 0.0 < record.confidence <= 1.0
    assert record.last_seen == 3.0
    assert record.first_seen == 0.0


def test_confidence_weighting_favours_the_better_observation() -> None:
    """A crisp close-up must outweigh a marginal long-range sighting."""
    model = WorldModel(min_observations=1)
    model.add_observation(observation("TARGET_1", (0.0, 0.0), confidence=0.05))
    model.add_observation(observation("TARGET_1", (1.0, 0.0), confidence=0.95))
    record = model.get_target("TARGET_1")
    assert record is not None
    assert record.position[0] > 0.8, "the high-confidence observation should dominate"


def test_duplicate_payload_at_two_places_is_flagged_ambiguous() -> None:
    """Two physical stations sharing a payload must never be planned against."""
    model = WorldModel(association_radius_m=1.5, min_observations=3)
    for index in range(4):
        model.add_observation(observation("TARGET_3", (2.0, 2.0), stamp=float(index)))
    assert model.has_confirmed("TARGET_3")

    # A second cluster 10 m away cannot be the same physical station.
    for index in range(4):
        model.add_observation(observation("TARGET_3", (-8.0, 2.0), stamp=float(10 + index)))

    record = model.get_target("TARGET_3")
    assert record is not None
    assert record.status is TargetStatus.AMBIGUOUS
    assert record.confidence == 0.0
    assert not model.has_confirmed("TARGET_3")
    assert model.summary()["targets_ambiguous"] == 2


def test_observations_within_the_association_radius_stay_one_target() -> None:
    """Sightings closer than the association radius must merge, not duplicate."""
    model = WorldModel(association_radius_m=1.5, min_observations=2)
    model.add_observation(observation("TARGET_1", (0.0, 0.0)))
    model.add_observation(observation("TARGET_1", (1.2, 0.0)))
    assert len(model.targets()) == 1
    assert model.get_target("TARGET_1").observation_count == 2


def test_scattered_observations_do_not_confirm_a_target() -> None:
    """Merging is not the same as trusting: wide scatter must stay TENTATIVE."""
    scattered = WorldModel(association_radius_m=1.5, min_observations=2, min_confidence=0.35)
    scattered.add_observation(observation("TARGET_1", (0.0, 0.0)))
    scattered.add_observation(observation("TARGET_1", (1.2, 0.0)))
    assert scattered.get_target("TARGET_1").position_spread_m > 0.5
    assert scattered.get_target("TARGET_1").status is TargetStatus.TENTATIVE

    tight = WorldModel(association_radius_m=1.5, min_observations=2, min_confidence=0.35)
    tight.add_observation(observation("TARGET_1", (0.0, 0.0)))
    tight.add_observation(observation("TARGET_1", (0.06, 0.0)))
    assert tight.get_target("TARGET_1").status is TargetStatus.CONFIRMED


def test_unknown_payload_returns_none() -> None:
    model = WorldModel()
    assert model.get_target("TARGET_9") is None
    assert not model.has_confirmed("TARGET_9")


def test_non_finite_observation_is_rejected_loudly() -> None:
    model = WorldModel()
    bad = observation("TARGET_1", (0.0, 0.0))
    bad.position_map = np.array([np.nan, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        model.add_observation(bad)
    assert model.rejected_observations == 1


def test_replace_targets_overwrites_wholesale() -> None:
    """The subscriber-side mirror must not merge with what it already had."""
    model = WorldModel(min_observations=1)
    model.add_observation(observation("TARGET_1", (0.0, 0.0)))
    other = WorldModel(min_observations=1)
    other.add_observation(observation("TARGET_2", (3.0, 3.0)))

    model.replace_targets(other.targets())
    assert {t.qr_id for t in model.targets()} == {"TARGET_2"}


# ---------------------------------------------------------------------------
# Occupancy mapping
# ---------------------------------------------------------------------------

def test_mapper_separates_ground_returns_from_obstacles() -> None:
    metadata = GridMetadata(0.2, 60, 60, -6.0, -6.0)
    mapper = OccupancyMapper(metadata, obstacle_min_height_m=0.25, min_hits=2)

    ground = np.array([[x, y, 0.0] for x in np.arange(-5, 5, 0.2) for y in (-1.0, 0.0)])
    obstacle = np.array([[1.0, 2.0, 1.2]] * 4 + [[1.1, 2.05, 1.1]] * 4)
    mapper.integrate(ground)
    mapper.integrate(obstacle)
    grid = mapper.build()

    assert grid.value_at((1.0, 2.0)) == OCCUPIED
    assert grid.value_at((0.0, 0.0)) == FREE
    # Never observed: must stay unknown rather than default to free.
    assert grid.value_at((-5.5, 5.5)) == -1
    assert mapper.height_map.max() == pytest.approx(1.2, abs=1e-5)


def test_mapper_ignores_a_single_spurious_return() -> None:
    """One stray high point must not become an obstacle."""
    mapper = OccupancyMapper(GridMetadata(0.2, 40, 40, -4.0, -4.0), min_hits=2)
    mapper.integrate(np.array([[0.0, 0.0, 0.0]] * 10))
    mapper.integrate(np.array([[0.0, 0.0, 2.0]]))
    assert mapper.build().value_at((0.0, 0.0)) == FREE


def test_world_model_extracts_obstacle_footprints() -> None:
    grid = OccupancyGrid(GridMetadata(0.2, 100, 100, -10.0, -10.0))
    grid.data[:] = FREE
    grid.mark_box(centre=(0.0, -6.0), size_xy=(6.0, 1.0))
    grid.mark_box(centre=(0.0, 0.0), size_xy=(1.0, 6.0))

    model = WorldModel()
    obstacles = model.set_occupancy(grid)

    assert len(obstacles) == 2
    by_width = sorted(obstacles, key=lambda o: -o.size_xy[0])
    assert by_width[0].size_xy[0] == pytest.approx(6.0, abs=0.3)
    assert by_width[0].centre[1] == pytest.approx(-6.0, abs=0.2)
    assert by_width[1].size_xy[1] == pytest.approx(6.0, abs=0.3)
    assert by_width[1].contains((0.0, 0.0))
    assert not by_width[1].contains((5.0, 0.0))


def test_summary_reports_the_digital_twin_state() -> None:
    model = WorldModel(min_observations=2)
    for index in range(3):
        model.add_observation(observation("TARGET_1", (1.0, 1.0), stamp=float(index)))
    model.add_observation(observation("TARGET_2", (5.0, 5.0)))
    grid = OccupancyGrid(GridMetadata(0.5, 20, 20, -5.0, -5.0))
    grid.data[:] = FREE
    grid.mark_box((0.0, 0.0), (2.0, 2.0))
    model.set_occupancy(grid)

    summary = model.summary()
    assert summary["targets_total"] == 2
    assert summary["targets_confirmed"] == 1
    assert summary["obstacles"] == 1
    assert summary["qr_ids"] == ["TARGET_1", "TARGET_2"]
    assert summary["map_known_fraction"] == pytest.approx(1.0)
