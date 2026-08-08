"""TEST 6 - full end-to-end mission integration, without Gazebo.

Runs the *production* pipeline against the synthetic arena:

    takeoff -> lawnmower scan -> rendered camera frames -> cv2 QR decode ->
    PnP -> TF into map -> world model fusion -> lidar occupancy mapping ->
    A* -> nav_msgs-shaped path -> pure pursuit -> rover drives -> rover
    camera QR verification -> automated validator

Ground truth is used only *after* the run, to score how accurate perception
was.  No decision in the loop reads it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mission_core.config import MissionConfig
from mission_core.errors import FailureReason
from mission_core.mission_state import MissionState
from mission_core.planner import path_intersects_obstacles

from offline_mission import OfflineMissionRunner, default_world
from sim_harness import OrientedBox, Station, SyntheticWorld

pytestmark = pytest.mark.integration


def run_mission(world=None, target="TARGET_2", config=None, **kwargs):
    runner = OfflineMissionRunner(
        world if world is not None else default_world(),
        config or MissionConfig(),
        requested_qr=target,
        **kwargs,
    )
    return runner, runner.run()


@pytest.fixture(scope="module")
def nominal_run():
    """One full successful mission, shared by the assertions below."""
    return run_mission(target="TARGET_2")


# ---------------------------------------------------------------------------
# The definition of done
# ---------------------------------------------------------------------------

def test_mission_succeeds_end_to_end(nominal_run) -> None:
    _, trace = nominal_run
    assert trace.state is MissionState.MISSION_SUCCESS, (
        f"mission ended in {trace.state.value} ({trace.failure_reason})"
    )
    assert trace.failure_reason == FailureReason.NONE.value
    assert trace.verified_qr == "TARGET_2"


def test_every_mandatory_validation_check_passes(nominal_run) -> None:
    _, trace = nominal_run
    assert trace.report is not None
    assert trace.report.passed, trace.report.render()
    names = {c.name for c in trace.report.checks}
    # The checks that make the result meaningful must all be present.
    assert {
        "requested_qr_discovered",
        "path_generated",
        "path_avoids_obstacles",
        "path_starts_at_rover",
        "rover_reached_goal",
        "rover_qr_verification",
    } <= names


def test_the_expected_log_sequence_appears(nominal_run) -> None:
    _, trace = nominal_run
    joined = "\n".join(trace.messages)
    for expected in (
        "[MISSION] Requested QR: TARGET_2",
        "[DRONE] Taking off",
        "[QR] TARGET_2 detected",
        "[MISSION] Requested target TARGET_2 found",
        "[PLANNER] Planning rover path",
        "[MISSION] Path sent to rover",
        "[ROVER] Navigation started",
        "[ROVER] Goal reached",
        "[QR] Rover detected TARGET_2",
        "[MISSION] QR verification successful",
        "[MISSION] SUCCESS",
    ):
        assert expected in joined, f"missing log line: {expected!r}\n---\n{joined}"


def test_all_three_stations_are_discovered_by_perception(nominal_run) -> None:
    """The digital twin must be complete, not just sufficient."""
    _, trace = nominal_run
    summary = trace.world_model.summary()
    assert summary["qr_ids"] == ["TARGET_1", "TARGET_2", "TARGET_3"]
    assert summary["targets_confirmed"] == 3
    assert summary["targets_ambiguous"] == 0


def test_perception_positions_are_accurate(nominal_run) -> None:
    """Scored against ground truth - the only place it is allowed."""
    runner, trace = nominal_run
    truth = runner.world.ground_truth_station_xy()
    errors = {}
    for payload, expected in truth.items():
        record = trace.world_model.get_target(payload)
        assert record is not None, f"{payload} was never discovered"
        errors[payload] = float(np.linalg.norm(record.position[:2] - expected))
    assert max(errors.values()) < 0.30, f"position errors: {errors}"


def test_obstacles_are_mapped_from_lidar(nominal_run) -> None:
    _, trace = nominal_run
    obstacles = trace.world_model.obstacles()
    # Two walls plus the three stations, all of which are real obstacles.
    assert len(obstacles) >= 5
    grid = trace.world_model.occupancy
    assert grid is not None
    assert grid.known_fraction > 0.90, "the scan left large parts of the arena unmapped"
    # The two walls must be present at their true footprints.
    assert grid.value_at((0.0, -6.0)) == 100
    assert grid.value_at((0.0, 0.0)) == 100
    assert grid.value_at((-4.0, -3.0)) == 0


def test_path_is_a_genuine_obstacle_free_detour(nominal_run) -> None:
    """The straight line collides; the planned path must not."""
    _, trace = nominal_run
    path = trace.path_xy
    grid = trace.world_model.occupancy
    assert path is not None and grid is not None and len(path) >= 2

    dense = []
    for start, end in zip(path[:-1], path[1:]):
        for t in np.linspace(0.0, 1.0, max(2, int(np.linalg.norm(end - start) / 0.05))):
            dense.append(start + t * (end - start))
    dense = np.asarray(dense)
    assert not path_intersects_obstacles(dense, grid, clearance_m=0.55), (
        "the followed path intersects a mapped obstacle"
    )
    # The direct line does collide, so the detour was necessary, not incidental.
    assert path_intersects_obstacles(
        np.array([path[0], path[-1]]), grid, clearance_m=0.55
    ), "obstacles were irrelevant to this route - the test proves nothing"


def test_rover_track_never_enters_an_obstacle(nominal_run) -> None:
    """Not just the plan: the trajectory the rover actually drove."""
    _, trace = nominal_run
    grid = trace.world_model.occupancy
    assert grid is not None and trace.rover_track
    track = np.asarray([p[:2] for p in trace.rover_track])
    # Checked against the raw grid inflated by the rover radius alone; the
    # extra safety margin is a planning buffer, not a hard collision bound.
    assert not path_intersects_obstacles(track, grid, clearance_m=0.30), (
        "the rover physically drove through a mapped obstacle"
    )


def test_rover_stops_within_the_goal_tolerance(nominal_run) -> None:
    _, trace = nominal_run
    config = MissionConfig()
    final = np.asarray(trace.rover_track[-1][:2])
    residual = float(np.linalg.norm(final - trace.path_xy[-1]))
    assert residual <= config.rover.goal_tolerance_m


def test_no_camera_frame_was_unusable(nominal_run) -> None:
    _, trace = nominal_run
    assert trace.drone_frames > 100, "the scan barely produced any imagery"
    assert trace.bad_frames == 0, f"{trace.bad_frames} unusable frames during the mission"


# ---------------------------------------------------------------------------
# The same pipeline for the other targets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", ["TARGET_1", "TARGET_3"])
def test_other_targets_also_complete(target: str) -> None:
    runner, trace = run_mission(target=target)
    assert trace.state is MissionState.MISSION_SUCCESS, (
        f"{target}: {trace.state.value} ({trace.failure_reason})"
    )
    assert trace.verified_qr == target
    assert trace.report is not None and trace.report.passed

    # The rover must end up at the *requested* station, not merely at some
    # station - scored against ground truth after the fact.
    truth = runner.world.ground_truth_station_xy()[target]
    final = np.asarray(trace.rover_track[-1][:2])
    distance = float(np.linalg.norm(final - truth))
    assert distance < 3.0, f"{target}: rover finished {distance:.2f} m from the station"


# ---------------------------------------------------------------------------
# End-to-end failure behaviour
# ---------------------------------------------------------------------------

def test_absent_target_fails_after_a_complete_scan() -> None:
    """Scan the whole arena for a station that is not there."""
    world = SyntheticWorld(
        stations=[Station("TARGET_1", (6.0, 6.0)), Station("TARGET_3", (-6.0, 6.0))],
        obstacles=[
            OrientedBox(np.array([0.0, -6.0, 0.75]), np.array([6.0, 1.0, 1.5])),
            OrientedBox(np.array([0.0, 0.0, 0.75]), np.array([1.0, 6.0, 1.5])),
        ],
    )
    _, trace = run_mission(world=world, target="TARGET_2")

    assert trace.state is MissionState.MISSION_FAILED
    assert trace.failure_reason == FailureReason.QR_NOT_DETECTED.value
    assert trace.path_xy is None
    # The other two must still have been discovered - the failure is specific.
    assert trace.world_model.summary()["qr_ids"] == ["TARGET_1", "TARGET_3"]


def test_wrong_rover_reading_fails_the_mission(monkeypatch) -> None:
    """TEST 5, end to end: right place, wrong code => MISSION_FAILED.

    The rover reaches the correct goal but its camera is made to report a
    different payload. Nothing downstream may paper over that.
    """
    runner = OfflineMissionRunner(default_world(), MissionConfig(), requested_qr="TARGET_2")
    monkeypatch.setattr(runner, "_verify_with_rover", lambda: "TARGET_1")
    trace = runner.run()

    assert trace.state is MissionState.MISSION_FAILED
    assert trace.failure_reason == FailureReason.QR_VERIFICATION_MISMATCH.value
    # Everything before verification must have worked - this is specifically a
    # verification failure, not a collapse earlier in the pipeline.
    assert trace.path_xy is not None and len(trace.path_xy) >= 2
    assert "[ROVER] Goal reached" in "\n".join(trace.messages)


def test_unknown_payload_is_refused_before_takeoff() -> None:
    _, trace = run_mission(target="TARGET_9")
    assert trace.state is MissionState.MISSION_FAILED
    assert trace.failure_reason == FailureReason.UNKNOWN_TARGET_REQUESTED.value
    assert trace.drone_frames == 0, "the drone should never have left the ground"


def test_scan_altitude_too_high_is_caught_before_flying() -> None:
    """Configuration validation must reject an unflyable mission at startup."""
    config = MissionConfig()
    unusable = replace(config, drone=replace(config.drone, scan_altitude_m=25.0))
    assert any("px per QR module" in p for p in unusable.validate())
