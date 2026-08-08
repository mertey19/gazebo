"""Pure-pursuit controller behaviour, driven against unicycle kinematics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mission_core.errors import FailureReason
from mission_core.path_following import (
    DifferentialDriveModel,
    FollowerState,
    PurePursuitController,
)


def make_controller(**overrides) -> PurePursuitController:
    kwargs = dict(
        lookahead_distance_m=0.9,
        min_lookahead_distance_m=0.35,
        max_linear_velocity=0.8,
        max_angular_velocity=1.4,
        heading_kp=2.0,
        rotate_in_place_rad=0.7,
        goal_tolerance_m=0.35,
        goal_yaw_tolerance_rad=0.30,
        max_cross_track_error_m=1.5,
        cross_track_grace_s=3.0,
    )
    kwargs.update(overrides)
    return PurePursuitController(**kwargs)


def drive(controller, rover, dt=0.05, max_time=200.0):
    """Run the closed loop; returns (final status, track, elapsed)."""
    track = [rover.pose.copy()]
    time = 0.0
    status = controller.compute(rover.pose, dt)
    while time < max_time and not status.is_terminal:
        rover.step(status.linear_velocity, status.angular_velocity, dt)
        track.append(rover.pose.copy())
        time += dt
        status = controller.compute(rover.pose, dt)
    return status, np.asarray(track), time


def test_straight_path_is_tracked_to_the_goal() -> None:
    controller = make_controller()
    controller.set_path(np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]]))
    rover = DifferentialDriveModel((0.0, 0.0, 0.0))

    status, track, elapsed = drive(controller, rover)
    assert status.state is FollowerState.GOAL_REACHED
    assert float(np.linalg.norm(rover.pose[:2] - np.array([8.0, 0.0]))) <= 0.35
    assert float(np.max(np.abs(track[:, 1]))) < 0.05, "drifted off a straight line"
    # 8 m at 0.8 m/s cannot be done faster than 10 s.
    assert elapsed >= 10.0


def test_multi_leg_path_with_a_right_angle_is_tracked() -> None:
    controller = make_controller()
    controller.set_path(
        np.array([[0.0, 0.0, 0.0], [6.0, 0.0, math.pi / 2], [6.0, 6.0, math.pi / 2]])
    )
    rover = DifferentialDriveModel((0.0, 0.0, 0.0))

    status, track, _ = drive(controller, rover)
    assert status.state is FollowerState.GOAL_REACHED
    assert float(np.linalg.norm(rover.pose[:2] - np.array([6.0, 6.0]))) <= 0.35
    # The corner is cut a little by the lookahead, but must stay bounded.
    assert float(np.max(track[:, 0])) < 7.0


def test_rover_starting_backwards_turns_in_place_first() -> None:
    """A 180 deg heading error must not produce a wide colliding arc."""
    controller = make_controller()
    controller.set_path(np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]]))
    rover = DifferentialDriveModel((0.0, 0.0, math.pi))

    first = controller.compute(rover.pose, 0.05)
    assert first.linear_velocity == 0.0, "should rotate in place, not drive an arc"
    assert abs(first.angular_velocity) > 0.5

    status, track, _ = drive(controller, rover)
    assert status.state is FollowerState.GOAL_REACHED
    assert float(np.min(track[:, 0])) > -0.6, "swung too far backwards before turning"


def test_goal_yaw_alignment_is_enforced_when_requested() -> None:
    """The rover must end up facing the station so its camera can see it."""
    controller = make_controller(align_to_goal_yaw=True, goal_yaw_tolerance_rad=0.15)
    controller.set_path(np.array([[0.0, 0.0, 0.0], [4.0, 0.0, math.pi / 2]]))
    rover = DifferentialDriveModel((0.0, 0.0, 0.0))

    status, _, _ = drive(controller, rover)
    assert status.state is FollowerState.GOAL_REACHED
    assert abs(math.atan2(math.sin(rover.pose[2] - math.pi / 2),
                          math.cos(rover.pose[2] - math.pi / 2))) <= 0.15


def test_velocity_limits_are_never_exceeded() -> None:
    controller = make_controller(max_linear_velocity=0.5, max_angular_velocity=0.9)
    controller.set_path(
        np.array([[0.0, 0.0, 0.0], [3.0, 2.0, 0.0], [6.0, -2.0, 0.0], [9.0, 0.0, 0.0]])
    )
    rover = DifferentialDriveModel((0.0, 0.0, 0.0))

    time = 0.0
    status = controller.compute(rover.pose, 0.05)
    while time < 200.0 and not status.is_terminal:
        assert abs(status.linear_velocity) <= 0.5 + 1e-9
        assert abs(status.angular_velocity) <= 0.9 + 1e-9
        rover.step(status.linear_velocity, status.angular_velocity, 0.05)
        time += 0.05
        status = controller.compute(rover.pose, 0.05)
    assert status.state is FollowerState.GOAL_REACHED


def test_sustained_cross_track_error_reports_a_tracking_failure() -> None:
    """A rover that cannot stay on the path must fail loudly, not wander."""
    controller = make_controller(max_cross_track_error_m=0.5, cross_track_grace_s=1.0)
    controller.set_path(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))

    # Hold the rover far off the path; the watchdog must trip after the grace.
    status = None
    for _ in range(40):
        status = controller.compute((5.0, 3.0, 0.0), 0.1)
        if status.state is FollowerState.FAILED:
            break
    assert status is not None
    assert status.state is FollowerState.FAILED
    assert status.failure_reason is FailureReason.PATH_TRACKING_FAILURE
    assert "cross-track" in status.detail


def test_transient_excursion_does_not_trip_the_watchdog() -> None:
    controller = make_controller(max_cross_track_error_m=0.5, cross_track_grace_s=3.0)
    controller.set_path(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    for _ in range(5):
        controller.compute((2.0, 2.0, 0.0), 0.1)  # 0.5 s off-path
    status = controller.compute((2.0, 0.0, 0.0), 0.1)  # back on
    assert status.state is FollowerState.TRACKING


def test_controller_without_a_path_commands_nothing() -> None:
    controller = make_controller()
    status = controller.compute((1.0, 1.0, 0.0), 0.1)
    assert status.state is FollowerState.IDLE
    assert status.linear_velocity == 0.0 and status.angular_velocity == 0.0


def test_malformed_paths_are_rejected() -> None:
    controller = make_controller()
    with pytest.raises(ValueError):
        controller.set_path(np.zeros((0, 3)))
    with pytest.raises(ValueError):
        controller.set_path(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="non-finite"):
        controller.set_path(np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 0.0]]))


def test_progress_index_only_moves_forward() -> None:
    """A self-crossing path must not make the rover snap back to an old leg."""
    controller = make_controller()
    controller.set_path(
        np.array(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 1.0, 0.0], [0.0, 1.0, 0.0],
             [0.0, 2.0, 0.0]]
        )
    )
    rover = DifferentialDriveModel((0.0, 0.0, 0.0))
    indices = []
    status = controller.compute(rover.pose, 0.05)
    for _ in range(4000):
        if status.is_terminal:
            break
        rover.step(status.linear_velocity, status.angular_velocity, 0.05)
        status = controller.compute(rover.pose, 0.05)
        indices.append(status.progress_index)
    assert indices == sorted(indices), "progress index went backwards"
    assert status.state is FollowerState.GOAL_REACHED
