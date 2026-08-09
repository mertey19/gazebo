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
    SafetyStopWatchdog,
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


def test_safety_stop_watchdog_requests_replan_only_after_sustained_blockage() -> None:
    watchdog = SafetyStopWatchdog(3.0)
    assert not watchdog.update(True, 1.0)
    assert not watchdog.update(True, 1.0)
    assert watchdog.update(True, 1.0)


def test_safety_stop_watchdog_resets_after_a_clear_scan() -> None:
    watchdog = SafetyStopWatchdog(2.0)
    assert not watchdog.update(True, 1.5)
    assert not watchdog.update(False, 0.1)
    assert watchdog.blocked_s == 0.0
    assert not watchdog.update(True, 1.5)


def test_safety_stop_watchdog_rejects_non_positive_delay() -> None:
    with pytest.raises(ValueError, match="positive"):
        SafetyStopWatchdog(0.0)


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


# ---------------------------------------------------------------------------
# Verification sweep
# ---------------------------------------------------------------------------

def test_sweep_covers_the_full_amplitude_both_ways() -> None:
    """The sweep must scan [-A, +A] and end back where it started."""
    from mission_core.path_following import VerificationSweep

    sweep = VerificationSweep(sweep_rad=1.0, yaw_rate_rad_s=0.5)
    dt = 0.05
    yaw = 0.0
    extremes = [0.0, 0.0]
    for _ in range(10000):
        if sweep.finished:
            break
        # step() returns a yaw *rate*, so integrate it.
        yaw += sweep.step(dt) * dt
        extremes[0] = min(extremes[0], yaw)
        extremes[1] = max(extremes[1], yaw)

    assert sweep.finished
    assert extremes[0] == pytest.approx(-1.0, abs=1e-6)
    assert extremes[1] == pytest.approx(1.0, abs=1e-6)
    assert yaw == pytest.approx(0.0, abs=1e-6), "the sweep must return to its start heading"


def test_sweep_respects_its_rate_and_never_overshoots_a_leg() -> None:
    from mission_core.path_following import VerificationSweep

    sweep = VerificationSweep(sweep_rad=0.6, yaw_rate_rad_s=0.4)
    yaw = 0.0
    while not sweep.finished:
        rate = sweep.step(0.1)
        assert abs(rate) <= 0.4 + 1e-9
        yaw += rate * 0.1
        assert -0.6 - 1e-9 <= yaw <= 0.6 + 1e-9, f"swept past the amplitude to {yaw}"


def test_finished_sweep_commands_nothing() -> None:
    from mission_core.path_following import VerificationSweep

    sweep = VerificationSweep(sweep_rad=0.2, yaw_rate_rad_s=1.0)
    while not sweep.finished:
        sweep.step(0.05)
    assert sweep.step(0.05) == 0.0
    assert sweep.progress == pytest.approx(1.0)

    sweep.reset()
    assert not sweep.finished
    assert sweep.step(0.05) != 0.0


def test_sweep_rejects_nonsense_parameters() -> None:
    from mission_core.path_following import VerificationSweep

    with pytest.raises(ValueError):
        VerificationSweep(sweep_rad=0.0, yaw_rate_rad_s=0.4)
    with pytest.raises(ValueError):
        VerificationSweep(sweep_rad=1.0, yaw_rate_rad_s=-0.1)


def test_sweep_amplitude_covers_plausible_odometry_yaw_drift(config) -> None:
    """The sweep must be wide enough to recover a station just out of frame.

    A station at the edge of the camera is at half the horizontal FOV; the
    sweep has to reach at least that far or it cannot fix the case it exists for.
    """
    half_fov = config.rover.camera.horizontal_fov_rad / 2.0
    assert config.verification.search_sweep_rad > half_fov, (
        f"sweep {config.verification.search_sweep_rad:.2f} rad does not exceed the "
        f"camera half-FOV {half_fov:.2f} rad"
    )


# ---------------------------------------------------------------------------
# Drone escort
# ---------------------------------------------------------------------------

def test_escort_station_is_where_the_camera_points(config) -> None:
    """The standoff must equal the distance the depressed camera looks ahead.

    Any other offset puts the rover at the edge of the frame or outside it,
    which defeats the point of escorting it at all.
    """
    from mission_core.exploration import EscortController

    escort = EscortController(
        altitude_m=config.drone.scan_altitude_m,
        depression_rad=config.drone.camera_depression_rad,
        speed_mps=config.drone.follow_speed_mps,
    )
    height = config.drone.scan_altitude_m - escort.rover_height_m
    assert escort.standoff_m == pytest.approx(
        height / math.tan(config.drone.camera_depression_rad), rel=1e-9
    )

    # Behind the rover, whichever way it is pointing.
    for yaw in (0.0, math.pi / 2, -2.0, math.pi):
        station = escort.station_for((3.0, -2.0, yaw))
        offset = station[:2] - np.array([3.0, -2.0])
        assert float(np.linalg.norm(offset)) == pytest.approx(escort.standoff_m, rel=1e-9)
        bearing = math.atan2(-offset[1], -offset[0])
        assert abs(math.atan2(math.sin(bearing - yaw), math.cos(bearing - yaw))) < 1e-9
        assert station[2] == pytest.approx(config.drone.scan_altitude_m)


def test_escort_converges_onto_a_moving_rover() -> None:
    """Closed loop: the drone must catch a rover and then hold station."""
    from mission_core.exploration import EscortController, KinematicDrone

    escort = EscortController(
        altitude_m=4.0, depression_rad=math.radians(30), speed_mps=2.2
    )
    drone = KinematicDrone((-14.0, 5.0, 4.0), yaw=0.0)
    rover = np.array([0.0, 0.0, 0.0])
    dt = 0.05

    for step in range(4000):
        # Rover drives forward at its cruise speed, turning gently.
        rover[2] += 0.05 * dt
        rover[0] += 0.8 * math.cos(rover[2]) * dt
        rover[1] += 0.8 * math.sin(rover[2]) * dt
        command = escort.compute(drone.position, drone.yaw, rover)
        drone.step(command.velocity_map, command.yaw_rate, dt)

    station = escort.station_for(rover)
    error = float(np.linalg.norm(drone.position[:2] - station[:2]))
    assert error < 1.0, f"drone settled {error:.2f} m from its station"
    assert abs(drone.position[2] - 4.0) < 0.2

    # And it must be looking at the rover, not merely near it.
    bearing = math.atan2(rover[1] - drone.position[1], rover[0] - drone.position[0])
    heading_error = abs(math.atan2(math.sin(bearing - drone.yaw), math.cos(bearing - drone.yaw)))
    assert heading_error < 0.25, f"drone yaw is {heading_error:.2f} rad off the rover"


def test_escort_never_exceeds_its_speed_limit() -> None:
    from mission_core.exploration import EscortController

    escort = EscortController(
        altitude_m=4.0, depression_rad=math.radians(30), speed_mps=1.5
    )
    for distance in (0.1, 1.0, 5.0, 50.0):
        command = escort.compute((-distance, 0.0, 4.0), 0.0, (0.0, 0.0, 0.0))
        assert float(np.linalg.norm(command.velocity_map[:2])) <= 1.5 + 1e-9


def test_escort_rejects_impossible_geometry() -> None:
    from mission_core.exploration import EscortController

    with pytest.raises(ValueError, match="above the rover"):
        EscortController(altitude_m=0.2, depression_rad=0.5, speed_mps=1.0)
    with pytest.raises(ValueError, match="depression"):
        EscortController(altitude_m=4.0, depression_rad=0.0, speed_mps=1.0)
