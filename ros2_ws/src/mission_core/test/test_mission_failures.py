"""TEST 4 and TEST 5 - failure handling and rover-side QR verification.

Every one of these asserts that the mission *stops* with a specific, named
reason.  Silently continuing past any of them would be the worst possible
outcome: a mission that reports success while the rover is at the wrong
station.
"""

from __future__ import annotations

import numpy as np
import pytest

from mission_core.config import MissionConfig
from mission_core.errors import FailureReason
from mission_core.mission_state import (
    InvalidTransition,
    MissionState,
    StateMachine,
)
from mission_core.occupancy import GridMetadata, OccupancyGrid
from mission_core.orchestrator import MissionCommand, MissionInputs, MissionOrchestrator
from mission_core.validation import MissionValidator
from mission_core.world_model import TargetObservation, TargetStatus, WorldModel


def confirmed_world_model(payload: str = "TARGET_2", xy=(7.0, -5.0)) -> WorldModel:
    """A world model with one confirmed target and two mapped obstacles."""
    model = WorldModel(min_observations=3, min_confidence=0.3)
    for index in range(4):
        model.add_observation(
            TargetObservation(
                qr_id=payload,
                position_map=np.array([xy[0], xy[1], 1.0]),
                normal_map=np.array([0.0, 0.0, 1.0]),
                confidence=0.9,
                stamp=float(index),
                observer_position_map=np.array([0.0, 0.0, 6.0]),
                source="drone_camera",
            )
        )
    grid = OccupancyGrid(GridMetadata(0.2, 110, 110, -11.0, -11.0))
    grid.data[:] = 0
    grid.mark_box((0.0, -6.0), (6.0, 1.0))
    grid.mark_box(xy, (0.8, 0.8))
    model.set_occupancy(grid)
    return model


def run_to_state(orchestrator: MissionOrchestrator, target: MissionState, **input_kwargs):
    """Tick the orchestrator until it reaches ``target`` or goes terminal.

    Acknowledges PUBLISH_PATH the way ``mission_manager_node`` does, so the
    handshake between SENDING_PATH and ROVER_NAVIGATING is exercised rather
    than bypassed.
    """
    defaults = dict(
        drone_ready=True,
        drone_at_scan_altitude=True,
        exploration_complete=True,
        rover_pose=np.array([-8.0, -8.0]),
    )
    defaults.update(input_kwargs)
    published = defaults.pop("path_published", False)
    outputs = None
    for tick in range(400):
        outputs = orchestrator.update(
            MissionInputs(now=tick * 0.1, path_published=published, **defaults)
        )
        if outputs.command is MissionCommand.PUBLISH_PATH and outputs.path is not None:
            published = True
        if orchestrator.state is target or orchestrator.machine.is_terminal:
            break
    return outputs


# ---------------------------------------------------------------------------
# TEST 4 - unknown / undiscovered targets
# ---------------------------------------------------------------------------

def test_requesting_an_unconfigured_payload_fails_immediately() -> None:
    orchestrator = MissionOrchestrator(
        MissionConfig(), confirmed_world_model(), requested_qr="TARGET_9"
    )
    orchestrator.update(MissionInputs(now=0.0, drone_ready=True))
    assert orchestrator.state is MissionState.MISSION_FAILED
    assert orchestrator.machine.failure_reason is FailureReason.UNKNOWN_TARGET_REQUESTED
    assert "TARGET_9" in orchestrator.machine.failure_detail


def test_never_decoded_payload_fails_after_the_scan_completes() -> None:
    """A full sweep that never saw the code must fail, not wait forever."""
    model = confirmed_world_model("TARGET_1", (6.0, 6.0))
    orchestrator = MissionOrchestrator(MissionConfig(), model, requested_qr="TARGET_2")
    run_to_state(orchestrator, MissionState.MISSION_FAILED)

    assert orchestrator.state is MissionState.MISSION_FAILED
    assert orchestrator.machine.failure_reason is FailureReason.QR_NOT_DETECTED


def test_tentative_target_is_not_good_enough_to_plan_against() -> None:
    """One sighting is not a confirmation; the mission must not act on it."""
    model = WorldModel(min_observations=3)
    model.add_observation(
        TargetObservation(
            qr_id="TARGET_2",
            position_map=np.array([7.0, -5.0, 1.0]),
            normal_map=np.array([0.0, 0.0, 1.0]),
            confidence=0.9,
            stamp=0.0,
            observer_position_map=np.zeros(3),
            source="drone_camera",
        )
    )
    orchestrator = MissionOrchestrator(MissionConfig(), model, requested_qr="TARGET_2")
    run_to_state(orchestrator, MissionState.MISSION_FAILED)

    assert orchestrator.machine.failure_reason is FailureReason.TARGET_NOT_DISCOVERED
    assert "TENTATIVE" in orchestrator.machine.failure_detail


def test_duplicate_qr_aborts_rather_than_guessing() -> None:
    model = confirmed_world_model("TARGET_2", (7.0, -5.0))
    for index in range(4):
        model.add_observation(
            TargetObservation(
                qr_id="TARGET_2",
                position_map=np.array([-7.0, 4.0, 1.0]),
                normal_map=np.array([0.0, 0.0, 1.0]),
                confidence=0.9,
                stamp=float(20 + index),
                observer_position_map=np.zeros(3),
                source="drone_camera",
            )
        )
    assert model.get_target("TARGET_2").status is TargetStatus.AMBIGUOUS

    orchestrator = MissionOrchestrator(MissionConfig(), model, requested_qr="TARGET_2")
    run_to_state(orchestrator, MissionState.MISSION_FAILED)
    assert orchestrator.machine.failure_reason is FailureReason.DUPLICATE_QR


def test_no_occupancy_grid_means_no_plan() -> None:
    """A confirmed target with no map must fail at PLANNING, not plan blind."""
    model = WorldModel(min_observations=3, min_confidence=0.3)
    for index in range(4):
        model.add_observation(
            TargetObservation(
                qr_id="TARGET_2",
                position_map=np.array([7.0, -5.0, 1.0]),
                normal_map=np.array([0.0, 0.0, 1.0]),
                confidence=0.9,
                stamp=float(index),
                observer_position_map=np.zeros(3),
                source="drone_camera",
            )
        )
    assert model.has_confirmed("TARGET_2") and model.occupancy is None
    orchestrator = MissionOrchestrator(MissionConfig(), model, requested_qr="TARGET_2")
    run_to_state(orchestrator, MissionState.MISSION_FAILED)
    assert orchestrator.machine.failure_reason is FailureReason.NO_VALID_PATH


def test_unreachable_target_fails_with_no_valid_path() -> None:
    """A walled-in station must abort planning, not emit a colliding path."""
    model = confirmed_world_model("TARGET_2", (7.0, -5.0))
    grid = model.occupancy
    assert grid is not None
    for centre, size in [
        ((5.0, -5.0), (0.4, 4.0)),
        ((9.0, -5.0), (0.4, 4.0)),
        ((7.0, -3.0), (4.4, 0.4)),
        ((7.0, -7.0), (4.4, 0.4)),
    ]:
        grid.mark_box(centre, size)
    model.set_occupancy(grid)

    orchestrator = MissionOrchestrator(MissionConfig(), model, requested_qr="TARGET_2")
    run_to_state(orchestrator, MissionState.MISSION_FAILED)
    assert orchestrator.machine.failure_reason is FailureReason.NO_VALID_PATH


def test_missing_rover_odometry_fails_with_localization_unavailable() -> None:
    orchestrator = MissionOrchestrator(
        MissionConfig(), confirmed_world_model(), requested_qr="TARGET_2"
    )
    run_to_state(orchestrator, MissionState.MISSION_FAILED, rover_pose=None)
    assert orchestrator.machine.failure_reason is FailureReason.LOCALIZATION_UNAVAILABLE


def test_navigation_timeout_is_enforced() -> None:
    config = MissionConfig()
    orchestrator = MissionOrchestrator(
        config, confirmed_world_model(), requested_qr="TARGET_2"
    )
    run_to_state(orchestrator, MissionState.ROVER_NAVIGATING)
    assert orchestrator.state is MissionState.ROVER_NAVIGATING

    # Jump past the navigation deadline without the rover ever arriving.
    orchestrator.update(
        MissionInputs(
            now=config.rover.navigation_timeout_s + 100.0,
            drone_ready=True,
            drone_at_scan_altitude=True,
            exploration_complete=True,
            rover_pose=np.array([-4.0, -8.0]),
            path_published=True,
        )
    )
    assert orchestrator.machine.failure_reason is FailureReason.NAVIGATION_TIMEOUT


def test_rover_tracking_failure_aborts_the_mission() -> None:
    orchestrator = MissionOrchestrator(
        MissionConfig(), confirmed_world_model(), requested_qr="TARGET_2"
    )
    run_to_state(orchestrator, MissionState.ROVER_NAVIGATING)
    orchestrator.update(
        MissionInputs(
            now=50.0,
            drone_ready=True,
            drone_at_scan_altitude=True,
            exploration_complete=True,
            rover_pose=np.array([-4.0, -8.0]),
            path_published=True,
            rover_tracking_failed=True,
            rover_failure_detail="cross-track error 2.10 m",
        )
    )
    assert orchestrator.machine.failure_reason is FailureReason.PATH_TRACKING_FAILURE
    assert "2.10" in orchestrator.machine.failure_detail


# ---------------------------------------------------------------------------
# TEST 5 - rover-side QR verification
# ---------------------------------------------------------------------------

def _reach_verification(requested="TARGET_2"):
    orchestrator = MissionOrchestrator(
        MissionConfig(), confirmed_world_model(), requested_qr=requested
    )
    run_to_state(orchestrator, MissionState.ROVER_NAVIGATING)
    base = dict(
        drone_ready=True,
        drone_at_scan_altitude=True,
        exploration_complete=True,
        rover_pose=orchestrator.path.xy[-1],
        path_published=True,
        rover_goal_reached=True,
    )
    orchestrator.update(MissionInputs(now=60.0, **base))
    assert orchestrator.state is MissionState.VERIFYING_TARGET
    return orchestrator, base


def test_verification_rejects_the_wrong_payload() -> None:
    """The decisive check: right place, wrong code => mission failure."""
    orchestrator, base = _reach_verification("TARGET_2")
    orchestrator.update(MissionInputs(now=61.0, verified_qr="TARGET_1", **base))

    assert orchestrator.state is MissionState.MISSION_FAILED
    assert orchestrator.machine.failure_reason is FailureReason.QR_VERIFICATION_MISMATCH
    assert "TARGET_1" in orchestrator.machine.failure_detail
    assert "TARGET_2" in orchestrator.machine.failure_detail


def test_verification_accepts_the_right_payload() -> None:
    orchestrator, base = _reach_verification("TARGET_2")
    outputs = orchestrator.update(MissionInputs(now=61.0, verified_qr="TARGET_2", **base))

    assert orchestrator.state is MissionState.MISSION_SUCCESS
    assert outputs.report is not None and outputs.report.passed
    assert "[MISSION] SUCCESS" in outputs.messages


def test_verification_times_out_when_no_code_is_readable() -> None:
    orchestrator, base = _reach_verification("TARGET_2")
    config = MissionConfig()
    orchestrator.update(
        MissionInputs(now=60.0 + config.verification.timeout_s + 1.0, **base)
    )
    assert orchestrator.machine.failure_reason is FailureReason.QR_VERIFICATION_TIMEOUT


def test_position_alone_never_produces_success() -> None:
    """Being at the goal is not success; only a verified code is."""
    orchestrator, base = _reach_verification("TARGET_2")
    for tick in range(10):
        orchestrator.update(MissionInputs(now=61.0 + tick, **base))
    assert orchestrator.state is MissionState.VERIFYING_TARGET
    assert not orchestrator.machine.succeeded


# ---------------------------------------------------------------------------
# Validator and state machine invariants
# ---------------------------------------------------------------------------

def test_validator_rejects_a_mismatched_verification() -> None:
    validator = MissionValidator(
        rover_radius_m=0.3,
        obstacle_safety_margin_m=0.25,
        goal_tolerance_m=0.35,
        approach_distance_m=1.8,
    )
    model = confirmed_world_model()
    path = np.array([[-8.0, -8.0], [3.0, -7.5], [5.2, -5.0]])
    report = validator.validate(
        requested_qr="TARGET_2",
        world_model=model,
        path_xy=path,
        occupancy=model.occupancy,
        rover_start_xy=(-8.0, -8.0),
        rover_final_xy=(5.2, -5.0),
        verified_qr="TARGET_3",
    )
    assert not report.passed
    assert report.failure_reason is FailureReason.QR_VERIFICATION_MISMATCH


def test_validator_rejects_a_path_through_an_obstacle() -> None:
    validator = MissionValidator(
        rover_radius_m=0.3,
        obstacle_safety_margin_m=0.25,
        goal_tolerance_m=0.35,
        approach_distance_m=1.8,
    )
    model = confirmed_world_model()
    straight = np.array([[-8.0, -8.0], [5.2, -5.0]])  # cuts through obstacle_a
    report = validator.validate(
        requested_qr="TARGET_2",
        world_model=model,
        path_xy=straight,
        occupancy=model.occupancy,
        rover_start_xy=(-8.0, -8.0),
        rover_final_xy=(5.2, -5.0),
        verified_qr="TARGET_2",
    )
    assert not report.passed
    assert any(c.name == "path_avoids_obstacles" and not c.passed for c in report.checks)


def test_validator_rejects_a_rover_that_stopped_short() -> None:
    validator = MissionValidator(
        rover_radius_m=0.3,
        obstacle_safety_margin_m=0.25,
        goal_tolerance_m=0.35,
        approach_distance_m=1.8,
    )
    model = confirmed_world_model()
    path = np.array([[-8.0, -8.0], [3.0, -7.5], [5.2, -5.0]])
    report = validator.validate(
        requested_qr="TARGET_2",
        world_model=model,
        path_xy=path,
        occupancy=model.occupancy,
        rover_start_xy=(-8.0, -8.0),
        rover_final_xy=(2.0, -6.0),
        verified_qr="TARGET_2",
    )
    assert not report.passed
    assert any(c.name == "rover_reached_goal" and not c.passed for c in report.checks)


def test_state_machine_refuses_illegal_transitions() -> None:
    machine = StateMachine()
    with pytest.raises(InvalidTransition):
        machine.transition(MissionState.ROVER_NAVIGATING, 0.0)
    machine.transition(MissionState.TAKEOFF, 0.0)
    machine.transition(MissionState.EXPLORING, 1.0)
    with pytest.raises(InvalidTransition):
        machine.transition(MissionState.MISSION_SUCCESS, 2.0)


def test_failure_requires_a_reason_and_is_terminal() -> None:
    machine = StateMachine()
    machine.transition(MissionState.TAKEOFF, 0.0)
    with pytest.raises(ValueError, match="FailureReason"):
        machine.transition(MissionState.MISSION_FAILED, 1.0)
    machine.fail(FailureReason.NAVIGATION_TIMEOUT, "took too long", 1.0)
    assert machine.is_terminal
    with pytest.raises(InvalidTransition):
        machine.fail(FailureReason.MISSION_TIMEOUT, "again", 2.0)


def test_state_trace_is_recorded_for_the_log() -> None:
    machine = StateMachine()
    machine.transition(MissionState.TAKEOFF, 0.0)
    machine.transition(MissionState.EXPLORING, 1.0)
    machine.fail(FailureReason.QR_NOT_DETECTED, "nothing found", 2.0)
    assert machine.describe() == "IDLE -> TAKEOFF -> EXPLORING -> MISSION_FAILED"


def test_mission_timeout_stops_a_stuck_mission() -> None:
    config = MissionConfig()
    orchestrator = MissionOrchestrator(
        config, confirmed_world_model(), requested_qr="TARGET_2"
    )
    orchestrator.update(MissionInputs(now=0.0, drone_ready=True))
    orchestrator.update(
        MissionInputs(now=config.mission.mission_timeout_s + 1.0, drone_ready=True)
    )
    assert orchestrator.machine.failure_reason is FailureReason.MISSION_TIMEOUT
