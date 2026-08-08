"""Mission orchestration: the state machine that actually runs the mission.

Kept ROS-free on purpose.  ``mission_manager_node`` collects topic data into a
:class:`MissionInputs` snapshot, calls :meth:`MissionOrchestrator.update`, and
acts on the returned :class:`MissionOutputs`.  The identical call sequence is
driven by the offline harness, so the orchestration logic that ships is the
orchestration logic that was tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

import numpy as np

from .config import MissionConfig
from .errors import FailureReason, PlanningError
from .mission_state import MissionState, StateMachine, StateTransition
from .planner import AStarPlanner, PlannedPath
from .validation import MissionValidator, ValidationReport
from .world_model import TargetStatus, WorldModel


class MissionCommand(str, Enum):
    """Side effect the manager should perform after an :meth:`update` call."""

    NONE = "NONE"
    START_TAKEOFF = "START_TAKEOFF"
    START_EXPLORATION = "START_EXPLORATION"
    PUBLISH_PATH = "PUBLISH_PATH"
    START_VERIFICATION = "START_VERIFICATION"
    STOP_ROVER = "STOP_ROVER"


@dataclass
class MissionInputs:
    """Everything the orchestrator needs to know about the world right now."""

    now: float
    drone_ready: bool = False
    drone_at_scan_altitude: bool = False
    exploration_complete: bool = False
    rover_pose: Optional[np.ndarray] = None
    rover_goal_reached: bool = False
    rover_tracking_failed: bool = False
    rover_failure_detail: str = ""
    #: Payload the rover's own camera has confirmed at the goal, if any.
    verified_qr: Optional[str] = None
    path_published: bool = False


@dataclass
class MissionOutputs:
    """What the manager should do, plus everything worth logging."""

    state: MissionState
    command: MissionCommand = MissionCommand.NONE
    transition: Optional[StateTransition] = None
    path: Optional[PlannedPath] = None
    report: Optional[ValidationReport] = None
    messages: List[str] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.state in (MissionState.MISSION_SUCCESS, MissionState.MISSION_FAILED)


class MissionOrchestrator:
    """Drives the mission from IDLE to a terminal state."""

    def __init__(
        self,
        config: MissionConfig,
        world_model: WorldModel,
        *,
        requested_qr: Optional[str] = None,
        planner: Optional[AStarPlanner] = None,
        validator: Optional[MissionValidator] = None,
    ) -> None:
        self.config = config
        self.world_model = world_model
        self.requested_qr = requested_qr or config.mission.target_qr
        self.planner = planner or AStarPlanner(
            rover_radius_m=config.planner.rover_radius_m,
            safety_margin_m=config.planner.obstacle_safety_margin_m,
            allow_unknown=config.planner.allow_unknown,
            heuristic_weight=config.planner.heuristic_weight,
            max_start_snap_m=config.planner.max_start_snap_m,
            shortcut=config.planner.shortcut,
        )
        self.validator = validator or MissionValidator(
            rover_radius_m=config.planner.rover_radius_m,
            obstacle_safety_margin_m=config.planner.obstacle_safety_margin_m,
            goal_tolerance_m=config.rover.goal_tolerance_m,
            approach_distance_m=config.planner.approach_distance_m,
        )
        self.machine = StateMachine()
        self.path: Optional[PlannedPath] = None
        self.report: Optional[ValidationReport] = None
        #: Rover pose captured when planning started - the path must begin here.
        self.rover_start_xy: Optional[np.ndarray] = None
        self.rover_final_xy: Optional[np.ndarray] = None
        self.verified_qr: Optional[str] = None
        self._started_at: Optional[float] = None
        self._exploration_started_at: Optional[float] = None
        self._navigation_started_at: Optional[float] = None
        self._verification_started_at: Optional[float] = None
        self._replans = 0

    # -- helpers ----------------------------------------------------------
    @property
    def state(self) -> MissionState:
        return self.machine.state

    def _validate_request(self) -> Optional[str]:
        known = self.config.mission.known_payloads
        if known and self.requested_qr not in known:
            return (
                f"requested QR {self.requested_qr!r} is not a configured mission target "
                f"(known: {known})"
            )
        return None

    def _fail(
        self, reason: FailureReason, detail: str, now: float, outputs: MissionOutputs
    ) -> MissionOutputs:
        outputs.transition = self.machine.fail(reason, detail, now)
        outputs.state = self.machine.state
        outputs.command = MissionCommand.STOP_ROVER
        outputs.messages.append(f"[MISSION] FAILED - {reason.value}: {detail}")
        outputs.report = self._build_report()
        self.report = outputs.report
        return outputs

    def _advance(
        self, target: MissionState, now: float, outputs: MissionOutputs, detail: str = ""
    ) -> None:
        outputs.transition = self.machine.transition(target, now, detail)
        outputs.state = self.machine.state

    def _build_report(self) -> ValidationReport:
        return self.validator.validate(
            requested_qr=self.requested_qr,
            world_model=self.world_model,
            path_xy=self.path.xy if self.path is not None else None,
            occupancy=self.world_model.occupancy,
            rover_start_xy=self.rover_start_xy,
            rover_final_xy=self.rover_final_xy,
            verified_qr=self.verified_qr,
        )

    # -- main loop --------------------------------------------------------
    def update(self, inputs: MissionInputs) -> MissionOutputs:
        """Advance the mission by one tick."""
        now = float(inputs.now)
        outputs = MissionOutputs(state=self.machine.state)
        if self.machine.is_terminal:
            return outputs

        if self._started_at is None:
            self._started_at = now
            self.machine.started_at = now
        if inputs.rover_pose is not None:
            self.rover_final_xy = np.asarray(inputs.rover_pose, dtype=float)[:2].copy()

        # Global watchdog: no state may run forever.
        if now - self._started_at > self.config.mission.mission_timeout_s:
            return self._fail(
                FailureReason.MISSION_TIMEOUT,
                f"mission exceeded {self.config.mission.mission_timeout_s:.0f} s "
                f"in state {self.machine.state.value}",
                now,
                outputs,
            )

        handler = getattr(self, f"_on_{self.machine.state.value.lower()}")
        return handler(inputs, now, outputs)

    # -- per-state handlers ------------------------------------------------
    def _on_idle(self, inputs: MissionInputs, now: float, outputs: MissionOutputs) -> MissionOutputs:
        problem = self._validate_request()
        if problem is not None:
            return self._fail(FailureReason.UNKNOWN_TARGET_REQUESTED, problem, now, outputs)
        if not inputs.drone_ready:
            return outputs
        outputs.messages.append(f"[MISSION] Requested QR: {self.requested_qr}")
        outputs.messages.append("[DRONE] Taking off")
        self._advance(MissionState.TAKEOFF, now, outputs, f"target={self.requested_qr}")
        outputs.command = MissionCommand.START_TAKEOFF
        return outputs

    def _on_takeoff(
        self, inputs: MissionInputs, now: float, outputs: MissionOutputs
    ) -> MissionOutputs:
        if not inputs.drone_at_scan_altitude:
            return outputs
        outputs.messages.append(
            f"[DRONE] Reached scan altitude {self.config.drone.scan_altitude_m:.1f} m; "
            "exploration started"
        )
        self._advance(MissionState.EXPLORING, now, outputs)
        self._exploration_started_at = now
        outputs.command = MissionCommand.START_EXPLORATION
        return outputs

    def _on_exploring(
        self, inputs: MissionInputs, now: float, outputs: MissionOutputs
    ) -> MissionOutputs:
        record = self.world_model.get_target(self.requested_qr)
        if record is not None and record.status is TargetStatus.AMBIGUOUS:
            return self._fail(
                FailureReason.DUPLICATE_QR,
                f"{self.requested_qr} was observed at more than one location; refusing to "
                "guess which physical station the operator meant",
                now,
                outputs,
            )

        confirmed = record is not None and record.is_usable
        # Finishing the sweep after the first confirmation yields a complete
        # digital twin (all three stations, both obstacles) instead of a map
        # that happens to contain only what was needed.
        wait_for_full_scan = self.config.drone.finish_scan_after_target_found
        ready = confirmed and (inputs.exploration_complete or not wait_for_full_scan)

        if ready and record is not None:
            summary = self.world_model.summary()
            outputs.messages.append(
                f"[WORLD_MODEL] {summary['targets_confirmed']} targets confirmed "
                f"{summary['qr_ids']}, {summary['obstacles']} obstacles mapped"
            )
            outputs.messages.append(
                f"[MISSION] Requested target {self.requested_qr} found at "
                f"({record.position[0]:.2f}, {record.position[1]:.2f}) "
                f"confidence {record.confidence:.2f}"
            )
            self._advance(MissionState.TARGET_FOUND, now, outputs)
            return outputs

        started = self._exploration_started_at or now
        if now - started > self.config.drone.exploration_timeout_s:
            return self._fail(
                FailureReason.TARGET_NOT_DISCOVERED,
                f"exploration timed out after {self.config.drone.exploration_timeout_s:.0f} s "
                f"without confirming {self.requested_qr}; "
                f"seen so far: {self.world_model.summary()['qr_ids']}",
                now,
                outputs,
            )
        if inputs.exploration_complete and not confirmed:
            reason = (
                FailureReason.QR_NOT_DETECTED
                if record is None
                else FailureReason.TARGET_NOT_DISCOVERED
            )
            return self._fail(
                reason,
                f"the full area was scanned but {self.requested_qr} is "
                + (
                    "not confirmed (status="
                    f"{record.status.value}, {record.observation_count} observations)"
                    if record is not None
                    else "never decoded"
                ),
                now,
                outputs,
            )
        return outputs

    def _on_target_found(
        self, inputs: MissionInputs, now: float, outputs: MissionOutputs
    ) -> MissionOutputs:
        if inputs.rover_pose is None:
            return self._fail(
                FailureReason.LOCALIZATION_UNAVAILABLE,
                "rover odometry is unavailable, cannot plan from its position",
                now,
                outputs,
            )
        self.rover_start_xy = np.asarray(inputs.rover_pose, dtype=float)[:2].copy()
        outputs.messages.append("[PLANNER] Planning rover path")
        self._advance(MissionState.PLANNING, now, outputs)
        return outputs

    def _on_planning(
        self, inputs: MissionInputs, now: float, outputs: MissionOutputs
    ) -> MissionOutputs:
        grid = self.world_model.occupancy
        if grid is None:
            return self._fail(
                FailureReason.NO_VALID_PATH,
                "no occupancy grid has been produced; the drone mapped nothing",
                now,
                outputs,
            )
        record = self.world_model.get_target(self.requested_qr)
        if record is None or not record.is_usable:
            return self._fail(
                FailureReason.TARGET_NOT_DISCOVERED,
                f"{self.requested_qr} is no longer a usable target at planning time",
                now,
                outputs,
            )
        start = self.rover_start_xy
        if start is None:
            return self._fail(
                FailureReason.LOCALIZATION_UNAVAILABLE,
                "rover start position was never captured",
                now,
                outputs,
            )

        try:
            # Inflate once and reuse: the approach search and A* must agree
            # exactly on what counts as free space.
            inflated = self.planner.inflate(grid)
            goal = self.planner.select_approach_pose(
                grid,
                record.position[:2],
                start,
                approach_distance_m=self.config.planner.approach_distance_m,
                target_clearance_m=self.config.planner.target_footprint_radius_m,
                inflated_grid=inflated,
                distance_tolerance_m=self.config.planner.approach_distance_tolerance_m,
                samples=self.config.planner.approach_samples,
            )
            path = self.planner.plan(
                inflated, start, goal[:2], goal_yaw=float(goal[2]), pre_inflated=True
            )
        except PlanningError as exc:
            return self._fail(exc.reason, exc.detail or str(exc), now, outputs)

        self.path = path
        outputs.path = path
        outputs.messages.append(
            f"[PLANNER] Path contains {len(path)} poses, {path.length_m:.2f} m, "
            f"{path.expanded_nodes} nodes expanded, "
            f"clearance {path.inflation_radius_m:.2f} m"
        )
        self._advance(MissionState.PATH_READY, now, outputs)
        return outputs

    def _on_path_ready(
        self, inputs: MissionInputs, now: float, outputs: MissionOutputs
    ) -> MissionOutputs:
        outputs.messages.append("[MISSION] Path sent to rover")
        self._advance(MissionState.SENDING_PATH, now, outputs)
        outputs.command = MissionCommand.PUBLISH_PATH
        outputs.path = self.path
        return outputs

    def _on_sending_path(
        self, inputs: MissionInputs, now: float, outputs: MissionOutputs
    ) -> MissionOutputs:
        if not inputs.path_published:
            # Re-issue rather than assume: the rover may not have been
            # subscribed when the first message went out.
            outputs.command = MissionCommand.PUBLISH_PATH
            outputs.path = self.path
            return outputs
        outputs.messages.append("[ROVER] Navigation started")
        self._advance(MissionState.ROVER_NAVIGATING, now, outputs)
        self._navigation_started_at = now
        return outputs

    def _on_rover_navigating(
        self, inputs: MissionInputs, now: float, outputs: MissionOutputs
    ) -> MissionOutputs:
        if inputs.rover_tracking_failed:
            return self._fail(
                FailureReason.PATH_TRACKING_FAILURE,
                inputs.rover_failure_detail or "the rover reported a tracking failure",
                now,
                outputs,
            )
        started = self._navigation_started_at or now
        if now - started > self.config.rover.navigation_timeout_s:
            return self._fail(
                FailureReason.NAVIGATION_TIMEOUT,
                f"the rover did not reach the goal within "
                f"{self.config.rover.navigation_timeout_s:.0f} s",
                now,
                outputs,
            )
        if not inputs.rover_goal_reached:
            return outputs
        outputs.messages.append("[ROVER] Goal reached")
        self._advance(MissionState.VERIFYING_TARGET, now, outputs)
        self._verification_started_at = now
        outputs.command = MissionCommand.START_VERIFICATION
        return outputs

    def _on_verifying_target(
        self, inputs: MissionInputs, now: float, outputs: MissionOutputs
    ) -> MissionOutputs:
        started = self._verification_started_at or now
        if inputs.verified_qr is None:
            if now - started > self.config.verification.timeout_s:
                return self._fail(
                    FailureReason.QR_VERIFICATION_TIMEOUT,
                    f"the rover camera decoded no QR code within "
                    f"{self.config.verification.timeout_s:.0f} s of arriving",
                    now,
                    outputs,
                )
            return outputs

        self.verified_qr = inputs.verified_qr
        outputs.messages.append(f"[QR] Rover detected {inputs.verified_qr}")
        if inputs.verified_qr != self.requested_qr:
            return self._fail(
                FailureReason.QR_VERIFICATION_MISMATCH,
                f"the rover reached a station carrying {inputs.verified_qr!r} but the mission "
                f"requested {self.requested_qr!r}",
                now,
                outputs,
            )

        report = self._build_report()
        self.report = report
        outputs.report = report
        if not report.passed:
            return self._fail(
                report.failure_reason or FailureReason.QR_VERIFICATION_MISMATCH,
                "QR matched but automated validation failed: "
                + "; ".join(c.name for c in report.failures),
                now,
                outputs,
            )
        outputs.messages.append("[MISSION] QR verification successful")
        outputs.messages.append("[MISSION] SUCCESS")
        self._advance(MissionState.MISSION_SUCCESS, now, outputs, "all validation checks passed")
        outputs.command = MissionCommand.STOP_ROVER
        return outputs

    # -- introspection -----------------------------------------------------
    def describe(self) -> str:
        return self.machine.describe()

    def status_dict(self) -> dict:
        """Snapshot published on ``/mission/status``."""
        return {
            "state": self.machine.state.value,
            "requested_qr": self.requested_qr,
            "verified_qr": self.verified_qr,
            "failure_reason": self.machine.failure_reason.value,
            "failure_detail": self.machine.failure_detail,
            "path_poses": len(self.path) if self.path is not None else 0,
            "path_length_m": float(self.path.length_m) if self.path is not None else 0.0,
            "world_model": self.world_model.summary(),
            "trace": self.machine.describe(),
        }
