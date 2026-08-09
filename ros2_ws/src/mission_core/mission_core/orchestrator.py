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
    """Side effect the manager should perform after an :meth:`update` call.

    Only side effects that something actually performs belong here. Entering
    EXPLORING needs no command, for instance: the explorer's ``start()``
    (triggered by ``START_TAKEOFF``) climbs *and* scans, and the transition is
    observable on ``/mission/status``.
    """

    NONE = "NONE"
    START_TAKEOFF = "START_TAKEOFF"
    PUBLISH_PATH = "PUBLISH_PATH"
    #: Ask the rover to sweep its heading while it looks for the code. Wheel
    #: odometry drifts in yaw, so arriving "facing the station" can still leave
    #: it outside the camera - the rover has to look around rather than assume.
    START_VERIFICATION = "START_VERIFICATION"
    #: Stop the current controller, discard its latched status and wait for a
    #: newly planned path. Kept distinct from STOP_ROVER so adapters can reset
    #: the path-publication handshake during recovery.
    PREPARE_REPLAN = "PREPARE_REPLAN"
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
        #: The occupancy grid the accepted path was planned against. The
        #: validator must judge the path on what was known when it was made,
        #: not on obstacles discovered afterwards - the rover's own lidar adds
        #: cells while it drives, and a path can only ever have been checked
        #: against the map that existed at planning time. A new obstacle on the
        #: remaining route is handled live, by the replan path, not by
        #: retroactively condemning a route the rover already completed.
        self.planning_occupancy = None
        self.verified_qr: Optional[str] = None
        self._started_at: Optional[float] = None
        self._exploration_started_at: Optional[float] = None
        self._navigation_started_at: Optional[float] = None
        self._verification_started_at: Optional[float] = None
        self._replans = 0
        self._verification_attempts = 0
        #: Payloads to visit, ordered at run time from where they were found.
        self.tour: List[str] = []
        self._leg = 0
        self._returning_home = False
        #: Where the rover began; also where it drives back to.
        self.home_xy: Optional[np.ndarray] = None
        #: Payload verified at each station the rover actually reached.
        self.verified_qrs: List[str] = []
        #: One validation report per completed leg, aggregated at the end.
        self.leg_reports: List[ValidationReport] = []

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

    @property
    def current_target(self) -> Optional[str]:
        """Payload of the leg being driven, or None on the way home."""
        if self._returning_home or self._leg >= len(self.tour):
            return None
        return self.tour[self._leg]

    def _plan_tour(self, start_xy) -> List[str]:
        """Order the confirmed targets by a greedy nearest-neighbour walk.

        The order comes from *discovered* positions, so it is a consequence of
        perception rather than a configured itinerary. Greedy is enough for
        three stations and, unlike an exact tour, is trivial to explain when
        someone asks why the rover went that way.
        """
        remaining = {
            record.qr_id: record.position[:2]
            for record in self.world_model.confirmed_targets()
            if record.qr_id in self.config.mission.known_payloads
        }
        order: List[str] = []
        cursor = np.asarray(start_xy, dtype=float)[:2]
        while remaining:
            nearest = min(remaining, key=lambda q: float(np.linalg.norm(remaining[q] - cursor)))
            cursor = remaining.pop(nearest)
            order.append(nearest)
        return order

    def _required_payloads(self) -> List[str]:
        """Payloads that must be CONFIRMED before the tour may start."""
        if self.config.mission.visit_all_targets:
            return list(self.config.mission.known_payloads)
        return [self.requested_qr]

    def _build_report(self) -> ValidationReport:
        return self.validator.validate(
            requested_qr=self.current_target or self.requested_qr,
            world_model=self.world_model,
            path_xy=self.path.xy if self.path is not None else None,
            occupancy=(
                self.planning_occupancy
                if self.planning_occupancy is not None
                else self.world_model.occupancy
            ),
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
        return outputs

    def _on_exploring(
        self, inputs: MissionInputs, now: float, outputs: MissionOutputs
    ) -> MissionOutputs:
        required = self._required_payloads()
        for payload in required:
            ambiguous = self.world_model.get_target(payload)
            if ambiguous is not None and ambiguous.status is TargetStatus.AMBIGUOUS:
                return self._fail(
                    FailureReason.DUPLICATE_QR,
                    f"{payload} was observed at more than one location; refusing to "
                    "guess which physical station the operator meant",
                    now,
                    outputs,
                )

        missing = [p for p in required if not self.world_model.has_confirmed(p)]
        confirmed = not missing
        record = self.world_model.get_target(self.requested_qr)
        # Finishing the sweep after the first confirmation yields a complete
        # digital twin (all three stations, both obstacles) instead of a map
        # that happens to contain only what was needed.
        wait_for_full_scan = self.config.drone.finish_scan_after_target_found
        ready = confirmed and (inputs.exploration_complete or not wait_for_full_scan)

        if ready:
            summary = self.world_model.summary()
            outputs.messages.append(
                f"[WORLD_MODEL] {summary['targets_confirmed']} targets confirmed "
                f"{summary['qr_ids']}, {summary['obstacles']} obstacles mapped"
            )
            for payload in required:
                found = self.world_model.get_target(payload)
                if found is not None:
                    outputs.messages.append(
                        f"[MISSION] {payload} found at "
                        f"({found.position[0]:.2f}, {found.position[1]:.2f}) "
                        f"confidence {found.confidence:.2f}"
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
            never_seen = [p for p in missing if self.world_model.get_target(p) is None]
            reason = (
                FailureReason.QR_NOT_DETECTED
                if never_seen
                else FailureReason.TARGET_NOT_DISCOVERED
            )
            details = []
            for payload in missing:
                found = self.world_model.get_target(payload)
                details.append(
                    f"{payload} never decoded"
                    if found is None
                    else f"{payload} is not confirmed (status={found.status.value}, "
                    f"{found.observation_count} observations)"
                )
            return self._fail(
                reason, "the full area was scanned but " + "; ".join(details), now, outputs
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
        if self.home_xy is None:
            self.home_xy = self.rover_start_xy.copy()
        if not self.tour:
            self.tour = (
                self._plan_tour(self.home_xy)
                if self.config.mission.visit_all_targets
                else [self.requested_qr]
            )
            outputs.messages.append(
                "[MISSION] Tour order: "
                + " -> ".join(self.tour)
                + (" -> HOME" if self.config.mission.return_home else "")
            )
        outputs.messages.append(f"[PLANNER] Planning rover path to {self.tour[0]}")
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
        target = self.current_target
        record = None
        if target is not None:
            record = self.world_model.get_target(target)
            if record is None or not record.is_usable:
                return self._fail(
                    FailureReason.TARGET_NOT_DISCOVERED,
                    f"{target} is no longer a usable target at planning time",
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
            if record is not None:
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
            else:
                # Home is a pose the rover already occupied, so it needs no
                # standoff - but it may sit inside the inflation of whatever
                # the drone mapped there (including the rover itself), which
                # the planner's bounded start/goal snapping handles.
                home = self.home_xy if self.home_xy is not None else start
                goal = np.array([home[0], home[1], 0.0])
            path = self.planner.plan(
                inflated, start, goal[:2], goal_yaw=float(goal[2]), pre_inflated=True
            )
        except PlanningError as exc:
            return self._fail(exc.reason, exc.detail or str(exc), now, outputs)

        self.path = path
        self.planning_occupancy = grid
        outputs.path = path
        leg = target if target is not None else "HOME"
        outputs.messages.append(
            f"[PLANNER] Path to {leg} contains {len(path)} poses, {path.length_m:.2f} m, "
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
        if self._returning_home:
            outputs.messages.append("[ROVER] Driving home")
            self._advance(MissionState.RETURNING_HOME, now, outputs)
        else:
            outputs.messages.append(
                f"[ROVER] Navigation started towards {self.current_target}"
            )
            self._advance(MissionState.ROVER_NAVIGATING, now, outputs)
        self._navigation_started_at = now
        return outputs

    def _on_rover_navigating(
        self, inputs: MissionInputs, now: float, outputs: MissionOutputs
    ) -> MissionOutputs:
        if inputs.rover_tracking_failed:
            if self._replans < self.config.rover.max_replans:
                if inputs.rover_pose is None:
                    return self._fail(
                        FailureReason.LOCALIZATION_UNAVAILABLE,
                        "rover pose is unavailable, cannot recover from tracking failure",
                        now,
                        outputs,
                    )
                self._replans += 1
                self.rover_start_xy = np.asarray(inputs.rover_pose, dtype=float)[:2].copy()
                self.path = None
                self._navigation_started_at = None
                outputs.messages.append(
                    f"[RECOVERY] Tracking failed ({inputs.rover_failure_detail or 'no detail'}); "
                    f"replanning from the live rover pose "
                    f"(attempt {self._replans}/{self.config.rover.max_replans})"
                )
                self._advance(
                    MissionState.PLANNING,
                    now,
                    outputs,
                    f"tracking recovery {self._replans}/{self.config.rover.max_replans}",
                )
                outputs.command = MissionCommand.PREPARE_REPLAN
                return outputs
            return self._fail(
                FailureReason.PATH_TRACKING_FAILURE,
                (inputs.rover_failure_detail or "the rover reported a tracking failure")
                + f"; replan budget exhausted ({self._replans}/"
                f"{self.config.rover.max_replans})",
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
        self._verification_attempts = 1
        outputs.command = MissionCommand.START_VERIFICATION
        return outputs

    def _on_verifying_target(
        self, inputs: MissionInputs, now: float, outputs: MissionOutputs
    ) -> MissionOutputs:
        started = self._verification_started_at or now
        if inputs.verified_qr is None:
            if now - started > self.config.verification.timeout_s:
                if self._verification_attempts < self.config.verification.max_attempts:
                    self._verification_attempts += 1
                    self._verification_started_at = now
                    outputs.messages.append(
                        "[RECOVERY] No QR decoded; repeating the verification sweep "
                        f"(attempt {self._verification_attempts}/"
                        f"{self.config.verification.max_attempts})"
                    )
                    outputs.command = MissionCommand.START_VERIFICATION
                    return outputs
                return self._fail(
                    FailureReason.QR_VERIFICATION_TIMEOUT,
                    f"the rover camera decoded no QR code after "
                    f"{self._verification_attempts} verification attempt(s)",
                    now,
                    outputs,
                )
            return outputs

        expected = self.current_target or self.requested_qr
        self.verified_qr = inputs.verified_qr
        outputs.messages.append(f"[QR] Rover detected {inputs.verified_qr}")
        if inputs.verified_qr != expected:
            return self._fail(
                FailureReason.QR_VERIFICATION_MISMATCH,
                f"the rover reached a station carrying {inputs.verified_qr!r} but this leg "
                f"was planned to {expected!r}",
                now,
                outputs,
            )

        leg_report = self._build_report()
        if not leg_report.passed:
            return self._fail(
                leg_report.failure_reason or FailureReason.QR_VERIFICATION_MISMATCH,
                "QR matched but automated validation failed: "
                + "; ".join(c.name for c in leg_report.failures),
                now,
                outputs,
            )

        self.leg_reports.append(leg_report)
        self.verified_qrs.append(expected)
        outputs.messages.append(
            f"[MISSION] {expected} verified "
            f"({len(self.verified_qrs)}/{len(self.tour)} stations)"
        )
        self._leg += 1

        if self._leg < len(self.tour):
            return self._start_next_leg(inputs, now, outputs, returning_home=False)
        if self.config.mission.return_home:
            return self._start_next_leg(inputs, now, outputs, returning_home=True)
        return self._succeed(now, outputs)

    def _start_next_leg(
        self,
        inputs: MissionInputs,
        now: float,
        outputs: MissionOutputs,
        *,
        returning_home: bool,
    ) -> MissionOutputs:
        """Re-plan from where the rover actually is and drive the next leg."""
        if inputs.rover_pose is None:
            return self._fail(
                FailureReason.LOCALIZATION_UNAVAILABLE,
                "rover pose is unavailable, cannot plan the next leg",
                now,
                outputs,
            )
        self._returning_home = returning_home
        # Every leg starts from the pose the rover is actually in, not from
        # where the previous leg's path happened to end.
        self.rover_start_xy = np.asarray(inputs.rover_pose, dtype=float)[:2].copy()
        self.path = None
        self.planning_occupancy = None
        self._navigation_started_at = None
        self._verification_started_at = None
        self._verification_attempts = 0
        self._replans = 0

        destination = "home" if returning_home else self.current_target
        outputs.messages.append(f"[MISSION] Next leg: {destination}")
        self._advance(MissionState.PLANNING, now, outputs, f"leg -> {destination}")
        # Clears the manager's latched tracking status, verified payload and
        # path handshake so the new leg starts from a clean slate.
        outputs.command = MissionCommand.PREPARE_REPLAN
        return outputs

    def _succeed(self, now: float, outputs: MissionOutputs) -> MissionOutputs:
        report = self._aggregate_report()
        self.report = report
        outputs.report = report
        if not report.passed:
            return self._fail(
                report.failure_reason or FailureReason.QR_VERIFICATION_MISMATCH,
                "the tour finished but validation failed: "
                + "; ".join(c.name for c in report.failures),
                now,
                outputs,
            )
        outputs.messages.append(
            f"[MISSION] All stations verified in order: {' -> '.join(self.verified_qrs)}"
        )
        outputs.messages.append("[MISSION] SUCCESS")
        self._advance(MissionState.MISSION_SUCCESS, now, outputs, "all validation checks passed")
        outputs.command = MissionCommand.STOP_ROVER
        return outputs

    def _on_returning_home(
        self, inputs: MissionInputs, now: float, outputs: MissionOutputs
    ) -> MissionOutputs:
        if inputs.rover_tracking_failed:
            return self._fail(
                FailureReason.PATH_TRACKING_FAILURE,
                inputs.rover_failure_detail or "the rover failed to track its route home",
                now,
                outputs,
            )
        started = self._navigation_started_at or now
        if now - started > self.config.rover.navigation_timeout_s:
            return self._fail(
                FailureReason.NAVIGATION_TIMEOUT,
                f"the rover did not get home within "
                f"{self.config.rover.navigation_timeout_s:.0f} s",
                now,
                outputs,
            )
        if not inputs.rover_goal_reached:
            return outputs
        outputs.messages.append("[ROVER] Home")
        return self._succeed(now, outputs)

    def _aggregate_report(self) -> ValidationReport:
        """One report for the whole tour: every leg, plus the drive home."""
        from .validation import Check

        report = ValidationReport()
        for payload, leg in zip(self.verified_qrs, self.leg_reports):
            for check in leg.checks:
                report.add(
                    Check(
                        f"{payload}:{check.name}",
                        check.passed,
                        check.detail,
                        check.mandatory,
                        check.failure_reason,
                    )
                )

        expected = list(self.tour)
        report.add(
            Check(
                "all_stations_verified",
                self.verified_qrs == expected,
                f"verified {self.verified_qrs} against the planned tour {expected}",
                failure_reason=FailureReason.QR_VERIFICATION_MISMATCH,
            )
        )
        if self.config.mission.return_home:
            home = self.home_xy
            final = self.rover_final_xy
            distance = (
                float(np.linalg.norm(np.asarray(final)[:2] - np.asarray(home)[:2]))
                if home is not None and final is not None
                else float("inf")
            )
            # Home is where the rover started, so it is judged by the same goal
            # tolerance as any other destination.
            report.add(
                Check(
                    "returned_home",
                    distance <= self.config.rover.goal_tolerance_m,
                    f"rover finished {distance:.2f} m from its start "
                    f"(tolerance {self.config.rover.goal_tolerance_m:.2f} m)",
                    failure_reason=FailureReason.NAVIGATION_TIMEOUT,
                )
            )
        return report

    # -- introspection -----------------------------------------------------
    def describe(self) -> str:
        return self.machine.describe()

    def status_dict(self) -> dict:
        """Snapshot published on ``/mission/status``."""
        return {
            "state": self.machine.state.value,
            "requested_qr": self.requested_qr,
            "tour": list(self.tour),
            "leg": self._leg,
            "verified_qrs": list(self.verified_qrs),
            "verified_qr": self.verified_qr,
            "failure_reason": self.machine.failure_reason.value,
            "failure_detail": self.machine.failure_detail,
            "path_poses": len(self.path) if self.path is not None else 0,
            "path_length_m": float(self.path.length_m) if self.path is not None else 0.0,
            "world_model": self.world_model.summary(),
            "trace": self.machine.describe(),
        }
