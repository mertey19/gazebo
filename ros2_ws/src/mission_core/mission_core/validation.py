"""Automated mission validation.

The mission manager may only declare ``MISSION_SUCCESS`` when every mandatory
check here passes.  The validator re-derives its own inflated grid and does not
share any state with the planner, so a planner bug cannot mark itself correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .errors import FailureReason
from .occupancy import OccupancyGrid
from .planner import path_intersects_obstacles
from .world_model import WorldModel


@dataclass(frozen=True)
class Check:
    """One named pass/fail assertion about the mission outcome."""

    name: str
    passed: bool
    detail: str = ""
    mandatory: bool = True
    failure_reason: FailureReason = FailureReason.NONE

    def __str__(self) -> str:
        mark = "PASS" if self.passed else ("FAIL" if self.mandatory else "WARN")
        suffix = f" - {self.detail}" if self.detail else ""
        return f"[{mark}] {self.name}{suffix}"


@dataclass
class ValidationReport:
    """Aggregated result of all mission checks."""

    checks: List[Check] = field(default_factory=list)

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.mandatory)

    @property
    def failures(self) -> List[Check]:
        return [c for c in self.checks if c.mandatory and not c.passed]

    @property
    def failure_reason(self) -> FailureReason:
        for check in self.failures:
            if check.failure_reason is not FailureReason.NONE:
                return check.failure_reason
        return FailureReason.NONE

    def render(self) -> str:
        header = "MISSION VALIDATION: " + ("PASSED" if self.passed else "FAILED")
        return "\n".join([header, *(f"  {c}" for c in self.checks)])


class MissionValidator:
    """Runs the mandatory success criteria against the recorded mission facts."""

    def __init__(
        self,
        *,
        rover_radius_m: float,
        obstacle_safety_margin_m: float,
        goal_tolerance_m: float,
        approach_distance_m: float,
        start_tolerance_m: float = 1.0,
    ) -> None:
        self.rover_radius_m = float(rover_radius_m)
        self.obstacle_safety_margin_m = float(obstacle_safety_margin_m)
        self.goal_tolerance_m = float(goal_tolerance_m)
        self.approach_distance_m = float(approach_distance_m)
        self.start_tolerance_m = float(start_tolerance_m)

    @property
    def clearance_m(self) -> float:
        return self.rover_radius_m + self.obstacle_safety_margin_m

    def validate(
        self,
        *,
        requested_qr: str,
        world_model: WorldModel,
        path_xy: Optional[np.ndarray],
        occupancy: Optional[OccupancyGrid],
        rover_start_xy: Optional[Sequence[float]],
        rover_final_xy: Optional[Sequence[float]],
        verified_qr: Optional[str],
    ) -> ValidationReport:
        report = ValidationReport()

        # 1 - the requested code was actually discovered by perception
        record = world_model.get_target(requested_qr)
        discovered = record is not None and record.is_usable
        report.add(
            Check(
                "requested_qr_discovered",
                discovered,
                (
                    f"{requested_qr} confirmed at "
                    f"({record.position[0]:.2f}, {record.position[1]:.2f}) "
                    f"from {record.observation_count} observations, "
                    f"confidence {record.confidence:.2f}"
                    if discovered and record is not None
                    else (
                        f"{requested_qr} status="
                        f"{record.status.value if record else 'NEVER_OBSERVED'}"
                    )
                ),
                failure_reason=(
                    FailureReason.NONE if discovered else FailureReason.TARGET_NOT_DISCOVERED
                ),
            )
        )

        # 2/3 - a usable path exists
        has_path = path_xy is not None and len(np.asarray(path_xy)) > 0
        report.add(
            Check(
                "path_generated",
                bool(has_path),
                f"{len(np.asarray(path_xy)) if has_path else 0} poses",
                failure_reason=FailureReason.NONE if has_path else FailureReason.NO_VALID_PATH,
            )
        )
        path = np.asarray(path_xy, dtype=float)[:, :2] if has_path else None
        report.add(
            Check(
                "path_non_empty",
                path is not None and len(path) >= 2,
                "a single-pose path cannot be followed",
                failure_reason=FailureReason.NO_VALID_PATH,
            )
        )

        # 4 - the path clears every mapped obstacle by the full robot clearance
        if path is not None and occupancy is not None and len(path) >= 2:
            collides = path_intersects_obstacles(
                path, occupancy, clearance_m=self.clearance_m, unknown_is_occupied=False
            )
            report.add(
                Check(
                    "path_avoids_obstacles",
                    not collides,
                    f"checked against observed grid inflated by {self.clearance_m:.2f} m",
                    failure_reason=FailureReason.PATH_INTERSECTS_OBSTACLE,
                )
            )
        else:
            report.add(
                Check(
                    "path_avoids_obstacles",
                    False,
                    "no path or no occupancy grid available to check",
                    failure_reason=FailureReason.NO_VALID_PATH,
                )
            )

        # 5 - the path really starts where the rover is (no teleporting)
        if path is not None and rover_start_xy is not None and len(path) >= 1:
            offset = float(np.linalg.norm(path[0] - np.asarray(rover_start_xy, dtype=float)[:2]))
            report.add(
                Check(
                    "path_starts_at_rover",
                    offset <= self.start_tolerance_m,
                    f"path start is {offset:.2f} m from the rover (limit {self.start_tolerance_m:.2f} m)",
                )
            )

        # 6 - the path ends at a sane standoff from the discovered station
        if path is not None and discovered and record is not None and len(path) >= 1:
            standoff = float(np.linalg.norm(path[-1] - record.position[:2]))
            # Allow a generous band: the approach ring is a target, not a hard
            # constraint, and the planner may snap the goal out of inflation.
            upper = self.approach_distance_m + self.goal_tolerance_m + self.clearance_m
            report.add(
                Check(
                    "path_ends_at_target",
                    standoff <= upper,
                    f"path end is {standoff:.2f} m from {requested_qr} (limit {upper:.2f} m)",
                )
            )

        # 7 - the rover physically arrived
        if path is not None and rover_final_xy is not None and len(path) >= 1:
            residual = float(
                np.linalg.norm(path[-1] - np.asarray(rover_final_xy, dtype=float)[:2])
            )
            report.add(
                Check(
                    "rover_reached_goal",
                    residual <= self.goal_tolerance_m,
                    f"rover stopped {residual:.2f} m from the goal "
                    f"(tolerance {self.goal_tolerance_m:.2f} m)",
                    failure_reason=FailureReason.NAVIGATION_TIMEOUT,
                )
            )
        else:
            report.add(
                Check(
                    "rover_reached_goal",
                    False,
                    "no final rover pose recorded",
                    failure_reason=FailureReason.NAVIGATION_TIMEOUT,
                )
            )

        # 8 - the decisive check: the rover's own camera confirms the station
        matched = verified_qr is not None and verified_qr == requested_qr
        report.add(
            Check(
                "rover_qr_verification",
                matched,
                (
                    f"rover camera decoded {verified_qr!r}, requested {requested_qr!r}"
                    if verified_qr is not None
                    else "rover camera never decoded a code at the goal"
                ),
                failure_reason=(
                    FailureReason.NONE
                    if matched
                    else (
                        FailureReason.QR_VERIFICATION_MISMATCH
                        if verified_qr
                        else FailureReason.QR_VERIFICATION_TIMEOUT
                    )
                ),
            )
        )

        # Non-mandatory: how complete the digital twin ended up being.
        summary = world_model.summary()
        report.add(
            Check(
                "world_model_populated",
                int(summary["targets_confirmed"]) > 0 and int(summary["obstacles"]) > 0,
                f"{summary['targets_confirmed']} confirmed targets, "
                f"{summary['obstacles']} obstacles, "
                f"{float(summary['map_known_fraction']):.0%} of the grid observed",
                mandatory=False,
            )
        )
        return report
