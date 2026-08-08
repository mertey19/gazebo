"""Pure-pursuit path follower for the differential-drive rover.

Every gain and threshold is a constructor argument sourced from YAML - there
are no tuned-in-place literals in the control law.  The controller is a pure
function of (path, pose, dt): it holds only the progress index and the
tracking-error timer, which makes it directly unit testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np

from .errors import FailureReason
from .geometry import normalize_angle


class FollowerState(str, Enum):
    """Outcome of one control step."""

    IDLE = "IDLE"
    TRACKING = "TRACKING"
    ALIGNING = "ALIGNING"
    GOAL_REACHED = "GOAL_REACHED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class TrackingStatus:
    """Command plus diagnostics for one control step."""

    state: FollowerState
    linear_velocity: float
    angular_velocity: float
    distance_to_goal_m: float
    cross_track_error_m: float
    heading_error_rad: float
    progress_index: int
    lookahead_point: Optional[np.ndarray] = None
    failure_reason: FailureReason = FailureReason.NONE
    detail: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.state in (FollowerState.GOAL_REACHED, FollowerState.FAILED)


def _closest_point_on_segment(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> tuple[np.ndarray, float]:
    """Projection of ``point`` onto segment ``start``-``end`` and its parameter."""
    segment = end - start
    length_sq = float(segment @ segment)
    if length_sq < 1e-12:
        return start.copy(), 0.0
    t = float(np.clip((point - start) @ segment / length_sq, 0.0, 1.0))
    return start + t * segment, t


class PurePursuitController:
    """Track a polyline with a differential-drive base.

    The classic pure-pursuit curvature law ``kappa = 2*y_local / L^2`` is
    augmented with two behaviours that matter in practice:

    * **rotate in place** when the lookahead point is far off-heading, because
      a differential drive can do that and it removes the large initial arc
      that would otherwise take the rover through an obstacle;
    * **speed scheduling** by heading error and remaining distance, so the
      rover actually stops inside ``goal_tolerance_m`` instead of orbiting it.
    """

    def __init__(
        self,
        *,
        lookahead_distance_m: float = 0.9,
        min_lookahead_distance_m: float = 0.35,
        lookahead_time_s: float = 0.6,
        max_linear_velocity: float = 0.8,
        max_angular_velocity: float = 1.4,
        heading_kp: float = 2.0,
        rotate_in_place_rad: float = 0.7,
        goal_tolerance_m: float = 0.35,
        goal_yaw_tolerance_rad: float = 0.25,
        align_to_goal_yaw: bool = True,
        max_cross_track_error_m: float = 1.5,
        cross_track_grace_s: float = 3.0,
        approach_slowdown_m: float = 1.0,
    ) -> None:
        if lookahead_distance_m <= 0.0 or min_lookahead_distance_m <= 0.0:
            raise ValueError("lookahead distances must be positive")
        if max_linear_velocity <= 0.0 or max_angular_velocity <= 0.0:
            raise ValueError("velocity limits must be positive")
        if goal_tolerance_m <= 0.0:
            raise ValueError("goal_tolerance_m must be positive")
        self.lookahead_distance_m = float(lookahead_distance_m)
        self.min_lookahead_distance_m = float(min_lookahead_distance_m)
        self.lookahead_time_s = float(lookahead_time_s)
        self.max_linear_velocity = float(max_linear_velocity)
        self.max_angular_velocity = float(max_angular_velocity)
        self.heading_kp = float(heading_kp)
        self.rotate_in_place_rad = float(rotate_in_place_rad)
        self.goal_tolerance_m = float(goal_tolerance_m)
        self.goal_yaw_tolerance_rad = float(goal_yaw_tolerance_rad)
        self.align_to_goal_yaw = bool(align_to_goal_yaw)
        self.max_cross_track_error_m = float(max_cross_track_error_m)
        self.cross_track_grace_s = float(cross_track_grace_s)
        self.approach_slowdown_m = float(approach_slowdown_m)

        self._path: Optional[np.ndarray] = None
        self._goal_yaw: float = 0.0
        self._progress_index: int = 0
        self._error_time_s: float = 0.0

    # -- lifecycle -------------------------------------------------------
    def set_path(self, path: np.ndarray) -> None:
        """Accept a new ``(N, >=2)`` path and reset progress."""
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[0] == 0 or path.shape[1] < 2:
            raise ValueError(f"path must have shape (N>=1, >=2), got {path.shape}")
        if not np.isfinite(path).all():
            raise ValueError("path contains non-finite coordinates")
        self._path = path
        self._goal_yaw = float(path[-1, 2]) if path.shape[1] >= 3 else 0.0
        self._progress_index = 0
        self._error_time_s = 0.0

    def reset(self) -> None:
        self._path = None
        self._progress_index = 0
        self._error_time_s = 0.0

    @property
    def has_path(self) -> bool:
        return self._path is not None and len(self._path) > 0

    @property
    def progress_index(self) -> int:
        return self._progress_index

    # -- control ---------------------------------------------------------
    def compute(self, pose: Sequence[float], dt: float) -> TrackingStatus:
        """One control step. ``pose`` is ``(x, y, yaw)`` in the planning frame."""
        if self._path is None:
            return TrackingStatus(FollowerState.IDLE, 0.0, 0.0, math.inf, 0.0, 0.0, 0)

        position = np.asarray(pose, dtype=float)[:2]
        yaw = float(pose[2])
        path_xy = self._path[:, :2]
        goal = path_xy[-1]
        distance_to_goal = float(np.linalg.norm(goal - position))

        cross_track, self._progress_index = self._update_progress(position)

        # -- terminal conditions
        if distance_to_goal <= self.goal_tolerance_m:
            yaw_error = normalize_angle(self._goal_yaw - yaw)
            if not self.align_to_goal_yaw or abs(yaw_error) <= self.goal_yaw_tolerance_rad:
                return TrackingStatus(
                    FollowerState.GOAL_REACHED,
                    0.0,
                    0.0,
                    distance_to_goal,
                    cross_track,
                    yaw_error,
                    self._progress_index,
                )
            # In the tolerance disc but pointing the wrong way: spin to face the
            # station so the camera can actually see the QR code.
            angular = float(
                np.clip(
                    self.heading_kp * yaw_error,
                    -self.max_angular_velocity,
                    self.max_angular_velocity,
                )
            )
            return TrackingStatus(
                FollowerState.ALIGNING,
                0.0,
                angular,
                distance_to_goal,
                cross_track,
                yaw_error,
                self._progress_index,
            )

        # -- tracking failure watchdog
        if cross_track > self.max_cross_track_error_m:
            self._error_time_s += max(0.0, float(dt))
            if self._error_time_s >= self.cross_track_grace_s:
                return TrackingStatus(
                    FollowerState.FAILED,
                    0.0,
                    0.0,
                    distance_to_goal,
                    cross_track,
                    0.0,
                    self._progress_index,
                    failure_reason=FailureReason.PATH_TRACKING_FAILURE,
                    detail=(
                        f"cross-track error {cross_track:.2f} m exceeded "
                        f"{self.max_cross_track_error_m:.2f} m for "
                        f"{self._error_time_s:.1f} s"
                    ),
                )
        else:
            self._error_time_s = 0.0

        lookahead = self._lookahead_point(position, distance_to_goal)
        to_lookahead = lookahead - position
        heading_error = normalize_angle(math.atan2(to_lookahead[1], to_lookahead[0]) - yaw)

        if abs(heading_error) > self.rotate_in_place_rad:
            angular = float(
                np.clip(
                    self.heading_kp * heading_error,
                    -self.max_angular_velocity,
                    self.max_angular_velocity,
                )
            )
            return TrackingStatus(
                FollowerState.TRACKING,
                0.0,
                angular,
                distance_to_goal,
                cross_track,
                heading_error,
                self._progress_index,
                lookahead_point=lookahead,
            )

        # Pure-pursuit curvature from the lookahead point in the body frame.
        local_y = -math.sin(yaw) * to_lookahead[0] + math.cos(yaw) * to_lookahead[1]
        chord = float(np.linalg.norm(to_lookahead))
        curvature = 2.0 * local_y / max(chord * chord, 1e-6)

        linear = self.max_linear_velocity
        # Ease off in turns and on final approach so the goal disc is captured
        # rather than overshot.
        linear *= float(np.clip(1.0 - abs(heading_error) / self.rotate_in_place_rad, 0.15, 1.0))
        if distance_to_goal < self.approach_slowdown_m:
            linear *= float(np.clip(distance_to_goal / self.approach_slowdown_m, 0.2, 1.0))
        angular = float(
            np.clip(linear * curvature, -self.max_angular_velocity, self.max_angular_velocity)
        )
        # Respect the angular limit by slowing down rather than by cutting the
        # turn, which would push the rover off the collision-checked path.
        if abs(linear * curvature) > self.max_angular_velocity and abs(curvature) > 1e-6:
            linear = min(linear, self.max_angular_velocity / abs(curvature))

        return TrackingStatus(
            FollowerState.TRACKING,
            linear,
            angular,
            distance_to_goal,
            cross_track,
            heading_error,
            self._progress_index,
            lookahead_point=lookahead,
        )

    # -- internals -------------------------------------------------------
    def _update_progress(self, position: np.ndarray) -> tuple[float, int]:
        """Closest point search restricted to forward progress along the path."""
        assert self._path is not None
        path_xy = self._path[:, :2]
        if len(path_xy) == 1:
            return float(np.linalg.norm(path_xy[0] - position)), 0

        best_distance = math.inf
        best_index = self._progress_index
        # Searching only forward from the current index prevents the rover from
        # "snapping back" to an earlier leg when the path crosses near itself.
        for index in range(self._progress_index, len(path_xy) - 1):
            projection, _ = _closest_point_on_segment(position, path_xy[index], path_xy[index + 1])
            distance = float(np.linalg.norm(projection - position))
            if distance < best_distance:
                best_distance = distance
                best_index = index
        return best_distance, best_index

    def _lookahead_point(self, position: np.ndarray, distance_to_goal: float) -> np.ndarray:
        """First path point at least ``lookahead`` metres ahead of the rover."""
        assert self._path is not None
        path_xy = self._path[:, :2]
        lookahead = max(
            self.min_lookahead_distance_m,
            min(self.lookahead_distance_m, distance_to_goal),
        )

        accumulated = 0.0
        projection, _ = _closest_point_on_segment(
            position,
            path_xy[self._progress_index],
            path_xy[min(self._progress_index + 1, len(path_xy) - 1)],
        )
        previous = projection
        for index in range(self._progress_index + 1, len(path_xy)):
            segment = path_xy[index] - previous
            length = float(np.linalg.norm(segment))
            if accumulated + length >= lookahead:
                remaining = lookahead - accumulated
                return previous + segment * (remaining / max(length, 1e-9))
            accumulated += length
            previous = path_xy[index]
        return path_xy[-1]

    def lookahead_for_speed(self, speed: float) -> float:
        """Speed-scheduled lookahead, exposed for tuning and tests."""
        return float(
            np.clip(
                speed * self.lookahead_time_s,
                self.min_lookahead_distance_m,
                self.lookahead_distance_m,
            )
        )


class VerificationSweep:
    """Rotate in place looking for a code that should already be in view.

    The rover arrives facing where its odometry *believes* the station is.
    Wheel odometry accumulates yaw error, so that belief can be off by more
    than the camera's half field of view, leaving the station just outside the
    frame - which is indistinguishable, from the logs, from "there is no code
    here".

    A bounded alternating sweep (right, then left through twice the amplitude,
    then back) resolves it in a few seconds and costs nothing when the heading
    was right all along, because verification succeeds on the first frames and
    the sweep is cancelled.
    """

    def __init__(self, sweep_rad: float, yaw_rate_rad_s: float) -> None:
        if sweep_rad <= 0.0 or yaw_rate_rad_s <= 0.0:
            raise ValueError("sweep amplitude and rate must be positive")
        self.sweep_rad = float(sweep_rad)
        self.yaw_rate_rad_s = float(yaw_rate_rad_s)
        # Right by A, left by 2A, right by A: ends where it started, having
        # covered [-A, +A] continuously.
        self._legs = (-self.sweep_rad, 2.0 * self.sweep_rad, -self.sweep_rad)
        self._leg = 0
        self._travelled = 0.0

    def reset(self) -> None:
        self._leg = 0
        self._travelled = 0.0

    @property
    def finished(self) -> bool:
        return self._leg >= len(self._legs)

    @property
    def progress(self) -> float:
        """Fraction of the total sweep completed, in ``[0, 1]``."""
        total = sum(abs(leg) for leg in self._legs)
        done = sum(abs(leg) for leg in self._legs[: self._leg]) + abs(self._travelled)
        return float(np.clip(done / total, 0.0, 1.0))

    def step(self, dt: float) -> float:
        """Yaw rate for this tick; 0.0 once the sweep is complete."""
        if self.finished or dt <= 0.0:
            return 0.0
        target = self._legs[self._leg]
        direction = 1.0 if target > 0.0 else -1.0
        remaining = abs(target) - abs(self._travelled)
        if remaining <= 0.0:
            self._leg += 1
            self._travelled = 0.0
            return self.step(dt)
        # Never overshoot the leg: the last tick is scaled to what is left.
        rate = direction * min(self.yaw_rate_rad_s, remaining / dt)
        self._travelled += rate * dt
        return float(rate)


class DifferentialDriveModel:
    """Unicycle kinematics used by the offline harness and controller tests.

    Not part of the ROS runtime - in simulation the ``DiffDrive`` plugin owns
    the real dynamics.  Keeping an explicit model here lets the controller be
    exercised deterministically without a simulator.
    """

    def __init__(
        self,
        pose: Sequence[float],
        *,
        max_linear_acceleration: float = 1.5,
        max_angular_acceleration: float = 4.0,
    ) -> None:
        self.pose = np.asarray(pose, dtype=float).reshape(3).copy()
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.max_linear_acceleration = float(max_linear_acceleration)
        self.max_angular_acceleration = float(max_angular_acceleration)

    def step(self, linear_command: float, angular_command: float, dt: float) -> np.ndarray:
        """Advance one timestep with first-order actuator rate limiting."""
        dt = float(dt)
        if dt <= 0.0:
            return self.pose.copy()
        max_dv = self.max_linear_acceleration * dt
        max_dw = self.max_angular_acceleration * dt
        self.linear_velocity += float(
            np.clip(linear_command - self.linear_velocity, -max_dv, max_dv)
        )
        self.angular_velocity += float(
            np.clip(angular_command - self.angular_velocity, -max_dw, max_dw)
        )
        # Midpoint integration: with the yaw taken at the half-step the rover
        # tracks arcs without the systematic outward bias of forward Euler.
        mid_yaw = self.pose[2] + 0.5 * self.angular_velocity * dt
        self.pose[0] += self.linear_velocity * math.cos(mid_yaw) * dt
        self.pose[1] += self.linear_velocity * math.sin(mid_yaw) * dt
        self.pose[2] = normalize_angle(self.pose[2] + self.angular_velocity * dt)
        return self.pose.copy()
