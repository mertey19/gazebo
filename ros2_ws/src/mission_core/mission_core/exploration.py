"""Drone exploration: coverage pattern generation and the flight controller.

Both the ROS ``drone_explorer`` node and the offline harness drive the drone
through this module, so a trajectory discrepancy between them is impossible by
construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence

import numpy as np

from .geometry import normalize_angle


class FlightPhase(str, Enum):
    """Coarse phase of the scout flight."""

    GROUNDED = "GROUNDED"
    CLIMBING = "CLIMBING"
    SCANNING = "SCANNING"
    COMPLETE = "COMPLETE"
    #: Holding station on the rover instead of flying the coverage pattern.
    ESCORTING = "ESCORTING"
    #: Flying back to the take-off point at cruise altitude.
    RETURNING = "RETURNING"
    #: Back on the ground where it started.
    LANDED = "LANDED"


@dataclass(frozen=True)
class FlightCommand:
    """Velocity command in the ``map`` frame plus the phase that produced it."""

    phase: FlightPhase
    velocity_map: np.ndarray
    yaw_rate: float
    waypoint_index: int
    distance_to_waypoint_m: float

    @property
    def is_complete(self) -> bool:
        return self.phase is FlightPhase.COMPLETE


def lawnmower_waypoints(
    area_min: Sequence[float],
    area_max: Sequence[float],
    altitude: float,
    lane_spacing: float,
    margin: float,
    forward_offset: float = 0.0,
) -> np.ndarray:
    """Boustrophedon coverage pattern over the mission area.

    Lane spacing must be smaller than the mapped ground swath or the scan
    leaves unobserved strips; :meth:`MissionConfig.validate` enforces that.

    ``forward_offset`` shifts every lane *backwards* along the scan axis by the
    distance between the vehicle and the nearest ground its camera can see.
    The pattern has to cover the area the sensor sweeps, not the area the
    vehicle overflies, and with a camera pitched down rather than a lidar
    pointing down those are not the same region: the near edge of a depressed
    camera's view is metres ahead of the aircraft, so an unshifted pattern
    leaves the first stretch of every lane - and, at the corner of the arena,
    the ground the rover is parked on - permanently unobserved.

    This is only a translation because the scan holds a fixed heading (see
    :class:`WaypointFlightController`); the camera always looks along +x, on
    the return legs as much as the outbound ones.
    """
    offset = float(forward_offset)
    min_x = float(area_min[0]) + margin - offset
    max_x = float(area_max[0]) - margin - offset
    min_y = float(area_min[1]) + margin
    max_y = float(area_max[1]) - margin
    if max_x <= min_x or max_y <= min_y:
        raise ValueError(
            f"scan margin {margin} m leaves no area to cover between "
            f"{list(area_min)} and {list(area_max)}"
        )
    if lane_spacing <= 0.0:
        raise ValueError("lane_spacing must be positive")

    lanes = max(2, int(math.ceil((max_y - min_y) / lane_spacing)) + 1)
    lane_ys = np.linspace(min_y, max_y, lanes)
    waypoints: List[List[float]] = []
    for index, lane_y in enumerate(lane_ys):
        # Alternate direction each lane so the drone never flies a dead leg.
        first, second = (min_x, max_x) if index % 2 == 0 else (max_x, min_x)
        waypoints.append([first, float(lane_y), float(altitude)])
        waypoints.append([second, float(lane_y), float(altitude)])
    return np.asarray(waypoints, dtype=float)


class WaypointFlightController:
    """Takes the drone off, flies the coverage pattern, then reports completion.

    Emits body-agnostic velocity setpoints in the ``map`` frame; the ROS node
    rotates them into the frame the ``VelocityControl`` plugin expects.  Yaw is
    held constant during the scan: a nadir camera gains nothing from turning,
    and a fixed heading keeps the TF chain trivially interpretable.
    """

    def __init__(
        self,
        waypoints: np.ndarray,
        *,
        scan_altitude_m: float,
        takeoff_speed_mps: float,
        scan_speed_mps: float,
        altitude_tolerance_m: float,
        waypoint_tolerance_m: float,
        hold_yaw_rad: float = 0.0,
        yaw_kp: float = 1.0,
        max_yaw_rate: float = 0.8,
    ) -> None:
        waypoints = np.asarray(waypoints, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3 or len(waypoints) == 0:
            raise ValueError(f"waypoints must have shape (N>0, 3), got {waypoints.shape}")
        self.waypoints = waypoints
        self.scan_altitude_m = float(scan_altitude_m)
        self.takeoff_speed_mps = float(takeoff_speed_mps)
        self.scan_speed_mps = float(scan_speed_mps)
        self.altitude_tolerance_m = float(altitude_tolerance_m)
        self.waypoint_tolerance_m = float(waypoint_tolerance_m)
        self.hold_yaw_rad = float(hold_yaw_rad)
        self.yaw_kp = float(yaw_kp)
        self.max_yaw_rate = float(max_yaw_rate)

        self._phase = FlightPhase.GROUNDED
        self._index = 0

    @property
    def phase(self) -> FlightPhase:
        return self._phase

    @property
    def waypoint_index(self) -> int:
        return self._index

    @property
    def coverage_fraction(self) -> float:
        return float(self._index) / float(len(self.waypoints))

    def start(self) -> None:
        self._phase = FlightPhase.CLIMBING
        self._index = 0

    def compute(self, position_map: Sequence[float], yaw: float) -> FlightCommand:
        """One control step from the drone's current pose."""
        position = np.asarray(position_map, dtype=float).reshape(3)
        yaw_error = normalize_angle(self.hold_yaw_rad - float(yaw))
        yaw_rate = float(
            np.clip(self.yaw_kp * yaw_error, -self.max_yaw_rate, self.max_yaw_rate)
        )

        if self._phase is FlightPhase.GROUNDED:
            return FlightCommand(self._phase, np.zeros(3), 0.0, self._index, math.inf)

        if self._phase is FlightPhase.CLIMBING:
            altitude_error = self.scan_altitude_m - position[2]
            if abs(altitude_error) <= self.altitude_tolerance_m:
                self._phase = FlightPhase.SCANNING
            else:
                climb = float(
                    np.clip(altitude_error, -self.takeoff_speed_mps, self.takeoff_speed_mps)
                )
                return FlightCommand(
                    FlightPhase.CLIMBING,
                    np.array([0.0, 0.0, climb]),
                    yaw_rate,
                    self._index,
                    abs(altitude_error),
                )

        if self._phase is FlightPhase.COMPLETE:
            return self._hold(position, yaw_rate)

        target = self.waypoints[self._index]
        delta = target - position
        horizontal_distance = float(np.linalg.norm(delta[:2]))
        if horizontal_distance <= self.waypoint_tolerance_m:
            self._index += 1
            if self._index >= len(self.waypoints):
                self._phase = FlightPhase.COMPLETE
                return self._hold(position, yaw_rate)
            target = self.waypoints[self._index]
            delta = target - position
            horizontal_distance = float(np.linalg.norm(delta[:2]))

        horizontal = delta[:2]
        if horizontal_distance > 1e-6:
            # Decelerate into the waypoint so the drone settles instead of
            # oscillating around it, which would smear the camera imagery.
            speed = min(self.scan_speed_mps, max(0.25, horizontal_distance))
            horizontal = horizontal / horizontal_distance * speed
        # Altitude is regulated continuously, not only during the climb, so a
        # disturbance mid-lane does not silently change the ground footprint.
        vertical = float(
            np.clip(target[2] - position[2], -self.takeoff_speed_mps, self.takeoff_speed_mps)
        )
        return FlightCommand(
            FlightPhase.SCANNING,
            np.array([horizontal[0], horizontal[1], vertical]),
            yaw_rate,
            self._index,
            horizontal_distance,
        )

    def _hold(self, position: np.ndarray, yaw_rate: float) -> FlightCommand:
        """Station-keep at the scan altitude once the pattern is finished."""
        vertical = float(
            np.clip(
                self.scan_altitude_m - position[2],
                -self.takeoff_speed_mps,
                self.takeoff_speed_mps,
            )
        )
        return FlightCommand(
            FlightPhase.COMPLETE, np.array([0.0, 0.0, vertical]), yaw_rate, self._index, 0.0
        )


class EscortController:
    """Hold station on the rover so it stays in the drone's camera.

    The drone parks *behind* the rover along its heading, by exactly the
    distance at which a camera depressed by ``depression_rad`` points at the
    rover: ``(altitude - rover_height) / tan(depression)``. Any other offset
    would put the rover at the edge of the frame or out of it entirely.

    Yaw tracks the bearing to the rover rather than the rover's own heading,
    so the rover stays centred horizontally even while it turns on the spot.
    """

    def __init__(
        self,
        *,
        altitude_m: float,
        depression_rad: float,
        speed_mps: float,
        distance_scale: float = 1.0,
        rover_height_m: float = 0.3,
        position_kp: float = 1.2,
        yaw_kp: float = 1.5,
        max_yaw_rate: float = 1.2,
        vertical_speed_mps: float = 1.2,
    ) -> None:
        if altitude_m <= rover_height_m:
            raise ValueError("the drone must fly above the rover")
        if not 0.0 < depression_rad <= math.pi / 2.0:
            raise ValueError("camera depression must lie in (0, pi/2]")
        if speed_mps <= 0.0:
            raise ValueError("escort speed must be positive")
        self.altitude_m = float(altitude_m)
        self.depression_rad = float(depression_rad)
        self.speed_mps = float(speed_mps)
        self.distance_scale = float(distance_scale)
        self.rover_height_m = float(rover_height_m)
        self.position_kp = float(position_kp)
        self.yaw_kp = float(yaw_kp)
        self.max_yaw_rate = float(max_yaw_rate)
        self.vertical_speed_mps = float(vertical_speed_mps)

    @property
    def standoff_m(self) -> float:
        """How far behind the rover the drone holds station."""
        height = self.altitude_m - self.rover_height_m
        return self.distance_scale * height / math.tan(self.depression_rad)

    def station_for(self, rover_pose: Sequence[float]) -> np.ndarray:
        """Where the drone should be, given the rover's ``(x, y, yaw)``."""
        pose = np.asarray(rover_pose, dtype=float)
        behind = np.array([math.cos(pose[2]), math.sin(pose[2])]) * self.standoff_m
        return np.array([pose[0] - behind[0], pose[1] - behind[1], self.altitude_m])

    def compute(
        self,
        drone_position: Sequence[float],
        drone_yaw: float,
        rover_pose: Sequence[float],
    ) -> FlightCommand:
        position = np.asarray(drone_position, dtype=float).reshape(3)
        rover = np.asarray(rover_pose, dtype=float)
        station = self.station_for(rover)

        error = station - position
        horizontal = error[:2]
        distance = float(np.linalg.norm(horizontal))
        if distance > 1e-6:
            # Proportional, saturated: close the gap quickly but never command
            # more than the escort speed, so the drone cannot outrun its own
            # camera and blur every frame.
            speed = min(self.speed_mps, self.position_kp * distance)
            horizontal = horizontal / distance * speed
        vertical = float(
            np.clip(error[2], -self.vertical_speed_mps, self.vertical_speed_mps)
        )

        bearing = math.atan2(rover[1] - position[1], rover[0] - position[0])
        yaw_rate = float(
            np.clip(
                self.yaw_kp * normalize_angle(bearing - float(drone_yaw)),
                -self.max_yaw_rate,
                self.max_yaw_rate,
            )
        )
        return FlightCommand(
            FlightPhase.ESCORTING,
            np.array([horizontal[0], horizontal[1], vertical]),
            yaw_rate,
            0,
            distance,
        )


class ReturnHomeController:
    """Fly back to the take-off point and land there.

    Deliberately two stages rather than one diagonal descent: crossing the
    arena at cruise altitude keeps the drone above everything it mapped, and
    only then does it come straight down. A descent that begins early would
    fly the drone through the airspace over obstacles and stations at an
    altitude nobody checked.
    """

    def __init__(
        self,
        home_xy: Sequence[float],
        *,
        cruise_altitude_m: float,
        speed_mps: float,
        descend_speed_mps: float = 0.8,
        arrive_tolerance_m: float = 0.4,
        landed_altitude_m: float = 0.20,
        position_kp: float = 1.0,
        yaw_kp: float = 1.0,
        max_yaw_rate: float = 0.8,
        hold_yaw_rad: float = 0.0,
    ) -> None:
        if cruise_altitude_m <= landed_altitude_m:
            raise ValueError("cruise altitude must be above the landing altitude")
        if speed_mps <= 0.0 or descend_speed_mps <= 0.0:
            raise ValueError("speeds must be positive")
        self.home_xy = np.asarray(home_xy, dtype=float)[:2].copy()
        self.cruise_altitude_m = float(cruise_altitude_m)
        self.speed_mps = float(speed_mps)
        self.descend_speed_mps = float(descend_speed_mps)
        self.arrive_tolerance_m = float(arrive_tolerance_m)
        self.landed_altitude_m = float(landed_altitude_m)
        self.position_kp = float(position_kp)
        self.yaw_kp = float(yaw_kp)
        self.max_yaw_rate = float(max_yaw_rate)
        self.hold_yaw_rad = float(hold_yaw_rad)
        self._landed = False

    @property
    def landed(self) -> bool:
        return self._landed

    def compute(self, position_map: Sequence[float], yaw: float) -> FlightCommand:
        position = np.asarray(position_map, dtype=float).reshape(3)
        offset = self.home_xy - position[:2]
        distance = float(np.linalg.norm(offset))
        yaw_rate = float(
            np.clip(
                self.yaw_kp * normalize_angle(self.hold_yaw_rad - float(yaw)),
                -self.max_yaw_rate,
                self.max_yaw_rate,
            )
        )

        if self._landed:
            return FlightCommand(FlightPhase.LANDED, np.zeros(3), 0.0, 0, distance)

        if distance > self.arrive_tolerance_m:
            speed = min(self.speed_mps, self.position_kp * distance)
            horizontal = offset / distance * speed
            # Hold cruise height while crossing; only descend once overhead.
            vertical = float(
                np.clip(
                    self.cruise_altitude_m - position[2],
                    -self.descend_speed_mps,
                    self.descend_speed_mps,
                )
            )
            return FlightCommand(
                FlightPhase.RETURNING,
                np.array([horizontal[0], horizontal[1], vertical]),
                yaw_rate,
                0,
                distance,
            )

        if position[2] <= self.landed_altitude_m:
            self._landed = True
            return FlightCommand(FlightPhase.LANDED, np.zeros(3), 0.0, 0, distance)

        # Overhead: come straight down, still nudging towards the exact spot.
        horizontal = offset * self.position_kp
        return FlightCommand(
            FlightPhase.RETURNING,
            np.array([horizontal[0], horizontal[1], -self.descend_speed_mps]),
            yaw_rate,
            0,
            distance,
        )


class KinematicDrone:
    """First-order velocity-tracking drone model for the offline harness.

    Mirrors what gz-sim's ``VelocityControl`` system does: it drives the link
    towards the commanded velocity rather than simulating rotor dynamics.  Not
    used at runtime.
    """

    def __init__(
        self,
        position: Sequence[float],
        yaw: float = 0.0,
        *,
        velocity_time_constant_s: float = 0.35,
    ) -> None:
        self.position = np.asarray(position, dtype=float).reshape(3).copy()
        self.yaw = float(yaw)
        self.velocity = np.zeros(3)
        self.velocity_time_constant_s = float(velocity_time_constant_s)

    def step(self, velocity_command: Sequence[float], yaw_rate: float, dt: float) -> None:
        dt = float(dt)
        if dt <= 0.0:
            return
        alpha = float(np.clip(dt / self.velocity_time_constant_s, 0.0, 1.0))
        self.velocity += alpha * (np.asarray(velocity_command, dtype=float) - self.velocity)
        self.position += self.velocity * dt
        # The drone must not sink through the ground while still on the pad.
        self.position[2] = max(self.position[2], 0.0)
        self.yaw = normalize_angle(self.yaw + float(yaw_rate) * dt)


def scan_pixels_per_module_ok(
    coverage_altitude_m: float, max_decodable_range_m: float
) -> bool:
    """Whether the configured scan altitude keeps codes inside decoding range."""
    return coverage_altitude_m <= max_decodable_range_m
