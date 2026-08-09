"""Typed mission configuration loaded from a single YAML file.

One file is the source of truth for every tunable value.  Each ROS node
declares the subset it needs as real ROS parameters whose *defaults* come from
this file, so operators can either edit the YAML or override a single value
with ``ros2 param set`` / a launch argument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised when the mission configuration is missing or inconsistent."""


@dataclass(frozen=True)
class FrameConfig:
    """TF frame identifiers. Changing these must never require a code change."""

    map_frame: str = "map"
    odom_frame: str = "odom"
    drone_base_frame: str = "drone/base_link"
    drone_camera_optical_frame: str = "drone/camera_optical_frame"
    drone_lidar_frame: str = "drone/lidar_link"
    rover_base_frame: str = "rover/base_link"
    rover_camera_optical_frame: str = "rover/camera_optical_frame"
    rover_lidar_frame: str = "rover/lidar_link"


@dataclass(frozen=True)
class CameraConfig:
    """Sensor geometry, mirrored by the SDF so simulation and code agree."""

    width: int = 1280
    height: int = 960
    horizontal_fov_rad: float = 1.047
    #: Matches perception.qr_detection_rate_hz: rendering frames the detector
    #: throttles away is wasted work, and dominates cost without a GPU.
    update_rate_hz: float = 6.0


@dataclass(frozen=True)
class MissionConfigSection:
    target_qr: str = "TARGET_2"
    #: Payloads the operator is allowed to request. Used only to reject typos
    #: early - the *positions* are never configured, only discovered.
    known_payloads: List[str] = field(default_factory=lambda: ["TARGET_1", "TARGET_2", "TARGET_3"])
    #: Side length of the printed plate on a station face, in metres. A
    #: physical property of the target hardware, exactly like an ArUco marker
    #: size - never a position prior.
    qr_plate_size_m: float = 0.80
    #: White border baked into the plate texture, in QR modules. Together with
    #: the plate size this fixes the code's physical size, which is what the
    #: detector's corners actually bound.
    qr_quiet_zone_modules: int = 4
    #: Height of a station's plate centre above the ground, in metres. A
    #: property of the target hardware (pedestal + half the cube), and what
    #: the slant range to a station is measured against.
    station_plate_centre_height_m: float = 0.65
    #: Visit every known payload in one run rather than only ``target_qr``.
    #: The order is decided at run time from the *discovered* positions, not
    #: configured - a fixed order would be a position prior in disguise.
    visit_all_targets: bool = True
    #: Drive back to where the rover started once the tour is complete.
    return_home: bool = True
    mission_timeout_s: float = 1800.0
    #: Where the mission area begins and ends, in the ``map`` frame.
    area_min_xy: List[float] = field(default_factory=lambda: [-11.0, -11.0])
    area_max_xy: List[float] = field(default_factory=lambda: [11.0, 11.0])


@dataclass(frozen=True)
class DroneConfig:
    #: 4.0 m with a 30 degree camera puts the station plates at a 6.7 m slant
    #: range, which resolves ~4.6 px per QR module. validate() enforces it.
    scan_altitude_m: float = 4.0
    takeoff_speed_mps: float = 1.2
    scan_speed_mps: float = 1.6
    #: Spacing between lawnmower legs. Must be < camera ground footprint.
    lane_spacing_m: float = 5.0
    #: Inset from the arena edge so the drone stays inside the mapped area.
    scan_margin_m: float = 2.0
    altitude_tolerance_m: float = 0.25
    waypoint_tolerance_m: float = 0.6
    #: Cap on how long the scan may run before the mission gives up.
    exploration_timeout_s: float = 300.0
    #: Keep flying after the requested target is confirmed, so the world model
    #: is complete rather than stopping at the first useful sighting.
    finish_scan_after_target_found: bool = True
    control_rate_hz: float = 30.0
    #: How far below horizontal the drone camera points. pi/2 is straight
    #: down; 30 degrees looks ahead of the vehicle and sees a station's
    #: vertical faces almost head-on instead of foreshortening its top plate.
    camera_depression_rad: float = 0.5235988
    #: Where the drone parks relative to the rover once it is escorting it:
    #: behind by (altitude / tan(depression)) so the rover sits in the middle
    #: of the frame, scaled by this factor.
    follow_distance_scale: float = 1.0
    follow_speed_mps: float = 2.2
    camera: CameraConfig = field(default_factory=CameraConfig)


@dataclass(frozen=True)
class RoverConfig:
    max_linear_velocity: float = 0.8
    max_angular_velocity: float = 1.4
    lookahead_distance_m: float = 0.9
    min_lookahead_distance_m: float = 0.35
    lookahead_time_s: float = 0.6
    heading_kp: float = 2.0
    rotate_in_place_rad: float = 0.7
    goal_tolerance_m: float = 0.35
    goal_yaw_tolerance_rad: float = 0.30
    align_to_goal_yaw: bool = True
    max_cross_track_error_m: float = 1.5
    cross_track_grace_s: float = 3.0
    approach_slowdown_m: float = 1.0
    navigation_timeout_s: float = 240.0
    control_rate_hz: float = 20.0
    #: Emergency stop range from the rover's 2D safety lidar. Must stay below
    #: the clearance the planner guarantees, measured from the lidar rather
    #: than from base_link - otherwise the stop fires on the rover's own
    #: correctly planned path instead of on a genuine surprise.
    obstacle_stop_distance_m: float = 0.30
    #: Forward mounting offset of that lidar; mirrors the SDF <pose>.
    safety_lidar_forward_offset_m: float = 0.24
    #: Half-width of the forward wedge the stop considers. Wide wedges see
    #: walls the rover is merely driving past, not heading into.
    obstacle_stop_half_angle_rad: float = 0.35
    #: How long an obstacle may continuously suppress forward motion before
    #: the follower asks the mission manager for a fresh plan.
    safety_stop_replan_delay_s: float = 3.0
    #: Number of fresh plans allowed after the controller reports that it can
    #: no longer track the current path. A finite budget prevents recovery
    #: from hiding a persistent localisation or control fault forever.
    max_replans: int = 1
    camera: CameraConfig = field(
        default_factory=lambda: CameraConfig(width=640, height=480, horizontal_fov_rad=1.20)
    )


@dataclass(frozen=True)
class PerceptionConfig:
    qr_detection_rate_hz: float = 6.0
    max_reprojection_error_px: float = 4.0
    min_quad_area_px: float = 400.0
    min_apparent_size_px: float = 40.0
    #: Below this the decoder is unreliable; used to sanity-check scan altitude.
    min_pixels_per_module: float = 3.0
    #: Consecutive unusable frames tolerated before the node reports a fault.
    max_consecutive_bad_frames: int = 30


@dataclass(frozen=True)
class WorldModelConfig:
    association_radius_m: float = 1.5
    min_observations: int = 3
    min_confidence: float = 0.35
    grid_resolution_m: float = 0.20
    obstacle_min_height_m: float = 0.25
    obstacle_max_height_m: float = 8.0
    mapper_min_hits: int = 2
    mapper_hit_ratio: float = 0.35
    publish_rate_hz: float = 2.0


@dataclass(frozen=True)
class PlannerConfig:
    rover_radius_m: float = 0.30
    #: Clearance added to the rover radius. Sized so pure pursuit can cut a
    #: corner by ~0.12 m and the safety lidar still reads more than
    #: rover.obstacle_stop_distance_m on a clean path.
    obstacle_safety_margin_m: float = 0.40
    planning_resolution_m: float = 0.20
    #: Fail closed: the rover may only plan through cells the mapping pass has
    #: actually observed. Operators can explicitly relax this for exploration.
    allow_unknown: bool = False
    heuristic_weight: float = 1.0
    shortcut: bool = True
    #: Standoff distance where the rover stops to read the station's QR code.
    approach_distance_m: float = 1.8
    #: How far the standoff may be moved in or out when the nominal ring is
    #: blocked, before the mission gives up on the approach.
    approach_distance_tolerance_m: float = 0.6
    approach_samples: int = 72
    #: Radius of a station's own footprint. Occupied cells this close to a
    #: target are the station itself, so they must not count as occluding it.
    #: A physical property of the target hardware, like the plate size.
    target_footprint_radius_m: float = 0.70
    max_start_snap_m: float = 1.5


@dataclass(frozen=True)
class VerificationConfig:
    #: How long the rover looks for the code once it has stopped.
    timeout_s: float = 30.0
    #: Consistent decodes required before the mission is declared a success.
    required_consecutive_reads: int = 3
    #: A code further away than this is some other station seen in the
    #: background, not the one the rover is parked in front of.
    max_range_m: float = 4.0
    #: On arrival the rover sweeps its heading to look for the code. Wheel
    #: odometry drifts in yaw, so "I am facing the station" is an estimate,
    #: not a fact - and a station 0.4 rad off-axis falls outside a 1.2 rad
    #: camera. Sweeping costs seconds and removes the whole failure mode.
    search_sweep_rad: float = 1.05
    search_yaw_rate_rad_s: float = 0.4
    #: Complete heading sweeps allowed before an unreadable code becomes a
    #: terminal failure. A decoded wrong payload is never retried.
    max_attempts: int = 2


@dataclass(frozen=True)
class MissionConfig:
    """Root configuration object."""

    mission: MissionConfigSection = field(default_factory=MissionConfigSection)
    frames: FrameConfig = field(default_factory=FrameConfig)
    drone: DroneConfig = field(default_factory=DroneConfig)
    rover: RoverConfig = field(default_factory=RoverConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)

    # -- derived helpers --------------------------------------------------
    @property
    def clearance_m(self) -> float:
        """Total inflation applied to obstacles for planning."""
        return self.planner.rover_radius_m + self.planner.obstacle_safety_margin_m

    def station_slant_range_m(self) -> float:
        """Distance from the drone camera to a station plate it is aimed at.

        With a depressed camera the plate is ahead of the vehicle, not below
        it, so every resolution and coverage figure has to come from this
        slant range rather than from the altitude.
        """
        height = self.drone.scan_altitude_m - self.mission.station_plate_centre_height_m
        return height / math.sin(self.drone.camera_depression_rad)

    def camera_ground_footprint_m(self) -> float:
        """Cross-track width of the camera swath where it meets a station."""
        return 2.0 * self.station_slant_range_m() * math.tan(
            self.drone.camera.horizontal_fov_rad / 2.0
        )

    def camera_ground_offset_m(self) -> float:
        """How far ahead of the drone the camera axis meets the ground."""
        height = self.drone.scan_altitude_m - self.mission.station_plate_centre_height_m
        return height / math.tan(self.drone.camera_depression_rad)

    def code_size_m(self, payload: str) -> float:
        """Physical side length of the code area for one payload."""
        from .qr import code_side_length_m

        return code_side_length_m(
            payload, self.mission.qr_plate_size_m, self.mission.qr_quiet_zone_modules
        )

    def validate(self) -> List[str]:
        """Return human-readable problems; empty list means the config is sane.

        Catching these offline is far cheaper than discovering mid-flight that
        the scan altitude makes every QR code unreadable.
        """
        problems: List[str] = []
        if self.mission.target_qr not in self.mission.known_payloads:
            problems.append(
                f"mission.target_qr {self.mission.target_qr!r} is not in known_payloads "
                f"{self.mission.known_payloads}"
            )
        if self.mission.qr_plate_size_m <= 0.0:
            problems.append("mission.qr_plate_size_m must be positive")

        footprint = self.camera_ground_footprint_m()
        if self.drone.lane_spacing_m >= footprint:
            problems.append(
                f"drone.lane_spacing_m ({self.drone.lane_spacing_m:.2f} m) is not smaller than "
                f"the camera ground footprint ({footprint:.2f} m): the scan would leave gaps"
            )

        # Resolution check: how many pixels per QR module at scan altitude, for
        # the payload that encodes into the *most* modules (the hardest to read).
        from .camera import PinholeCamera  # local import keeps config import light
        from .qr import expected_pixels_per_module

        camera = PinholeCamera.from_hfov(
            self.drone.camera.width, self.drone.camera.height, self.drone.camera.horizontal_fov_rad
        )
        for payload in self.mission.known_payloads:
            pixels_per_module = expected_pixels_per_module(
                camera, self.code_size_m(payload), self.station_slant_range_m(), payload
            )
            if pixels_per_module < self.perception.min_pixels_per_module:
                problems.append(
                    f"at a {self.station_slant_range_m():.1f} m slant range the drone camera "
                    f"resolves only "
                    f"{pixels_per_module:.2f} px per QR module for {payload!r} (minimum "
                    f"{self.perception.min_pixels_per_module:.1f}): lower the altitude, widen "
                    f"the plate, or raise the resolution"
                )
        if not 0.05 <= self.drone.camera_depression_rad <= math.pi / 2.0:
            problems.append(
                "drone.camera_depression_rad must lie between 0.05 rad and pi/2"
            )
        # The lidar, not the camera, is what fills the occupancy grid, and it
        # looks straight down - so coverage is limited by its cone, not by the
        # camera swath.
        lidar_radius = self.drone.scan_altitude_m  # 45 deg outer ray
        if self.drone.lane_spacing_m >= 2.0 * lidar_radius:
            problems.append(
                f"drone.lane_spacing_m ({self.drone.lane_spacing_m:.2f} m) exceeds the "
                f"downward lidar's {2.0 * lidar_radius:.2f} m ground diameter at "
                f"{self.drone.scan_altitude_m:.1f} m: the occupancy grid would have gaps"
            )

        if self.planner.planning_resolution_m > self.planner.rover_radius_m:
            problems.append(
                "planner.planning_resolution_m is coarser than the rover radius; obstacle "
                "inflation would be quantised too aggressively"
            )
        if self.rover.goal_tolerance_m >= self.planner.approach_distance_m:
            problems.append(
                "rover.goal_tolerance_m must be smaller than planner.approach_distance_m, "
                "otherwise the goal disc reaches the station itself"
            )
        if self.verification.max_range_m <= self.planner.approach_distance_m:
            problems.append(
                "verification.max_range_m must exceed planner.approach_distance_m or the "
                "rover will reject the very code it parked in front of"
            )
        # The emergency stop must never be able to fire on a path the planner
        # itself produced; it exists for obstacles the map did not contain.
        lidar_clearance = self.clearance_m - self.rover.safety_lidar_forward_offset_m
        if self.rover.obstacle_stop_distance_m >= lidar_clearance:
            problems.append(
                f"rover.obstacle_stop_distance_m ({self.rover.obstacle_stop_distance_m:.2f} m) "
                f"is not below the clearance visible to the safety lidar "
                f"({lidar_clearance:.2f} m = rover radius + safety margin - lidar offset); "
                "the emergency stop would trigger on the rover's own planned path"
            )
        if self.rover.max_replans < 0:
            problems.append("rover.max_replans must not be negative")
        if self.rover.safety_stop_replan_delay_s <= 0.0:
            problems.append("rover.safety_stop_replan_delay_s must be positive")
        if self.verification.max_attempts < 1:
            problems.append("verification.max_attempts must be at least 1")
        area_min = self.mission.area_min_xy
        area_max = self.mission.area_max_xy
        if area_max[0] <= area_min[0] or area_max[1] <= area_min[1]:
            problems.append("mission.area_max_xy must be greater than mission.area_min_xy")
        return problems

    def require_valid(self) -> "MissionConfig":
        problems = self.validate()
        if problems:
            raise ConfigError("invalid mission configuration:\n  - " + "\n  - ".join(problems))
        return self


def _build(cls: Any, data: Mapping[str, Any], path: str = "") -> Any:
    """Recursively construct a (frozen) dataclass from a plain mapping."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path or 'root'}: expected a mapping, got {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        # Typos in a config file are a classic silent-misconfiguration source.
        raise ConfigError(
            f"{path or 'root'}: unknown key(s) {sorted(unknown)}; "
            f"expected one of {sorted(known)}"
        )
    # ``from __future__ import annotations`` turns field.type into a string, so
    # the nested dataclass types have to be resolved explicitly.
    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for name in known:
        if name not in data:
            continue
        value = data[name]
        field_type = hints.get(name)
        if isinstance(field_type, type) and is_dataclass(field_type):
            kwargs[name] = _build(field_type, value, f"{path}.{name}" if path else name)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def mission_config_from_dict(data: Mapping[str, Any]) -> MissionConfig:
    return _build(MissionConfig, data)


def load_mission_config(path: str | Path, *, validate: bool = True) -> MissionConfig:
    """Read ``path`` and return a :class:`MissionConfig`."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"mission configuration not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{file_path}: top level must be a mapping")
    config = mission_config_from_dict(raw)
    return config.require_valid() if validate else config


def config_to_dict(config: Any) -> Dict[str, Any]:
    """Inverse of :func:`mission_config_from_dict`, for logging and dumping."""
    result: Dict[str, Any] = {}
    for spec in fields(config):
        value = getattr(config, spec.name)
        if is_dataclass(value) and not isinstance(value, type):
            result[spec.name] = config_to_dict(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            result[spec.name] = list(value)
        else:
            result[spec.name] = value
    return result
