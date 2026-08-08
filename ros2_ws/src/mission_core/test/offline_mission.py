"""Headless end-to-end mission runner.

Wires the *production* mission_core components to the synthetic world in
:mod:`sim_harness` and runs the complete flow:

    takeoff -> lawnmower scan -> camera frames -> QR decode + PnP -> TF into
    map -> world model fusion -> lidar occupancy mapping -> A* -> path ->
    pure pursuit -> rover drives -> rover camera QR verification -> validator

Nothing here reads a ground-truth pose to make a decision; ground truth is only
consulted afterwards, by the tests, to score how accurate perception was.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from mission_core.camera import PinholeCamera
from mission_core.config import MissionConfig
from mission_core.errors import PerceptionError
from mission_core.exploration import (
    FlightPhase,
    KinematicDrone,
    WaypointFlightController,
    lawnmower_waypoints,
)
from mission_core.geometry import (
    R_BODY_TO_FORWARD_OPTICAL,
    R_BODY_TO_NADIR_OPTICAL,
    Transform,
)
from mission_core.mission_state import MissionState
from mission_core.occupancy import GridMetadata, OccupancyMapper
from mission_core.orchestrator import MissionCommand, MissionInputs, MissionOrchestrator
from mission_core.path_following import (
    DifferentialDriveModel,
    FollowerState,
    PurePursuitController,
)
from mission_core.qr import QrDetector
from mission_core.world_model import TargetObservation, WorldModel

from sim_harness import SyntheticWorld, camera_pose_from_body, nadir_lidar_directions

#: Sensor mount offsets in each vehicle's ``base_link``.  These mirror the
#: ``<pose>`` entries of the corresponding SDF sensors exactly; if one changes,
#: the other must too.
DRONE_CAMERA_MOUNT = (0.10, 0.0, -0.08)
DRONE_LIDAR_MOUNT = (0.0, 0.0, -0.06)
ROVER_CAMERA_MOUNT = (0.22, 0.0, 0.55)


@dataclass
class MissionTrace:
    """Everything the run produced, for assertions and for printing."""

    state: MissionState = MissionState.IDLE
    messages: List[str] = field(default_factory=list)
    path_xy: Optional[np.ndarray] = None
    rover_track: List[np.ndarray] = field(default_factory=list)
    drone_track: List[np.ndarray] = field(default_factory=list)
    detections: Dict[str, int] = field(default_factory=dict)
    world_model: Optional[WorldModel] = None
    report = None
    verified_qr: Optional[str] = None
    failure_reason: str = "NONE"
    sim_time_s: float = 0.0
    drone_frames: int = 0
    rover_frames: int = 0
    bad_frames: int = 0

    @property
    def succeeded(self) -> bool:
        return self.state is MissionState.MISSION_SUCCESS


class OfflineMissionRunner:
    """Runs one complete mission against a :class:`SyntheticWorld`."""

    def __init__(
        self,
        world: SyntheticWorld,
        config: MissionConfig,
        *,
        requested_qr: Optional[str] = None,
        drone_start: Sequence[float] = (-8.0, -6.5, 0.0),
        rover_start: Sequence[float] = (-8.0, -8.0, 0.0),
        dt: float = 0.1,
        max_sim_time_s: float = 900.0,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.world = world
        self.config = config
        self.requested_qr = requested_qr or config.mission.target_qr
        self.dt = float(dt)
        self.max_sim_time_s = float(max_sim_time_s)
        self.logger = logger

        self.drone_camera = PinholeCamera.from_hfov(
            config.drone.camera.width,
            config.drone.camera.height,
            config.drone.camera.horizontal_fov_rad,
        )
        self.rover_camera = PinholeCamera.from_hfov(
            config.rover.camera.width,
            config.rover.camera.height,
            config.rover.camera.horizontal_fov_rad,
        )
        self.detector = QrDetector(
            config.mission.qr_plate_size_m,
            quiet_zone_modules=config.mission.qr_quiet_zone_modules,
            max_reprojection_error_px=config.perception.max_reprojection_error_px,
            min_quad_area_px=config.perception.min_quad_area_px,
            min_apparent_size_px=config.perception.min_apparent_size_px,
        )

        self.world_model = WorldModel(
            association_radius_m=config.world_model.association_radius_m,
            min_observations=config.world_model.min_observations,
            min_confidence=config.world_model.min_confidence,
        )
        area_min = config.mission.area_min_xy
        area_max = config.mission.area_max_xy
        resolution = config.world_model.grid_resolution_m
        self.mapper = OccupancyMapper(
            GridMetadata(
                resolution,
                int(math.ceil((area_max[0] - area_min[0]) / resolution)),
                int(math.ceil((area_max[1] - area_min[1]) / resolution)),
                float(area_min[0]),
                float(area_min[1]),
            ),
            obstacle_min_height_m=config.world_model.obstacle_min_height_m,
            obstacle_max_height_m=config.world_model.obstacle_max_height_m,
            min_hits=config.world_model.mapper_min_hits,
            hit_ratio_threshold=config.world_model.mapper_hit_ratio,
        )

        self.drone = KinematicDrone(drone_start)
        self.rover = DifferentialDriveModel(rover_start)
        self.flight = WaypointFlightController(
            lawnmower_waypoints(
                area_min,
                area_max,
                config.drone.scan_altitude_m,
                config.drone.lane_spacing_m,
                config.drone.scan_margin_m,
            ),
            scan_altitude_m=config.drone.scan_altitude_m,
            takeoff_speed_mps=config.drone.takeoff_speed_mps,
            scan_speed_mps=config.drone.scan_speed_mps,
            altitude_tolerance_m=config.drone.altitude_tolerance_m,
            waypoint_tolerance_m=config.drone.waypoint_tolerance_m,
        )
        self.follower = PurePursuitController(
            lookahead_distance_m=config.rover.lookahead_distance_m,
            min_lookahead_distance_m=config.rover.min_lookahead_distance_m,
            lookahead_time_s=config.rover.lookahead_time_s,
            max_linear_velocity=config.rover.max_linear_velocity,
            max_angular_velocity=config.rover.max_angular_velocity,
            heading_kp=config.rover.heading_kp,
            rotate_in_place_rad=config.rover.rotate_in_place_rad,
            goal_tolerance_m=config.rover.goal_tolerance_m,
            goal_yaw_tolerance_rad=config.rover.goal_yaw_tolerance_rad,
            align_to_goal_yaw=config.rover.align_to_goal_yaw,
            max_cross_track_error_m=config.rover.max_cross_track_error_m,
            cross_track_grace_s=config.rover.cross_track_grace_s,
            approach_slowdown_m=config.rover.approach_slowdown_m,
        )
        self.orchestrator = MissionOrchestrator(
            config, self.world_model, requested_qr=self.requested_qr
        )

        self._lidar_directions = nadir_lidar_directions()
        self._perception_period = 1.0 / max(config.perception.qr_detection_rate_hz, 1e-3)
        self._last_perception_t = -math.inf
        self._last_map_publish_t = -math.inf
        self._verification_hits: List[str] = []
        self.trace = MissionTrace(world_model=self.world_model)

    # -- logging ----------------------------------------------------------
    def _log(self, message: str) -> None:
        self.trace.messages.append(message)
        if self.logger is not None:
            self.logger(message)

    # -- sensing ----------------------------------------------------------
    def _drone_camera_pose(self) -> Transform:
        return camera_pose_from_body(
            self.drone.position, self.drone.yaw, DRONE_CAMERA_MOUNT, R_BODY_TO_NADIR_OPTICAL
        )

    def _rover_camera_pose(self) -> Transform:
        return camera_pose_from_body(
            (self.rover.pose[0], self.rover.pose[1], 0.0),
            self.rover.pose[2],
            ROVER_CAMERA_MOUNT,
            R_BODY_TO_FORWARD_OPTICAL,
        )

    def _observe_with_drone(self, now: float) -> None:
        """Render one drone frame, decode it, and fold results into the model."""
        pose = self._drone_camera_pose()
        image = self.world.render(pose, self.drone_camera)
        self.trace.drone_frames += 1
        try:
            detections = self.detector.detect(image, self.drone_camera)
        except PerceptionError:
            # A bad frame is logged and skipped; it must never abort the scan.
            self.trace.bad_frames += 1
            return
        for detection in detections:
            # Camera optical -> map. This is the TF step; getting it wrong is
            # what mixes camera and world coordinates.
            position_map = pose.apply(detection.position_optical)
            normal_map = pose.rotation @ detection.rotation_optical @ np.array([0.0, 0.0, 1.0])
            self.world_model.add_observation(
                TargetObservation(
                    qr_id=detection.payload,
                    position_map=position_map,
                    normal_map=normal_map,
                    confidence=detection.confidence,
                    stamp=now,
                    observer_position_map=pose.translation.copy(),
                    source="drone_camera",
                )
            )
            self.trace.detections[detection.payload] = (
                self.trace.detections.get(detection.payload, 0) + 1
            )
            if self.trace.detections[detection.payload] == 1:
                self._log(
                    f"[QR] {detection.payload} detected at "
                    f"({position_map[0]:.2f}, {position_map[1]:.2f}) "
                    f"range {detection.range_m:.2f} m"
                )

    def _scan_with_lidar(self) -> None:
        origin = self.drone.position + np.asarray(DRONE_LIDAR_MOUNT)
        points = self.world.raycast(origin, self._lidar_directions)
        if len(points):
            self.mapper.integrate(points)

    def _refresh_occupancy(self) -> None:
        self.world_model.set_occupancy(self.mapper.build(), self.mapper.height_map)

    def _verify_with_rover(self) -> Optional[str]:
        """Look for a QR code from the rover's stopped position."""
        pose = self._rover_camera_pose()
        image = self.world.render(pose, self.rover_camera)
        self.trace.rover_frames += 1
        try:
            detections = self.detector.detect(image, self.rover_camera)
        except PerceptionError:
            self.trace.bad_frames += 1
            return None
        # Only the station the rover is parked in front of counts; a code
        # legible in the background is a different station.
        near = [d for d in detections if d.range_m <= self.config.verification.max_range_m]
        if not near:
            self._verification_hits.clear()
            return None
        closest = min(near, key=lambda d: d.range_m)
        self._verification_hits.append(closest.payload)
        required = self.config.verification.required_consecutive_reads
        recent = self._verification_hits[-required:]
        if len(recent) >= required and len(set(recent)) == 1:
            return recent[0]
        return None

    # -- main loop --------------------------------------------------------
    def run(self) -> MissionTrace:
        now = 0.0
        rover_goal_reached = False
        rover_tracking_failed = False
        rover_failure_detail = ""
        path_published = False
        verified_qr: Optional[str] = None

        while now < self.max_sim_time_s:
            state = self.orchestrator.state

            # ---- drone: fly, sense, map
            if state in (
                MissionState.TAKEOFF,
                MissionState.EXPLORING,
                MissionState.TARGET_FOUND,
                MissionState.PLANNING,
                MissionState.PATH_READY,
                MissionState.SENDING_PATH,
                MissionState.ROVER_NAVIGATING,
                MissionState.VERIFYING_TARGET,
            ):
                if self.flight.phase is FlightPhase.GROUNDED:
                    self.flight.start()
                command = self.flight.compute(self.drone.position, self.drone.yaw)
                self.drone.step(command.velocity_map, command.yaw_rate, self.dt)
                self.trace.drone_track.append(self.drone.position.copy())

                if self.flight.phase in (FlightPhase.SCANNING, FlightPhase.COMPLETE):
                    if now - self._last_perception_t >= self._perception_period:
                        self._last_perception_t = now
                        self._observe_with_drone(now)
                        self._scan_with_lidar()
                    if now - self._last_map_publish_t >= 1.0 / max(
                        self.config.world_model.publish_rate_hz, 1e-3
                    ):
                        self._last_map_publish_t = now
                        self._refresh_occupancy()

            # ---- rover: follow the path once one has been received
            if state is MissionState.ROVER_NAVIGATING and self.follower.has_path:
                status = self.follower.compute(self.rover.pose, self.dt)
                if status.state is FollowerState.GOAL_REACHED:
                    rover_goal_reached = True
                    self.rover.step(0.0, 0.0, self.dt)
                elif status.state is FollowerState.FAILED:
                    rover_tracking_failed = True
                    rover_failure_detail = status.detail
                else:
                    self.rover.step(status.linear_velocity, status.angular_velocity, self.dt)
                self.trace.rover_track.append(self.rover.pose.copy())

            # ---- rover camera verification
            if state is MissionState.VERIFYING_TARGET and verified_qr is None:
                verified_qr = self._verify_with_rover()

            # ---- orchestrator tick
            outputs = self.orchestrator.update(
                MissionInputs(
                    now=now,
                    drone_ready=True,
                    drone_at_scan_altitude=self.flight.phase
                    in (FlightPhase.SCANNING, FlightPhase.COMPLETE),
                    exploration_complete=self.flight.phase is FlightPhase.COMPLETE,
                    rover_pose=self.rover.pose[:2].copy(),
                    rover_goal_reached=rover_goal_reached,
                    rover_tracking_failed=rover_tracking_failed,
                    rover_failure_detail=rover_failure_detail,
                    verified_qr=verified_qr,
                    path_published=path_published,
                )
            )
            for message in outputs.messages:
                self._log(message)

            if outputs.command is MissionCommand.PUBLISH_PATH and outputs.path is not None:
                # Stand-in for the /mission/rover_path publish + rover subscribe.
                self.follower.set_path(outputs.path.poses)
                self.trace.path_xy = outputs.path.xy.copy()
                path_published = True

            if outputs.is_terminal:
                break
            now += self.dt

        self.trace.state = self.orchestrator.state
        self.trace.report = self.orchestrator.report
        self.trace.verified_qr = self.orchestrator.verified_qr
        self.trace.failure_reason = self.orchestrator.machine.failure_reason.value
        self.trace.sim_time_s = now
        if self.trace.state not in (MissionState.MISSION_SUCCESS, MissionState.MISSION_FAILED):
            self._log(
                f"[MISSION] harness stopped at {now:.1f} s in state {self.trace.state.value}"
            )
        return self.trace


def default_world() -> SyntheticWorld:
    """The mission arena, mirroring ``mission_arena.sdf``.

    Station and obstacle placements here must match the SDF world; the tests
    use this to check the *logic*, Gazebo exercises the same layout for real.
    """
    from sim_harness import OrientedBox, Station

    return SyntheticWorld(
        stations=[
            Station("TARGET_1", (6.0, 6.0)),
            Station("TARGET_2", (7.0, -5.0)),
            Station("TARGET_3", (-6.0, 6.0)),
        ],
        obstacles=[
            # Blocks the straight line from the rover start to TARGET_2.
            OrientedBox(np.array([0.0, -6.0, 0.75]), np.array([6.0, 1.0, 1.5]), name="obstacle_a"),
            # Blocks the straight line from the rover start to TARGET_1.
            OrientedBox(np.array([0.0, 0.0, 0.75]), np.array([1.0, 6.0, 1.5]), name="obstacle_b"),
        ],
    )
