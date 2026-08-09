"""Scout UAV flight node.

Takes off, flies the lawnmower coverage pattern from
:mod:`mission_core.exploration`, and reports progress.  The mission manager
starts it and watches ``/drone/exploration_status``; this node knows nothing
about targets, planning or the rover.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_srvs.srv import Trigger

from mission_core.exploration import (
    EscortController,
    FlightPhase,
    ReturnHomeController,
    WaypointFlightController,
    lawnmower_waypoints,
)
from mission_interfaces.msg import ExplorationStatus

from .common import (
    DEFAULT_QOS,
    LATCHED_QOS,
    SENSOR_QOS,
    declare_mission_config,
    make_header,
    odometry_pose_in_frame,
    odometry_position_in_frame,
)


class DroneExplorerNode(Node):
    """Flies the coverage pattern that lets the drone see every station."""

    def __init__(self) -> None:
        super().__init__("drone_explorer")
        self.config = declare_mission_config(self)

        self.declare_parameter("odometry_topic", "/drone/odometry")
        self.declare_parameter("cmd_vel_topic", "/drone/cmd_vel")
        self.declare_parameter("status_topic", "/drone/exploration_status")
        self.declare_parameter("rover_odometry_topic", "/rover/odometry")
        # Autostart is the default so the single launch command produces a
        # complete mission; the service exists for manual/staged runs.
        self.declare_parameter("auto_start", True)

        self.waypoints = lawnmower_waypoints(
            self.config.mission.area_min_xy,
            self.config.mission.area_max_xy,
            self.config.drone.scan_altitude_m,
            self.config.drone.lane_spacing_m,
            self.config.drone.scan_margin_m,
        )
        self.controller = WaypointFlightController(
            self.waypoints,
            scan_altitude_m=self.config.drone.scan_altitude_m,
            takeoff_speed_mps=self.config.drone.takeoff_speed_mps,
            scan_speed_mps=self.config.drone.scan_speed_mps,
            altitude_tolerance_m=self.config.drone.altitude_tolerance_m,
            waypoint_tolerance_m=self.config.drone.waypoint_tolerance_m,
        )

        from tf2_ros import Buffer, TransformListener

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.escort = EscortController(
            altitude_m=self.config.drone.scan_altitude_m,
            depression_rad=self.config.drone.camera_depression_rad,
            speed_mps=self.config.drone.follow_speed_mps,
            distance_scale=self.config.drone.follow_distance_scale,
        )
        self._escorting = False
        self._rover_pose: Optional[np.ndarray] = None
        #: Built once the drone's own take-off point is known, which is simply
        #: where it was standing when the first odometry arrived.
        self.home_controller: Optional[ReturnHomeController] = None
        self._returning = False
        self._takeoff_xy: Optional[np.ndarray] = None

        self._position: Optional[np.ndarray] = None
        self._yaw = 0.0
        self._last_waypoint_logged = -1
        self._last_phase = FlightPhase.GROUNDED

        self.cmd_pub = self.create_publisher(
            Twist, str(self.get_parameter("cmd_vel_topic").value), DEFAULT_QOS
        )
        self.status_pub = self.create_publisher(
            ExplorationStatus, str(self.get_parameter("status_topic").value), LATCHED_QOS
        )
        self.create_subscription(
            Odometry, str(self.get_parameter("odometry_topic").value), self._on_odometry, SENSOR_QOS
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("rover_odometry_topic").value),
            self._on_rover_odometry,
            SENSOR_QOS,
        )
        self.create_service(Trigger, "~/start", self._on_start)
        self.create_service(Trigger, "~/follow", self._on_follow)
        self.create_service(Trigger, "~/return_home", self._on_return_home)
        self.create_service(Trigger, "~/hold", self._on_hold)
        self.create_timer(1.0 / max(self.config.drone.control_rate_hz, 1e-3), self._control_step)

        if bool(self.get_parameter("auto_start").value):
            self.controller.start()
        self.get_logger().info(
            f"[DRONE] explorer ready: {len(self.waypoints)} waypoints, "
            f"{self.config.drone.lane_spacing_m:.1f} m lanes at "
            f"{self.config.drone.scan_altitude_m:.1f} m, "
            f"scan speed {self.config.drone.scan_speed_mps:.2f} m/s, "
            f"finish_scan_after_target_found="
            f"{self.config.drone.finish_scan_after_target_found}, "
            f"camera {self.config.drone.camera.width}x{self.config.drone.camera.height}"
            f"@{self.config.drone.camera.update_rate_hz:.1f}Hz "
            f"(footprint {self.config.camera_ground_footprint_m():.2f} m)"
        )

    # -- callbacks ---------------------------------------------------------
    def _on_odometry(self, msg: Odometry) -> None:
        # The lawnmower waypoints are map-frame coordinates, so the pose flown
        # against them must be too. map -> drone/odom happens to be identity
        # here, but relying on that is how the rover ended up a spawn offset
        # from its goal.
        position, yaw, error = odometry_position_in_frame(
            self.tf_buffer, msg, self.config.frames.map_frame
        )
        if position is None:
            self.get_logger().warn(
                f"[DRONE] cannot express odometry in {self.config.frames.map_frame}: {error}",
                throttle_duration_sec=5.0,
            )
            return
        self._position = position
        self._yaw = yaw
        if self._takeoff_xy is None:
            # Wherever the drone was standing when it first reported in is the
            # spot it has to come back to. Nothing else needs to know it.
            self._takeoff_xy = position[:2].copy()
            self.get_logger().info(
                f"[DRONE] take-off point recorded at "
                f"({self._takeoff_xy[0]:.2f}, {self._takeoff_xy[1]:.2f})"
            )

    def _on_start(self, _request, response):
        self.controller.start()
        response.success = True
        response.message = "exploration started"
        self.get_logger().info("[DRONE] exploration start requested")
        return response

    def _on_rover_odometry(self, msg: Odometry) -> None:
        pose, error = odometry_pose_in_frame(
            self.tf_buffer, msg, self.config.frames.map_frame
        )
        if pose is None:
            self.get_logger().warn(
                f"[DRONE] cannot express rover odometry in the map frame: {error}",
                throttle_duration_sec=5.0,
            )
            return
        self._rover_pose = pose

    def _on_follow(self, _request, response):
        """Stop the coverage pattern and hold station on the rover."""
        self._escorting = True
        response.success = True
        response.message = (
            f"escorting the rover from {self.escort.standoff_m:.2f} m behind it"
        )
        self.get_logger().info(f"[DRONE] {response.message}")
        return response

    def _on_return_home(self, _request, response):
        """Break station and fly back to the take-off point."""
        if self._takeoff_xy is None:
            response.success = False
            response.message = "no odometry yet, so the take-off point is unknown"
            self.get_logger().error(f"[DRONE] {response.message}")
            return response
        self.home_controller = ReturnHomeController(
            self._takeoff_xy,
            cruise_altitude_m=self.config.drone.scan_altitude_m,
            speed_mps=self.config.drone.return_speed_mps,
            descend_speed_mps=self.config.drone.descend_speed_mps,
            landed_altitude_m=self.config.drone.landed_altitude_m,
        )
        self._escorting = False
        self._returning = True
        response.success = True
        response.message = (
            f"returning to ({self._takeoff_xy[0]:.2f}, {self._takeoff_xy[1]:.2f})"
        )
        self.get_logger().info(f"[DRONE] {response.message}")
        return response

    def _on_hold(self, _request, response):
        self.controller._phase = FlightPhase.COMPLETE  # station-keep at altitude
        response.success = True
        response.message = "holding at scan altitude"
        self.get_logger().info("[DRONE] hold requested")
        return response

    # -- control -----------------------------------------------------------
    def _control_step(self) -> None:
        if self._position is None:
            self.get_logger().warn(
                "[DRONE] no odometry yet; holding zero velocity", throttle_duration_sec=5.0
            )
            self.cmd_pub.publish(Twist())
            return

        if self._returning and self.home_controller is not None:
            command = self.home_controller.compute(self._position, self._yaw)
        elif self._escorting:
            if self._rover_pose is None:
                self.get_logger().warn(
                    "[DRONE] escorting but the rover pose is unknown; holding",
                    throttle_duration_sec=5.0,
                )
                self.cmd_pub.publish(Twist())
                return
            command = self.escort.compute(self._position, self._yaw, self._rover_pose)
        else:
            command = self.controller.compute(self._position, self._yaw)

        # gz-sim's VelocityControl interprets the twist in the *body* frame, so
        # the map-frame setpoint must be rotated by the inverse of the current
        # yaw. Skipping this works only while yaw is exactly zero and fails
        # silently (as a slow drift) the moment it is not.
        cos_y, sin_y = math.cos(self._yaw), math.sin(self._yaw)
        body_x = cos_y * command.velocity_map[0] + sin_y * command.velocity_map[1]
        body_y = -sin_y * command.velocity_map[0] + cos_y * command.velocity_map[1]

        twist = Twist()
        twist.linear.x = float(body_x)
        twist.linear.y = float(body_y)
        twist.linear.z = float(command.velocity_map[2])
        twist.angular.z = float(command.yaw_rate)
        self.cmd_pub.publish(twist)

        self._publish_status(command)
        self._log_progress(command)

    def _publish_status(self, command) -> None:
        status = ExplorationStatus()
        status.header = make_header(self, self.config.frames.map_frame)
        status.phase = command.phase.value
        status.at_scan_altitude = command.phase in (
            FlightPhase.SCANNING,
            FlightPhase.COMPLETE,
            FlightPhase.ESCORTING,
        )
        # Escorting means the sweep is behind us: the mission has already left
        # EXPLORING, and reporting "incomplete" here would stall a re-entry.
        status.complete = command.phase in (
            FlightPhase.COMPLETE,
            FlightPhase.ESCORTING,
            FlightPhase.RETURNING,
            FlightPhase.LANDED,
        )
        status.escorting = command.phase is FlightPhase.ESCORTING
        status.returned_home = command.phase is FlightPhase.LANDED
        status.waypoint_index = int(command.waypoint_index)
        status.waypoint_count = int(len(self.waypoints))
        status.coverage_fraction = float(self.controller.coverage_fraction)
        status.distance_to_waypoint_m = float(
            command.distance_to_waypoint_m if math.isfinite(command.distance_to_waypoint_m) else 0.0
        )
        status.altitude_m = float(self._position[2]) if self._position is not None else 0.0
        self.status_pub.publish(status)

    def _log_progress(self, command) -> None:
        if command.phase is not self._last_phase:
            self.get_logger().info(
                f"[DRONE] {self._last_phase.value} -> {command.phase.value} "
                f"at altitude {self._position[2]:.2f} m"
            )
            self._last_phase = command.phase
        if (
            command.phase is FlightPhase.SCANNING
            and command.waypoint_index != self._last_waypoint_logged
        ):
            self._last_waypoint_logged = command.waypoint_index
            target = self.waypoints[min(command.waypoint_index, len(self.waypoints) - 1)]
            self.get_logger().info(
                f"[DRONE] waypoint {command.waypoint_index + 1}/{len(self.waypoints)} "
                f"-> ({target[0]:.1f}, {target[1]:.1f}) "
                f"[{self.controller.coverage_fraction:.0%} covered]"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DroneExplorerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
