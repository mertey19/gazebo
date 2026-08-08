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
    FlightPhase,
    WaypointFlightController,
    lawnmower_waypoints,
)
from mission_interfaces.msg import ExplorationStatus

from .common import DEFAULT_QOS, LATCHED_QOS, SENSOR_QOS, declare_mission_config, make_header


class DroneExplorerNode(Node):
    """Flies the coverage pattern that lets the drone see every station."""

    def __init__(self) -> None:
        super().__init__("drone_explorer")
        self.config = declare_mission_config(self)

        self.declare_parameter("odometry_topic", "/drone/odometry")
        self.declare_parameter("cmd_vel_topic", "/drone/cmd_vel")
        self.declare_parameter("status_topic", "/drone/exploration_status")
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
        self.create_service(Trigger, "~/start", self._on_start)
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
        position = msg.pose.pose.position
        self._position = np.array([position.x, position.y, position.z], dtype=float)
        q = msg.pose.pose.orientation
        self._yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def _on_start(self, _request, response):
        self.controller.start()
        response.success = True
        response.message = "exploration started"
        self.get_logger().info("[DRONE] exploration start requested")
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
        status.at_scan_altitude = command.phase in (FlightPhase.SCANNING, FlightPhase.COMPLETE)
        status.complete = command.phase is FlightPhase.COMPLETE
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
