"""Rover path-following node.

Consumes ``nav_msgs/Path`` on ``/mission/rover_path`` - the path is *received*
from the mission manager, never computed here - and drives the differential
base along it with pure pursuit, publishing tracking diagnostics as it goes.

Also runs a last-resort safety stop from the rover's own planar lidar.  That
stop is a safety net, not the obstacle-avoidance strategy: avoidance happens in
the planner, using the map the drone built.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Trigger

from mission_core.errors import FailureReason
from mission_core.path_following import (
    FollowerState,
    PurePursuitController,
    SafetyStopWatchdog,
    TrackingStatus,
    VerificationSweep,
)

from mission_interfaces.msg import TrackingStatus as TrackingStatusMsg

from .common import (
    DEFAULT_QOS,
    LATCHED_QOS,
    SENSOR_QOS,
    declare_mission_config,
    make_header,
    node_time_seconds,
    odometry_pose_in_frame,
    path_msg_to_array,
)


class RoverPathFollowerNode(Node):
    """Drives the rover along an externally supplied path."""

    def __init__(self) -> None:
        super().__init__("rover_path_follower")
        self.config = declare_mission_config(self)

        self.declare_parameter("path_topic", "/mission/rover_path")
        self.declare_parameter("odometry_topic", "/rover/odometry")
        self.declare_parameter("cmd_vel_topic", "/rover/cmd_vel")
        self.declare_parameter("scan_topic", "/rover/scan")
        self.declare_parameter("status_topic", "/rover/tracking_status")
        self.declare_parameter("use_safety_stop", True)

        rover = self.config.rover
        self.controller = PurePursuitController(
            lookahead_distance_m=rover.lookahead_distance_m,
            min_lookahead_distance_m=rover.min_lookahead_distance_m,
            lookahead_time_s=rover.lookahead_time_s,
            max_linear_velocity=rover.max_linear_velocity,
            max_angular_velocity=rover.max_angular_velocity,
            heading_kp=rover.heading_kp,
            rotate_in_place_rad=rover.rotate_in_place_rad,
            goal_tolerance_m=rover.goal_tolerance_m,
            goal_yaw_tolerance_rad=rover.goal_yaw_tolerance_rad,
            align_to_goal_yaw=rover.align_to_goal_yaw,
            max_cross_track_error_m=rover.max_cross_track_error_m,
            cross_track_grace_s=rover.cross_track_grace_s,
            approach_slowdown_m=rover.approach_slowdown_m,
        )

        # tf2 is imported lazily so the module stays importable for linting
        # on a machine without a full ROS installation.
        from tf2_ros import Buffer, TransformListener

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._sweep = VerificationSweep(
            self.config.verification.search_sweep_rad,
            self.config.verification.search_yaw_rate_rad_s,
        )
        self._safety_watchdog = SafetyStopWatchdog(
            rover.safety_stop_replan_delay_s
        )
        self._searching = False

        self._pose: Optional[np.ndarray] = None
        self._pose_failures = 0
        self._last_step_s: Optional[float] = None
        self._terminal_logged = False
        self._safety_blocked = False
        self._min_forward_range = math.inf
        self._path_frame = self.config.frames.map_frame

        self.cmd_pub = self.create_publisher(
            Twist, str(self.get_parameter("cmd_vel_topic").value), DEFAULT_QOS
        )
        self.status_pub = self.create_publisher(
            TrackingStatusMsg, str(self.get_parameter("status_topic").value), LATCHED_QOS
        )
        self.create_subscription(
            Path, str(self.get_parameter("path_topic").value), self._on_path, LATCHED_QOS
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odometry_topic").value),
            self._on_odometry,
            SENSOR_QOS,
        )
        if bool(self.get_parameter("use_safety_stop").value):
            self.create_subscription(
                LaserScan, str(self.get_parameter("scan_topic").value), self._on_scan, SENSOR_QOS
            )
        self.create_service(Trigger, "~/stop", self._on_stop)
        self.create_service(Trigger, "~/search", self._on_search)

        self.create_timer(1.0 / max(rover.control_rate_hz, 1e-3), self._control_step)
        self.get_logger().info(
            f"[ROVER] follower ready: v<={rover.max_linear_velocity:.2f} m/s, "
            f"w<={rover.max_angular_velocity:.2f} rad/s, "
            f"goal tolerance {rover.goal_tolerance_m:.2f} m"
        )

    # -- callbacks ---------------------------------------------------------
    def _on_path(self, msg: Path) -> None:
        poses = path_msg_to_array(msg)
        if len(poses) == 0:
            self.get_logger().error("[ROVER] received an empty path; ignoring it")
            return
        if msg.header.frame_id and msg.header.frame_id != self._path_frame:
            # Following a path expressed in a different frame would drive the
            # rover somewhere plausible-looking and wrong. Refuse instead.
            self.get_logger().error(
                f"[ROVER] path is in frame {msg.header.frame_id!r} but the rover plans in "
                f"{self._path_frame!r}; refusing to follow it"
            )
            return
        self.controller.set_path(poses)
        self._safety_watchdog.reset()
        self._terminal_logged = False
        self._last_step_s = None
        self.get_logger().info(
            f"[ROVER] accepted path with {len(poses)} poses, "
            f"{float(np.sum(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1))):.2f} m; "
            "navigation started"
        )

    def _on_odometry(self, msg: Odometry) -> None:
        # The path is expressed in the planning frame, so the pose used to
        # follow it must be too. Odometry is in the rover's odom frame, which
        # starts at the spawn pose; using it raw drove the rover a full spawn
        # offset away from the goal while every log line looked healthy.
        pose, error = odometry_pose_in_frame(self.tf_buffer, msg, self._path_frame)
        if pose is None:
            self._pose_failures += 1
            self.get_logger().warn(
                f"[ROVER] cannot express odometry in {self._path_frame}: {error}",
                throttle_duration_sec=5.0,
            )
            return
        self._pose = pose

    def _on_scan(self, msg: LaserScan) -> None:
        """Emergency stop when something is inside the safety radius ahead."""
        ranges = np.asarray(msg.ranges, dtype=float)
        if ranges.size == 0:
            return
        angles = msg.angle_min + np.arange(ranges.size) * msg.angle_increment
        # Only a narrow forward wedge matters: the rover cannot strafe, so a
        # return off to the side is not on a collision course.
        forward = np.abs(angles) <= math.radians(35.0)
        valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
        candidates = ranges[forward & valid]
        self._min_forward_range = float(candidates.min()) if candidates.size else math.inf
        blocked = self._min_forward_range < self.config.rover.obstacle_stop_distance_m
        if blocked and not self._safety_blocked:
            self.get_logger().warn(
                f"[ROVER] safety stop: obstacle {self._min_forward_range:.2f} m ahead "
                f"(< {self.config.rover.obstacle_stop_distance_m:.2f} m)"
            )
        self._safety_blocked = blocked

    def _on_search(self, _request, response):
        """Sweep the heading while the QR detector looks for the code."""
        self._sweep.reset()
        self._searching = True
        response.success = True
        response.message = (
            f"sweeping +/-{self.config.verification.search_sweep_rad:.2f} rad at "
            f"{self.config.verification.search_yaw_rate_rad_s:.2f} rad/s"
        )
        self.get_logger().info(f"[ROVER] verification sweep started: {response.message}")
        return response

    def _on_stop(self, _request, response):
        self.controller.reset()
        self._safety_watchdog.reset()
        self._searching = False
        self.cmd_pub.publish(Twist())
        response.success = True
        response.message = "path cleared, rover stopped"
        self.get_logger().info("[ROVER] stop requested")
        return response

    # -- control -----------------------------------------------------------
    def _control_step(self) -> None:
        if self._searching:
            self._sweep_step()
            return
        if not self.controller.has_path:
            return
        if self._pose is None:
            self.get_logger().warn(
                "[ROVER] no odometry yet; holding still", throttle_duration_sec=5.0
            )
            self.cmd_pub.publish(Twist())
            return

        now = node_time_seconds(self)
        dt = now - self._last_step_s if self._last_step_s is not None else 0.0
        self._last_step_s = now

        status = self.controller.compute(self._pose, dt)
        linear, angular = status.linear_velocity, status.angular_velocity

        # The safety stop suppresses forward motion but leaves rotation
        # available, so the rover can still turn away rather than deadlock.
        blocked_forward_motion = (
            self._safety_blocked and linear > 0.0 and not status.is_terminal
        )
        if blocked_forward_motion:
            linear = 0.0
        if self._safety_watchdog.update(blocked_forward_motion, dt):
            status = TrackingStatus(
                state=FollowerState.FAILED,
                linear_velocity=0.0,
                angular_velocity=0.0,
                distance_to_goal_m=status.distance_to_goal_m,
                cross_track_error_m=status.cross_track_error_m,
                heading_error_rad=status.heading_error_rad,
                progress_index=status.progress_index,
                failure_reason=FailureReason.PATH_TRACKING_FAILURE,
                detail=(
                    f"safety lidar blocked forward motion for "
                    f"{self._safety_watchdog.blocked_s:.1f} s "
                    f"at {self._min_forward_range:.2f} m; requesting a replan"
                ),
            )
            linear = 0.0
            angular = 0.0

        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        self.cmd_pub.publish(twist)

        self._publish_status(status, linear, angular)

        if status.is_terminal and not self._terminal_logged:
            self._terminal_logged = True
            if status.state is FollowerState.GOAL_REACHED:
                self.get_logger().info(
                    f"[ROVER] Goal reached: {status.distance_to_goal_m:.2f} m from the goal, "
                    f"cross-track {status.cross_track_error_m:.2f} m"
                )
            else:
                self.get_logger().error(f"[ROVER] tracking FAILED: {status.detail}")

    def _sweep_step(self) -> None:
        """Rotate in place looking for the code; stop when the sweep is done."""
        dt = 1.0 / max(self.config.rover.control_rate_hz, 1e-3)
        yaw_rate = self._sweep.step(dt)
        twist = Twist()
        twist.angular.z = float(
            np.clip(yaw_rate, -self.config.rover.max_angular_velocity,
                    self.config.rover.max_angular_velocity)
        )
        self.cmd_pub.publish(twist)
        if self._sweep.finished:
            self._searching = False
            self.cmd_pub.publish(Twist())
            self.get_logger().info(
                "[ROVER] verification sweep complete; holding position"
            )

    def _publish_status(self, status, linear: float, angular: float) -> None:
        msg = TrackingStatusMsg()
        msg.header = make_header(self, self._path_frame)
        msg.state = status.state.value
        msg.distance_to_goal_m = float(status.distance_to_goal_m)
        msg.cross_track_error_m = float(status.cross_track_error_m)
        msg.heading_error_rad = float(status.heading_error_rad)
        msg.progress_index = int(status.progress_index)
        msg.commanded_linear = float(linear)
        msg.commanded_angular = float(angular)
        msg.goal_reached = status.state is FollowerState.GOAL_REACHED
        msg.failed = status.state is FollowerState.FAILED
        msg.failure_reason = (
            status.failure_reason.value
            if status.failure_reason is not FailureReason.NONE
            else FailureReason.NONE.value
        )
        msg.detail = status.detail
        self.status_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RoverPathFollowerNode()
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
