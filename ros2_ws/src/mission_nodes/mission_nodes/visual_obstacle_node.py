"""Monocular obstacle perception node - the mission's only ranging sensor.

One instance runs per camera, mirroring the QR detectors.  It segments each
frame into floor and not-floor, intersects the bottom of every obstacle
silhouette with the known ground plane, and publishes the resulting contact
points, their measured heights and the surrounding free floor as a
``mission_interfaces/GroundObservation`` in the planning frame.

Consumers:

* ``world_model_node`` folds the drone's observations into the occupancy grid,
  and the rover's contacts in as runtime obstacle evidence;
* ``rover_path_follower_node`` uses ``nearest_forward_range_m`` for its
  emergency stop.

The algorithm lives in :mod:`mission_core.vision_mapping`; this node is the
adapter that supplies it with intrinsics from ``CameraInfo`` and a pose from
TF, and that keeps other vehicles out of its own map.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from mission_core.camera import PinholeCamera
from mission_core.vision_mapping import MonocularObstacleDetector

from mission_interfaces.msg import GroundObservation

from .common import (
    DEFAULT_QOS,
    SENSOR_QOS,
    declare_mission_config,
    lookup_transform,
    stamp_to_seconds,
)


class VisualObstacleNode(Node):
    """Turns camera frames into ground-plane obstacle evidence."""

    def __init__(self) -> None:
        super().__init__("visual_obstacle_detector")
        self.config = declare_mission_config(self)
        vision = self.config.vision

        self.declare_parameter("camera_name", "drone")
        self.declare_parameter("image_topic", "/drone/camera/image")
        self.declare_parameter("camera_info_topic", "/drone/camera/camera_info")
        self.declare_parameter("observation_topic", "/perception/drone/ground_observations")
        self.declare_parameter(
            "camera_optical_frame", self.config.frames.drone_camera_optical_frame
        )
        self.declare_parameter("max_range_m", vision.drone_max_range_m)
        self.declare_parameter("forward_half_angle_rad", self.config.rover.obstacle_stop_half_angle_rad)
        # Frames of other vehicles. They are real objects, correctly detected,
        # and mapping them would put an obstacle on the route the mission is
        # driving: the drone escorts the rover from behind and has it in view
        # for the whole navigation phase.
        self.declare_parameter("self_filter_frames", [self.config.frames.rover_base_frame])
        self.declare_parameter("rate_hz", self.config.perception.qr_detection_rate_hz)

        self.camera_name = str(self.get_parameter("camera_name").value)
        self.camera_optical_frame = str(self.get_parameter("camera_optical_frame").value)
        self.map_frame = self.config.frames.map_frame
        self.self_filter_frames = [
            frame for frame in self.get_parameter("self_filter_frames").value if frame
        ]

        self.detector = MonocularObstacleDetector(
            ground_z=vision.ground_z_m,
            max_range_m=float(self.get_parameter("max_range_m").value),
            min_obstacle_height_m=vision.min_obstacle_height_m,
            max_obstacle_height_m=vision.max_obstacle_height_m,
            column_stride=vision.column_stride_px,
            free_stride=vision.free_stride_px,
            forward_half_angle_rad=float(self.get_parameter("forward_half_angle_rad").value),
            chroma_sigma=vision.chroma_sigma,
            bright_luma_margin=vision.bright_luma_margin,
            min_blob_area_px=vision.min_blob_area_px,
            downsample=vision.segmentation_downsample,
            max_non_ground_fraction=vision.max_non_ground_fraction,
        )
        self.bridge = CvBridge()
        self.camera: Optional[PinholeCamera] = None
        self._last_processed_s = -math.inf
        self._min_period_s = 1.0 / max(float(self.get_parameter("rate_hz").value), 1e-3)
        self._tf_failures = 0
        self._rejected_frames = 0

        from tf2_ros import Buffer, TransformListener

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.observation_pub = self.create_publisher(
            GroundObservation,
            str(self.get_parameter("observation_topic").value),
            DEFAULT_QOS,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._on_camera_info,
            SENSOR_QOS,
        )
        self.create_subscription(
            Image, str(self.get_parameter("image_topic").value), self._on_image, SENSOR_QOS
        )
        self.get_logger().info(
            f"[VISION] {self.camera_name} obstacle mapper up: ground plane at "
            f"z={self.detector.ground_z:.2f} m, trusted to "
            f"{self.detector.max_range_m:.1f} m, heights "
            f"{self.detector.min_obstacle_height_m:.2f}-"
            f"{self.detector.max_obstacle_height_m:.1f} m"
        )

    # -- callbacks ---------------------------------------------------------
    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self.camera is not None:
            return
        self.camera = PinholeCamera.from_camera_info_k(msg.width, msg.height, msg.k)
        self.get_logger().info(
            f"[VISION] {self.camera_name} intrinsics: {msg.width}x{msg.height} "
            f"fx={self.camera.fx:.1f} fy={self.camera.fy:.1f}"
        )

    def _on_image(self, msg: Image) -> None:
        if self.camera is None:
            self.get_logger().warn(
                "[VISION] no CameraInfo yet; a pixel cannot be turned into a ray "
                "without intrinsics",
                throttle_duration_sec=5.0,
            )
            return
        stamp_s = stamp_to_seconds(msg.header.stamp)
        if stamp_s - self._last_processed_s < self._min_period_s:
            return
        self._last_processed_s = stamp_s

        map_from_camera, error = lookup_transform(
            self.tf_buffer, self.map_frame, self.camera_optical_frame, msg.header.stamp
        )
        if map_from_camera is None:
            self._tf_failures += 1
            self.get_logger().warn(
                f"[VISION] dropping frame: TF unavailable ({error}); "
                f"{self._tf_failures} total",
                throttle_duration_sec=5.0,
            )
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warn(
                f"[VISION] cv_bridge conversion failed: {exc}", throttle_duration_sec=5.0
            )
            return

        observation = self.detector.process(
            frame,
            self.camera,
            map_from_camera,
            exclude_centres_xy=self._other_vehicles(msg),
            exclude_radius_m=self.config.vision.self_filter_radius_m,
        )
        if not observation.usable:
            self._rejected_frames += 1
            self.get_logger().warn(
                f"[VISION] {self.camera_name} frame not mapped: no floor reference yet "
                f"and {observation.non_ground_fraction:.0%} of this frame is not floor, "
                f"so it cannot establish one ({self._rejected_frames} total)",
                throttle_duration_sec=5.0,
            )

        out = GroundObservation()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.map_frame
        out.source = f"{self.camera_name}_camera"
        out.contacts = observation.contacts.astype(np.float32).reshape(-1)
        out.free_space = observation.free.astype(np.float32).reshape(-1)
        out.nearest_forward_range_m = float(observation.nearest_forward_range_m)
        out.columns_evaluated = int(observation.columns_evaluated)
        out.non_ground_fraction = float(observation.non_ground_fraction)
        out.usable = bool(observation.usable)
        self.observation_pub.publish(out)

    def _other_vehicles(self, msg: Image) -> Optional[np.ndarray]:
        """Positions of the vehicles that must not be mapped, in ``map``."""
        centres: List[np.ndarray] = []
        for frame in self.self_filter_frames:
            transform, _ = lookup_transform(
                self.tf_buffer, self.map_frame, frame, msg.header.stamp
            )
            if transform is not None:
                centres.append(transform.translation[:2])
        return np.asarray(centres, dtype=float) if centres else None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualObstacleNode()
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
