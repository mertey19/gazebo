"""QR perception node.

One instance runs per camera (drone and rover).  It decodes QR codes from the
image stream, recovers each marker's pose with PnP, transforms that pose into
the planning frame using TF, and publishes a
``mission_interfaces/QrObservation``.

It never consults simulation state.  The payload comes from the image and the
position comes from the marker's measured size and the TF chain.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from mission_core.camera import PinholeCamera
from mission_core.errors import PerceptionError
from mission_core.geometry import matrix_to_quaternion
from mission_core.qr import QrDetector

from mission_interfaces.msg import QrObservation

from .common import (
    DEFAULT_QOS,
    SENSOR_QOS,
    declare_mission_config,
    lookup_transform,
    point_msg,
    stamp_to_seconds,
    vector_msg,
)


class QrDetectorNode(Node):
    """Turns camera frames into world-referenced QR observations."""

    def __init__(self) -> None:
        super().__init__("qr_detector")
        self.config = declare_mission_config(self)

        self.declare_parameter("camera_name", "drone")
        self.declare_parameter("image_topic", "/drone/camera/image")
        self.declare_parameter("camera_info_topic", "/drone/camera/camera_info")
        self.declare_parameter("observation_topic", "/perception/drone/qr_observations")
        # The optical frame is configured rather than taken from the image
        # header: a bridge that forwards the wrong frame_id would otherwise
        # silently place every target in the wrong spot.
        self.declare_parameter(
            "camera_optical_frame", self.config.frames.drone_camera_optical_frame
        )
        self.declare_parameter("publish_annotated_image", False)

        self.camera_name = str(self.get_parameter("camera_name").value)
        self.camera_optical_frame = str(self.get_parameter("camera_optical_frame").value)
        self.map_frame = self.config.frames.map_frame

        self.detector = QrDetector(
            self.config.mission.qr_plate_size_m,
            quiet_zone_modules=self.config.mission.qr_quiet_zone_modules,
            max_reprojection_error_px=self.config.perception.max_reprojection_error_px,
            min_quad_area_px=self.config.perception.min_quad_area_px,
            min_apparent_size_px=self.config.perception.min_apparent_size_px,
        )
        self.bridge = CvBridge()
        self.camera: Optional[PinholeCamera] = None
        self._consecutive_bad_frames = 0
        self._last_processed_s = -float("inf")
        self._min_period_s = 1.0 / max(self.config.perception.qr_detection_rate_hz, 1e-3)
        self._tf_failures = 0

        # tf2 is imported lazily so the module can be imported (for linting or
        # doc generation) on a machine without a full ROS installation.
        from tf2_ros import Buffer, TransformListener

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.observation_pub = self.create_publisher(
            QrObservation, str(self.get_parameter("observation_topic").value), DEFAULT_QOS
        )
        self.annotated_pub = (
            self.create_publisher(Image, "~/annotated_image", SENSOR_QOS)
            if bool(self.get_parameter("publish_annotated_image").value)
            else None
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
            f"[QR] {self.camera_name} detector up: plate "
            f"{self.config.mission.qr_plate_size_m:.2f} m, throttled to "
            f"{self.config.perception.qr_detection_rate_hz:.1f} Hz, "
            f"optical frame {self.camera_optical_frame}"
        )

    # -- callbacks ---------------------------------------------------------
    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self.camera is not None:
            return
        self.camera = PinholeCamera.from_camera_info_k(msg.width, msg.height, msg.k)
        self.get_logger().info(
            f"[QR] {self.camera_name} intrinsics: {msg.width}x{msg.height} "
            f"fx={self.camera.fx:.1f} fy={self.camera.fy:.1f} "
            f"cx={self.camera.cx:.1f} cy={self.camera.cy:.1f}"
        )

    def _on_image(self, msg: Image) -> None:
        if self.camera is None:
            self.get_logger().warn(
                "[QR] no CameraInfo yet; cannot solve marker pose without intrinsics",
                throttle_duration_sec=5.0,
            )
            return

        stamp_s = stamp_to_seconds(msg.header.stamp)
        # Detection is the expensive part of this node; throttling here rather
        # than lowering the sensor rate keeps the raw stream available for RViz.
        if stamp_s - self._last_processed_s < self._min_period_s:
            return
        self._last_processed_s = stamp_s

        if msg.header.frame_id and msg.header.frame_id != self.camera_optical_frame:
            self.get_logger().warn(
                f"[QR] image frame_id {msg.header.frame_id!r} differs from the configured "
                f"optical frame {self.camera_optical_frame!r}; using the configured one",
                throttle_duration_sec=30.0,
            )

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self._note_bad_frame(f"cv_bridge conversion failed: {exc}")
            return

        try:
            detections = self.detector.detect(frame, self.camera)
        except PerceptionError as exc:
            self._note_bad_frame(str(exc))
            return
        self._consecutive_bad_frames = 0

        if not detections:
            return

        map_from_camera, error = lookup_transform(
            self.tf_buffer, self.map_frame, self.camera_optical_frame, msg.header.stamp
        )
        if map_from_camera is None:
            self._tf_failures += 1
            self.get_logger().warn(
                f"[QR] dropping {len(detections)} detection(s): TF unavailable ({error}); "
                f"{self._tf_failures} total",
                throttle_duration_sec=5.0,
            )
            return

        for detection in detections:
            self.observation_pub.publish(
                self._build_observation(detection, map_from_camera, msg)
            )

        if self.annotated_pub is not None:
            self._publish_annotated(frame, detections, msg)

    # -- helpers -----------------------------------------------------------
    def _note_bad_frame(self, detail: str) -> None:
        self._consecutive_bad_frames += 1
        text = (
            f"[QR] {self.camera_name} unusable frame ({self._consecutive_bad_frames} "
            f"consecutive): {detail}"
        )
        # Two call sites, not one bound method chosen at runtime: rclpy keys
        # its logger cache by caller location and rejects a severity change
        # from the same line.
        if self._consecutive_bad_frames >= self.config.perception.max_consecutive_bad_frames:
            self.get_logger().error(text, throttle_duration_sec=2.0)
        else:
            self.get_logger().warn(text, throttle_duration_sec=2.0)

    def _build_observation(self, detection, map_from_camera, image_msg) -> QrObservation:
        position_map = map_from_camera.apply(detection.position_optical)
        # The marker's outward normal is its local +Z; rotating it through the
        # camera and then through TF is what makes "which way does the plate
        # face" a world-frame statement rather than a camera-frame one.
        normal_map = (
            map_from_camera.rotation @ detection.rotation_optical @ np.array([0.0, 0.0, 1.0])
        )

        msg = QrObservation()
        msg.header = image_msg.header
        msg.header.frame_id = self.camera_optical_frame
        msg.qr_id = detection.payload
        msg.source = f"{self.camera_name}_camera"
        msg.pose_camera.position = point_msg(detection.position_optical)
        qx, qy, qz, qw = matrix_to_quaternion(detection.rotation_optical)
        msg.pose_camera.orientation.x = float(qx)
        msg.pose_camera.orientation.y = float(qy)
        msg.pose_camera.orientation.z = float(qz)
        msg.pose_camera.orientation.w = float(qw)
        msg.position_map = point_msg(position_map)
        msg.normal_map = vector_msg(normal_map)
        msg.map_frame_id = self.map_frame
        msg.range_m = float(detection.range_m)
        msg.code_size_m = float(self.detector.code_size_m(detection.payload))
        msg.apparent_size_px = float(detection.apparent_size_px)
        msg.reprojection_error_px = float(detection.reprojection_error_px)
        msg.ambiguity_ratio = float(detection.ambiguity_ratio)
        msg.confidence = float(detection.confidence)
        return msg

    def _publish_annotated(self, frame, detections, image_msg) -> None:
        import cv2

        annotated = frame.copy()
        for detection in detections:
            corners = detection.corners_px.astype(int).reshape(-1, 1, 2)
            cv2.polylines(annotated, [corners], True, (0, 220, 0), 2)
            cv2.putText(
                annotated,
                f"{detection.payload} {detection.range_m:.1f}m",
                tuple(detection.corners_px[0].astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 0),
                2,
            )
        out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out.header = image_msg.header
        self.annotated_pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = QrDetectorNode()
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
