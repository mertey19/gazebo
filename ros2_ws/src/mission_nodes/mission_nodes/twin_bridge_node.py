"""Ground-station bridge: the mission, as the Simurgh digital twin hears it.

Subscribes to what the mission already publishes and forwards it to the
station's ``DigitalTwinUdpIngress`` as ``DigitalTwinMessageV1`` JSON over UDP.
Nothing on the station side changes: it already has phases, route waypoints,
target/obstacle/voxel deltas, a QR gallery and vehicle trails - this node just
speaks to it.

Read-only by construction.  It subscribes and sends; it publishes no ROS topic,
offers no service and touches no vehicle.  A ground station that disappears
mid-mission therefore cannot affect the mission, which is why the socket is
also unconnected UDP: there is nothing to block on and nothing to time out.

What each vehicle trail is made of: nothing here.  The station lengthens a
vehicle's track from successive pose messages on its own, so a trail topic
would only duplicate state it already keeps.
"""

from __future__ import annotations

import base64
import socket
from typing import Dict, List, Optional

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import Image

from mission_core.mission_state import MissionState
from mission_core.occupancy import GridMetadata, OccupancyGrid
from mission_core.twin_telemetry import (
    VEHICLE_ROVER,
    VEHICLE_UAV,
    GeoAnchor,
    TwinMessageBuilder,
    iter_json_datagrams,
    qr_photo_message,
)

from mission_interfaces.msg import MissionStatus, ObstacleArray, TargetArray

from .common import (
    DEFAULT_QOS,
    LATCHED_QOS,
    SENSOR_QOS,
    declare_mission_config,
    lookup_transform,
    node_time_seconds,
    odometry_pose_in_frame,
    path_msg_to_array,
)


class _Record:
    """Duck-typed stand-in for the core records the builder expects."""

    def __init__(self, **fields) -> None:
        self.__dict__.update(fields)


class TwinBridgeNode(Node):
    """Forwards live mission state to the digital-twin ground station."""

    def __init__(self) -> None:
        super().__init__("twin_bridge")
        self.config = declare_mission_config(self)
        station = self.config.ground_station

        self.declare_parameter("drone_odometry_topic", "/drone/odometry")
        self.declare_parameter("rover_odometry_topic", "/rover/odometry")
        self.declare_parameter("mission_status_topic", "/mission/status")
        self.declare_parameter("targets_topic", "/world_model/targets")
        self.declare_parameter("obstacles_topic", "/world_model/obstacles")
        self.declare_parameter("occupancy_topic", "/world_model/occupancy_grid")
        self.declare_parameter("path_topic", "/mission/rover_path")
        self.declare_parameter("rover_image_topic", "/rover/camera/image")

        self.builder = TwinMessageBuilder(
            anchor=GeoAnchor(
                latitude=station.anchor_latitude,
                longitude=station.anchor_longitude,
                altitude_m=station.anchor_altitude_m,
            ),
            source_id=station.source_id,
            auth_token=station.auth_token,
            voxel_batch=station.voxel_batch,
        )
        self.map_frame = self.config.frames.map_frame
        self.target = (station.host, station.port)
        self.max_datagram_bytes = station.max_datagram_bytes
        self.send_voxels = station.send_voxels
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self._drone: Optional[Dict] = None
        self._rover: Optional[Dict] = None
        self._state = MissionState.IDLE
        self._status = "running"
        self._warning = ""
        self._note = ""
        self._verified: List[str] = []
        self._targets: List[_Record] = []
        self._obstacles: List[_Record] = []
        self._grid: Optional[OccupancyGrid] = None
        self._path: Optional[np.ndarray] = None
        self._last_rover_image: Optional[Image] = None
        self._sent_photos: set = set()
        self._datagrams = 0
        self._send_failures = 0
        self._pose_failures = 0

        from tf2_ros import Buffer, TransformListener

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            Odometry, str(self.get_parameter("drone_odometry_topic").value),
            lambda msg: self._on_odometry(msg, "_drone"), SENSOR_QOS,
        )
        self.create_subscription(
            Odometry, str(self.get_parameter("rover_odometry_topic").value),
            lambda msg: self._on_odometry(msg, "_rover"), SENSOR_QOS,
        )
        self.create_subscription(
            MissionStatus, str(self.get_parameter("mission_status_topic").value),
            self._on_status, LATCHED_QOS,
        )
        self.create_subscription(
            TargetArray, str(self.get_parameter("targets_topic").value),
            self._on_targets, LATCHED_QOS,
        )
        self.create_subscription(
            ObstacleArray, str(self.get_parameter("obstacles_topic").value),
            self._on_obstacles, LATCHED_QOS,
        )
        self.create_subscription(
            OccupancyGridMsg, str(self.get_parameter("occupancy_topic").value),
            self._on_grid, LATCHED_QOS,
        )
        self.create_subscription(
            Path, str(self.get_parameter("path_topic").value), self._on_path, LATCHED_QOS,
        )
        self.create_subscription(
            Image, str(self.get_parameter("rover_image_topic").value),
            self._on_rover_image, SENSOR_QOS,
        )

        self.create_timer(1.0 / max(station.publish_rate_hz, 1e-3), self._publish)
        self.get_logger().info(
            f"[TWIN] streaming to {station.host}:{station.port} at "
            f"{station.publish_rate_hz:.1f} Hz, map origin anchored at "
            f"({station.anchor_latitude:.6f}, {station.anchor_longitude:.6f})"
        )

    # -- inputs ------------------------------------------------------------
    def _on_odometry(self, msg: Odometry, slot: str) -> None:
        """Odometry, expressed in the planning frame before it is sent anywhere.

        Odometry is in the vehicle's ``odom`` frame, and reading it as a map
        coordinate is the mistake that once drove a correctly planned path to a
        point one whole spawn offset from the station. A console that plots the
        vehicle a spawn offset from where it is would be the same defect with a
        friendlier face, so this goes through TF like every other consumer.
        """
        pose, error = odometry_pose_in_frame(self.tf_buffer, msg, self.map_frame)
        if pose is None:
            self._pose_failures += 1
            self.get_logger().warn(
                f"[TWIN] cannot express odometry in {self.map_frame}: {error}",
                throttle_duration_sec=10.0,
            )
            return
        altitude_m = float(msg.pose.pose.position.z)
        map_from_odom, _ = lookup_transform(
            self.tf_buffer, self.map_frame, msg.header.frame_id, msg.header.stamp
        )
        if map_from_odom is not None:
            altitude_m = float(
                map_from_odom.apply(
                    np.array(
                        [
                            msg.pose.pose.position.x,
                            msg.pose.pose.position.y,
                            msg.pose.pose.position.z,
                        ]
                    )
                )[2]
            )
        linear = msg.twist.twist.linear
        setattr(
            self,
            slot,
            {
                "xy": (float(pose[0]), float(pose[1])),
                "altitude_m": altitude_m,
                "yaw_rad": float(pose[2]),
                "speed_mps": float(np.hypot(linear.x, linear.y)),
            },
        )

    def _on_status(self, msg: MissionStatus) -> None:
        try:
            self._state = MissionState(msg.state)
        except ValueError:
            self.get_logger().warn(f"[TWIN] unknown mission state {msg.state!r}")
            return
        if self._state is MissionState.MISSION_SUCCESS:
            self._status = "success"
        elif self._state is MissionState.MISSION_FAILED:
            self._status = "failed"
        else:
            self._status = "running"
        self._warning = "" if msg.failure_reason in ("", "NONE") else msg.failure_reason
        self._note = msg.failure_detail or msg.trace
        if msg.verified_qr and msg.verified_qr not in self._verified:
            self._verified.append(msg.verified_qr)

    def _on_targets(self, msg: TargetArray) -> None:
        self._targets = [
            _Record(
                qr_id=target.qr_id,
                position=np.array([target.position.x, target.position.y, target.position.z]),
                confidence=float(target.confidence),
                status="CONFIRMED" if target.status == 1 else "TENTATIVE",
                reached=target.qr_id in self._verified,
            )
            for target in msg.targets
        ]

    def _on_obstacles(self, msg: ObstacleArray) -> None:
        self._obstacles = [
            _Record(
                obstacle_id=obstacle.id,
                centre=np.array([obstacle.centre.x, obstacle.centre.y]),
                radius=0.5
                * float(np.hypot(obstacle.size.x, obstacle.size.y)),
            )
            for obstacle in msg.obstacles
        ]

    def _on_grid(self, msg: OccupancyGridMsg) -> None:
        if not self.send_voxels:
            return
        metadata = GridMetadata(
            resolution=msg.info.resolution,
            width=msg.info.width,
            height=msg.info.height,
            origin_x=msg.info.origin.position.x,
            origin_y=msg.info.origin.position.y,
        )
        data = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self._grid = OccupancyGrid(metadata, data)

    def _on_path(self, msg: Path) -> None:
        poses = path_msg_to_array(msg)
        self._path = poses[:, :2] if len(poses) else None

    def _on_rover_image(self, msg: Image) -> None:
        self._last_rover_image = msg

    # -- output ------------------------------------------------------------
    def _publish(self) -> None:
        timestamp_ms = int(node_time_seconds(self) * 1000.0)
        replanning = "replan" in (self._note or "").lower()

        # The drone carries the map deltas: it is the vehicle that discovered
        # them, and splitting them across both streams would only make the
        # station's ordering harder to reason about.
        if self._drone is not None:
            self._send(
                self.builder.build(
                    timestamp_ms=timestamp_ms,
                    vehicle=VEHICLE_UAV,
                    state=self._state,
                    replanning=replanning,
                    status=self._status,
                    warning=self._warning,
                    note=self._note,
                    targets=self._targets,
                    obstacles=self._obstacles,
                    grid=self._grid,
                    path_xy=self._path,
                    **self._drone,
                )
            )
        if self._rover is not None:
            self._send(
                self.builder.build(
                    timestamp_ms=timestamp_ms,
                    vehicle=VEHICLE_ROVER,
                    state=self._state,
                    replanning=replanning,
                    status=self._status,
                    warning=self._warning,
                    note=self._note,
                    **self._rover,
                )
            )
        self._send_pending_photo(timestamp_ms)

    def _send_pending_photo(self, timestamp_ms: int) -> None:
        """One rover close-up per verified station, for the QR gallery."""
        pending = [q for q in self._verified if q not in self._sent_photos]
        if not pending or self._last_rover_image is None:
            return
        payload = pending[0]
        try:
            import cv2
            from cv_bridge import CvBridge

            frame = CvBridge().imgmsg_to_cv2(self._last_rover_image, desired_encoding="bgr8")
            # Downscaled and JPEG-compressed on purpose: the gallery wants
            # evidence a referee can read, not a lossless frame, and the whole
            # thing has to survive one datagram.
            scale = 640.0 / max(frame.shape[1], 1)
            if scale < 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ok:
                return
        except Exception as exc:  # noqa: BLE001 - a console feed must not kill the node
            self.get_logger().warn(f"[TWIN] could not encode the QR photo: {exc}")
            self._sent_photos.add(payload)
            return

        self._sent_photos.add(payload)
        self._send(
            qr_photo_message(
                timestamp_ms=timestamp_ms,
                source_id=self.builder.source_id,
                target_id=payload,
                image_base64=base64.b64encode(buffer.tobytes()).decode("ascii"),
                auth_token=self.builder.auth_token,
            )
        )
        self.get_logger().info(f"[TWIN] sent the rover's {payload} photo to the gallery")

    def _send(self, message: Dict) -> None:
        for datagram in iter_json_datagrams(message, max_bytes=self.max_datagram_bytes):
            try:
                self._socket.sendto(datagram.encode("utf-8"), self.target)
                self._datagrams += 1
            except OSError as exc:
                self._send_failures += 1
                self.get_logger().warn(
                    f"[TWIN] datagram dropped: {exc} ({self._send_failures} total)",
                    throttle_duration_sec=10.0,
                )
                return

    def destroy_node(self) -> bool:
        self._socket.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TwinBridgeNode()
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
