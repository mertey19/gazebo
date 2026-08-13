"""World model node - the digital twin.

Fuses QR observations into per-station estimates and monocular ground
observations into a 2D occupancy grid, then publishes both as ROS messages so
every other node works from the same picture of the world.

Everything published here was *observed*, by a camera.  The node has no access
to the simulator's model list or ground-truth poses, and there is no range
sensor anywhere in the stack.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from mission_core.occupancy import GridMetadata, OccupancyMapper
from mission_core.world_model import TargetObservation, TargetStatus, WorldModel

from mission_interfaces.msg import (
    GroundObservation,
    ObstacleArray,
    QrObservation,
    TargetArray,
    WorldModelStatus,
)
from mission_interfaces.srv import GetTarget

from .common import (
    DEFAULT_QOS,
    LATCHED_QOS,
    declare_mission_config,
    make_header,
    obstacle_to_msg,
    occupancy_to_msg,
    point_msg,
    stamp_to_seconds,
    target_record_to_msg,
)


def _points_to_array(flat) -> np.ndarray:
    """A flat ``[x, y, z, ...]`` message array -> ``(N, 3)``."""
    points = np.asarray(flat, dtype=float)
    if points.size < 3:
        return np.zeros((0, 3), dtype=float)
    return points[: points.size - points.size % 3].reshape(-1, 3)


class WorldModelNode(Node):
    """Maintains and publishes the mission's internal world representation."""

    def __init__(self) -> None:
        super().__init__("world_model")
        self.config = declare_mission_config(self)

        # Drone observations only. The rover's camera exists to *verify* a
        # station at close range, and the mission manager consumes its
        # observations directly for that. Feeding them into the shared map
        # instead injects the rover's own localisation error into target
        # positions, which is how one station once became two clusters and
        # tripped DUPLICATE_QR after a perfectly good verification.
        self.declare_parameter(
            "observation_topics", ["/perception/drone/qr_observations"]
        )
        # The drone's frames build the shared map; the rover's are local
        # evidence that something has changed on a route already mapped.
        self.declare_parameter(
            "mapping_observation_topic", "/perception/drone/ground_observations"
        )
        self.declare_parameter(
            "runtime_observation_topic", "/perception/rover/ground_observations"
        )
        self.declare_parameter("publish_markers", True)

        frames = self.config.frames
        self.map_frame = frames.map_frame

        self.world_model = WorldModel(
            association_radius_m=self.config.world_model.association_radius_m,
            min_observations=self.config.world_model.min_observations,
            min_confidence=self.config.world_model.min_confidence,
        )
        self.mapper = OccupancyMapper(
            self._grid_metadata(),
            min_hits=self.config.world_model.mapper_min_hits,
            hit_ratio_threshold=self.config.world_model.mapper_hit_ratio,
        )

        self.targets_pub = self.create_publisher(TargetArray, "/world_model/targets", LATCHED_QOS)
        self.obstacles_pub = self.create_publisher(
            ObstacleArray, "/world_model/obstacles", LATCHED_QOS
        )
        self.grid_pub = self.create_publisher(
            OccupancyGridMsg, "/world_model/occupancy_grid", LATCHED_QOS
        )
        self.status_pub = self.create_publisher(
            WorldModelStatus, "/world_model/status", LATCHED_QOS
        )
        self.marker_pub = (
            self.create_publisher(MarkerArray, "/world_model/markers", LATCHED_QOS)
            if bool(self.get_parameter("publish_markers").value)
            else None
        )

        for topic in list(self.get_parameter("observation_topics").value):
            self.create_subscription(QrObservation, topic, self._on_observation, DEFAULT_QOS)
            self.get_logger().info(f"[WORLD_MODEL] consuming observations from {topic}")

        self.create_subscription(
            GroundObservation,
            str(self.get_parameter("mapping_observation_topic").value),
            self._on_mapping_observation,
            DEFAULT_QOS,
        )
        self.create_subscription(
            GroundObservation,
            str(self.get_parameter("runtime_observation_topic").value),
            self._on_runtime_observation,
            DEFAULT_QOS,
        )

        self.create_service(GetTarget, "/world_model/get_target", self._on_get_target)
        self.create_timer(
            1.0 / max(self.config.world_model.publish_rate_hz, 1e-3), self._publish
        )

        self._rejected_frames = 0
        self._last_summary = ""
        self.get_logger().info(
            f"[WORLD_MODEL] grid {self.mapper.metadata.width}x{self.mapper.metadata.height} "
            f"@ {self.mapper.metadata.resolution:.2f} m/cell, origin "
            f"({self.mapper.metadata.origin_x:.1f}, {self.mapper.metadata.origin_y:.1f})"
        )

    def _grid_metadata(self) -> GridMetadata:
        area_min = self.config.mission.area_min_xy
        area_max = self.config.mission.area_max_xy
        resolution = self.config.world_model.grid_resolution_m
        return GridMetadata(
            resolution=resolution,
            width=int(math.ceil((area_max[0] - area_min[0]) / resolution)),
            height=int(math.ceil((area_max[1] - area_min[1]) / resolution)),
            origin_x=float(area_min[0]),
            origin_y=float(area_min[1]),
        )

    # -- callbacks ---------------------------------------------------------
    def _on_observation(self, msg: QrObservation) -> None:
        if msg.map_frame_id and msg.map_frame_id != self.map_frame:
            self.get_logger().error(
                f"[WORLD_MODEL] rejecting {msg.qr_id} observation expressed in "
                f"{msg.map_frame_id!r}; this node plans in {self.map_frame!r}"
            )
            return
        try:
            record = self.world_model.add_observation(
                TargetObservation(
                    qr_id=msg.qr_id,
                    position_map=np.array(
                        [msg.position_map.x, msg.position_map.y, msg.position_map.z]
                    ),
                    normal_map=np.array(
                        [msg.normal_map.x, msg.normal_map.y, msg.normal_map.z]
                    ),
                    confidence=float(msg.confidence),
                    stamp=stamp_to_seconds(msg.header.stamp),
                    observer_position_map=np.zeros(3),
                    source=msg.source,
                )
            )
        except ValueError as exc:
            self.get_logger().warn(f"[WORLD_MODEL] rejected observation: {exc}")
            return

        if record.observation_count == 1:
            self.get_logger().info(
                f"[QR] {record.qr_id} first seen at "
                f"({record.position[0]:.2f}, {record.position[1]:.2f}) via {msg.source}"
            )
        elif record.observation_count == self.config.world_model.min_observations:
            self.get_logger().info(
                f"[WORLD_MODEL] {record.qr_id} CONFIRMED at "
                f"({record.position[0]:.2f}, {record.position[1]:.2f}) "
                f"confidence {record.confidence:.2f} spread {record.position_spread_m:.2f} m"
            )
        if record.status is TargetStatus.AMBIGUOUS:
            self.get_logger().error(
                f"[WORLD_MODEL] {record.qr_id} is AMBIGUOUS: the same payload was observed at "
                f"more than one location. Refusing to treat it as a usable target."
            )

    def _reject_foreign_frame(self, msg: GroundObservation) -> bool:
        """Guard against evidence expressed in something other than ``map``."""
        if msg.header.frame_id and msg.header.frame_id != self.map_frame:
            self.get_logger().error(
                f"[WORLD_MODEL] rejecting {msg.source} ground observation expressed in "
                f"{msg.header.frame_id!r}; this node maps in {self.map_frame!r}"
            )
            return True
        return False

    def _on_mapping_observation(self, msg: GroundObservation) -> None:
        """Fold one drone frame's floor geometry into the occupancy grid."""
        if self._reject_foreign_frame(msg):
            return
        if not msg.usable:
            self._rejected_frames += 1
            self.get_logger().warn(
                f"[WORLD_MODEL] {msg.source} reported an unusable frame "
                f"({msg.non_ground_fraction:.0%} not floor); {self._rejected_frames} total",
                throttle_duration_sec=5.0,
            )
            return
        self.mapper.integrate_ground_observation(
            _points_to_array(msg.contacts), _points_to_array(msg.free_space)
        )

    def _on_runtime_observation(self, msg: GroundObservation) -> None:
        """Add obstacles the rover meets on a route the drone already mapped.

        Only the contacts are taken. The rover's camera is 0.55 m off the
        ground and looks along it, so its free-space estimate is far weaker
        than the drone's - and the drone owns the map by design.
        """
        if self._reject_foreign_frame(msg) or not msg.usable:
            return
        contacts = _points_to_array(msg.contacts)
        if contacts.size:
            self.mapper.integrate_runtime_obstacle_hits(contacts)

    def _on_get_target(self, request, response):
        record = self.world_model.get_target(request.qr_id)
        if record is None:
            response.found = False
            response.usable = False
            response.message = f"{request.qr_id} has never been observed"
            return response
        response.found = True
        response.usable = bool(record.is_usable)
        response.target = target_record_to_msg(record)
        response.message = (
            "confirmed"
            if record.is_usable
            else f"status={record.status.value}, {record.observation_count} observation(s)"
        )
        return response

    # -- publishing --------------------------------------------------------
    def _publish(self) -> None:
        grid = self.mapper.build()
        obstacles = self.world_model.set_occupancy(grid, self.mapper.height_map)

        targets_msg = TargetArray()
        targets_msg.header = make_header(self, self.map_frame)
        targets_msg.targets = [target_record_to_msg(t) for t in self.world_model.targets()]
        self.targets_pub.publish(targets_msg)

        obstacles_msg = ObstacleArray()
        obstacles_msg.header = make_header(self, self.map_frame)
        obstacles_msg.obstacles = [obstacle_to_msg(o) for o in obstacles]
        self.obstacles_pub.publish(obstacles_msg)

        self.grid_pub.publish(occupancy_to_msg(grid, self, self.map_frame))

        summary = self.world_model.summary()
        status = WorldModelStatus()
        status.header = make_header(self, self.map_frame)
        status.targets_total = int(summary["targets_total"])
        status.targets_confirmed = int(summary["targets_confirmed"])
        status.targets_ambiguous = int(summary["targets_ambiguous"])
        status.obstacles = int(summary["obstacles"])
        status.rejected_observations = int(summary["rejected_observations"])
        status.map_known_fraction = float(grid.known_fraction)
        status.map_occupied_fraction = float(grid.occupied_fraction)
        status.qr_ids = list(summary["qr_ids"])
        self.status_pub.publish(status)

        # Log only on change: a 2 Hz unconditional log buries everything else.
        rendered = (
            f"{status.targets_confirmed}/{status.targets_total} confirmed "
            f"{status.qr_ids}, {status.obstacles} obstacles, "
            f"{status.map_known_fraction:.0%} mapped"
        )
        if rendered != self._last_summary:
            self._last_summary = rendered
            self.get_logger().info(f"[WORLD_MODEL] {rendered}")

        if self.marker_pub is not None:
            self.marker_pub.publish(self._build_markers())

    def _build_markers(self) -> MarkerArray:
        """RViz overlay. Debug only - no mission logic depends on it."""
        array = MarkerArray()
        for index, record in enumerate(self.world_model.targets()):
            marker = Marker()
            marker.header = make_header(self, self.map_frame)
            marker.ns = "targets"
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = point_msg(record.position)
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.6
            if record.status is TargetStatus.CONFIRMED:
                marker.color.r, marker.color.g, marker.color.b = 0.1, 0.9, 0.2
            elif record.status is TargetStatus.AMBIGUOUS:
                marker.color.r, marker.color.g, marker.color.b = 0.95, 0.1, 0.1
            else:
                marker.color.r, marker.color.g, marker.color.b = 0.95, 0.8, 0.1
            marker.color.a = 0.85
            array.markers.append(marker)

            label = Marker()
            label.header = marker.header
            label.ns = "target_labels"
            label.id = index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position = point_msg(
                (record.position[0], record.position[1], record.position[2] + 0.9)
            )
            label.pose.orientation.w = 1.0
            label.scale.z = 0.45
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = f"{record.qr_id} ({record.confidence:.2f})"
            array.markers.append(label)

        for index, obstacle in enumerate(self.world_model.obstacles()):
            marker = Marker()
            marker.header = make_header(self, self.map_frame)
            marker.ns = "obstacles"
            marker.id = index
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position = point_msg(
                (obstacle.centre[0], obstacle.centre[1], max(obstacle.height_m, 0.2) / 2.0)
            )
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(obstacle.size_xy[0])
            marker.scale.y = float(obstacle.size_xy[1])
            marker.scale.z = float(max(obstacle.height_m, 0.2))
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.4, 0.5, 0.9, 0.35
            array.markers.append(marker)
        return array


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WorldModelNode()
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
