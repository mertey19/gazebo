"""World model node - the digital twin.

Fuses QR observations into per-station estimates and lidar returns into a 2D
occupancy grid, then publishes both as ROS messages so every other node works
from the same picture of the world.

Everything published here was *observed*.  The node has no access to the
simulator's model list or ground-truth poses.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray

from mission_core.occupancy import GridMetadata, OccupancyMapper, planar_scan_hit_points
from mission_core.world_model import TargetObservation, TargetStatus, WorldModel

from mission_interfaces.msg import (
    ObstacleArray,
    QrObservation,
    TargetArray,
    WorldModelStatus,
)
from mission_interfaces.srv import GetTarget

from .common import (
    DEFAULT_QOS,
    LATCHED_QOS,
    SENSOR_QOS,
    declare_mission_config,
    lookup_transform,
    make_header,
    obstacle_to_msg,
    occupancy_to_msg,
    point_msg,
    stamp_to_seconds,
    target_record_to_msg,
)


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
        self.declare_parameter("point_cloud_topic", "/drone/scan/points")
        self.declare_parameter("laser_scan_topic", "/rover/scan")
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
            obstacle_min_height_m=self.config.world_model.obstacle_min_height_m,
            obstacle_max_height_m=self.config.world_model.obstacle_max_height_m,
            min_hits=self.config.world_model.mapper_min_hits,
            hit_ratio_threshold=self.config.world_model.mapper_hit_ratio,
        )

        from tf2_ros import Buffer, TransformListener

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

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
            PointCloud2,
            str(self.get_parameter("point_cloud_topic").value),
            self._on_point_cloud,
            SENSOR_QOS,
        )
        self.create_subscription(
            LaserScan,
            str(self.get_parameter("laser_scan_topic").value),
            self._on_laser_scan,
            SENSOR_QOS,
        )

        self.create_service(GetTarget, "/world_model/get_target", self._on_get_target)
        self.create_timer(
            1.0 / max(self.config.world_model.publish_rate_hz, 1e-3), self._publish
        )

        self._cloud_frame_failures = 0
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

    def _on_point_cloud(self, msg: PointCloud2) -> None:
        """Fold a lidar sweep into the occupancy grid, in the map frame."""
        map_from_sensor, error = lookup_transform(
            self.tf_buffer, self.map_frame, msg.header.frame_id, msg.header.stamp
        )
        if map_from_sensor is None:
            self._cloud_frame_failures += 1
            self.get_logger().warn(
                f"[WORLD_MODEL] dropping lidar sweep: {error}", throttle_duration_sec=5.0
            )
            return
        try:
            points = point_cloud2.read_points_numpy(
                msg, field_names=("x", "y", "z"), skip_nans=True
            )
        except Exception as exc:  # malformed cloud
            self.get_logger().warn(f"[WORLD_MODEL] unreadable point cloud: {exc}")
            return
        if points.size == 0:
            return
        # Rays that hit nothing come back at max range; those points are not
        # evidence of anything and would smear fake free space across the map.
        finite = np.isfinite(points).all(axis=1)
        self.mapper.integrate(map_from_sensor.apply(points[finite]))

    def _on_laser_scan(self, msg: LaserScan) -> None:
        """Add newly encountered rover-height obstacles to the shared map."""
        map_from_sensor, error = lookup_transform(
            self.tf_buffer, self.map_frame, msg.header.frame_id, msg.header.stamp
        )
        if map_from_sensor is None:
            self.get_logger().warn(
                f"[WORLD_MODEL] dropping rover lidar sweep: {error}",
                throttle_duration_sec=5.0,
            )
            return
        points = planar_scan_hit_points(
            msg.ranges,
            angle_min=msg.angle_min,
            angle_increment=msg.angle_increment,
            range_min=msg.range_min,
            range_max=msg.range_max,
        )
        if points.size:
            self.mapper.integrate_runtime_obstacle_hits(map_from_sensor.apply(points))

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
