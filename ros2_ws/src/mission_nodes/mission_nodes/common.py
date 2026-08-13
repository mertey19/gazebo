"""Shared plumbing for the mission ROS 2 nodes.

Keeps three concerns out of the individual nodes: turning the YAML mission
configuration into real ROS parameters, choosing the right QoS for each kind
of topic, and converting between ``mission_core`` types and ROS messages.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Point, PoseStamped, Quaternion, Vector3
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from nav_msgs.msg import Path as PathMsg
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from std_msgs.msg import Header

from mission_core.config import (
    MissionConfig,
    config_to_dict,
    load_mission_config,
    mission_config_from_dict,
)
from mission_core.geometry import Transform, quaternion_from_yaw
from mission_core.occupancy import GridMetadata, OccupancyGrid
from mission_core.planner import PlannedPath
from mission_core.world_model import ObstacleRecord, TargetRecord, TargetStatus

from mission_interfaces.msg import Obstacle as ObstacleMsg
from mission_interfaces.msg import TargetRecord as TargetRecordMsg

#: Sensor streams: lossy is fine, latency is not.
SENSOR_QOS = qos_profile_sensor_data

#: State that a late subscriber must still receive (map, path, mission status).
#: Transient-local means RViz or an operator tool started after the mission
#: still sees the current value instead of waiting for the next publish.
LATCHED_QOS = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

#: Ordinary reliable command/telemetry streams.
DEFAULT_QOS = QoSProfile(
    depth=10,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

_STATUS_TO_MSG = {
    TargetStatus.TENTATIVE: TargetRecordMsg.STATUS_TENTATIVE,
    TargetStatus.CONFIRMED: TargetRecordMsg.STATUS_CONFIRMED,
    TargetStatus.AMBIGUOUS: TargetRecordMsg.STATUS_AMBIGUOUS,
}
_MSG_TO_STATUS = {value: key for key, value in _STATUS_TO_MSG.items()}


# ---------------------------------------------------------------------------
# Configuration <-> ROS parameters
# ---------------------------------------------------------------------------

def _flatten(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        else:
            flat[name] = value
    return flat


def _unflatten(flat: Dict[str, Any]) -> Dict[str, Any]:
    nested: Dict[str, Any] = {}
    for name, value in flat.items():
        parts = name.split(".")
        cursor = nested
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return nested


def declare_mission_config(node: Node, default_config_file: str = "") -> MissionConfig:
    """Load the mission YAML and re-expose every value as a ROS parameter.

    The YAML is the source of truth for defaults; ROS parameters are the
    override mechanism.  Declaring all of them means ``ros2 param list`` shows
    the complete tunable surface of a node, and a launch argument can change
    any single value without editing the file.
    """
    node.declare_parameter("config_file", default_config_file)
    config_file = str(node.get_parameter("config_file").value)
    if not config_file:
        raise RuntimeError(
            "parameter 'config_file' is empty; point it at config/mission.yaml"
        )
    config = load_mission_config(config_file)
    # Say which file won. "Is this node running the overlay or the defaults?"
    # should be answerable from the log, not inferred from behaviour.
    node.get_logger().info(f"[CONFIG] loaded {config_file}")

    flat = _flatten(config_to_dict(config))
    for name, value in flat.items():
        if not node.has_parameter(name):
            node.declare_parameter(name, value)
    resolved = {name: node.get_parameter(name).value for name in flat}
    # Rebuilding through the dataclass re-runs validate(), so an override that
    # makes the configuration incoherent (an unreadable scan altitude, say)
    # fails at startup rather than mid-flight.
    merged = mission_config_from_dict(_unflatten(resolved)).require_valid()

    changed = {k: (flat[k], resolved[k]) for k in flat if flat[k] != resolved[k]}
    if changed:
        node.get_logger().info(
            "mission config overridden by ROS parameters: "
            + ", ".join(f"{k}={new} (file: {old})" for k, (old, new) in sorted(changed.items()))
        )
    return merged


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def stamp_to_seconds(stamp: TimeMsg) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def node_time_seconds(node: Node) -> float:
    return stamp_to_seconds(node.get_clock().now().to_msg())


def make_header(node: Node, frame_id: str, stamp: Optional[TimeMsg] = None) -> Header:
    header = Header()
    header.stamp = stamp if stamp is not None else node.get_clock().now().to_msg()
    header.frame_id = frame_id
    return header


# ---------------------------------------------------------------------------
# Geometry conversions
# ---------------------------------------------------------------------------

def transform_from_msg(transform_stamped) -> Transform:
    """``geometry_msgs/TransformStamped`` -> :class:`Transform`."""
    translation = transform_stamped.transform.translation
    rotation = transform_stamped.transform.rotation
    return Transform.from_quaternion(
        (translation.x, translation.y, translation.z),
        (rotation.x, rotation.y, rotation.z, rotation.w),
    )


def pose_to_xy_yaw(pose) -> np.ndarray:
    """``geometry_msgs/Pose`` -> ``(x, y, yaw)``."""
    q = pose.orientation
    yaw = math.atan2(
        2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )
    return np.array([pose.position.x, pose.position.y, yaw], dtype=float)


def quaternion_msg_from_yaw(yaw: float) -> Quaternion:
    x, y, z, w = quaternion_from_yaw(yaw)
    return Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))


def point_msg(values: Sequence[float]) -> Point:
    values = list(values) + [0.0] * (3 - len(values))
    return Point(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def vector_msg(values: Sequence[float]) -> Vector3:
    values = list(values) + [0.0] * (3 - len(values))
    return Vector3(x=float(values[0]), y=float(values[1]), z=float(values[2]))


# ---------------------------------------------------------------------------
# mission_core <-> mission_interfaces
# ---------------------------------------------------------------------------

def target_record_to_msg(record: TargetRecord) -> TargetRecordMsg:
    msg = TargetRecordMsg()
    msg.qr_id = record.qr_id
    msg.position = point_msg(record.position)
    msg.normal = vector_msg(record.normal)
    msg.status = _STATUS_TO_MSG[record.status]
    msg.observation_count = int(record.observation_count)
    msg.confidence = float(record.confidence)
    msg.position_spread_m = float(record.position_spread_m)
    msg.first_seen = seconds_to_stamp(record.first_seen)
    msg.last_seen = seconds_to_stamp(record.last_seen)
    return msg


def target_record_from_msg(msg: TargetRecordMsg) -> TargetRecord:
    """Rebuild a core record from a message.

    Used by the mission manager, which consumes the world model over topics
    rather than owning it.  ``weight_sum`` and the raw samples are not
    transported: the manager only reads the fused estimate, and confidence
    travels explicitly so the two nodes cannot disagree about it.
    """
    record = TargetRecord(
        qr_id=msg.qr_id,
        position=np.array([msg.position.x, msg.position.y, msg.position.z]),
        normal=np.array([msg.normal.x, msg.normal.y, msg.normal.z]),
        observation_count=int(msg.observation_count),
        first_seen=stamp_to_seconds(msg.first_seen),
        last_seen=stamp_to_seconds(msg.last_seen),
        status=_MSG_TO_STATUS.get(msg.status, TargetStatus.TENTATIVE),
        position_spread_m=float(msg.position_spread_m),
    )
    record.weight_sum = float(msg.observation_count)
    return record


def seconds_to_stamp(seconds: float) -> TimeMsg:
    seconds = max(0.0, float(seconds))
    sec = int(seconds)
    return TimeMsg(sec=sec, nanosec=int(round((seconds - sec) * 1e9)))


def obstacle_to_msg(record: ObstacleRecord) -> ObstacleMsg:
    msg = ObstacleMsg()
    msg.id = record.obstacle_id
    msg.centre = point_msg(record.centre)
    msg.size = vector_msg([record.size_xy[0], record.size_xy[1], record.height_m])
    msg.height_m = float(record.height_m)
    msg.cell_count = int(record.cell_count)
    return msg


def occupancy_to_msg(
    grid: OccupancyGrid, node: Node, frame_id: str
) -> OccupancyGridMsg:
    msg = OccupancyGridMsg()
    msg.header = make_header(node, frame_id)
    msg.info.resolution = float(grid.metadata.resolution)
    msg.info.width = int(grid.metadata.width)
    msg.info.height = int(grid.metadata.height)
    msg.info.origin.position.x = float(grid.metadata.origin_x)
    msg.info.origin.position.y = float(grid.metadata.origin_y)
    msg.info.origin.orientation.w = 1.0
    msg.data = grid.to_row_major_list()
    return msg


def occupancy_from_msg(msg: OccupancyGridMsg) -> OccupancyGrid:
    metadata = GridMetadata(
        resolution=float(msg.info.resolution),
        width=int(msg.info.width),
        height=int(msg.info.height),
        origin_x=float(msg.info.origin.position.x),
        origin_y=float(msg.info.origin.position.y),
    )
    data = np.asarray(msg.data, dtype=np.int8).reshape(metadata.height, metadata.width)
    return OccupancyGrid(metadata, data)


def planned_path_to_msg(path: PlannedPath, node: Node, frame_id: str) -> PathMsg:
    msg = PathMsg()
    msg.header = make_header(node, frame_id)
    for pose in path.poses:
        stamped = PoseStamped()
        stamped.header = msg.header
        stamped.pose.position = point_msg((pose[0], pose[1], 0.0))
        stamped.pose.orientation = quaternion_msg_from_yaw(float(pose[2]))
        msg.poses.append(stamped)
    return msg


def path_msg_to_array(msg: PathMsg) -> np.ndarray:
    """``nav_msgs/Path`` -> ``(N, 3)`` array of ``(x, y, yaw)``."""
    if not msg.poses:
        return np.zeros((0, 3))
    return np.asarray([pose_to_xy_yaw(p.pose) for p in msg.poses], dtype=float)


# ---------------------------------------------------------------------------
# TF helpers
# ---------------------------------------------------------------------------

def lookup_transform(
    buffer_,
    target_frame: str,
    source_frame: str,
    stamp: TimeMsg,
    timeout_s: float = 0.0,
    allow_latest_fallback: bool = True,
) -> Tuple[Optional[Transform], str]:
    """Look up ``target <- source`` at ``stamp``, falling back to the latest.

    Returns the error instead of raising so the caller can rate-limit its own
    logging: a missing transform during startup is normal, a persistent one is
    a fault, and only the caller knows which it is looking at.

    Two deliberate choices, both learned the hard way in Gazebo:

    * ``timeout_s`` defaults to zero. Blocking inside a callback of a
      single-threaded executor cannot work: the TF listener is a callback on
      the same executor, so nothing can arrive while the wait is in progress
      and the lookup is guaranteed to time out. It merely wastes the wait.
    * A sensor message is routinely stamped a few milliseconds ahead of the
      newest transform, which raises "extrapolation into the future". Dropping
      the data for that is far worse than using a transform 20 ms old: at rover
      and drone speeds that is under 3 cm. One run integrated *zero* camera
      sweeps for exactly this reason and produced a completely empty map.
    """
    try:
        stamped = buffer_.lookup_transform(
            target_frame, source_frame, stamp, timeout=Duration(seconds=timeout_s)
        )
        return transform_from_msg(stamped), ""
    except Exception as exc:  # tf2 raises several unrelated exception types
        first_error = exc

    if not allow_latest_fallback:
        return None, f"{target_frame} <- {source_frame}: {first_error}"
    try:
        import rclpy.time

        stamped = buffer_.lookup_transform(target_frame, source_frame, rclpy.time.Time())
    except Exception as exc:
        # Report the *stamped* failure: it is the more informative of the two,
        # and a fallback failure usually just means the frame does not exist.
        return None, f"{target_frame} <- {source_frame}: {first_error}"
    return transform_from_msg(stamped), ""


def odometry_pose_in_frame(
    buffer_, msg, target_frame: str
) -> Tuple[Optional[np.ndarray], str]:
    """``nav_msgs/Odometry`` -> ``(x, y, yaw)`` in ``target_frame``.

    Odometry is expressed in the vehicle's ``odom`` frame, which is **not** the
    planning frame: gz's DiffDrive dead-reckons from zero at spawn, so a rover
    spawned at (-8, -8) reports (0, 0) while standing still. Consuming that
    pose directly as a map coordinate offsets the entire mission by the spawn
    pose - the rover then drives a correctly planned path to entirely the wrong
    place, and nothing in the logs looks wrong.

    The lookup uses the latest available transform rather than the message
    stamp: ``map -> odom`` is static in this stack, so the result is exact and
    it cannot fail with an extrapolation error inside a control loop.
    """
    import rclpy.time

    source_frame = msg.header.frame_id
    if not source_frame:
        return None, "odometry message has an empty frame_id"
    pose = pose_to_xy_yaw(msg.pose.pose)
    if source_frame == target_frame:
        return pose, ""
    try:
        stamped = buffer_.lookup_transform(target_frame, source_frame, rclpy.time.Time())
    except Exception as exc:  # tf2 raises several unrelated exception types
        return None, f"{target_frame} <- {source_frame}: {exc}"

    transform = transform_from_msg(stamped)
    position = transform.apply(np.array([pose[0], pose[1], 0.0]))
    return np.array([position[0], position[1], pose[2] + transform.yaw]), ""


def odometry_position_in_frame(
    buffer_, msg, target_frame: str
) -> Tuple[Optional[np.ndarray], float, str]:
    """Same as :func:`odometry_pose_in_frame` but keeps the z coordinate.

    Used by the drone, whose altitude is the whole point of the pose.
    """
    import rclpy.time

    source_frame = msg.header.frame_id
    if not source_frame:
        return None, 0.0, "odometry message has an empty frame_id"
    p = msg.pose.pose.position
    position = np.array([p.x, p.y, p.z], dtype=float)
    yaw = float(pose_to_xy_yaw(msg.pose.pose)[2])
    if source_frame == target_frame:
        return position, yaw, ""
    try:
        stamped = buffer_.lookup_transform(target_frame, source_frame, rclpy.time.Time())
    except Exception as exc:
        return None, 0.0, f"{target_frame} <- {source_frame}: {exc}"

    transform = transform_from_msg(stamped)
    return transform.apply(position), yaw + transform.yaw, ""


__all__: List[str] = [
    "odometry_pose_in_frame",
    "odometry_position_in_frame",
    "DEFAULT_QOS",
    "LATCHED_QOS",
    "SENSOR_QOS",
    "declare_mission_config",
    "lookup_transform",
    "make_header",
    "node_time_seconds",
    "obstacle_to_msg",
    "occupancy_from_msg",
    "occupancy_to_msg",
    "path_msg_to_array",
    "planned_path_to_msg",
    "point_msg",
    "pose_to_xy_yaw",
    "quaternion_msg_from_yaw",
    "seconds_to_stamp",
    "stamp_to_seconds",
    "target_record_from_msg",
    "target_record_to_msg",
    "transform_from_msg",
    "vector_msg",
]
