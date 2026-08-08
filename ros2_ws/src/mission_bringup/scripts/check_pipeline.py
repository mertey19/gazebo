#!/usr/bin/env python3
"""Assert that every stage of the live pipeline is actually producing data.

Walks the mission data path from the simulator outwards and reports which
stage is the first to go quiet.  That is the difference between "the mission
did not succeed" and "the drone camera never rendered a frame" - and it is the
question worth answering first on a machine where Gazebo has never been run.

    ros2 run mission_bringup check_pipeline.py --timeout 120

Exit code 0 only if every required stage produced data.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, Image, LaserScan, PointCloud2
from tf2_msgs.msg import TFMessage

from mission_interfaces.msg import (
    ExplorationStatus,
    QrObservation,
    TargetArray,
    WorldModelStatus,
)

LATCHED = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


@dataclass
class Stage:
    """One checkpoint on the data path."""

    name: str
    topic: str
    msg_type: Any
    qos: Any
    #: Stages the mission cannot run without. Non-required ones are reported
    #: but do not fail the check - they may legitimately be quiet early on.
    required: bool = True
    count: int = 0
    first_seen: Optional[float] = None
    detail: str = ""
    #: Optional predicate that must hold for the stage to count as healthy.
    validator: Optional[Any] = field(default=None)


def _describe_grid(msg: OccupancyGrid) -> str:
    known = sum(1 for v in msg.data if v >= 0)
    occupied = sum(1 for v in msg.data if v >= 100)
    total = max(len(msg.data), 1)
    return (
        f"{msg.info.width}x{msg.info.height} @ {msg.info.resolution:.2f} m, "
        f"{known / total:.0%} observed, {occupied} occupied cells"
    )


def build_stages() -> List[Stage]:
    return [
        Stage("drone camera", "/drone/camera/image", Image, qos_profile_sensor_data,
              validator=lambda m: (m.width > 0 and len(m.data) > 0, f"{m.width}x{m.height} {m.encoding}")),
        Stage("drone camera_info", "/drone/camera/camera_info", CameraInfo, qos_profile_sensor_data,
              validator=lambda m: (m.k[0] > 0.0, f"fx={m.k[0]:.1f} cx={m.k[2]:.1f}")),
        Stage("drone lidar", "/drone/scan/points", PointCloud2, qos_profile_sensor_data,
              validator=lambda m: (m.width * m.height > 0, f"{m.width * m.height} points, frame={m.header.frame_id}")),
        Stage("drone odometry", "/drone/odometry", Odometry, qos_profile_sensor_data,
              validator=lambda m: (True, f"z={m.pose.pose.position.z:+.2f} m")),
        Stage("rover camera", "/rover/camera/image", Image, qos_profile_sensor_data,
              validator=lambda m: (m.width > 0 and len(m.data) > 0, f"{m.width}x{m.height} {m.encoding}")),
        Stage("rover odometry", "/rover/odometry", Odometry, qos_profile_sensor_data,
              validator=lambda m: (True, f"x={m.pose.pose.position.x:+.2f} y={m.pose.pose.position.y:+.2f}")),
        Stage("rover lidar", "/rover/scan", LaserScan, qos_profile_sensor_data,
              validator=lambda m: (len(m.ranges) > 0, f"{len(m.ranges)} beams")),
        Stage("tf", "/tf", TFMessage, qos_profile_sensor_data,
              validator=lambda m: (len(m.transforms) > 0,
                                   ",".join(t.child_frame_id for t in m.transforms))),
        Stage("exploration", "/drone/exploration_status", ExplorationStatus, LATCHED,
              validator=lambda m: (True, f"phase={m.phase} alt={m.altitude_m:.2f} m "
                                         f"wp {m.waypoint_index}/{m.waypoint_count}")),
        # The decisive perception stage: a QR code decoded from a Gazebo frame.
        Stage("QR observations", "/perception/drone/qr_observations", QrObservation, 10,
              validator=lambda m: (bool(m.qr_id),
                                   f"{m.qr_id} at ({m.position_map.x:+.2f}, {m.position_map.y:+.2f}) "
                                   f"range {m.range_m:.2f} m err {m.reprojection_error_px:.2f} px")),
        Stage("occupancy grid", "/world_model/occupancy_grid", OccupancyGrid, LATCHED,
              validator=lambda m: (m.info.width > 0, _describe_grid(m))),
        Stage("world model", "/world_model/status", WorldModelStatus, LATCHED,
              validator=lambda m: (True, f"{m.targets_confirmed}/{m.targets_total} confirmed "
                                         f"{list(m.qr_ids)}, {m.obstacles} obstacles, "
                                         f"{m.map_known_fraction:.0%} mapped")),
        Stage("targets", "/world_model/targets", TargetArray, LATCHED, required=False,
              validator=lambda m: (True, f"{len(m.targets)} records")),
    ]


class PipelineChecker(Node):
    def __init__(self, stages: List[Stage]) -> None:
        super().__init__("check_pipeline")
        self.stages = stages
        self._started = time.monotonic()
        for stage in stages:
            self.create_subscription(
                stage.msg_type, stage.topic, self._make_callback(stage), stage.qos
            )

    def _make_callback(self, stage: Stage):
        def callback(msg) -> None:
            stage.count += 1
            if stage.first_seen is None:
                stage.first_seen = time.monotonic() - self._started
            if stage.validator is not None:
                try:
                    ok, detail = stage.validator(msg)
                except Exception as exc:  # a malformed message is a real finding
                    ok, detail = False, f"validator raised: {exc}"
                stage.detail = detail
                if not ok:
                    stage.detail = f"INVALID: {detail}"
        return callback

    @property
    def all_required_seen(self) -> bool:
        return all(s.count > 0 for s in self.stages if s.required)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=180.0)
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    checker = PipelineChecker(build_stages())
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and time.monotonic() < deadline and not checker.all_required_seen:
            rclpy.spin_once(checker, timeout_sec=0.5)
        # Keep listening briefly so the reported detail reflects a settled
        # pipeline rather than the very first message of each stage.
        settle = time.monotonic() + 5.0
        while rclpy.ok() and time.monotonic() < settle:
            rclpy.spin_once(checker, timeout_sec=0.2)
    finally:
        stages = checker.stages
        checker.destroy_node()
        rclpy.shutdown()

    print()
    print(f"{'stage':<20s} {'msgs':>6s} {'t0':>7s}  detail")
    print("-" * 100)
    failures: List[str] = []
    for stage in stages:
        first = f"{stage.first_seen:.1f}s" if stage.first_seen is not None else "-"
        mark = "ok " if stage.count else ("FAIL" if stage.required else "warn")
        print(f"[{mark}] {stage.name:<14s} {stage.count:>6d} {first:>7s}  {stage.detail}")
        if stage.required and stage.count == 0:
            failures.append(f"{stage.name} ({stage.topic}) produced no data")
        elif stage.detail.startswith("INVALID"):
            failures.append(f"{stage.name}: {stage.detail}")

    print()
    if failures:
        print("PIPELINE CHECK FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("PIPELINE CHECK PASSED: every required stage produced valid data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
