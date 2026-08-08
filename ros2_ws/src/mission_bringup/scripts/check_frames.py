#!/usr/bin/env python3
"""Verify the TF tree and the map<-odom assumptions against live odometry.

The two ``map -> */odom`` transforms in ``mission.launch.py`` encode an
assumption about where each gz plugin puts its odometry origin:
``OdometryPublisher`` reports a world pose, ``DiffDrive`` dead-reckons from
zero at spawn.  If a future Gazebo version changes either, every position in
the mission shifts by a constant offset - which is exactly the class of bug
that is invisible in logs and obvious here.

This tool compares, for both vehicles, the pose TF reports in ``map`` against
the pose implied by odometry, and prints the residual.  Run it a few seconds
after launch, before the drone has moved:

    ros2 run mission_bringup check_frames.py
"""

from __future__ import annotations

import sys
from typing import Dict, Optional

import rclpy
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import Buffer, TransformListener

#: Spawn poses declared in worlds/mission_arena.sdf.
EXPECTED_SPAWN_XY: Dict[str, tuple] = {
    "drone": (-8.0, -6.5),
    "rover": (-8.0, -8.0),
}
BASE_FRAMES = {"drone": "drone/base_link", "rover": "rover/base_link"}
#: Any residual above this is a real frame error, not solver noise.
TOLERANCE_M = 0.25


class FrameChecker(Node):
    def __init__(self) -> None:
        super().__init__("check_frames")
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.odometry: Dict[str, Optional[Odometry]] = {"drone": None, "rover": None}
        for name in self.odometry:
            self.create_subscription(
                Odometry,
                f"/{name}/odometry",
                lambda msg, key=name: self.odometry.__setitem__(key, msg),
                qos_profile_sensor_data,
            )

    def report(self) -> int:
        failures = 0
        print(f"{'vehicle':8s} {'source':22s} {'x':>8s} {'y':>8s} {'z':>8s}")
        print("-" * 60)
        for name, expected in EXPECTED_SPAWN_XY.items():
            odom = self.odometry[name]
            if odom is None:
                print(f"{name:8s} NO ODOMETRY RECEIVED on /{name}/odometry")
                failures += 1
                continue
            p = odom.pose.pose.position
            print(f"{name:8s} {'odometry':22s} {p.x:8.3f} {p.y:8.3f} {p.z:8.3f}")

            try:
                tf = self.buffer.lookup_transform(
                    "map", BASE_FRAMES[name], rclpy.time.Time(), timeout=Duration(seconds=2.0)
                )
            except Exception as exc:
                print(f"{name:8s} TF map <- {BASE_FRAMES[name]} FAILED: {exc}")
                failures += 1
                continue
            t = tf.transform.translation
            print(f"{name:8s} {'tf (map <- base)':22s} {t.x:8.3f} {t.y:8.3f} {t.z:8.3f}")
            print(f"{name:8s} {'expected spawn':22s} {expected[0]:8.3f} {expected[1]:8.3f}")

            error = max(abs(t.x - expected[0]), abs(t.y - expected[1]))
            verdict = "OK" if error <= TOLERANCE_M else "MISMATCH"
            if error > TOLERANCE_M:
                failures += 1
                print(
                    f"{name:8s} -> {verdict}: TF places the vehicle {error:.3f} m from its spawn "
                    f"pose. Fix the map -> {name}/odom static transform in mission.launch.py."
                )
            else:
                print(f"{name:8s} -> {verdict} ({error:.3f} m residual)")
            print()
        return failures


def main() -> int:
    rclpy.init()
    node = FrameChecker()
    # Let TF and odometry accumulate before judging anything.
    end = node.get_clock().now() + Duration(seconds=5.0)
    while rclpy.ok() and node.get_clock().now() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    failures = node.report()
    node.destroy_node()
    rclpy.shutdown()
    if failures:
        print(f"{failures} frame check(s) FAILED")
    else:
        print("all frame checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
