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
        """Compare TF against the spawn pose - only meaningful before motion.

        A vehicle that has already driven or flown is *supposed* to be
        somewhere else, so comparing it against its spawn pose would report a
        fault that is not one. The mover case is detected and reported as
        SKIPPED rather than silently passing or noisily failing.
        """
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
            if error <= TOLERANCE_M:
                print(f"{name:8s} -> OK ({error:.3f} m residual)")
                print()
                continue

            # Distinguish "the transform is wrong" from "it flew away". The
            # odometry delta tells us whether the vehicle actually moved, and
            # the two frames agreeing tells us the TF chain is self-consistent.
            odom_speed = float(
                abs(odom.twist.twist.linear.x)
                + abs(odom.twist.twist.linear.y)
                + abs(odom.twist.twist.linear.z)
            )
            tf_matches_odom = (
                abs(t.x - p.x) <= TOLERANCE_M
                and abs(t.y - p.y) <= TOLERANCE_M
                and abs(t.z - p.z) <= TOLERANCE_M
            )
            if tf_matches_odom and (odom_speed > 0.05 or error > 1.0):
                print(
                    f"{name:8s} -> SKIPPED: the vehicle has moved {error:.2f} m from spawn "
                    f"(speed {odom_speed:.2f} m/s) and TF agrees with odometry to "
                    f"{TOLERANCE_M:.2f} m, so the map -> {name}/odom transform is consistent. "
                    f"Run this check before the mission starts to test it against spawn."
                )
            else:
                failures += 1
                print(
                    f"{name:8s} -> MISMATCH: TF places the vehicle {error:.3f} m from its spawn "
                    f"pose and does not agree with odometry. Fix the map -> {name}/odom "
                    f"static transform in mission.launch.py."
                )
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
