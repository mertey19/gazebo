#!/usr/bin/env python3
"""Block until the mission reaches a terminal state, then exit with its verdict.

Turns "did the mission succeed?" into a process exit code, so a CI job or a
shell script can assert on a real Gazebo run instead of a human reading logs.

    ros2 run mission_bringup await_mission.py --timeout 1800

Exit codes:
    0  MISSION_SUCCESS
    1  MISSION_FAILED   (the failure reason is printed)
    2  timed out before reaching any terminal state
    3  no /mission/status message ever arrived
"""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from mission_interfaces.msg import MissionStatus

TERMINAL_SUCCESS = "MISSION_SUCCESS"
TERMINAL_FAILURE = "MISSION_FAILED"

# Must match the publisher in mission_manager_node, and transient-local also
# delivers the last state published before this process started.
STATUS_QOS = QoSProfile(
    depth=10,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class MissionAwaiter(Node):
    """Prints every mission state change and records the terminal one."""

    def __init__(self) -> None:
        super().__init__("await_mission")
        self.latest: MissionStatus | None = None
        self.terminal: MissionStatus | None = None
        self._last_state = ""
        self._started = time.monotonic()
        self.create_subscription(MissionStatus, "/mission/status", self._on_status, STATUS_QOS)

    def _on_status(self, msg: MissionStatus) -> None:
        self.latest = msg
        if msg.state != self._last_state:
            elapsed = time.monotonic() - self._started
            print(
                f"[{elapsed:7.1f}s] state={msg.state:<18s} target={msg.requested_qr}"
                + (f" verified={msg.verified_qr}" if msg.verified_qr else "")
                + (
                    f" reason={msg.failure_reason}"
                    if msg.failure_reason and msg.failure_reason != "NONE"
                    else ""
                ),
                flush=True,
            )
            self._last_state = msg.state
        if msg.state in (TERMINAL_SUCCESS, TERMINAL_FAILURE):
            self.terminal = msg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=1800.0, help="wall-clock seconds")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = MissionAwaiter()
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and node.terminal is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.5)
    finally:
        terminal, latest = node.terminal, node.latest
        node.destroy_node()
        rclpy.shutdown()

    print()
    if terminal is None:
        if latest is None:
            print("FAIL: no /mission/status message was ever received.")
            print("      The mission manager is not running, or QoS does not match.")
            return 3
        print(f"FAIL: timed out after {args.timeout:.0f} s in state {latest.state}")
        print(f"      trace: {latest.trace}")
        return 2

    print(f"state          : {terminal.state}")
    print(f"requested QR   : {terminal.requested_qr}")
    print(f"verified QR    : {terminal.verified_qr or '(none)'}")
    print(f"failure reason : {terminal.failure_reason}")
    print(f"path           : {terminal.path_poses} poses, {terminal.path_length_m:.2f} m")
    print(f"elapsed        : {terminal.elapsed_s:.1f} s of mission time")
    print(f"trace          : {terminal.trace}")

    if terminal.state == TERMINAL_SUCCESS:
        # The manager only reaches this state once every mandatory validator
        # check has passed, including the rover-side QR match.
        print("\nMISSION SUCCESS")
        return 0
    print(f"\nMISSION FAILED: {terminal.failure_reason} - {terminal.failure_detail}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
