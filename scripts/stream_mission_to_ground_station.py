#!/usr/bin/env python3
"""Fly a whole mission and stream it to the Simurgh ground station, no ROS.

Runs the *production* mission pipeline against the offline harness and sends
the same ``DigitalTwinMessageV1`` messages ``twin_bridge_node`` sends on the
robot, to the station's UDP ingress.  The station cannot tell the difference:
it is the same builder, the same schema and the same delta stream.

What that buys is a ground-station demo on a machine with no ROS, no Gazebo and
no Ubuntu - which is the machine this repository is written on.

    python scripts/stream_mission_to_ground_station.py --host 127.0.0.1
    python scripts/stream_mission_to_ground_station.py --target TARGET_1 --speed 2

Open the Unity project, press Play, then start this. Targets and obstacles
appear on the map as the drone discovers them, and both vehicles draw their own
trails from the pose stream.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE = REPO_ROOT / "ros2_ws" / "src" / "mission_core"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(CORE / "test"))

import numpy as np  # noqa: E402

from mission_core.config import load_mission_config  # noqa: E402
from mission_core.mission_state import MissionState  # noqa: E402
from mission_core.twin_telemetry import (  # noqa: E402
    VEHICLE_ROVER,
    VEHICLE_UAV,
    GeoAnchor,
    TwinMessageBuilder,
    iter_json_datagrams,
)

from offline_mission import OfflineMissionRunner, default_world  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "ros2_ws" / "src" / "mission_bringup" / "config" / "mission.yaml"


class GroundStationStreamer:
    """Sends one message per vehicle every ``period_s`` of simulated time."""

    def __init__(self, runner: OfflineMissionRunner, args) -> None:
        self.runner = runner
        station = runner.config.ground_station
        self.builder = TwinMessageBuilder(
            anchor=GeoAnchor(
                latitude=args.anchor_lat if args.anchor_lat is not None else station.anchor_latitude,
                longitude=args.anchor_lon if args.anchor_lon is not None else station.anchor_longitude,
                altitude_m=station.anchor_altitude_m,
            ),
            source_id=station.source_id,
            auth_token=args.token or station.auth_token,
            voxel_batch=station.voxel_batch,
        )
        self.target = (args.host, args.port)
        self.max_bytes = station.max_datagram_bytes
        self.period_s = 1.0 / max(args.rate, 1e-3)
        self.speed = max(args.speed, 1e-3)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.datagrams = 0
        self.messages = 0
        self._next_at = 0.0
        self._wall_started = time.perf_counter()
        self._verbose = args.verbose

    def __call__(self, runner: OfflineMissionRunner, now: float) -> None:
        if now < self._next_at:
            return
        self._next_at = now + self.period_s
        # Play at (roughly) wall-clock speed, so the operator sees a mission
        # unfold rather than a whole flight arriving in one frame.
        target_wall = self._wall_started + now / self.speed
        delay = target_wall - time.perf_counter()
        if delay > 0:
            time.sleep(min(delay, 1.0))

        orchestrator = runner.orchestrator
        state = orchestrator.state
        verified = set(orchestrator.verified_qrs)
        targets = [
            _target_view(record, record.qr_id in verified)
            for record in runner.world_model.targets()
        ]
        replanning = any(
            "RECOVERY" in message for message in runner.trace.messages[-4:]
        )
        timestamp_ms = int((time.time()) * 1000.0)

        drone = runner.drone
        self._send(
            self.builder.build(
                timestamp_ms=timestamp_ms,
                vehicle=VEHICLE_UAV,
                xy=drone.position[:2],
                altitude_m=float(drone.position[2]),
                yaw_rad=float(drone.yaw),
                speed_mps=float(np.linalg.norm(getattr(drone, "velocity", np.zeros(3))[:2])),
                state=state,
                replanning=replanning,
                status=_status_for(state),
                warning=_warning_for(orchestrator),
                note=_note_for(runner),
                targets=targets,
                obstacles=runner.world_model.obstacles(),
                grid=runner.world_model.occupancy,
                path_xy=runner.trace.path_xy,
            )
        )
        pose = runner.rover.pose
        self._send(
            self.builder.build(
                timestamp_ms=timestamp_ms,
                vehicle=VEHICLE_ROVER,
                xy=pose[:2],
                altitude_m=0.0,
                yaw_rad=float(pose[2]),
                speed_mps=float(getattr(runner.follower, "last_linear_velocity", 0.0)),
                state=state,
                replanning=replanning,
                status=_status_for(state),
                warning=_warning_for(orchestrator),
                note=_note_for(runner),
            )
        )

    def _send(self, message) -> None:
        self.messages += 1
        for datagram in iter_json_datagrams(message, max_bytes=self.max_bytes):
            self.socket.sendto(datagram.encode("utf-8"), self.target)
            self.datagrams += 1
            if self._verbose:
                print(f"  -> {len(datagram):5d} B  {datagram[:110]}")

    def close(self) -> None:
        self.socket.close()


class _TargetView:
    def __init__(self, record, reached: bool) -> None:
        self.qr_id = record.qr_id
        self.position = record.position
        self.confidence = record.confidence
        self.status = record.status
        self.reached = reached


def _target_view(record, reached: bool) -> _TargetView:
    return _TargetView(record, reached)


def _status_for(state: MissionState) -> str:
    if state is MissionState.MISSION_SUCCESS:
        return "success"
    if state is MissionState.MISSION_FAILED:
        return "failed"
    return "running"


def _warning_for(orchestrator) -> str:
    reason = orchestrator.machine.failure_reason.value
    return "" if reason == "NONE" else reason


def _note_for(runner: OfflineMissionRunner) -> str:
    return runner.trace.messages[-1] if runner.trace.messages else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=None, help="QR payload to reach")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default=None, help="ground station address")
    parser.add_argument("--port", type=int, default=None, help="DigitalTwinUdpIngress port")
    parser.add_argument("--rate", type=float, default=5.0, help="messages per vehicle per second")
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="playback speed; 2 means twice real time, 0 means as fast as possible",
    )
    parser.add_argument("--anchor-lat", type=float, default=None)
    parser.add_argument("--anchor-lon", type=float, default=None)
    parser.add_argument("--token", default="", help="authToken, if the station checks one")
    parser.add_argument("--verbose", action="store_true", help="print every datagram")
    args = parser.parse_args()

    config = load_mission_config(args.config)
    args.host = args.host or config.ground_station.host
    args.port = args.port or config.ground_station.port
    if args.speed <= 0:
        args.speed = 1e6

    runner = OfflineMissionRunner(
        default_world(), config, requested_qr=args.target,
        logger=lambda m: print(m, flush=True),
    )
    streamer = GroundStationStreamer(runner, args)
    runner.on_step = streamer

    anchor = streamer.builder.anchor
    print(f"streaming to {args.host}:{args.port}, map origin at "
          f"({anchor.latitude:.6f}, {anchor.longitude:.6f})", flush=True)
    started = time.perf_counter()
    try:
        trace = runner.run()
    finally:
        streamer.close()

    print()
    print(f"final state : {trace.state.value}")
    print(f"sent        : {streamer.messages} messages, {streamer.datagrams} datagrams "
          f"in {time.perf_counter() - started:.1f} s")
    print(f"targets     : {[t.qr_id for t in runner.world_model.targets()]}")
    print(f"obstacles   : {len(runner.world_model.obstacles())}")
    return 0 if trace.state is MissionState.MISSION_SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
