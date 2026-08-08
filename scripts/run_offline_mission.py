#!/usr/bin/env python3
"""Run a complete mission without Gazebo, and print the log and the verdict.

Drives the production ``mission_core`` pipeline against the synthetic arena in
``test/sim_harness.py``: real QR textures rendered through a real pinhole
camera model, decoded by OpenCV, posed by ``solvePnP``, mapped by a ray-cast
lidar, planned by A*, tracked by pure pursuit, and verified by the rover's own
camera.

Useful for reproducing the verification results on a machine with no ROS
installation, and for debugging mission logic without waiting for a simulator.

    python scripts/run_offline_mission.py
    python scripts/run_offline_mission.py --target TARGET_1 --save-frames out/
"""

from __future__ import annotations

import argparse
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
from mission_core.planner import path_intersects_obstacles  # noqa: E402

from offline_mission import OfflineMissionRunner, default_world  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "ros2_ws" / "src" / "mission_bringup" / "config" / "mission.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=None, help="QR payload to reach (default: from config)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dt", type=float, default=0.1, help="simulation timestep in seconds")
    parser.add_argument(
        "--save-frames",
        type=Path,
        default=None,
        help="write one drone frame and one rover frame here for inspection",
    )
    args = parser.parse_args()

    config = load_mission_config(args.config)
    world = default_world()
    started = time.perf_counter()

    runner = OfflineMissionRunner(
        world, config, requested_qr=args.target, dt=args.dt, logger=lambda m: print(m, flush=True)
    )
    trace = runner.run()
    wall_s = time.perf_counter() - started

    print()
    print("=" * 72)
    print(f"final state      : {trace.state.value}")
    print(f"failure reason   : {trace.failure_reason}")
    print(f"verified QR      : {trace.verified_qr}")
    print(f"simulated time   : {trace.sim_time_s:.1f} s   (wall clock {wall_s:.1f} s)")
    print(
        f"camera frames    : {trace.drone_frames} drone, {trace.rover_frames} rover, "
        f"{trace.bad_frames} unusable"
    )
    print(f"world model      : {trace.world_model.summary()}")

    if trace.path_xy is not None and len(trace.path_xy) >= 2:
        length = float(np.sum(np.linalg.norm(np.diff(trace.path_xy, axis=0), axis=1)))
        print(f"path             : {len(trace.path_xy)} poses, {length:.2f} m")
        print(f"                   {[tuple(np.round(p, 2)) for p in trace.path_xy]}")
        grid = trace.world_model.occupancy
        if grid is not None:
            collides = path_intersects_obstacles(
                trace.path_xy, grid, clearance_m=config.clearance_m
            )
            print(f"path collides    : {collides}")

    # Perception accuracy, scored against ground truth *after* the fact. This
    # is the only place ground truth is read, and nothing in the mission loop
    # ever sees it.
    print()
    print("perception error vs ground truth (evaluation only):")
    for payload, truth in sorted(world.ground_truth_station_xy().items()):
        record = trace.world_model.get_target(payload)
        if record is None:
            print(f"  {payload}: NOT DISCOVERED")
            continue
        error = float(np.linalg.norm(record.position[:2] - truth))
        print(
            f"  {payload}: estimated ({record.position[0]:+.3f}, {record.position[1]:+.3f}) "
            f"true ({truth[0]:+.1f}, {truth[1]:+.1f})  error {error:.3f} m  "
            f"[{record.status.value}, {record.observation_count} obs, "
            f"conf {record.confidence:.2f}]"
        )

    if trace.report is not None:
        print()
        print(trace.report.render())

    if args.save_frames is not None:
        _save_frames(runner, args.save_frames)

    print("=" * 72)
    return 0 if trace.state is MissionState.MISSION_SUCCESS else 1


def _save_frames(runner: OfflineMissionRunner, directory: Path) -> None:
    """Dump the last drone and rover viewpoints, for eyeballing the imagery."""
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    drone = runner.world.render(runner._drone_camera_pose(), runner.drone_camera)
    rover = runner.world.render(runner._rover_camera_pose(), runner.rover_camera)
    cv2.imwrite(str(directory / "drone_view.png"), drone)
    cv2.imwrite(str(directory / "rover_view.png"), rover)
    print(f"\nwrote drone_view.png and rover_view.png to {directory}")


if __name__ == "__main__":
    raise SystemExit(main())
