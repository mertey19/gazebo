#!/usr/bin/env python3
"""Record a complete mission as an MP4, for people who would rather watch it.

Runs the *production* mission pipeline against the offline harness and draws,
every few simulated steps:

* a top-down view of the arena - the occupancy grid exactly as the drone has
  discovered it so far, the stations once their QR codes are decoded, the
  planned path, and both vehicles;
* the drone's and the rover's live camera images, which are the same pixels
  the QR detector is working on;
* the mission state machine and the last few log lines.

This is the offline harness, not Gazebo. Nothing here is mocked - the QR codes
are decoded by OpenCV from rendered imagery, the map comes from a ray-cast
camera, the route from A*, the driving from pure pursuit - but the physics and
the renderer are simplified. The Gazebo evidence lives in CI; use
`--gazebo-frames` to build a video out of frames captured from a real run.

    python scripts/record_mission_video.py --target TARGET_2 -o mission.mp4
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE = REPO_ROOT / "ros2_ws" / "src" / "mission_core"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(CORE / "test"))

from mission_core.config import load_mission_config  # noqa: E402
from mission_core.mission_state import MissionState  # noqa: E402
from mission_core.occupancy import OCCUPIED, UNKNOWN  # noqa: E402
from mission_core.world_model import TargetStatus  # noqa: E402

from offline_mission import OfflineMissionRunner, default_world  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "ros2_ws" / "src" / "mission_bringup" / "config" / "mission.yaml"

# Layout
MAP_PX = 760
PANEL_W = 620
HEIGHT = MAP_PX
WIDTH = MAP_PX + PANEL_W

# Colours are BGR.
BG = (26, 26, 30)
INK = (235, 235, 240)
MUTED = (150, 150, 160)
GRID_FREE = (58, 62, 58)
GRID_OCC = (60, 70, 190)
OBSTACLE_C = (80, 110, 235)
VISITED_C = (150, 150, 155)
ACCENT = (90, 200, 255)
DRONE_C = (60, 190, 255)
ROVER_C = (255, 190, 60)
PATH_C = (110, 240, 140)
OK_C = (110, 240, 140)
BAD_C = (90, 90, 245)


def put(img, text, org, scale=0.5, colour=INK, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


class MissionRecorder:
    """Draws one video frame per call from the runner's live state."""

    def __init__(self, runner: OfflineMissionRunner, every_n_steps: int = 4) -> None:
        self.runner = runner
        self.every_n_steps = max(1, every_n_steps)
        self.frames: List[np.ndarray] = []
        self.log: Deque[str] = deque(maxlen=7)
        self._steps = 0
        self._drone_trail: List[np.ndarray] = []
        self._rover_trail: List[np.ndarray] = []
        self._consumed = 0

        area_min = runner.config.mission.area_min_xy
        area_max = runner.config.mission.area_max_xy
        self._min = np.array(area_min, dtype=float)
        span = np.array(area_max, dtype=float) - self._min
        # Uniform scale so the arena is not distorted.
        self._scale = (MAP_PX - 40) / float(max(span))

    # -- coordinate helpers ------------------------------------------------
    def to_px(self, xy) -> Tuple[int, int]:
        """Map metres to pixels, with +y upwards as a person expects."""
        rel = (np.asarray(xy, dtype=float)[:2] - self._min) * self._scale
        return int(20 + rel[0]), int(MAP_PX - 20 - rel[1])

    # -- drawing -----------------------------------------------------------
    def _draw_map(self) -> np.ndarray:
        panel = np.full((MAP_PX, MAP_PX, 3), BG, np.uint8)
        runner = self.runner
        grid = runner.world_model.occupancy

        if grid is not None:
            # Occupancy, drawn as the drone has actually observed it.
            data = grid.data
            rgb = np.full((data.shape[0], data.shape[1], 3), BG, np.uint8)
            rgb[data == 0] = GRID_FREE
            rgb[data >= OCCUPIED] = GRID_OCC
            rgb = cv2.flip(rgb, 0)  # grid row 0 is the south edge
            side = int(round(grid.metadata.width * grid.metadata.resolution * self._scale))
            rgb = cv2.resize(rgb, (side, side), interpolation=cv2.INTER_NEAREST)
            x0, y0 = self.to_px((self._min[0], self._min[1] + grid.metadata.height *
                                 grid.metadata.resolution))
            panel[y0:y0 + side, x0:x0 + side] = rgb[
                : max(0, min(side, MAP_PX - y0)), : max(0, min(side, MAP_PX - x0))
            ]

        cv2.rectangle(panel, self.to_px(self._min),
                      self.to_px(self._min + np.array([22.0, 22.0])), (70, 70, 78), 1)

        # Planned path.
        if runner.follower.has_path:
            pts = np.array([self.to_px(p) for p in runner.follower._path[:, :2]], np.int32)
            cv2.polylines(panel, [pts], False, PATH_C, 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(panel, tuple(p), 4, PATH_C, -1, cv2.LINE_AA)

        # Trails.
        for trail, colour in ((self._drone_trail, DRONE_C), (self._rover_trail, ROVER_C)):
            if len(trail) > 1:
                pts = np.array([self.to_px(p) for p in trail], np.int32)
                cv2.polylines(panel, [pts], False, tuple(int(c * 0.45) for c in colour),
                              1, cv2.LINE_AA)

        # Mapped obstacles, outlined and named. The occupancy cells alone read
        # as scenery; a reviewer should be able to see at a glance that the
        # arena contains obstacles and that the route goes around them.
        stations = [record.position[:2] for record in runner.world_model.targets()]
        for obstacle in runner.world_model.obstacles():
            centre = np.asarray(obstacle.centre, dtype=float)
            half = np.asarray(obstacle.size_xy, dtype=float) / 2.0
            # An extracted blob sitting on a discovered station *is* that
            # station, not a separate obstacle.
            if any(float(np.linalg.norm(centre - s)) < 1.5 for s in stations):
                continue
            if max(obstacle.size_xy) < 1.2:
                continue  # the rover's own footprint, seen from above
            top_left = self.to_px(centre - half)
            bottom_right = self.to_px(centre + half)
            cv2.rectangle(panel, top_left, bottom_right, OBSTACLE_C, 2, cv2.LINE_AA)
            label_at = (min(top_left[0], bottom_right[0]) + 4,
                        min(top_left[1], bottom_right[1]) - 6)
            put(panel, "OBSTACLE", label_at, 0.40, OBSTACLE_C)

        # Discovered stations - only the ones perception has actually decoded.
        verified = set(runner.orchestrator.verified_qrs)
        target_now = runner.orchestrator.current_target
        for record in runner.world_model.targets():
            px = self.to_px(record.position)
            if record.qr_id in verified:
                colour, ring = VISITED_C, 2
            elif record.qr_id == target_now:
                colour, ring = ACCENT, 3
            elif record.status is TargetStatus.CONFIRMED:
                colour, ring = OK_C, 2
            else:
                colour, ring = (60, 200, 255), 2
            cv2.circle(panel, px, 11, colour, ring, cv2.LINE_AA)
            suffix = " [done]" if record.qr_id in verified else ""
            put(panel, record.qr_id + suffix, (px[0] + 15, px[1] + 5), 0.45, colour)
            if record.status is not TargetStatus.CONFIRMED:
                put(panel, f"{record.observation_count}/3", (px[0] + 15, px[1] + 20), 0.36, MUTED)

        dpad = runner.drone_home.home_xy
        dpx_home = self.to_px(dpad)
        cv2.drawMarker(panel, dpx_home, DRONE_C, cv2.MARKER_SQUARE, 14, 2)
        put(panel, "DRONE PAD", (dpx_home[0] + 12, dpx_home[1] + 18), 0.40, DRONE_C)

        home = runner.orchestrator.home_xy
        if home is not None:
            hpx = self.to_px(home)
            cv2.drawMarker(panel, hpx, PATH_C, cv2.MARKER_TILTED_CROSS, 16, 2)
            put(panel, "HOME", (hpx[0] + 12, hpx[1] - 8), 0.40, PATH_C)

        # Vehicles.
        drone = runner.drone.position
        dpx = self.to_px(drone)
        footprint = runner.config.camera_ground_footprint_m() * self._scale / 2.0
        cv2.rectangle(panel, (int(dpx[0] - footprint), int(dpx[1] - footprint)),
                      (int(dpx[0] + footprint), int(dpx[1] + footprint)),
                      tuple(int(c * 0.5) for c in DRONE_C), 1)
        cv2.circle(panel, dpx, 7, DRONE_C, -1, cv2.LINE_AA)
        put(panel, f"drone {drone[2]:.1f} m", (dpx[0] + 12, dpx[1] - 10), 0.42, DRONE_C)

        pose = runner.rover.pose
        rpx = self.to_px(pose[:2])
        nose = self.to_px(pose[:2] + 0.9 * np.array([np.cos(pose[2]), np.sin(pose[2])]))
        cv2.circle(panel, rpx, 7, ROVER_C, -1, cv2.LINE_AA)
        cv2.line(panel, rpx, nose, ROVER_C, 2, cv2.LINE_AA)
        put(panel, "rover", (rpx[0] + 12, rpx[1] + 18), 0.42, ROVER_C)

        put(panel, "map frame (top-down)  -  occupancy as discovered by the drone",
            (20, MAP_PX - 12), 0.42, MUTED)
        return panel

    def _draw_panel(self, now: float) -> np.ndarray:
        panel = np.full((HEIGHT, PANEL_W, 3), BG, np.uint8)
        runner = self.runner
        state = runner.orchestrator.state

        put(panel, "AUTONOMOUS UAV + UGV QR MISSION", (18, 30), 0.58, INK, 2)
        put(panel, f"requested target: {runner.requested_qr}", (18, 54), 0.46, ACCENT)

        colour = OK_C if state is MissionState.MISSION_SUCCESS else (
            BAD_C if state is MissionState.MISSION_FAILED else ACCENT)
        put(panel, f"{state.value}", (18, 84), 0.62, colour, 2)
        put(panel, f"t = {now:6.1f} s (simulated)", (18, 106), 0.42, MUTED)

        # Camera views: the actual pixels the detector sees.
        y = 124
        for label, image, width in (
            (
                f"drone camera ({math.degrees(runner.config.drone.camera_depression_rad):.0f} deg down)",
                runner._last_drone_frame,
                280,
            ),
            ("rover camera (forward)", runner._last_rover_frame, 280),
        ):
            put(panel, label, (18, y + 12), 0.42, MUTED)
            y += 20
            if image is not None:
                h = int(width * image.shape[0] / image.shape[1])
                panel[y:y + h, 18:18 + width] = cv2.resize(image, (width, h))
                cv2.rectangle(panel, (18, y), (18 + width, y + h), (70, 70, 78), 1)
                y += h + 14
            else:
                cv2.rectangle(panel, (18, y), (18 + width, y + 100), (50, 50, 56), -1)
                put(panel, "not streaming yet", (30, y + 55), 0.42, MUTED)
                y += 114

        orchestrator = runner.orchestrator
        if orchestrator.tour:
            legs = []
            for payload in orchestrator.tour:
                if payload in orchestrator.verified_qrs:
                    legs.append(payload + " OK")
                elif payload == orchestrator.current_target:
                    legs.append(">" + payload)
                else:
                    legs.append(payload)
            if runner.config.mission.return_home:
                legs.append("HOME")
            put(panel, "tour: " + "  ".join(legs), (18, y + 6), 0.40, ACCENT)
            y += 18
        if runner.drone_home.landed:
            drone_state = "landed at its take-off point"
        elif runner._returning_drone:
            drone_state = "returning home"
        elif runner._escorting:
            drone_state = "escorting the rover"
        else:
            drone_state = "scanning"
        put(panel, f"drone: {drone_state}"
                   f"   camera {math.degrees(runner.config.drone.camera_depression_rad):.0f} deg down",
            (18, y + 6), 0.40, DRONE_C)
        y += 18

        summary = runner.world_model.summary()
        put(panel, f"targets confirmed : {summary['targets_confirmed']}/3   "
                   f"obstacles: {summary['obstacles']}", (18, y + 6), 0.42, INK)
        put(panel, f"map observed      : {summary['map_known_fraction']:.0%}",
            (18, y + 24), 0.42, INK)
        if runner.orchestrator.path is not None:
            put(panel, f"path              : {len(runner.orchestrator.path)} poses, "
                       f"{runner.orchestrator.path.length_m:.1f} m", (18, y + 42), 0.42, PATH_C)
        y += 66

        for line in self.log:
            put(panel, line[:70], (18, y), 0.38, MUTED)
            y += 16
        return panel

    # -- hook --------------------------------------------------------------
    def __call__(self, runner: OfflineMissionRunner, now: float) -> None:
        self._steps += 1
        self._drone_trail.append(runner.drone.position[:2].copy())
        if self._steps % self.every_n_steps == 0:
            # Display only - the mission renders the rover camera solely during
            # verification, but an empty pane for two thirds of the video is
            # just confusing.
            runner._last_rover_frame = runner.world.render(
                runner._rover_camera_pose(), runner.rover_camera
            )
        self._rover_trail.append(runner.rover.pose[:2].copy())
        while len(runner.trace.messages) > len(self.log) + self._consumed:
            self.log.append(runner.trace.messages[self._consumed])
            self._consumed += 1
        if self._steps % self.every_n_steps:
            return
        frame = np.full((HEIGHT, WIDTH, 3), BG, np.uint8)
        frame[:, :MAP_PX] = self._draw_map()
        frame[:, MAP_PX:] = self._draw_panel(now)
        self.frames.append(frame)


def build_gazebo_card(directory: Path) -> Optional[np.ndarray]:
    """Closing card showing imagery from a real Gazebo run.

    The body of the video is the offline harness. This makes the distinction
    explicit rather than letting a viewer assume the whole thing is Gazebo.
    """
    drone = directory / "gz_drone.png"
    rover = directory / "gz_rover.png"
    if not drone.is_file() or not rover.is_file():
        print(f"no Gazebo frames in {directory}, skipping the closing card")
        return None

    card = np.full((HEIGHT, WIDTH, 3), BG, np.uint8)
    put(card, "AND THE SAME PIPELINE IN REAL GAZEBO HARMONIC", (40, 56), 0.85, OK_C, 2)
    put(card, "Ubuntu 24.04 + ROS 2 Jazzy + Gazebo Harmonic, run headless on GitHub Actions.",
        (40, 92), 0.52, INK)
    put(card, "These two images are actual gz-sim camera frames from that run.",
        (40, 118), 0.52, MUTED)

    bottom = 150
    for index, (path, label) in enumerate(((drone, "drone camera (nadir), 1280x960"),
                                           (rover, "rover camera (forward), 640x480"))):
        image = cv2.imread(str(path))
        if image is None:
            continue
        width = 620
        height = int(width * image.shape[0] / image.shape[1])
        x = 40 + index * (width + 40)
        cv2.rectangle(card, (x - 2, 148), (x + width + 2, 152 + height), (70, 70, 78), 1)
        card[150:150 + height, x:x + width] = cv2.resize(image, (width, height))
        put(card, label, (x, 150 + height + 22), 0.46, MUTED)
        bottom = max(bottom, 150 + height + 22)

    verdict = directory / "gz_mission.txt"
    y = bottom + 34
    if verdict.is_file():
        for line in verdict.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("["):
                continue
            put(card, line, (40, y), 0.5, OK_C if "SUCCESS" in line else INK)
            y += 22
            if y > HEIGHT - 20:
                break
    return card


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("-o", "--output", type=Path, default=Path("mission.mp4"))
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--every", type=int, default=6, help="steps per recorded frame")
    parser.add_argument("--hold-s", type=float, default=3.0, help="freeze on the verdict")
    parser.add_argument(
        "--gazebo-frames",
        type=Path,
        default=None,
        help="directory with real Gazebo camera frames to append as a closing card",
    )
    args = parser.parse_args()

    config = load_mission_config(args.config)
    runner = OfflineMissionRunner(
        default_world(), config, requested_qr=args.target,
        logger=lambda m: print(m, flush=True),
    )
    recorder = MissionRecorder(runner, every_n_steps=args.every)
    runner.on_step = recorder

    trace = runner.run()
    # Hold the final frame so the verdict is readable.
    if recorder.frames:
        recorder(runner, trace.sim_time_s)
        recorder.frames.extend([recorder.frames[-1]] * int(args.fps * args.hold_s))

    if args.gazebo_frames is not None:
        card = build_gazebo_card(args.gazebo_frames)
        if card is not None:
            recorder.frames.extend([card] * int(args.fps * 6.0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        print("could not open the video writer", file=sys.stderr)
        return 2
    for frame in recorder.frames:
        writer.write(frame)
    writer.release()

    print()
    print(f"wrote {args.output}  ({len(recorder.frames)} frames, "
          f"{len(recorder.frames) / args.fps:.1f} s at {args.fps} fps)")
    print(f"final state: {trace.state.value}   verified: {trace.verified_qr}")
    return 0 if trace.state is MissionState.MISSION_SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
