#!/usr/bin/env python3
"""Probe what this machine's OpenCV build can actually decode.

The QR front end is the one part of the mission that depends heavily on the
OpenCV version: the detector was substantially rewritten across 4.6 -> 4.8, and
a build that cannot decode a small, distant code fails *silently* - it simply
reports no detections, which looks identical to "there was nothing to see".

This script renders the same imagery the mission uses and reports, per
strategy, whether the code came back. Run it on any target machine before
blaming the world file:

    python3 scripts/diagnose_qr_backend.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE = REPO_ROOT / "ros2_ws" / "src" / "mission_core"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(CORE / "test"))

from mission_core.camera import PinholeCamera  # noqa: E402
from mission_core.config import load_mission_config  # noqa: E402
from mission_core.geometry import R_BODY_TO_NADIR_OPTICAL  # noqa: E402
from mission_core.qr import render_qr_image  # noqa: E402

from sim_harness import Station, SyntheticWorld, camera_pose_from_body  # noqa: E402

CONFIG = REPO_ROOT / "ros2_ws" / "src" / "mission_bringup" / "config" / "mission.yaml"
PAYLOAD = "TARGET_2"


def strategies() -> List[Tuple[str, Callable[[np.ndarray], List[str]]]]:
    """Candidate decode paths, cheapest first."""
    detector = cv2.QRCodeDetector()

    def plain_multi(image: np.ndarray) -> List[str]:
        ok, decoded, _, _ = detector.detectAndDecodeMulti(image)
        return [d for d in decoded if d] if ok else []

    def plain_single(image: np.ndarray) -> List[str]:
        decoded, points, _ = detector.detectAndDecode(image)
        return [decoded] if decoded else []

    def upscaled(factor: int):
        def run(image: np.ndarray) -> List[str]:
            big = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
            ok, decoded, _, _ = detector.detectAndDecodeMulti(big)
            return [d for d in decoded if d] if ok else []
        return run

    def detect_then_crop(image: np.ndarray) -> List[str]:
        """Locate first, then decode a generously upscaled crop of each hit."""
        found, corners = detector.detectMulti(image)
        if not found or corners is None:
            return []
        results: List[str] = []
        for quad in corners:
            quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
            side = int(max(np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)))
            side = max(side * 4, 200)
            target = np.array(
                [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype=np.float32
            )
            warped = cv2.warpPerspective(
                image, cv2.getPerspectiveTransform(quad, target), (side, side)
            )
            payload, _, _ = detector.detectAndDecode(warped)
            if payload:
                results.append(payload)
        return results

    def as_gray(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    def gray_multi(image: np.ndarray) -> List[str]:
        ok, decoded, _, _ = detector.detectAndDecodeMulti(as_gray(image))
        return [d for d in decoded if d] if ok else []

    def otsu(image: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(
            as_gray(image), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary

    def otsu_multi(image: np.ndarray) -> List[str]:
        ok, decoded, _, _ = detector.detectAndDecodeMulti(otsu(image))
        return [d for d in decoded if d] if ok else []

    def otsu_upscaled(image: np.ndarray) -> List[str]:
        big = cv2.resize(otsu(image), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        ok, decoded, _, _ = detector.detectAndDecodeMulti(big)
        return [d for d in decoded if d] if ok else []

    def adaptive_multi(image: np.ndarray) -> List[str]:
        binary = cv2.adaptiveThreshold(
            as_gray(image), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
        )
        ok, decoded, _, _ = detector.detectAndDecodeMulti(binary)
        return [d for d in decoded if d] if ok else []

    def sharpened_multi(image: np.ndarray) -> List[str]:
        gray = as_gray(image)
        blurred = cv2.GaussianBlur(gray, (0, 0), 1.2)
        sharp = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)
        ok, decoded, _, _ = detector.detectAndDecodeMulti(sharp)
        return [d for d in decoded if d] if ok else []

    candidates: List[Tuple[str, Callable[[np.ndarray], List[str]]]] = [
        ("detectAndDecodeMulti", plain_multi),
        ("gray + multi", gray_multi),
        ("detectAndDecode", plain_single),
        ("upscale x2 + multi", upscaled(2)),
        ("upscale x3 + multi", upscaled(3)),
        ("otsu + multi", otsu_multi),
        ("otsu + upscale x2", otsu_upscaled),
        ("adaptive + multi", adaptive_multi),
        ("sharpen + multi", sharpened_multi),
        ("detectMulti + warped crop", detect_then_crop),
    ]

    aruco_cls = getattr(cv2, "QRCodeDetectorAruco", None)
    if aruco_cls is not None:
        aruco = aruco_cls()

        def aruco_multi(image: np.ndarray) -> List[str]:
            ok, decoded, _, _ = aruco.detectAndDecodeMulti(image)
            return [d for d in decoded if d] if ok else []

        candidates.append(("QRCodeDetectorAruco", aruco_multi))
    return candidates


def main() -> int:
    config = load_mission_config(CONFIG)
    camera = PinholeCamera.from_hfov(
        config.drone.camera.width, config.drone.camera.height,
        config.drone.camera.horizontal_fov_rad,
    )

    print(f"opencv                : {cv2.__version__}")
    print(f"numpy                 : {np.__version__}")
    print(f"QRCodeEncoder         : {hasattr(cv2, 'QRCodeEncoder')}")
    print(f"QRCodeDetectorAruco   : {hasattr(cv2, 'QRCodeDetectorAruco')}")
    print(f"scan altitude         : {config.drone.scan_altitude_m} m")
    print(f"code side             : {config.code_size_m(PAYLOAD):.4f} m")
    print()

    # 1 - the texture itself, at full resolution. If this fails, the encoder
    #     and decoder in this build do not even agree with each other.
    texture = render_qr_image(PAYLOAD, 512, config.mission.qr_quiet_zone_modules)
    ok, decoded, _, _ = cv2.QRCodeDetector().detectAndDecodeMulti(texture)
    print(f"[texture 512px] decoded = {list(decoded) if ok else 'NOTHING'}")
    print()

    # 2 - the code as the drone actually sees it, at a few altitudes.
    world = SyntheticWorld([Station(PAYLOAD, (0.0, 0.0))], [])
    altitudes = [config.drone.scan_altitude_m, 4.0, 3.0, 2.0]
    names = [name for name, _ in strategies()]

    # Keep the frames so a failing build's actual imagery can be inspected
    # rather than guessed at.
    dump_dir = Path("qr-diagnostic")
    dump_dir.mkdir(exist_ok=True)

    print(f"{'altitude':>9s} {'code px':>8s}  " + "  ".join(f"{n:>22s}" for n in names))
    print("-" * (20 + 24 * len(names)))
    any_success_at_scan_altitude = False
    for altitude in altitudes:
        pose = camera_pose_from_body(
            (0.0, 0.0, altitude), 0.0, (0.10, 0.0, -0.08), R_BODY_TO_NADIR_OPTICAL
        )
        frame = world.render(pose, camera)
        cv2.imwrite(str(dump_dir / f"frame_{altitude:.0f}m.png"), frame)
        code_px = camera.fx * config.code_size_m(PAYLOAD) / (altitude - 1.05)
        cells = []
        for name, run in strategies():
            try:
                hits = run(frame)
            except cv2.error as exc:
                cells.append(f"{'cv2.error':>22s}")
                continue
            mark = "OK" if PAYLOAD in hits else "-"
            cells.append(f"{mark:>22s}")
            if mark == "OK" and abs(altitude - config.drone.scan_altitude_m) < 1e-6:
                any_success_at_scan_altitude = True
        print(f"{altitude:9.1f} {code_px:8.0f}  " + "  ".join(cells))

    cv2.imwrite(str(dump_dir / "texture.png"), texture)
    print()
    print(f"wrote diagnostic frames to {dump_dir}/")
    if any_success_at_scan_altitude:
        print("At least one strategy decodes at the configured scan altitude.")
        return 0
    print("NO strategy decodes at the configured scan altitude on this OpenCV build.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
