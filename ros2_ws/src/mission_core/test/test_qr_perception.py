"""TEST 1 - the QR decoder and marker-pose pipeline.

Exercises the real ``cv2.QRCodeDetector`` against rendered imagery, then checks
that the recovered pose, transformed through the same TF chain the ROS node
uses, lands on the station.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mission_core.camera import PinholeCamera
from mission_core.errors import PerceptionError
from mission_core.geometry import (
    R_BODY_TO_FORWARD_OPTICAL,
    R_BODY_TO_NADIR_OPTICAL,
    Transform,
)
from mission_core.qr import (
    QrDetector,
    code_fraction_of_plate,
    code_side_length_m,
    expected_pixels_per_module,
    qr_module_count,
    render_qr_image,
    validate_frame,
)

from sim_harness import Station, SyntheticWorld, camera_pose_from_body

PAYLOADS = ["TARGET_1", "TARGET_2", "TARGET_3"]
DRONE_MOUNT = (0.10, 0.0, -0.08)
ROVER_MOUNT = (0.22, 0.0, 0.55)


# ---------------------------------------------------------------------------
# Encoder / decoder round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", PAYLOADS)
def test_rendered_qr_decodes_to_its_payload(payload: str) -> None:
    """A rendered plate texture must decode back to exactly its payload."""
    image = render_qr_image(payload, 512)
    ok, decoded, _, _ = cv2.QRCodeDetector().detectAndDecodeMulti(image)
    assert ok, f"{payload}: detector found no code in its own rendered texture"
    assert list(decoded) == [payload]


def test_payloads_produce_distinct_textures() -> None:
    """Two stations must never be visually identical."""
    images = {p: render_qr_image(p, 256) for p in PAYLOADS}
    for first in PAYLOADS:
        for second in PAYLOADS:
            if first < second:
                assert not np.array_equal(images[first], images[second]), (
                    f"{first} and {second} render to identical textures"
                )


def test_code_fraction_matches_module_geometry() -> None:
    """The plate/code size relation must follow the module counts exactly."""
    modules = qr_module_count("TARGET_1")
    quiet = 4
    assert code_fraction_of_plate("TARGET_1", quiet) == pytest.approx(
        modules / (modules + 2 * quiet)
    )
    assert code_side_length_m("TARGET_1", 0.80, quiet) == pytest.approx(
        0.80 * modules / (modules + 2 * quiet)
    )


# ---------------------------------------------------------------------------
# Frame validation
# ---------------------------------------------------------------------------

def test_validate_frame_rejects_unusable_input() -> None:
    with pytest.raises(PerceptionError, match="None"):
        validate_frame(None)
    with pytest.raises(PerceptionError, match="empty"):
        validate_frame(np.zeros((0, 0), dtype=np.uint8))
    with pytest.raises(PerceptionError, match="uniform"):
        validate_frame(np.full((64, 64), 128, dtype=np.uint8))
    with pytest.raises(PerceptionError, match="too small"):
        validate_frame(np.array([[0, 255], [255, 0]], dtype=np.uint8))


def test_validate_frame_accepts_colour_and_grayscale() -> None:
    colour = np.zeros((32, 32, 3), dtype=np.uint8)
    colour[:16] = 255
    assert validate_frame(colour).shape == (32, 32)
    assert validate_frame(colour[:, :, 0]).shape == (32, 32)


# ---------------------------------------------------------------------------
# Pose recovery through a rendered camera
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload,xy", [("TARGET_1", (6.0, 6.0)), ("TARGET_2", (7.0, -5.0))])
def test_nadir_detection_recovers_station_position(
    payload: str, xy, drone_camera: PinholeCamera, config
) -> None:
    """Decode + PnP + TF must place the station within a few centimetres."""
    world = SyntheticWorld([Station(payload, xy)], [])
    detector = QrDetector(
        config.mission.qr_plate_size_m,
        quiet_zone_modules=config.mission.qr_quiet_zone_modules,
        min_apparent_size_px=config.perception.min_apparent_size_px,
    )
    pose = camera_pose_from_body(
        (xy[0], xy[1], config.drone.scan_altitude_m), 0.0, DRONE_MOUNT, R_BODY_TO_NADIR_OPTICAL
    )
    detections = detector.detect(world.render(pose, drone_camera), drone_camera)

    assert len(detections) == 1, f"expected exactly one detection, got {len(detections)}"
    detection = detections[0]
    assert detection.payload == payload
    position_map = pose.apply(detection.position_optical)
    # The top plate sits at pedestal + cube height; only x/y feed the planner.
    assert position_map[0] == pytest.approx(xy[0], abs=0.10)
    assert position_map[1] == pytest.approx(xy[1], abs=0.10)
    assert detection.reprojection_error_px < config.perception.max_reprojection_error_px


def test_detection_survives_yaw_and_lateral_offset(drone_camera, config) -> None:
    """Perception must not depend on flying exactly over a station.

    Also covers the fronto-parallel case, where SOLVEPNP_IPPE_SQUARE alone
    returns NaN and the iterative fallback has to take over.
    """
    world = SyntheticWorld([Station("TARGET_1", (6.0, 6.0))], [])
    detector = QrDetector(config.mission.qr_plate_size_m, min_apparent_size_px=40.0)
    truth = np.array([6.0, 6.0])

    errors = []
    for yaw in (0.0, 0.4, np.pi / 2, np.pi, 4.0):
        for offset in ((0.0, 0.0), (1.0, 0.0), (0.0, -1.0), (-0.8, 0.8)):
            pose = camera_pose_from_body(
                (6.0 + offset[0], 6.0 + offset[1], 6.0), yaw, DRONE_MOUNT,
                R_BODY_TO_NADIR_OPTICAL,
            )
            detections = detector.detect(world.render(pose, drone_camera), drone_camera)
            assert detections, f"missed the station at yaw={yaw:.2f} offset={offset}"
            position = pose.apply(detections[0].position_optical)
            errors.append(float(np.linalg.norm(position[:2] - truth)))

    assert max(errors) < 0.20, f"worst position error {max(errors):.3f} m"
    assert float(np.mean(errors)) < 0.10


def test_pose_is_never_nan_for_fronto_parallel_marker(drone_camera, config) -> None:
    """Regression: a perfectly nadir view once produced a NaN pose silently."""
    world = SyntheticWorld([Station("TARGET_3", (0.0, 0.0))], [])
    detector = QrDetector(config.mission.qr_plate_size_m, min_apparent_size_px=40.0)
    pose = camera_pose_from_body((0.0, 0.0, 6.0), 0.0, (0.0, 0.0, 0.0), R_BODY_TO_NADIR_OPTICAL)
    detections = detector.detect(world.render(pose, drone_camera), drone_camera)
    assert detections, "fronto-parallel marker was dropped entirely"
    assert np.isfinite(detections[0].position_optical).all()
    assert np.isfinite(detections[0].reprojection_error_px)


def test_rover_camera_reads_a_station_side_face(rover_camera, config) -> None:
    """The verification camera must read the code from the planned standoff."""
    world = SyntheticWorld([Station("TARGET_2", (7.0, -5.0))], [])
    detector = QrDetector(config.mission.qr_plate_size_m, min_apparent_size_px=40.0)
    standoff = config.planner.approach_distance_m
    pose = camera_pose_from_body(
        (7.0 - standoff, -5.0, 0.0), 0.0, ROVER_MOUNT, R_BODY_TO_FORWARD_OPTICAL
    )
    detections = detector.detect(world.render(pose, rover_camera), rover_camera)

    assert detections, f"rover saw no code from {standoff:.2f} m"
    assert detections[0].payload == "TARGET_2"
    assert detections[0].range_m < config.verification.max_range_m


def test_multiple_stations_in_one_frame_are_all_decoded(drone_camera, config) -> None:
    """Two stations in view must not shadow each other."""
    world = SyntheticWorld(
        [Station("TARGET_1", (-1.4, 0.0)), Station("TARGET_2", (1.4, 0.0))], []
    )
    detector = QrDetector(config.mission.qr_plate_size_m, min_apparent_size_px=40.0)
    pose = camera_pose_from_body((0.0, 0.0, 5.0), 0.0, (0.0, 0.0, 0.0), R_BODY_TO_NADIR_OPTICAL)
    payloads = {d.payload for d in detector.detect(world.render(pose, drone_camera), drone_camera)}
    assert payloads == {"TARGET_1", "TARGET_2"}


def test_altitude_budget_matches_configuration(drone_camera, config) -> None:
    """The configured scan altitude must leave decoding headroom."""
    for payload in config.mission.known_payloads:
        ppm = expected_pixels_per_module(
            drone_camera, config.code_size_m(payload), config.drone.scan_altitude_m, payload
        )
        assert ppm >= config.perception.min_pixels_per_module, (
            f"{payload}: only {ppm:.2f} px/module at "
            f"{config.drone.scan_altitude_m:.1f} m"
        )


def test_rendered_faces_are_not_mirrored() -> None:
    """Every textured face must use a right-handed basis with its normal.

    Regression: the harness derived texture-right from the inward view
    direction, which mirrors every face.  A mirrored QR code has its third
    finder pattern in the bottom-*right* corner and is unreadable by OpenCV
    4.6 (what Ubuntu 24.04 and ROS Jazzy ship), while OpenCV >= 4.7 decodes it
    happily - so the defect was invisible locally and fatal on the target.

    Asserted geometrically rather than by decoding, because a decoder that
    tolerates mirroring cannot detect mirroring.
    """
    from sim_harness import OrientedBox

    box = OrientedBox(np.zeros(3), np.array([1.0, 1.0, 1.0]))
    for quad in box.faces(None, (0, 0, 0)):
        right = quad.corners[1] - quad.corners[0]
        up = quad.corners[0] - quad.corners[3]
        right /= np.linalg.norm(right)
        up /= np.linalg.norm(up)
        assert np.allclose(np.cross(right, up), quad.normal, atol=1e-9), (
            f"face with normal {quad.normal} is mirrored: "
            f"right x up = {np.cross(right, up)}"
        )


def test_rendered_code_has_canonical_finder_layout(drone_camera, config) -> None:
    """A rendered QR must have finder patterns at TL, TR and BL - not BR.

    Rectifies the plate straight out of the rendered frame using the *known*
    projected geometry (no decoder involved) and then checks where the three
    7x7 finder patterns actually are.
    """
    world = SyntheticWorld([Station("TARGET_2", (0.0, 0.0))], [])
    pose = camera_pose_from_body((0.0, 0.0, 3.0), 0.0, (0.0, 0.0, 0.0), R_BODY_TO_NADIR_OPTICAL)
    frame = cv2.cvtColor(world.render(pose, drone_camera), cv2.COLOR_BGR2GRAY)

    # Take the top face's corners straight from the harness - they are already
    # in texture order TL, TR, BR, BL - so this test checks what is actually
    # rendered rather than re-stating the convention it is meant to police.
    station = world.stations[0]
    top_face = max(station.cube.faces(None, (0, 0, 0)), key=lambda q: q.normal[2])
    assert top_face.normal[2] > 0.99, "expected an upward-facing top plate"
    pixels = drone_camera.project(pose.inverse().apply(top_face.corners)).astype(np.float32)

    size = 210  # 21 modules x 10 px, plus the quiet zone handled below
    modules = qr_module_count("TARGET_2")
    quiet = config.mission.qr_quiet_zone_modules
    total = modules + 2 * quiet
    plate_px = int(size * total / modules)
    rectified = cv2.warpPerspective(
        frame,
        cv2.getPerspectiveTransform(
            pixels,
            np.array(
                [[0, 0], [plate_px - 1, 0], [plate_px - 1, plate_px - 1], [0, plate_px - 1]],
                dtype=np.float32,
            ),
        ),
        (plate_px, plate_px),
    )
    module_px = plate_px / total

    def is_finder(corner: str) -> bool:
        """A finder pattern is a dark 7x7 block with a light 5x5 ring inside."""
        row = quiet if corner[0] == "t" else total - quiet - 7
        col = quiet if corner[1] == "l" else total - quiet - 7
        def module(r: int, c: int) -> bool:
            y = int((row + r + 0.5) * module_px)
            x = int((col + c + 0.5) * module_px)
            return rectified[y, x] < 128  # dark
        outer = all(module(0, c) for c in range(7)) and all(module(6, c) for c in range(7))
        ring = not module(1, 1) and not module(1, 5) and not module(5, 1)
        core = module(3, 3)
        return outer and ring and core

    assert is_finder("tl"), "no finder pattern at the top-left"
    assert is_finder("tr"), "no finder pattern at the top-right"
    assert is_finder("bl"), "no finder pattern at the bottom-left - the code is MIRRORED"
    assert not is_finder("br"), "a finder pattern at the bottom-right means a mirrored code"


def test_transform_round_trip_is_exact() -> None:
    """A pose composed and inverted must return the original point."""
    pose = camera_pose_from_body((3.0, -2.0, 5.0), 0.9, DRONE_MOUNT, R_BODY_TO_NADIR_OPTICAL)
    point_map = np.array([1.0, 2.0, 0.5])
    point_optical = pose.inverse().apply(point_map)
    assert np.allclose(pose.apply(point_optical), point_map, atol=1e-12)
    assert isinstance(pose, Transform)
