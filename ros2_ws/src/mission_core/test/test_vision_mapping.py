"""TEST 7 - monocular obstacle mapping, the mission's only ranging sensor.

No vehicle carries a lidar or a depth camera, so every obstacle position comes
out of an RGB frame plus the known ground plane.  These tests hold that
construction to the standard the removed sensor met: the geometry has to be
*measured*, not approximated, and the failure modes a camera has and a range
sensor does not - shadows, sky, a view with no floor in it - must not turn into
obstacles.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mission_core.camera import PinholeCamera
from mission_core.geometry import (
    R_BODY_TO_FORWARD_OPTICAL,
    Transform,
    optical_from_depression,
)
from mission_core.vision_mapping import (
    MonocularObstacleDetector,
    below_horizon_mask,
    segment_non_ground,
)

from sim_harness import GROUND_BGR, OrientedBox, SyntheticWorld, camera_pose_from_body

DEPRESSION = 0.5235988  # 30 degrees, the shipped drone camera pitch


@pytest.fixture(scope="module")
def detector() -> MonocularObstacleDetector:
    return MonocularObstacleDetector(max_range_m=12.0, min_blob_area_px=200.0)


def drone_pose(position, yaw: float = 0.0) -> Transform:
    return camera_pose_from_body(
        np.asarray(position, dtype=float), yaw, (0.0, 0.0, 0.0),
        optical_from_depression(DEPRESSION),
    )


def flat_floor(camera: PinholeCamera) -> np.ndarray:
    """An image of nothing but floor, with the harness's own floor colour."""
    frame = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    frame[:, :] = GROUND_BGR
    # Real floor is never perfectly uniform; give the robust statistics
    # something to measure a spread from.
    rng = np.random.default_rng(7)
    noise = rng.integers(-6, 7, size=frame.shape, dtype=np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# The geometry: a pixel plus a plane is a position
# ---------------------------------------------------------------------------

def test_contact_and_height_are_measured_not_approximated(drone_camera) -> None:
    """A box of known size, placed by hand, must come back with its own numbers.

    The mask is synthesised from the exact projection of a 1.5 m wall standing
    5 m in front of the camera, so any error here is the mapper's own and not
    the renderer's.
    """
    camera = drone_camera
    pose = drone_pose((0.0, 0.0, 4.0))
    camera_from_world = pose.inverse()

    wall_distance, wall_height = 5.0, 1.5
    base = camera_from_world.apply(np.array([wall_distance, 0.0, 0.0]))
    top = camera_from_world.apply(np.array([wall_distance, 0.0, wall_height]))
    base_row = int(round(camera.project(base)[0][1]))
    top_row = int(round(camera.project(top)[0][1]))

    mask = np.zeros((camera.height, camera.width), dtype=bool)
    mask[top_row:base_row, 400:880] = True

    observation = detector_for(camera).project(mask, camera, pose)
    assert observation.contact_count > 0
    contacts = observation.contacts
    # The centre column looks straight along +x; its contact is the one whose
    # geometry is exactly the hand-computed case.
    centre = contacts[np.argmin(np.abs(contacts[:, 1]))]
    assert centre[0] == pytest.approx(wall_distance, abs=0.05)
    assert centre[2] == pytest.approx(wall_height, abs=0.10)


def detector_for(camera: PinholeCamera) -> MonocularObstacleDetector:
    return MonocularObstacleDetector(max_range_m=12.0, column_stride=8, free_stride=16)


def test_pixels_above_the_horizon_can_never_be_floor(rover_camera) -> None:
    """The rover camera looks along the ground, so half its frame is sky."""
    pose = camera_pose_from_body((0.0, 0.0, 0.0), 0.0, (0.22, 0.0, 0.55), R_BODY_TO_FORWARD_OPTICAL)
    mask = below_horizon_mask(rover_camera, pose)
    assert mask.any() and not mask.all(), "a level camera sees both floor and sky"
    # The split is a horizontal line through the principal point for a camera
    # with no roll and no pitch.
    assert not mask[: rover_camera.height // 2 - 2].any()
    assert mask[rover_camera.height // 2 + 2 :].all()

    # A camera below the plane it is supposed to be looking at has no valid
    # pixels at all, rather than a half-plane of nonsense.
    underground = Transform(np.array([0.0, 0.0, -1.0]), pose.rotation)
    assert not below_horizon_mask(rover_camera, underground).any()


def test_sky_is_not_an_obstacle_standing_at_the_horizon(rover_camera) -> None:
    """The lower edge of the sky is the horizon, which is not a contact point.

    Segmenting the whole frame would also let the sky vote in the floor-colour
    statistics, and a blue sky against grey ground moves that median a long
    way.
    """
    frame = flat_floor(rover_camera)
    frame[: rover_camera.height // 2] = (200, 130, 60)  # sky blue in BGR
    pose = camera_pose_from_body((0.0, 0.0, 0.0), 0.0, (0.22, 0.0, 0.55), R_BODY_TO_FORWARD_OPTICAL)

    detector = MonocularObstacleDetector(max_range_m=6.0)
    observation = detector.process(frame, rover_camera, pose)
    assert observation.usable
    assert observation.contact_count == 0, "the horizon was mapped as a wall"
    assert observation.free_count > 0, "the floor below the horizon was not mapped"


def test_a_shadow_is_not_a_wall(drone_camera) -> None:
    """Shadows lose lightness but keep the floor's hue; obstacles change hue.

    A brightness-based segmenter maps every shadow as an obstacle, and in an
    arena lit by one sun that means a permanent phantom beside every real
    object.
    """
    frame = flat_floor(drone_camera)
    shadow = (np.asarray(GROUND_BGR) * 0.45).astype(np.uint8)
    frame[500:800, 300:900] = shadow
    mask, fraction, _ = segment_non_ground(frame)
    assert fraction < 0.01, f"{fraction:.0%} of a shadowed floor was called an obstacle"
    assert not mask[600, 600]


def test_a_white_plate_is_found_although_it_has_no_hue(drone_camera) -> None:
    """Chroma cannot see an achromatic object; the one-sided luma test can."""
    frame = flat_floor(drone_camera)
    frame[300:600, 500:800] = (250, 250, 250)
    mask, fraction, _ = segment_non_ground(frame)
    assert fraction > 0.05
    assert mask[450, 650]


def test_the_first_frame_must_be_mostly_floor_or_it_is_refused(drone_camera) -> None:
    """With no reference yet, the floor has to be the majority of the view.

    A first frame that is half obstacle could define either surface as the
    background, and nothing in the image says which. Rather than pick, the
    detector refuses the frame and keeps looking for one it can trust.
    """
    frame = flat_floor(drone_camera)
    frame[400:, :] = (40, 40, 200)  # 58% of the frame, and no floor model yet
    detector = MonocularObstacleDetector()
    observation = detector.process(frame, drone_camera, drone_pose((0.0, 0.0, 4.0)))
    assert not observation.usable
    assert observation.contact_count == 0 and observation.free_count == 0
    assert detector.floor_model is None, "a refused frame must not become the reference"


def test_a_close_obstacle_does_not_redefine_the_floor(drone_camera) -> None:
    """Once the floor is known, a wall filling the view is a wall.

    This is the case a per-frame colour model gets exactly backwards: the wall
    becomes the majority, so it becomes "background", and the strip of real
    floor beside it is mapped as the obstacle - at the moment the vehicle is
    closest to something solid.
    """
    detector = MonocularObstacleDetector(max_range_m=12.0)
    pose = drone_pose((0.0, 0.0, 4.0))
    clear = detector.process(flat_floor(drone_camera), drone_camera, pose)
    assert clear.usable and clear.contact_count == 0
    assert detector.floor_model is not None

    blocked = flat_floor(drone_camera)
    blocked[300:, :] = (40, 40, 200)  # a wall across three quarters of the view
    observation = detector.process(blocked, drone_camera, pose)
    assert observation.usable, "a known floor plus a big obstacle is a usable frame"
    assert observation.contact_count > 0, "the wall was not detected at all"
    # Free samples may only come from the floor strip above the wall, never
    # from the wall itself.
    assert observation.free_count < clear.free_count / 2.0


def test_the_forward_range_only_reports_the_wedge_ahead(drone_camera) -> None:
    """An obstacle beside the vehicle is not on a collision course."""
    camera = drone_camera
    pose = drone_pose((0.0, 0.0, 4.0))
    detector = MonocularObstacleDetector(max_range_m=12.0, forward_half_angle_rad=0.35)

    ahead = np.array([[6.0, 0.0, 1.0]])
    beside = np.array([[6.0, 6.0, 1.0]])
    assert detector._nearest_forward(ahead, pose, np.array([6.0])) == pytest.approx(6.0)
    assert not math.isfinite(detector._nearest_forward(beside, pose, np.array([8.5])))


def test_another_vehicle_is_filtered_out_of_the_map(drone_camera) -> None:
    """The drone escorts the rover and has it in frame the whole way.

    The rover is a real object and detecting it is correct; *mapping* it puts
    an obstacle on the route the mission is currently driving.
    """
    camera = drone_camera
    pose = drone_pose((0.0, 0.0, 4.0))
    camera_from_world = pose.inverse()
    rover_xy = np.array([6.0, 0.0])
    base = camera_from_world.apply(np.array([rover_xy[0], rover_xy[1], 0.0]))
    top = camera_from_world.apply(np.array([rover_xy[0], rover_xy[1], 0.5]))
    base_row = int(round(camera.project(base)[0][1]))
    top_row = int(round(camera.project(top)[0][1]))
    mask = np.zeros((camera.height, camera.width), dtype=bool)
    mask[top_row:base_row, 560:720] = True

    detector = MonocularObstacleDetector(max_range_m=12.0)
    assert detector.project(mask, camera, pose).contact_count > 0
    filtered = detector.project(
        mask, camera, pose, exclude_centres_xy=np.array([rover_xy]), exclude_radius_m=1.0
    )
    assert filtered.contact_count == 0


# ---------------------------------------------------------------------------
# Against the rendered arena
# ---------------------------------------------------------------------------

def test_a_rendered_wall_is_mapped_where_it_actually_stands(drone_camera) -> None:
    """End to end on one frame: render, segment, project, compare to truth."""
    world = SyntheticWorld(
        stations=[],
        obstacles=[
            OrientedBox(np.array([0.0, -6.0, 0.75]), np.array([6.0, 1.0, 1.5]), name="wall")
        ],
    )
    pose = drone_pose((0.0, -12.0, 4.0), yaw=math.pi / 2.0)
    frame = world.render(pose, drone_camera)
    observation = MonocularObstacleDetector(max_range_m=10.0).process(
        frame, drone_camera, pose
    )

    assert observation.contact_count >= 20, "the wall's foot was barely found"
    contacts = observation.contacts
    # The near face is at y = -6.5; contacts on it must sit on that line. The
    # lateral ends of a silhouette are not ground contacts at all (the bottom
    # pixel there belongs to the far top edge), so score the central columns.
    central = contacts[np.abs(contacts[:, 0]) < 2.0]
    assert len(central) >= 10
    assert np.median(central[:, 1]) == pytest.approx(-6.5, abs=0.15)
    # The measured height is an upper bound: seen from above, the top of a
    # silhouette is the object's *far* top edge, which one view cannot
    # separate from a taller object standing at the contact point.
    height = float(np.median(central[:, 2]))
    assert 1.5 - 0.15 <= height <= 2.1, f"measured height {height:.2f} m"
    # Nothing may be mapped in front of the wall: that is where the rover drives.
    assert np.all(contacts[:, 1] <= -3.5)
