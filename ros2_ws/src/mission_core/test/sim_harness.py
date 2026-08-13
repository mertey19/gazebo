"""Offline optical + kinematic harness used by the mission_core test suite.

This is **not** part of the robot runtime.  It exists so the full perception ->
world-model -> planning -> control -> verification chain can be executed
deterministically on a developer machine, without Gazebo.

What it reproduces faithfully:

* a pinhole camera rendering real QR textures onto real 3D quads, so
  ``cv2.QRCodeDetector`` has to decode genuine imagery;
* surface colours copied from ``mission_arena.sdf``, because the obstacle
  mapper is now a *vision* algorithm: it has to separate floor from not-floor
  in these frames, so the frames have to carry the same separation the real
  arena does - no more and no less;
* unicycle rover kinematics.

What it deliberately does not reproduce: contact physics, motor dynamics,
lighting, lens distortion and sensor noise.  Those are Gazebo's job; a green
run here means the *logic* is right, not that the vehicle flies.

There is no range sensor to model, because neither vehicle carries one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from mission_core.camera import PinholeCamera
from mission_core.geometry import Transform
from mission_core.qr import render_qr_image

GROUND_Z = 0.0

#: Surface colours in OpenCV BGR, converted from the ``<diffuse>`` entries of
#: ``mission_arena.sdf`` and the station models.  They are mirrored rather than
#: invented because the obstacle mapper separates floor from not-floor by
#: colour: an offline scene with an easier separation than the real world would
#: silently make every mapping test meaningless.  ``test_simulation_consistency``
#: pins them to the SDF.
GROUND_BGR = (128, 138, 128)  # <diffuse>0.50 0.54 0.50</diffuse>
OBSTACLE_BGR = (115, 102, 97)  # <diffuse>0.38 0.40 0.45</diffuse>
PEDESTAL_BGR = (61, 51, 51)  # <diffuse>0.20 0.20 0.24</diffuse>
#: How far the two checker tones sit either side of the floor colour. The floor
#: in Gazebo is a flat material; the harness needs *some* texture or the QR
#: front-end correctly rejects the frame as "sensor not streaming".
CHECKER_CONTRAST = 0.08


#: Anything closer than this to the image plane cannot be projected.
NEAR_PLANE_M = 0.05


def _yaw_matrix(yaw: float) -> np.ndarray:
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    return np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])


def _clip_to_near_plane(polygon_cam: np.ndarray, near: float) -> np.ndarray:
    """Sutherland-Hodgman clip of a convex polygon against ``z >= near``."""
    if np.all(polygon_cam[:, 2] >= near):
        return polygon_cam
    output: List[np.ndarray] = []
    count = len(polygon_cam)
    for index in range(count):
        current = polygon_cam[index]
        following = polygon_cam[(index + 1) % count]
        current_in = current[2] >= near
        if current_in:
            output.append(current)
        if current_in != (following[2] >= near):
            span = following[2] - current[2]
            if abs(span) > 1e-12:
                output.append(current + (near - current[2]) / span * (following - current))
    return np.asarray(output, dtype=float) if output else np.zeros((0, 3))


@dataclass
class Quad:
    """A textured or flat-shaded 3D rectangle, corners ordered TL, TR, BR, BL."""

    corners: np.ndarray
    texture: Optional[np.ndarray] = None
    colour: Tuple[int, int, int] = (120, 120, 120)

    @property
    def centre(self) -> np.ndarray:
        return self.corners.mean(axis=0)

    @property
    def normal(self) -> np.ndarray:
        """Outward normal, defined as ``right x up`` of the texture basis.

        Corners are stored in texture order TL, TR, BR, BL, so ``right`` is
        ``TR - TL`` and ``up`` is ``TL - BL``.  Taking ``TL - BL`` (rather than
        ``BL - TL``) is what keeps this normal consistent with an unmirrored
        texture: the two must agree, or fixing one flips the other and the face
        gets back-face culled instead of rendered.
        """
        right = self.corners[1] - self.corners[0]
        up = self.corners[0] - self.corners[3]
        normal = np.cross(right, up)
        length = float(np.linalg.norm(normal))
        return normal / length if length > 1e-12 else normal


@dataclass
class OrientedBox:
    """An axis-aligned-in-its-own-frame box, yawed about +Z in the world."""

    centre: np.ndarray
    size: np.ndarray
    yaw: float = 0.0
    name: str = "box"

    def rotation(self) -> np.ndarray:
        return _yaw_matrix(self.yaw)

    def faces(self, texture: Optional[np.ndarray], colour: Tuple[int, int, int]) -> List[Quad]:
        """The six faces as :class:`Quad` objects, wound outward."""
        half = np.asarray(self.size, dtype=float) / 2.0
        rot = self.rotation()
        centre = np.asarray(self.centre, dtype=float)

        # (outward normal, texture-up) pairs in the box's local frame.  Texture
        # right is derived as `up x normal`, which is the condition
        # `right x up == normal` - i.e. a right-handed basis with the *outward*
        # normal.  Using the inward view direction instead flips the handedness
        # and renders every texture mirrored, which for a QR code moves the
        # third finder pattern from bottom-left to bottom-right.  OpenCV >= 4.7
        # happens to decode mirrored codes, so that bug stays invisible until
        # the stack meets an older build (Ubuntu 24.04 ships OpenCV 4.6).
        specs = [
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), half[0], (half[1], half[2])),
            (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), half[0], (half[1], half[2])),
            (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), half[1], (half[0], half[2])),
            (np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0]), half[1], (half[0], half[2])),
            (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), half[2], (half[0], half[1])),
            (np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]), half[2], (half[0], half[1])),
        ]
        quads: List[Quad] = []
        for normal, up, offset, extents in specs:
            right = np.cross(up, normal)
            right_norm = float(np.linalg.norm(right))
            if right_norm < 1e-9:  # pragma: no cover - guarded by the spec table
                continue
            right = right / right_norm
            # Half-extent along the face's right/up axes.
            half_right = float(abs(right @ np.array([half[0], half[1], half[2]])))
            half_up = float(abs(up @ np.array([half[0], half[1], half[2]])))
            face_centre_local = normal * offset
            local_corners = np.array(
                [
                    face_centre_local - right * half_right + up * half_up,
                    face_centre_local + right * half_right + up * half_up,
                    face_centre_local + right * half_right - up * half_up,
                    face_centre_local - right * half_right - up * half_up,
                ]
            )
            quads.append(Quad(local_corners @ rot.T + centre, texture, colour))
        return quads


@dataclass
class Station:
    """A QR-coded target station: a textured cube on a short pedestal."""

    payload: str
    xy: Tuple[float, float]
    cube_size: float = 0.80
    pedestal_height: float = 0.25
    yaw: float = 0.0
    _texture: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._texture = cv2.cvtColor(render_qr_image(self.payload, 512), cv2.COLOR_GRAY2BGR)

    @property
    def centre(self) -> np.ndarray:
        return np.array(
            [self.xy[0], self.xy[1], self.pedestal_height + self.cube_size / 2.0], dtype=float
        )

    @property
    def cube(self) -> OrientedBox:
        return OrientedBox(
            self.centre,
            np.array([self.cube_size] * 3),
            self.yaw,
            f"station_{self.payload}",
        )

    @property
    def pedestal(self) -> OrientedBox:
        return OrientedBox(
            np.array([self.xy[0], self.xy[1], self.pedestal_height / 2.0]),
            np.array([self.cube_size * 0.55, self.cube_size * 0.55, self.pedestal_height]),
            self.yaw,
            f"pedestal_{self.payload}",
        )

    def quads(self) -> List[Quad]:
        return self.cube.faces(self._texture, (255, 255, 255)) + self.pedestal.faces(
            None, PEDESTAL_BGR
        )


class SyntheticWorld:
    """Static scene: ground plane, QR stations and box obstacles."""

    def __init__(
        self,
        stations: Sequence[Station],
        obstacles: Sequence[OrientedBox],
        *,
        # Wider than the 22 x 22 m arena on purpose: the drone camera looks
        # ahead of the vehicle, and where the finite floor patch ends the
        # renderer draws background, which the obstacle segmenter would be
        # right to call not-floor. Gazebo's ground plane is 60 x 60 m.
        ground_extent: float = 16.0,
        checker_size_m: float = 1.0,
        ground_render_scale: int = 4,
    ) -> None:
        self.stations = list(stations)
        self.obstacles = list(obstacles)
        self.ground_extent = float(ground_extent)
        self.checker_size_m = float(checker_size_m)
        # The ground exists only to keep frames non-uniform and to give the
        # decoder realistic surroundings; nothing measures it. Rendering it at
        # 1/N resolution and upscaling cuts the dominant per-frame cost with no
        # effect on QR detection.
        self.ground_render_scale = max(1, int(ground_render_scale))
        # Per-pixel viewing rays depend only on the intrinsics, so they are
        # built once per camera instead of once per frame (the whole scan
        # renders hundreds of frames).
        self._ray_cache: Dict[Tuple[int, int, float, float], np.ndarray] = {}
        self._quads: List[Quad] = []
        for station in self.stations:
            self._quads.extend(station.quads())
        for obstacle in self.obstacles:
            self._quads.extend(obstacle.faces(None, OBSTACLE_BGR))

    # -- ground truth (tests and evaluation only) -------------------------
    def ground_truth_station_xy(self) -> Dict[str, np.ndarray]:
        """Actual station positions. Used **only** to score perception error."""
        return {s.payload: np.array(s.xy, dtype=float) for s in self.stations}

    # -- rendering ---------------------------------------------------------
    def _pixel_rays(self, camera: PinholeCamera, stride: int) -> np.ndarray:
        """Cached ``(H/stride, W/stride, 3)`` optical-frame ray directions."""
        key = (camera.width, camera.height, camera.fx, camera.fy, stride)
        cached = self._ray_cache.get(key)
        if cached is None:
            rows, cols = np.mgrid[0 : camera.height : stride, 0 : camera.width : stride]
            cached = np.stack(
                [
                    (cols - camera.cx) / camera.fx,
                    (rows - camera.cy) / camera.fy,
                    np.ones_like(cols, dtype=np.float32),
                ],
                axis=-1,
            ).astype(np.float32)
            self._ray_cache[key] = cached
        return cached

    def _ground_image(self, camera_pose: Transform, camera: PinholeCamera) -> np.ndarray:
        """Procedural checkerboard ground, drawn by back-projecting each pixel.

        A flat colour would make empty frames uniform, which the perception
        front-end correctly rejects as "sensor not streaming"; real ground has
        texture, so the harness provides some.

        The two tones differ only in lightness, both sitting on the SDF floor's
        hue.  That is the case the obstacle segmenter has to survive: a floor
        with brightness variation across it, whose *colour* is nevertheless one
        thing.
        """
        stride = self.ground_render_scale
        dirs = self._pixel_rays(camera, stride)
        small = np.full((dirs.shape[0], dirs.shape[1], 3), 40, dtype=np.uint8)
        dirs_map = dirs @ camera_pose.rotation.T.astype(np.float32)
        origin = camera_pose.translation.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (np.float32(GROUND_Z) - origin[2]) / dirs_map[..., 2]
        hits = np.isfinite(t) & (t > 0.0)
        if np.any(hits):
            world = origin[:2] + dirs_map[..., :2] * t[..., None]
            checker = (
                np.floor(world[..., 0] / self.checker_size_m).astype(np.int32)
                + np.floor(world[..., 1] / self.checker_size_m).astype(np.int32)
            ) % 2
            inside = (
                hits
                & (np.abs(world[..., 0]) < self.ground_extent)
                & (np.abs(world[..., 1]) < self.ground_extent)
            )
            dark = tuple(int(round(c * (1.0 - CHECKER_CONTRAST))) for c in GROUND_BGR)
            light = tuple(int(round(c * (1.0 + CHECKER_CONTRAST))) for c in GROUND_BGR)
            small[inside & (checker == 0)] = dark
            small[inside & (checker == 1)] = light
        if stride == 1:
            return small
        return cv2.resize(
            small, (camera.width, camera.height), interpolation=cv2.INTER_NEAREST
        )

    def render(self, camera_pose: Transform, camera: PinholeCamera) -> np.ndarray:
        """Render the scene from ``camera_pose`` (pose of the *optical* frame)."""
        image = self._ground_image(camera_pose, camera)
        world_from_camera = camera_pose
        camera_from_world = camera_pose.inverse()
        eye = world_from_camera.translation

        # Painter's algorithm: far faces first so near geometry overwrites it.
        visible: List[Tuple[float, Quad]] = []
        for quad in self._quads:
            if quad.normal @ (eye - quad.centre) <= 1e-6:
                continue  # back-facing
            visible.append((float(np.linalg.norm(quad.centre - eye)), quad))
        visible.sort(key=lambda item: -item[0])

        for _, quad in visible:
            corners_cam = camera_from_world.apply(quad.corners)
            # Faces are clipped, not discarded. Dropping a whole face because
            # one of its corners has passed behind the camera deletes exactly
            # the geometry a *close* obstacle presents - and then the mapper
            # sees floor where a wall is and marks the wall's own footprint
            # free. Gazebo clips; so does this.
            polygon_cam = _clip_to_near_plane(corners_cam, NEAR_PLANE_M)
            if len(polygon_cam) < 3:
                continue
            pixels = camera.project(polygon_cam).astype(np.float32)
            if not self._overlaps_image(pixels, camera):
                continue
            if quad.texture is None:
                cv2.fillConvexPoly(
                    image, pixels.astype(np.int32), quad.colour, lineType=cv2.LINE_AA
                )
                continue
            # The texture homography is built from the quad's geometry rather
            # than from its four projected corners, because a clipped face no
            # longer has four of them - and a corner behind the camera has no
            # projection at all.
            transform = self._texture_homography(quad, camera, camera_from_world)
            if transform is None:
                continue
            warped = cv2.warpPerspective(
                quad.texture,
                transform,
                (camera.width, camera.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            mask = np.zeros((camera.height, camera.width), dtype=np.uint8)
            cv2.fillConvexPoly(mask, pixels.astype(np.int32), 255)
            image[mask > 0] = warped[mask > 0]
        return image

    @staticmethod
    def _texture_homography(
        quad: Quad, camera: PinholeCamera, camera_from_world: Transform
    ) -> Optional[np.ndarray]:
        """Map texture pixels onto image pixels for one textured quad.

        A quad is planar, so texture -> image is a homography.  Composing it
        from the quad's own basis (origin ``TL``, edges ``TR - TL`` and
        ``BL - TL``) keeps it defined even when some corners lie behind the
        camera; the visible extent is then decided by the clipped polygon mask,
        not by this matrix.
        """
        tex_h, tex_w = quad.texture.shape[:2]
        origin = camera_from_world.apply(quad.corners[0])
        right = camera_from_world.rotation @ (quad.corners[1] - quad.corners[0]) / (tex_w - 1)
        down = camera_from_world.rotation @ (quad.corners[3] - quad.corners[0]) / (tex_h - 1)
        matrix = camera.matrix @ np.column_stack((right, down, origin))
        if abs(float(np.linalg.det(matrix))) < 1e-12:
            return None
        return matrix

    @staticmethod
    def _overlaps_image(pixels: np.ndarray, camera: PinholeCamera) -> bool:
        return bool(
            pixels[:, 0].max() >= 0
            and pixels[:, 0].min() <= camera.width - 1
            and pixels[:, 1].max() >= 0
            and pixels[:, 1].min() <= camera.height - 1
            # Reject degenerate projections that would make the homography singular.
            and (pixels[:, 0].max() - pixels[:, 0].min()) > 1.0
            and (pixels[:, 1].max() - pixels[:, 1].min()) > 1.0
        )


def camera_pose_from_body(
    body_position: Sequence[float],
    body_yaw: float,
    mount_offset: Sequence[float],
    body_to_optical: np.ndarray,
) -> Transform:
    """Compose ``map -> base_link -> camera_optical`` the same way TF would."""
    map_from_body = Transform.from_yaw(body_position, body_yaw)
    body_from_optical = Transform(np.asarray(mount_offset, dtype=float), body_to_optical)
    return map_from_body @ body_from_optical
