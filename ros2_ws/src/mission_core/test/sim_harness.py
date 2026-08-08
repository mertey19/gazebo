"""Offline optical + kinematic harness used by the mission_core test suite.

This is **not** part of the robot runtime.  It exists so the full perception ->
world-model -> planning -> control -> verification chain can be executed
deterministically on a developer machine, without Gazebo.

What it reproduces faithfully:

* a pinhole camera rendering real QR textures onto real 3D quads, so
  ``cv2.QRCodeDetector`` has to decode genuine imagery;
* a downward lidar that ray-casts against oriented boxes;
* unicycle rover kinematics.

What it deliberately does not reproduce: contact physics, motor dynamics,
lighting, lens distortion and sensor noise.  Those are Gazebo's job; a green
run here means the *logic* is right, not that the vehicle flies.
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


def _yaw_matrix(yaw: float) -> np.ndarray:
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    return np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])


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

    def ray_intersection(self, origin: np.ndarray, direction: np.ndarray) -> Optional[float]:
        """Slab-method ray/OBB intersection; returns the entry distance."""
        distances = self.ray_intersections(
            np.asarray(origin, dtype=float), np.asarray(direction, dtype=float).reshape(1, 3)
        )
        value = float(distances[0])
        return value if math.isfinite(value) else None

    def ray_intersections(self, origin: np.ndarray, directions: np.ndarray) -> np.ndarray:
        """Vectorised slab test for ``(N, 3)`` directions; ``inf`` where missed.

        Vectorised because a single lidar sweep casts hundreds of rays against
        every box in the scene, several times a second, for the whole flight.
        """
        rot = self.rotation()
        local_origin = rot.T @ (np.asarray(origin, dtype=float) - np.asarray(self.centre, float))
        local_dirs = np.asarray(directions, dtype=float) @ rot
        half = np.asarray(self.size, dtype=float) / 2.0

        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-half - local_origin) / local_dirs
            t2 = (half - local_origin) / local_dirs
        t_low = np.minimum(t1, t2)
        t_high = np.maximum(t1, t2)
        # A ray parallel to a slab either misses entirely or is unconstrained
        # by that slab; encode both cases instead of dividing by zero.
        parallel = np.abs(local_dirs) < 1e-12
        outside = parallel & (np.abs(local_origin) > half)
        t_low = np.where(parallel, -np.inf, t_low)
        t_high = np.where(parallel, np.inf, t_high)

        t_min = np.max(t_low, axis=1)
        t_max = np.min(t_high, axis=1)
        hit = (t_min <= t_max) & (t_max > 1e-6) & ~np.any(outside, axis=1)
        entry = np.where(t_min > 1e-6, t_min, t_max)
        return np.where(hit, entry, np.inf)

    def footprint_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.centre, dtype=float)[:2], np.asarray(self.size, dtype=float)[:2]


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
            None, (60, 60, 65)
        )


class SyntheticWorld:
    """Static scene: ground plane, QR stations and box obstacles."""

    def __init__(
        self,
        stations: Sequence[Station],
        obstacles: Sequence[OrientedBox],
        *,
        ground_extent: float = 12.0,
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
            self._quads.extend(obstacle.faces(None, (95, 100, 110)))

    # -- ground truth (tests and evaluation only) -------------------------
    def ground_truth_station_xy(self) -> Dict[str, np.ndarray]:
        """Actual station positions. Used **only** to score perception error."""
        return {s.payload: np.array(s.xy, dtype=float) for s in self.stations}

    def collision_boxes(self) -> List[OrientedBox]:
        return [*self.obstacles, *(s.cube for s in self.stations), *(s.pedestal for s in self.stations)]

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
            small[inside & (checker == 0)] = (118, 124, 118)
            small[inside & (checker == 1)] = (138, 146, 138)
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
            corners_cam = camera_from_world.apply(quad.corners)
            if np.any(corners_cam[:, 2] <= 0.05):
                continue  # crosses the near plane; no clipping implemented
            visible.append((float(np.linalg.norm(quad.centre - eye)), quad))
        visible.sort(key=lambda item: -item[0])

        for _, quad in visible:
            corners_cam = camera_from_world.apply(quad.corners)
            pixels = camera.project(corners_cam).astype(np.float32)
            if not self._overlaps_image(pixels, camera):
                continue
            if quad.texture is None:
                cv2.fillConvexPoly(
                    image, pixels.astype(np.int32), quad.colour, lineType=cv2.LINE_AA
                )
                continue
            tex_h, tex_w = quad.texture.shape[:2]
            source = np.array(
                [[0, 0], [tex_w - 1, 0], [tex_w - 1, tex_h - 1], [0, tex_h - 1]], dtype=np.float32
            )
            transform = cv2.getPerspectiveTransform(source, pixels)
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

    # -- lidar --------------------------------------------------------------
    def raycast(self, origin: Sequence[float], directions: np.ndarray) -> np.ndarray:
        """Cast rays from ``origin``; returns ``(N, 3)`` hit points in ``map``.

        Rays that hit nothing (neither geometry nor the finite ground patch)
        are dropped, mirroring a lidar's max-range behaviour.
        """
        origin = np.asarray(origin, dtype=float)
        dirs = np.atleast_2d(np.asarray(directions, dtype=float))
        norms = np.linalg.norm(dirs, axis=1)
        dirs = dirs[norms > 1e-12] / norms[norms > 1e-12, None]
        if len(dirs) == 0:
            return np.zeros((0, 3))

        best = np.full(len(dirs), np.inf)
        for box in self.collision_boxes():
            best = np.minimum(best, box.ray_intersections(origin, dirs))

        # The ground is a finite patch, so a ray may fly past its edge.
        with np.errstate(divide="ignore", invalid="ignore"):
            ground_t = (GROUND_Z - origin[2]) / dirs[:, 2]
        ground_hit = np.isfinite(ground_t) & (ground_t > 0.0) & (ground_t < best)
        if np.any(ground_hit):
            candidates = origin + dirs * ground_t[:, None]
            inside = (
                (np.abs(candidates[:, 0]) < self.ground_extent)
                & (np.abs(candidates[:, 1]) < self.ground_extent)
                & ground_hit
            )
            best = np.where(inside, ground_t, best)

        valid = np.isfinite(best)
        if not np.any(valid):
            return np.zeros((0, 3))
        return origin + dirs[valid] * best[valid, None]


def nadir_lidar_directions(
    half_angle_rad: float = 0.55, rings: int = 12, per_ring: int = 36
) -> np.ndarray:
    """Downward cone of ray directions approximating a nadir 3D lidar."""
    directions = [np.array([0.0, 0.0, -1.0])]
    for ring in range(1, rings + 1):
        theta = half_angle_rad * ring / rings
        for index in range(per_ring):
            phi = 2.0 * math.pi * index / per_ring
            directions.append(
                np.array(
                    [
                        math.sin(theta) * math.cos(phi),
                        math.sin(theta) * math.sin(phi),
                        -math.cos(theta),
                    ]
                )
            )
    return np.asarray(directions, dtype=float)


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
