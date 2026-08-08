"""Pinhole camera model shared by the perception nodes and the test harness."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PinholeCamera:
    """Intrinsics of an ideal pinhole camera.

    Gazebo's ``camera`` sensor is a perfect pinhole unless a distortion plugin
    is added, so the mission stack carries zero distortion coefficients.  They
    are still passed explicitly to OpenCV so that swapping in a distorted
    (real) camera later only requires filling this struct from ``CameraInfo``.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera resolution must be positive")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("focal lengths must be positive")

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_rad: float) -> "PinholeCamera":
        """Build intrinsics the way Gazebo derives them from ``<horizontal_fov>``."""
        if not 0.0 < hfov_rad < math.pi:
            raise ValueError("horizontal FOV must lie in (0, pi)")
        fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
        # gz-sim uses square pixels: fy == fx, and the principal point is the
        # exact image centre using the OpenCV (w-1)/2 pixel-centre convention.
        return cls(width, height, fx, fx, (width - 1) / 2.0, (height - 1) / 2.0)

    @classmethod
    def from_camera_info_k(cls, width: int, height: int, k: Sequence[float]) -> "PinholeCamera":
        """Build intrinsics from a ``sensor_msgs/CameraInfo`` ``k`` array."""
        k = list(k)
        if len(k) != 9:
            raise ValueError("CameraInfo.k must contain 9 elements")
        return cls(width, height, k[0], k[4], k[2], k[5])

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=float,
        )

    @property
    def distortion(self) -> np.ndarray:
        return np.zeros((5, 1), dtype=float)

    def project(self, points_optical: np.ndarray) -> np.ndarray:
        """Project optical-frame points ``(N, 3)`` to pixels ``(N, 2)``.

        Points at or behind the image plane have no valid projection; callers
        must cull them first (``visible_mask``).
        """
        pts = np.atleast_2d(np.asarray(points_optical, dtype=float))
        z = pts[:, 2]
        if np.any(z <= 1e-9):
            raise ValueError("cannot project points at or behind the camera")
        u = self.fx * pts[:, 0] / z + self.cx
        v = self.fy * pts[:, 1] / z + self.cy
        return np.column_stack((u, v))

    def visible_mask(self, points_optical: np.ndarray, margin_px: float = 0.0) -> np.ndarray:
        """Boolean mask of points that fall in front of the camera and on-sensor."""
        pts = np.atleast_2d(np.asarray(points_optical, dtype=float))
        in_front = pts[:, 2] > 1e-6
        mask = np.zeros(len(pts), dtype=bool)
        if not np.any(in_front):
            return mask
        uv = np.full((len(pts), 2), np.nan)
        uv[in_front] = self.project(pts[in_front])
        on_sensor = (
            (uv[:, 0] >= -margin_px)
            & (uv[:, 0] <= self.width - 1 + margin_px)
            & (uv[:, 1] >= -margin_px)
            & (uv[:, 1] <= self.height - 1 + margin_px)
        )
        mask[in_front] = on_sensor[in_front]
        return mask

    def ray(self, pixel: Sequence[float]) -> np.ndarray:
        """Unit viewing ray in the optical frame through a pixel."""
        u, v = float(pixel[0]), float(pixel[1])
        direction = np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
        return direction / np.linalg.norm(direction)

    def ground_sample_distance(self, range_m: float) -> float:
        """Metres per pixel at ``range_m`` along the optical axis."""
        if range_m <= 0.0:
            raise ValueError("range must be positive")
        return range_m / self.fx
