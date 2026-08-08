"""Minimal rigid-body transform utilities.

Deliberately dependency-free (numpy only) so that the mission algorithms can be
unit tested without a ROS installation.  Conventions follow REP-103:

* world / ``map`` frame: ENU  (x east, y north, z up)
* body  / ``base_link``:      (x forward, y left, z up)
* camera optical frame:       (x right, y down, z forward)

Quaternions are stored in ROS order ``(x, y, z, w)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# A rotation whose determinant deviates from 1 by more than this is rejected.
# 1e-6 comfortably absorbs float32 round-trips through ROS messages while still
# catching genuine mistakes such as a mirrored (left-handed) basis.
_ROTATION_TOLERANCE = 1e-6


def quaternion_to_matrix(quat: Sequence[float]) -> np.ndarray:
    """Convert a ``(x, y, z, w)`` quaternion into a 3x3 rotation matrix."""
    x, y, z, w = (float(v) for v in quat)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("cannot normalise a zero-length quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quaternion(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix into a ``(x, y, z, w)`` quaternion.

    Uses the branch-on-largest-diagonal form of Shepperd's method; the naive
    ``w``-first formula loses all precision when ``w`` approaches zero (a 180
    degree rotation), which happens routinely for down-looking cameras.
    """
    rot = np.asarray(rot, dtype=float)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=float)


def quaternion_from_yaw(yaw: float) -> np.ndarray:
    """Quaternion for a rotation of ``yaw`` radians about +Z."""
    half = 0.5 * float(yaw)
    return np.array([0.0, 0.0, math.sin(half), math.cos(half)], dtype=float)


def yaw_from_quaternion(quat: Sequence[float]) -> float:
    """Extract the ENU yaw (rotation about +Z) from a ``(x, y, z, w)`` quaternion."""
    x, y, z, w = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def normalize_angle(angle: float) -> float:
    """Wrap an angle into ``(-pi, pi]``."""
    wrapped = math.fmod(float(angle) + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


@dataclass(frozen=True)
class Transform:
    """An SE(3) rigid transform expressed as ``translation`` + ``rotation``.

    A ``Transform`` always means "pose of the child frame expressed in the
    parent frame", which is also the operator that maps a point from child
    coordinates into parent coordinates.  Keeping that single interpretation
    everywhere is what stops camera and world coordinates from being mixed up.
    """

    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self) -> None:
        translation = np.asarray(self.translation, dtype=float).reshape(3)
        rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)
        if abs(float(np.linalg.det(rotation)) - 1.0) > _ROTATION_TOLERANCE:
            raise ValueError(
                "rotation is not a proper rotation matrix "
                f"(det={float(np.linalg.det(rotation)):.9f})"
            )
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "rotation", rotation)

    # -- constructors ----------------------------------------------------
    @classmethod
    def identity(cls) -> "Transform":
        return cls(np.zeros(3), np.eye(3))

    @classmethod
    def from_quaternion(cls, translation: Sequence[float], quat: Sequence[float]) -> "Transform":
        return cls(np.asarray(translation, dtype=float), quaternion_to_matrix(quat))

    @classmethod
    def from_yaw(cls, translation: Sequence[float], yaw: float) -> "Transform":
        return cls.from_quaternion(translation, quaternion_from_yaw(yaw))

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "Transform":
        matrix = np.asarray(matrix, dtype=float).reshape(4, 4)
        return cls(matrix[:3, 3], matrix[:3, :3])

    # -- accessors -------------------------------------------------------
    @property
    def quaternion(self) -> np.ndarray:
        return matrix_to_quaternion(self.rotation)

    @property
    def yaw(self) -> float:
        return yaw_from_quaternion(self.quaternion)

    def as_matrix(self) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    # -- algebra ---------------------------------------------------------
    def compose(self, other: "Transform") -> "Transform":
        """Return ``self * other`` (apply ``other`` first, then ``self``)."""
        return Transform(
            self.rotation @ other.translation + self.translation,
            self.rotation @ other.rotation,
        )

    def __matmul__(self, other: "Transform") -> "Transform":
        return self.compose(other)

    def inverse(self) -> "Transform":
        rot_t = self.rotation.T
        return Transform(-rot_t @ self.translation, rot_t)

    def apply(self, points: Iterable[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Transform a point ``(3,)`` or a batch of points ``(N, 3)``."""
        pts = np.asarray(points, dtype=float)
        single = pts.ndim == 1
        pts = np.atleast_2d(pts)
        if pts.shape[1] != 3:
            raise ValueError(f"expected points of shape (N, 3), got {pts.shape}")
        out = pts @ self.rotation.T + self.translation
        return out[0] if single else out


# ---------------------------------------------------------------------------
# Fixed frame conventions
# ---------------------------------------------------------------------------

def optical_from_body_rotation(
    right_in_body: Sequence[float],
    down_in_body: Sequence[float],
) -> np.ndarray:
    """Build ``R_body_optical`` from the camera's right/down axes in body coords.

    The optical Z axis (viewing direction) is derived as ``right x down`` so the
    result is guaranteed right-handed; supplying non-orthogonal axes raises.
    """
    right = np.asarray(right_in_body, dtype=float)
    down = np.asarray(down_in_body, dtype=float)
    right = right / np.linalg.norm(right)
    down = down / np.linalg.norm(down)
    if abs(float(right @ down)) > 1e-9:
        raise ValueError("camera right and down axes must be orthogonal")
    forward = np.cross(right, down)
    return np.column_stack((right, down, forward))


#: Optical frame of a nadir (straight-down) camera relative to ``base_link``.
#: Image-right is the vehicle's right (-Y body), image-down is vehicle-backward
#: (-X body) so that vehicle-forward appears "up" in the picture.
R_BODY_TO_NADIR_OPTICAL = optical_from_body_rotation((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0))

#: Optical frame of a forward-looking camera relative to ``base_link``
#: (the standard REP-103 camera_link -> camera_optical_frame rotation).
R_BODY_TO_FORWARD_OPTICAL = optical_from_body_rotation((0.0, -1.0, 0.0), (0.0, 0.0, -1.0))
