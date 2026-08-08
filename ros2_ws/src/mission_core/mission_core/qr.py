"""QR detection and 6-DoF marker pose estimation.

This is the *only* place in the stack that turns pixels into a target
observation.  Nothing here ever consults Gazebo model names, ground-truth
poses, or a lookup table of expected payloads: the payload string and the
marker pose both come out of the image.

Pipeline
--------
``image`` -> :meth:`cv2.QRCodeDetector.detectAndDecodeMulti` -> 4 image corners
per code -> :func:`cv2.solvePnPGeneric` with ``SOLVEPNP_IPPE_SQUARE`` against
the *physically measured* marker side length -> marker pose in the camera
optical frame.

The marker side length is a property of the printed target (exactly like an
ArUco marker size) and is a configuration parameter, not a position prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from .camera import PinholeCamera
from .errors import PerceptionError


@dataclass(frozen=True)
class QrDetection:
    """A single decoded QR code together with its pose in the camera frame."""

    payload: str
    corners_px: np.ndarray
    #: Marker centre expressed in the camera *optical* frame (x right, y down, z fwd).
    position_optical: np.ndarray
    #: Marker orientation as ``R_optical_marker``.
    rotation_optical: np.ndarray
    #: RMS reprojection error of the accepted PnP solution, in pixels.
    reprojection_error_px: float
    #: ``err_best / err_second_best`` from the two IPPE solutions.  Values close
    #: to 1.0 mean the planar-pose ambiguity was not resolved and the
    #: orientation should not be trusted (the position still is).
    ambiguity_ratio: float
    #: Straight-line distance from camera centre to marker centre, in metres.
    range_m: float
    #: Apparent marker side length in pixels - a direct measure of how well the
    #: code was resolved by the sensor.
    apparent_size_px: float

    @property
    def confidence(self) -> float:
        """Heuristic per-observation quality in ``[0, 1]``.

        Combines geometric fit (low reprojection error) with how many pixels
        the code actually covered.  Used to weight multi-observation fusion.
        """
        fit = 1.0 / (1.0 + self.reprojection_error_px)
        # 60 px of apparent side length is where a version-1..3 QR becomes
        # comfortably decodable; beyond ~200 px extra size adds no information.
        resolution = float(np.clip((self.apparent_size_px - 60.0) / 140.0, 0.0, 1.0))
        return float(np.clip(0.5 * fit + 0.5 * resolution, 0.0, 1.0))


def _quad_area(corners: np.ndarray) -> float:
    """Shoelace area of a 4-point polygon."""
    x = corners[:, 0]
    y = corners[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _mean_side_length(corners: np.ndarray) -> float:
    sides = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
    return float(np.mean(sides))


def validate_frame(image: np.ndarray) -> np.ndarray:
    """Reject unusable camera frames early with a precise diagnostic.

    Returns a single-channel 8-bit view suitable for the detector.
    """
    if image is None:
        raise PerceptionError("camera frame is None")
    if not isinstance(image, np.ndarray):
        raise PerceptionError(f"camera frame has type {type(image).__name__}, expected ndarray")
    if image.size == 0:
        raise PerceptionError("camera frame is empty")
    if image.ndim == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 1:
            gray = image[:, :, 0]
        else:
            raise PerceptionError(f"unsupported channel count {image.shape[2]}")
    elif image.ndim == 2:
        gray = image
    else:
        raise PerceptionError(f"unsupported frame rank {image.ndim}")

    if gray.dtype != np.uint8:
        if not np.isfinite(gray).all():
            raise PerceptionError("camera frame contains non-finite values")
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if gray.shape[0] < 8 or gray.shape[1] < 8:
        raise PerceptionError(f"camera frame too small: {gray.shape}")
    # A constant frame means the sensor produced nothing (bridge not up, render
    # failure, lens cap).  Detecting it here avoids burning CPU every tick and
    # gives the operator a real message instead of "no detections".
    if int(gray.max()) - int(gray.min()) < 2:
        raise PerceptionError("camera frame is uniform - sensor is probably not streaming")
    return np.ascontiguousarray(gray)


class QrDetector:
    """Decode QR codes and recover their pose relative to the camera."""

    def __init__(
        self,
        plate_size_m: float,
        *,
        quiet_zone_modules: int = 4,
        max_reprojection_error_px: float = 4.0,
        min_quad_area_px: float = 400.0,
        min_apparent_size_px: float = 25.0,
    ) -> None:
        if plate_size_m <= 0.0:
            raise ValueError("plate_size_m must be positive")
        self.plate_size_m = float(plate_size_m)
        self.quiet_zone_modules = int(quiet_zone_modules)
        self.max_reprojection_error_px = float(max_reprojection_error_px)
        self.min_quad_area_px = float(min_quad_area_px)
        self.min_apparent_size_px = float(min_apparent_size_px)
        self._detector = cv2.QRCodeDetector()
        self._object_point_cache: dict[str, np.ndarray] = {}

    def code_size_m(self, payload: str) -> float:
        """Physical side length of the code area for a decoded payload."""
        return code_side_length_m(payload, self.plate_size_m, self.quiet_zone_modules)

    def _object_points(self, payload: str) -> np.ndarray:
        """Marker-frame corners, ordered TL, TR, BR, BL.

        That order is what ``QRCodeDetector`` returns *and* what
        ``SOLVEPNP_IPPE_SQUARE`` requires, so the two line up without any
        re-shuffling.  Marker frame: X right, Y up, Z out of the printed face.
        The module count - and therefore the physical size - depends on the
        payload, so the points are cached per payload rather than fixed.
        """
        cached = self._object_point_cache.get(payload)
        if cached is None:
            half = self.code_size_m(payload) / 2.0
            cached = np.array(
                [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
                dtype=np.float64,
            )
            self._object_point_cache[payload] = cached
        return cached

    # -- decoding --------------------------------------------------------
    def decode(self, image: np.ndarray) -> List[tuple[str, np.ndarray]]:
        """Return ``(payload, corners)`` for every readable code in the frame."""
        gray = validate_frame(image)
        try:
            ok, payloads, corner_sets, _ = self._detector.detectAndDecodeMulti(gray)
        except cv2.error as exc:  # pragma: no cover - OpenCV internal failure
            raise PerceptionError(f"OpenCV QR detector failed: {exc}") from exc
        if not ok or corner_sets is None:
            return []
        results: List[tuple[str, np.ndarray]] = []
        for payload, corners in zip(payloads, corner_sets):
            # detectAndDecodeMulti reports located-but-undecodable codes as an
            # empty payload; those are not usable observations.
            if not payload:
                continue
            results.append((payload, np.asarray(corners, dtype=np.float64).reshape(4, 2)))
        return results

    # -- pose ------------------------------------------------------------
    def _reprojection_rms_px(
        self,
        object_points: np.ndarray,
        corners: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera: PinholeCamera,
    ) -> float:
        """RMS reprojection error, computed here rather than trusted from OpenCV.

        ``solvePnPGeneric`` can return NaN residuals alongside NaN poses, and a
        NaN silently passes every ``error > threshold`` test.  Recomputing is
        cheap and makes the quality gate actually hold.
        """
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera.matrix, camera.distortion
        )
        residual = projected.reshape(4, 2) - corners.reshape(4, 2)
        return float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))

    def _solve_pose(self, corners: np.ndarray, camera: PinholeCamera, payload: str):
        """Estimate the marker pose, with a fallback for degenerate geometry.

        ``SOLVEPNP_IPPE_SQUARE`` is the right tool for a square fiducial and
        supplies both planar-ambiguity solutions - but it degenerates when the
        marker is almost exactly fronto-parallel, which is precisely the view a
        nadir drone camera has of a flat target.  Observed failure modes there
        are a NaN pose and, worse, a *finite* pose with an identity rotation
        that silently ignores the marker's in-plane orientation.

        So both solvers are always run and every candidate is scored by its
        own recomputed reprojection error; the global best wins.  Trusting
        either solver's success flag, or stopping at the first one that returns
        finite numbers, is what let the identity-rotation solution through.
        """
        object_points = self._object_points(payload)
        candidates: list[tuple[np.ndarray, np.ndarray]] = []
        for flags in (cv2.SOLVEPNP_IPPE_SQUARE, cv2.SOLVEPNP_ITERATIVE):
            try:
                retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    object_points,
                    corners.reshape(4, 1, 2),
                    camera.matrix,
                    camera.distortion,
                    flags=flags,
                )
            except cv2.error:
                continue
            if not retval or not tvecs:
                continue
            candidates.extend(
                (np.asarray(r, dtype=float), np.asarray(t, dtype=float))
                for r, t in zip(rvecs, tvecs)
                if np.isfinite(r).all() and np.isfinite(t).all()
            )
        if not candidates:
            return None

        scored = [
            (self._reprojection_rms_px(object_points, corners, rvec, tvec, camera), rvec, tvec)
            for rvec, tvec in candidates
        ]
        scored.sort(key=lambda item: item[0])
        best_error, best_rvec, best_tvec = scored[0]
        # Ratio near 1.0 => the two planar-pose solutions fit equally well and
        # the orientation should not be trusted (the position still can be).
        ambiguity = best_error / scored[1][0] if len(scored) > 1 and scored[1][0] > 1e-9 else 0.0
        rot, _ = cv2.Rodrigues(best_rvec)
        return best_tvec.reshape(3), rot, best_error, float(np.clip(ambiguity, 0.0, 1.0))

    def detect(self, image: np.ndarray, camera: PinholeCamera) -> List[QrDetection]:
        """Full pipeline: decode every code and estimate its pose."""
        detections: List[QrDetection] = []
        for payload, corners in self.decode(image):
            area = _quad_area(corners)
            side_px = _mean_side_length(corners)
            if area < self.min_quad_area_px or side_px < self.min_apparent_size_px:
                # Decoded but too small for a trustworthy pose - drop it rather
                # than pollute the world model with a low-quality position.
                continue
            solution = self._solve_pose(corners, camera, payload)
            if solution is None:
                continue
            translation, rotation, error, ambiguity = solution
            if not np.isfinite(translation).all() or translation[2] <= 0.0:
                continue
            # ``not (error <= threshold)`` rather than ``error > threshold`` so
            # a NaN that survived everything above is still rejected.
            if not error <= self.max_reprojection_error_px:
                continue
            detections.append(
                QrDetection(
                    payload=payload,
                    corners_px=corners,
                    position_optical=translation,
                    rotation_optical=rotation,
                    reprojection_error_px=error,
                    ambiguity_ratio=ambiguity,
                    range_m=float(np.linalg.norm(translation)),
                    apparent_size_px=side_px,
                )
            )
        return detections


def encode_qr_modules(payload: str) -> np.ndarray:
    """Encode ``payload`` and return **only** the code area, one pixel per module.

    ``cv2.QRCodeEncoder`` wraps its output in a quiet zone of its own, whose
    width is an implementation detail.  Cropping to the bounding box of the
    dark modules is exact: a QR code's finder patterns sit in three of its four
    corners, so that bounding box is the code boundary by construction.  Every
    downstream size calculation depends on knowing this boundary precisely.
    """
    if not payload:
        raise ValueError("QR payload must be a non-empty string")
    raw = cv2.QRCodeEncoder.create().encode(payload)
    if raw is None or raw.size == 0:  # pragma: no cover - encoder failure
        raise RuntimeError(f"failed to encode QR payload {payload!r}")
    dark_rows, dark_cols = np.where(raw == 0)
    if dark_rows.size == 0:  # pragma: no cover - impossible for a valid code
        raise RuntimeError(f"encoded QR for {payload!r} contains no dark modules")
    code = raw[
        dark_rows.min() : dark_rows.max() + 1, dark_cols.min() : dark_cols.max() + 1
    ]
    if code.shape[0] != code.shape[1]:  # pragma: no cover - defensive
        raise RuntimeError(f"cropped QR for {payload!r} is not square: {code.shape}")
    return code


def qr_module_count(payload: str) -> int:
    """Side length of the code in modules (21 for version 1, 25 for version 2...)."""
    return int(encode_qr_modules(payload).shape[0])


def code_fraction_of_plate(payload: str, quiet_zone_modules: int = 4) -> float:
    """Fraction of the printed plate occupied by the code proper.

    The QR *code* is what the detector's corners bound, but the *plate* is what
    is physically 0.8 m wide in the world.  Confusing the two scales every PnP
    range by ~1.6x, which then walks every target position along its viewing
    ray.  Keeping the conversion in one named function makes that impossible.
    """
    if quiet_zone_modules < 0:
        raise ValueError("quiet_zone_modules must not be negative")
    modules = qr_module_count(payload)
    return modules / float(modules + 2 * quiet_zone_modules)


def code_side_length_m(payload: str, plate_size_m: float, quiet_zone_modules: int = 4) -> float:
    """Physical side length of the code area on a ``plate_size_m`` wide plate."""
    return float(plate_size_m) * code_fraction_of_plate(payload, quiet_zone_modules)


def render_qr_image(payload: str, pixel_size: int = 512, quiet_zone_modules: int = 4) -> np.ndarray:
    """Render ``payload`` as a square 8-bit plate texture with a quiet zone.

    The result is what gets pasted onto the target model in Gazebo and onto the
    synthetic quads in the test harness, so the code-to-plate ratio is
    guaranteed identical in simulation and in the size maths.
    """
    code = encode_qr_modules(payload)
    bordered = cv2.copyMakeBorder(
        code,
        quiet_zone_modules,
        quiet_zone_modules,
        quiet_zone_modules,
        quiet_zone_modules,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    # INTER_NEAREST keeps module edges perfectly sharp; any smoothing here would
    # be re-introduced later by the camera and hurts decoding at long range.
    # An integer upscale factor also keeps every module exactly the same width.
    scale = max(1, pixel_size // bordered.shape[0])
    upscaled = cv2.resize(
        bordered,
        (bordered.shape[1] * scale, bordered.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    return upscaled


def expected_pixels_per_module(
    camera: PinholeCamera, code_size_m: float, range_m: float, payload: str
) -> float:
    """How many image pixels one QR module spans at ``range_m``.

    Below roughly 3 px/module the decoder becomes unreliable, so mission
    configuration (scan altitude, marker size, resolution) can be sanity
    checked offline instead of discovered during a flight.
    """
    code_px = camera.fx * float(code_size_m) / float(range_m)
    return code_px / float(qr_module_count(payload))


def max_scan_altitude_for_decoding(
    camera: PinholeCamera,
    code_size_m: float,
    payload: str,
    min_pixels_per_module: float = 3.0,
) -> float:
    """Largest nadir range at which ``payload`` still decodes reliably."""
    return camera.fx * float(code_size_m) / (min_pixels_per_module * qr_module_count(payload))


__all__ = [
    "QrDetection",
    "QrDetector",
    "code_fraction_of_plate",
    "code_side_length_m",
    "encode_qr_modules",
    "expected_pixels_per_module",
    "max_scan_altitude_for_decoding",
    "qr_module_count",
    "render_qr_image",
    "validate_frame",
]
