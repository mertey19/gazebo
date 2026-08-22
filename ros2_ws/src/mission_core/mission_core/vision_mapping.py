"""Monocular obstacle mapping - the camera-only replacement for a lidar.

The mission carries **no ranging sensor**: both vehicles have exactly one RGB
camera each, and every metre of obstacle geometry in the occupancy grid is
recovered from those images.  Two facts make that possible without depth:

1. the arena floor is a plane of known height (``ground_z``), and
2. the pose of the camera above that plane is known from TF.

Together they turn a pixel into a ray/plane intersection, which is exact - the
classic inverse-perspective (IPM) construction.  The only approximation left is
*which* pixels lie on the floor, and that is what :func:`segment_non_ground`
decides.

The geometry is applied where it is valid and nowhere else:

* an obstacle's **ground-contact pixel** - the bottom of its silhouette - is on
  the floor by definition, so its intersection is the true base position;
* pixels *above* that contact are on a vertical face and are deliberately
  **not** intersected with the floor.  Doing so is the standard IPM smear: it
  would paint a fake footprint stretching away from the camera.  Instead the
  top of the silhouette is intersected with the *vertical line through the
  contact point*, which measures the obstacle's height;
* pixels classified as floor are intersected and reported as free space;
* everything hidden behind an obstacle produces no pixels at all, so it stays
  ``UNKNOWN`` in the grid.  With ``planner.allow_unknown = false`` an obstacle's
  unseen interior is therefore never planned through, even though only its
  visible rim is ever marked occupied.

Nothing here knows what a "station" or an "obstacle_a" is: the output is a set
of contact points, their measured heights, and a set of free-space samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .camera import PinholeCamera
from .geometry import Transform

#: Lab is 8-bit in OpenCV: L in [0, 255], a/b in [0, 255] with 128 neutral, so
#: every threshold below is in those units and means the same on any frame.
_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class FloorModel:
    """What the arena floor looks like, in Lab.

    Kept *between* frames on purpose.  Re-deriving it from every image assumes
    the floor is the majority of every image, and the one moment that stops
    being true is the moment it matters most: a vehicle close enough to a wall
    for the wall to fill its view.  Estimated from frames the floor does
    dominate and then carried forward, the reference stays the floor.
    """

    chroma_median: np.ndarray
    chroma_scale: np.ndarray
    luma_median: float

    def blended_with(self, other: "FloorModel", weight: float) -> "FloorModel":
        """Move ``weight`` of the way towards a freshly measured model.

        Lighting drifts; the floor does not become a different colour between
        one frame and the next. A slow blend tracks the first without ever
        letting a single frame redefine the reference.
        """
        weight = float(np.clip(weight, 0.0, 1.0))
        return FloorModel(
            chroma_median=(1.0 - weight) * self.chroma_median + weight * other.chroma_median,
            chroma_scale=(1.0 - weight) * self.chroma_scale + weight * other.chroma_scale,
            luma_median=(1.0 - weight) * self.luma_median + weight * other.luma_median,
        )


@dataclass(frozen=True)
class GroundObservation:
    """One frame's worth of monocular obstacle evidence, in the ``map`` frame.

    ``contacts`` are ``(N, 3)``: ``x``/``y`` on the floor where an obstacle was
    seen to meet it, ``z`` the height measured for that obstacle - not the
    height of the point itself, which is zero by construction.  Carrying the
    height here is what lets :class:`~mission_core.occupancy.OccupancyMapper`
    fill its height map without a second sensor model.
    """

    contacts: np.ndarray
    free: np.ndarray
    #: Horizontal distance from the camera to the nearest contact inside the
    #: forward wedge, in metres; ``inf`` when the wedge is clear.  Computed in
    #: the sensor's own geometry so the rover's emergency stop never depends on
    #: how well the vehicle is localised in ``map``.
    nearest_forward_range_m: float = math.inf
    columns_evaluated: int = 0
    non_ground_fraction: float = 0.0
    #: False when the frame was rejected wholesale (see ``max_non_ground_fraction``).
    usable: bool = True
    #: The floor reference this frame was segmented against.
    floor_model: Optional[FloorModel] = None

    @property
    def contact_count(self) -> int:
        return int(self.contacts.shape[0])

    @property
    def free_count(self) -> int:
        return int(self.free.shape[0])


def below_horizon_mask(
    camera: PinholeCamera, camera_pose_map: Transform, ground_z: float = 0.0
) -> np.ndarray:
    """Pixels whose viewing ray can reach the floor at all.

    The vanishing line of the ground plane splits every image into a half that
    can see the floor and a half that provably cannot, and it follows from the
    camera pose alone.  The rover's camera looks straight ahead, so on it that
    line is real and central: roughly half of every frame is sky.  Leaving that
    half in ruins the dominant-surface statistics - sky is not a *second* kind
    of floor, it is not floor at all - and its lower boundary is not a ground
    contact but the horizon, which sits at infinity.
    """
    if camera_pose_map.translation[2] <= float(ground_z):
        return np.zeros((camera.height, camera.width), dtype=bool)
    row = camera_pose_map.rotation[2, :]
    us = (np.arange(camera.width, dtype=np.float32) - camera.cx) / camera.fx
    vs = (np.arange(camera.height, dtype=np.float32) - camera.cy) / camera.fy
    return (row[0] * us[None, :] + row[1] * vs[:, None] + row[2]) < 0.0


def segment_non_ground(
    image: np.ndarray,
    *,
    region: Optional[np.ndarray] = None,
    model: Optional[FloorModel] = None,
    chroma_sigma: float = 3.5,
    chroma_scale_floor: float = 2.0,
    bright_luma_margin: float = 40.0,
    min_blob_area_px: float = 200.0,
    downsample: int = 2,
) -> Tuple[np.ndarray, float, Optional[FloorModel]]:
    """Label the pixels that are *not* the arena floor.

    The floor is not described by a hardcoded colour - that would be a prior
    about the world, and it would not survive a change of lighting or of
    simulator.  It is identified as the **dominant surface**: the per-frame
    median of the Lab chroma channels is taken as the floor's colour and the
    MAD as its natural spread, so a pixel is non-ground when its chroma sits
    ``chroma_sigma`` robust deviations away from that median.

    Working in chroma rather than in brightness is what makes shadows harmless:
    the sun and the ambient term are both white, so a shadowed patch of floor
    keeps the floor's hue and only loses lightness.  A second, deliberately
    one-sided luminance test catches achromatic obstacles that chroma cannot see
    - a white QR plate is far *brighter* than the floor - while never firing on
    a shadow, which is darker.

    ``region`` restricts both the statistics and the answer to the part of the
    frame that could contain floor at all - see :func:`below_horizon_mask`.
    ``model`` supplies a floor reference measured earlier; without one the
    reference is taken from this frame, which is only sound while the floor is
    the majority of it.

    Returns the full-resolution boolean mask, the fraction of the region it
    covers, and the floor model *measured from this frame* - which the caller
    can carry forward whether or not it was the one used here.  That model is
    ``None`` when this frame had nothing to measure: a frame with no usable
    region says nothing about the floor's colour, and inventing a placeholder
    for it poisons every frame that follows.
    """
    if image is None or image.size == 0:
        raise ValueError("cannot segment an empty image")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected a BGR image, got shape {image.shape}")
    step = max(1, int(downsample))
    height, width = image.shape[:2]

    small = image
    if step > 1:
        small = cv2.resize(
            image, (max(1, width // step), max(1, height // step)), interpolation=cv2.INTER_AREA
        )
    small_region: Optional[np.ndarray] = None
    if region is not None:
        if region.shape[:2] != (height, width):
            raise ValueError(
                f"region shape {region.shape[:2]} does not match image {(height, width)}"
            )
        small_region = region
        if step > 1:
            small_region = cv2.resize(
                region.astype(np.uint8),
                (small.shape[1], small.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if not small_region.any():
            # No pixel in this frame can be floor, so there is nothing to
            # measure and nothing to say. Returning a fabricated model here is
            # how a drone still sitting on its pad - camera below the ground
            # plane, region empty - once defined the floor as pure black and
            # then classified the entire arena as obstacle for the rest of the
            # flight.
            return np.zeros((height, width), dtype=bool), 0.0, None

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    luma = lab[:, :, 0].astype(np.float32)
    chroma = lab[:, :, 1:].astype(np.float32)

    sample = (chroma if small_region is None else chroma[small_region]).reshape(-1, 2)
    luma_sample = luma if small_region is None else luma[small_region]
    median = np.median(sample, axis=0)
    # A perfectly uniform floor has a MAD of zero, which would make every pixel
    # infinitely deviant. The floor on the scale is what keeps the test finite
    # and is expressed in Lab units, so it means the same thing on any frame.
    scale = np.maximum(
        _MAD_TO_SIGMA * np.median(np.abs(sample - median), axis=0), float(chroma_scale_floor)
    )
    measured = FloorModel(median, scale, float(np.median(luma_sample)))

    reference = model or measured
    mask = np.max(np.abs(chroma - reference.chroma_median) / reference.chroma_scale, axis=2) > float(
        chroma_sigma
    )
    mask |= (luma - reference.luma_median) > float(bright_luma_margin)
    if small_region is not None:
        mask &= small_region

    mask = mask.astype(np.uint8)
    # Opening removes single-pixel speckle from texture and compression noise
    # before the area filter, which would otherwise have to be far coarser.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    min_area = max(1.0, float(min_blob_area_px) / float(step * step))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count > 1:
        keep = np.zeros(count, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        mask = keep[labels]
    else:
        mask = mask.astype(bool)

    considered = mask.size if small_region is None else int(np.count_nonzero(small_region))
    fraction = float(np.count_nonzero(mask)) / float(max(considered, 1))
    if step > 1:
        mask = cv2.resize(
            mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    return mask, fraction, measured


def _intersect_ground(
    pixels: np.ndarray,
    camera: PinholeCamera,
    camera_pose_map: Transform,
    ground_z: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Intersect the rays through ``pixels`` ``(N, 2)`` with the ground plane.

    Returns the ``(N, 3)`` intersection points and a validity mask; a ray that
    points up, or along the plane, has no usable intersection.
    """
    pixels = np.atleast_2d(np.asarray(pixels, dtype=float))
    directions = np.column_stack(
        (
            (pixels[:, 0] - camera.cx) / camera.fx,
            (pixels[:, 1] - camera.cy) / camera.fy,
            np.ones(len(pixels)),
        )
    )
    directions = directions @ camera_pose_map.rotation.T
    origin = camera_pose_map.translation
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = (float(ground_z) - origin[2]) / directions[:, 2]
    valid = (directions[:, 2] < -1e-9) & np.isfinite(distance) & (distance > 0.0)
    points = origin + directions * np.where(valid, distance, 0.0)[:, None]
    return points, valid


def _run_tops(mask_columns: np.ndarray, bottoms: np.ndarray) -> np.ndarray:
    """First row of the non-ground run that ends at ``bottoms`` in each column."""
    height = mask_columns.shape[0]
    rows = np.arange(height, dtype=np.int32)[:, None]
    # Highest row index at or below each row that is *ground*; the run therefore
    # starts one row below it.  Accumulating is far cheaper than walking every
    # column, and the -1 fill makes a run that reaches the top of the image
    # start at row 0.
    last_ground = np.maximum.accumulate(np.where(mask_columns, -1, rows), axis=0)
    return last_ground[bottoms, np.arange(mask_columns.shape[1])] + 1


class MonocularObstacleDetector:
    """Turns one camera frame into ground-contact and free-space evidence.

    One instance serves one camera.  The drone's frames build the shared
    occupancy grid; the rover's frames are the local safety sensor, which is
    why ``forward_half_angle_rad`` and the nearest-range output exist.
    """

    def __init__(
        self,
        *,
        ground_z: float = 0.0,
        max_range_m: float = 10.0,
        min_obstacle_height_m: float = 0.25,
        max_obstacle_height_m: float = 8.0,
        column_stride: int = 8,
        free_stride: int = 16,
        forward_half_angle_rad: float = 0.35,
        chroma_sigma: float = 3.5,
        bright_luma_margin: float = 40.0,
        min_blob_area_px: float = 200.0,
        downsample: int = 2,
        max_non_ground_fraction: float = 0.35,
        model_blend: float = 0.15,
        stale_model_fraction: float = 0.9,
        stale_model_frames: int = 8,
    ) -> None:
        if max_range_m <= 0.0:
            raise ValueError("max_range_m must be positive")
        if column_stride < 1 or free_stride < 1:
            raise ValueError("pixel strides must be at least 1")
        if min_obstacle_height_m < 0.0:
            raise ValueError("min_obstacle_height_m must not be negative")
        if max_obstacle_height_m <= min_obstacle_height_m:
            raise ValueError("max_obstacle_height_m must exceed min_obstacle_height_m")
        self.ground_z = float(ground_z)
        self.max_range_m = float(max_range_m)
        self.min_obstacle_height_m = float(min_obstacle_height_m)
        self.max_obstacle_height_m = float(max_obstacle_height_m)
        self.column_stride = int(column_stride)
        self.free_stride = int(free_stride)
        self.forward_half_angle_rad = float(forward_half_angle_rad)
        self.chroma_sigma = float(chroma_sigma)
        self.bright_luma_margin = float(bright_luma_margin)
        self.min_blob_area_px = float(min_blob_area_px)
        self.downsample = int(downsample)
        self.max_non_ground_fraction = float(max_non_ground_fraction)
        self.model_blend = float(model_blend)
        self.stale_model_fraction = float(stale_model_fraction)
        self.stale_model_frames = int(stale_model_frames)
        #: Floor reference, learned from frames the floor demonstrably owns.
        self.floor_model: Optional[FloorModel] = None
        self._starved_frames = 0
        #: How many times the reference was discarded as stale. A healthy run
        #: leaves this at zero; anything else is worth a look in the logs.
        self.model_resets = 0

    # -- segmentation ------------------------------------------------------
    def segment(
        self, image: np.ndarray, region: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """Segment one frame, maintaining the floor reference as it goes.

        The reference is only updated from frames in which the floor is
        comfortably dominant, so a vehicle that drives up to a wall keeps
        measuring against the floor it came from instead of adopting the wall
        as the new definition of "background".

        A reference that is never allowed to change is a reference that can be
        wrong forever, so there is one way back: when frame after frame reports
        that virtually nothing is floor, the more likely explanation is a bad
        reference than a world made entirely of obstacle, and it is discarded.
        """
        mask, fraction, measured = segment_non_ground(
            image,
            region=region,
            model=self.floor_model,
            chroma_sigma=self.chroma_sigma,
            bright_luma_margin=self.bright_luma_margin,
            min_blob_area_px=self.min_blob_area_px,
            downsample=self.downsample,
        )
        if measured is not None and fraction <= self.max_non_ground_fraction:
            self.floor_model = (
                measured
                if self.floor_model is None
                else self.floor_model.blended_with(measured, self.model_blend)
            )
            self._starved_frames = 0
            return mask, fraction

        if self.floor_model is not None and fraction >= self.stale_model_fraction:
            self._starved_frames += 1
            if self._starved_frames >= self.stale_model_frames:
                self.floor_model = None
                self._starved_frames = 0
                self.model_resets += 1
                # This frame was read against the reference now being thrown
                # away, so its mask is not evidence of anything.
                return np.zeros(mask.shape, dtype=bool), fraction
        elif fraction < self.stale_model_fraction:
            self._starved_frames = 0
        return mask, fraction

    # -- projection --------------------------------------------------------
    def project(
        self,
        mask: np.ndarray,
        camera: PinholeCamera,
        camera_pose_map: Transform,
        *,
        exclude_centres_xy: Optional[np.ndarray] = None,
        exclude_radius_m: float = 0.0,
    ) -> GroundObservation:
        """Lift a segmentation mask into ``map``-frame evidence.

        ``exclude_centres_xy`` removes contacts within ``exclude_radius_m`` of
        the given positions.  That is the self-filter: the drone flies escort
        behind the rover, so the rover is a large, correctly detected,
        completely unwanted obstacle sitting exactly on the route it is driving.
        """
        if mask.shape[:2] != (camera.height, camera.width):
            raise ValueError(
                f"mask shape {mask.shape[:2]} does not match camera "
                f"{(camera.height, camera.width)}"
            )
        columns = np.arange(0, camera.width, self.column_stride, dtype=np.int32)
        sub = np.asarray(mask, dtype=bool)[:, columns]
        occupied_columns = sub.any(axis=0)
        contacts = np.zeros((0, 3), dtype=float)
        nearest = math.inf

        if np.any(occupied_columns):
            sub = sub[:, occupied_columns]
            active = columns[occupied_columns]
            bottoms = camera.height - 1 - np.argmax(sub[::-1, :], axis=0)
            tops = _run_tops(sub, bottoms)

            # The silhouette meets the floor at the *lower* edge of the bottom
            # pixel and leaves the obstacle at the *upper* edge of the top one.
            base_px = np.column_stack((active.astype(float), bottoms + 0.5))
            top_px = np.column_stack((active.astype(float), tops - 0.5))
            base_points, base_valid = _intersect_ground(
                base_px, camera, camera_pose_map, self.ground_z
            )
            origin = camera_pose_map.translation
            radial = np.linalg.norm(base_points[:, :2] - origin[:2], axis=1)
            valid = base_valid & (radial <= self.max_range_m) & (radial > 1e-6)

            heights = self._measure_heights(top_px, camera, camera_pose_map, radial)
            valid &= (heights >= self.min_obstacle_height_m) & (
                heights <= self.max_obstacle_height_m
            )
            if exclude_centres_xy is not None and exclude_radius_m > 0.0:
                centres = np.atleast_2d(np.asarray(exclude_centres_xy, dtype=float))[:, :2]
                if centres.size:
                    gap = np.linalg.norm(
                        base_points[:, None, :2] - centres[None, :, :], axis=2
                    )
                    valid &= np.min(gap, axis=1) > float(exclude_radius_m)

            if np.any(valid):
                contacts = np.column_stack(
                    (base_points[valid, 0], base_points[valid, 1], heights[valid])
                )
                nearest = self._nearest_forward(
                    base_points[valid], camera_pose_map, radial[valid]
                )

        free = self._project_free(mask, camera, camera_pose_map)
        return GroundObservation(
            contacts=contacts,
            free=free,
            nearest_forward_range_m=nearest,
            columns_evaluated=int(columns.size),
        )

    def _measure_heights(
        self,
        top_pixels: np.ndarray,
        camera: PinholeCamera,
        camera_pose_map: Transform,
        radial: np.ndarray,
    ) -> np.ndarray:
        """Height of each silhouette, from the ray through its topmost pixel.

        The contact point fixes *where* the obstacle stands, so the ray through
        the top of its silhouette can be followed until it is above that spot;
        the height it has reached there is the obstacle's height.  This is a
        real measurement from one image - it needs no second view and no
        assumption about the obstacle's shape beyond "it stands on the floor".
        """
        directions = np.column_stack(
            (
                (top_pixels[:, 0] - camera.cx) / camera.fx,
                (top_pixels[:, 1] - camera.cy) / camera.fy,
                np.ones(len(top_pixels)),
            )
        ) @ camera_pose_map.rotation.T
        horizontal = np.linalg.norm(directions[:, :2], axis=1)
        origin = camera_pose_map.translation
        with np.errstate(divide="ignore", invalid="ignore"):
            travel = np.where(horizontal > 1e-9, radial / horizontal, np.nan)
        heights = origin[2] + travel * directions[:, 2] - self.ground_z
        return np.where(np.isfinite(heights), heights, -1.0)

    def _nearest_forward(
        self,
        points: np.ndarray,
        camera_pose_map: Transform,
        radial: np.ndarray,
    ) -> float:
        """Closest contact inside the forward wedge, as a horizontal distance."""
        origin = camera_pose_map.translation
        # +Z of an optical frame is the viewing direction; its ground projection
        # is where the vehicle is heading.
        axis = camera_pose_map.rotation[:, 2][:2]
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            return math.inf
        axis = axis / norm
        offsets = points[:, :2] - origin[:2]
        along = offsets @ axis
        across = np.abs(offsets @ np.array([-axis[1], axis[0]]))
        bearing = np.arctan2(across, along)
        wedge = (along > 0.0) & (bearing <= self.forward_half_angle_rad)
        return float(radial[wedge].min()) if np.any(wedge) else math.inf

    def _project_free(
        self, mask: np.ndarray, camera: PinholeCamera, camera_pose_map: Transform
    ) -> np.ndarray:
        """Ground-classified pixels, on a coarse grid, lifted onto the floor."""
        rows = np.arange(0, camera.height, self.free_stride, dtype=np.int32)
        cols = np.arange(0, camera.width, self.free_stride, dtype=np.int32)
        patch = np.asarray(mask, dtype=bool)[np.ix_(rows, cols)]
        grid_rows, grid_cols = np.nonzero(~patch)
        if grid_rows.size == 0:
            return np.zeros((0, 3), dtype=float)
        pixels = np.column_stack(
            (cols[grid_cols].astype(float), rows[grid_rows].astype(float))
        )
        points, valid = _intersect_ground(pixels, camera, camera_pose_map, self.ground_z)
        radial = np.linalg.norm(points[:, :2] - camera_pose_map.translation[:2], axis=1)
        valid &= radial <= self.max_range_m
        return points[valid]

    # -- convenience -------------------------------------------------------
    def process(
        self,
        image: np.ndarray,
        camera: PinholeCamera,
        camera_pose_map: Transform,
        *,
        exclude_centres_xy: Optional[np.ndarray] = None,
        exclude_radius_m: float = 0.0,
    ) -> GroundObservation:
        """Segment ``image`` and project it in one call."""
        mask, fraction = self.segment(
            image, below_horizon_mask(camera, camera_pose_map, self.ground_z)
        )
        # A frame is evidence only once a floor reference exists. Until one
        # does, a view that is not overwhelmingly floor could just as well
        # define the *obstacle* as the background - and nothing in the image
        # says which reading is right, so neither is taken. Once a reference
        # exists, a view full of obstacle is not a broken frame: it is a close
        # obstacle, the most important measurement the sensor ever makes.
        if self.floor_model is None:
            return GroundObservation(
                contacts=np.zeros((0, 3)),
                free=np.zeros((0, 3)),
                non_ground_fraction=fraction,
                usable=False,
            )
        observation = self.project(
            mask,
            camera,
            camera_pose_map,
            exclude_centres_xy=exclude_centres_xy,
            exclude_radius_m=exclude_radius_m,
        )
        return GroundObservation(
            contacts=observation.contacts,
            free=observation.free,
            nearest_forward_range_m=observation.nearest_forward_range_m,
            columns_evaluated=observation.columns_evaluated,
            non_ground_fraction=fraction,
            usable=True,
            floor_model=self.floor_model,
        )
