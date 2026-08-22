"""Ground-station telemetry: the mission as the digital twin expects to hear it.

The Simurgh ground station already speaks a complete mission dialect over UDP
- phases, route waypoints, obstacle/target/voxel *deltas*, decoded QR content,
imagery - so nothing in it needs changing to display this mission.  What is
needed is a translator, and this module is its ROS-free half: it turns the
mission's own state into the station's ``DigitalTwinMessageV1`` dictionaries.

Three things the wire format forces on us, each of which is a design decision
rather than a formatting detail:

* **Geography.** The mission plans in ``map`` metres (ENU); the station plots on
  a Mapbox globe and wants degrees. One anchor converts between them, and it is
  configuration - the arena's real position - not a constant.
* **Deltas, not snapshots.** A 110 x 110 occupancy grid is 12 100 cells. Sending
  it whole at the world model's rate would be megabytes a second down a link
  the field station shares with video. Only what *changed* is sent, which is
  also exactly how the station's mission engine wants to receive it: it applies
  ``upsert`` operations to what it already holds.
* **Datagram limits.** One JSON object per UDP packet, and a packet that
  exceeds the path MTU is fragmented or dropped. Voxel deltas are therefore
  emitted in batches instead of all at once.

Vehicle trails are deliberately *not* sent: the station draws them itself from
successive pose messages, so a trail topic would duplicate state it already
maintains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .mission_state import MissionState

SCHEMA_VERSION = "1.0"

#: Phases the station's mission engine knows (TwinOperationPhases).
PHASE_SCAN = "scan"
PHASE_JOINT_OPERATION = "joint_operation"
PHASE_DYNAMIC_REPLAN = "dynamic_replan"
PHASE_COMPLETE = "complete"

VEHICLE_UAV = "uav"
VEHICLE_ROVER = "rover"

#: WGS-84 metres per degree of latitude, at the equator-ish approximation the
#: station's own feeder uses (Tools/SlamFeeder/trajectory.py). Good to a few
#: parts in a thousand over an arena tens of metres across.
_M_PER_DEG_LAT = 111_320.0

_PHASE_BY_STATE = {
    MissionState.IDLE: PHASE_SCAN,
    MissionState.TAKEOFF: PHASE_SCAN,
    MissionState.EXPLORING: PHASE_SCAN,
    MissionState.TARGET_FOUND: PHASE_JOINT_OPERATION,
    MissionState.PLANNING: PHASE_JOINT_OPERATION,
    MissionState.PATH_READY: PHASE_JOINT_OPERATION,
    MissionState.SENDING_PATH: PHASE_JOINT_OPERATION,
    MissionState.ROVER_NAVIGATING: PHASE_JOINT_OPERATION,
    MissionState.VERIFYING_TARGET: PHASE_JOINT_OPERATION,
    MissionState.RETURNING_HOME: PHASE_JOINT_OPERATION,
    MissionState.MISSION_SUCCESS: PHASE_COMPLETE,
    MissionState.MISSION_FAILED: PHASE_COMPLETE,
}


def mission_phase(state: MissionState, *, replanning: bool = False) -> str:
    """Map a mission state onto the station's four operation phases.

    ``replanning`` wins over the state, because a replan is a *reason* the
    rover is navigating rather than a state of its own - the station colours
    the whole panel differently for it, which is the point.
    """
    if replanning and state not in (
        MissionState.MISSION_SUCCESS,
        MissionState.MISSION_FAILED,
    ):
        return PHASE_DYNAMIC_REPLAN
    return _PHASE_BY_STATE.get(state, PHASE_SCAN)


@dataclass(frozen=True)
class GeoAnchor:
    """Where the ``map`` frame's origin sits on the Earth.

    Everything the mission computes is in local ENU metres; the station plots
    on a map. This is the single place the two meet, and it is the only piece
    of the bridge that has to be told something about the physical site.
    """

    latitude: float = 0.0
    longitude: float = 0.0
    #: Height of the ``map`` origin above sea level. Only shifts what the
    #: station displays as altitude; nothing in the mission depends on it.
    altitude_m: float = 0.0

    def to_geo(self, east_m: float, north_m: float) -> Tuple[float, float]:
        latitude = self.latitude + float(north_m) / _M_PER_DEG_LAT
        longitude = self.longitude + float(east_m) / (
            _M_PER_DEG_LAT * math.cos(math.radians(self.latitude))
        )
        return latitude, longitude

    def point(self, xy: Sequence[float]) -> Dict[str, float]:
        latitude, longitude = self.to_geo(float(xy[0]), float(xy[1]))
        return {"latitude": round(latitude, 9), "longitude": round(longitude, 9)}


def _yaw_deg(yaw_rad: float) -> float:
    """ENU yaw (CCW from east) -> compass heading in degrees, as the station shows it."""
    return float((90.0 - math.degrees(float(yaw_rad))) % 360.0)


def pose_block(anchor: GeoAnchor, xy: Sequence[float], altitude_m: float, yaw_rad: float) -> Dict:
    block = anchor.point(xy)
    block.update(
        {
            "altitudeM": round(float(altitude_m) + anchor.altitude_m, 3),
            "yawDeg": round(_yaw_deg(yaw_rad), 2),
            "pitchDeg": 0.0,
            "rollDeg": 0.0,
        }
    )
    return block


@dataclass
class TwinDeltaTracker:
    """Remembers what the station has already been told.

    The station holds the map; this holds a mirror of it. Anything that has not
    moved further than a tolerance, or changed state, is simply not resent -
    which is what keeps a full arena scan inside a field radio's budget.
    """

    #: How far a target or obstacle must move before it is worth resending.
    position_tolerance_m: float = 0.15
    _targets: Dict[str, Tuple] = field(default_factory=dict, repr=False)
    _obstacles: Dict[str, Tuple] = field(default_factory=dict, repr=False)
    _voxels: Dict[Tuple[int, int], float] = field(default_factory=dict, repr=False)
    _route: Optional[Tuple] = field(default=None, repr=False)

    def reset(self) -> None:
        self._targets.clear()
        self._obstacles.clear()
        self._voxels.clear()
        self._route = None

    # -- targets ---------------------------------------------------------
    def target_deltas(self, anchor: GeoAnchor, records: Iterable) -> List[Dict]:
        """``TwinTargetDelta`` entries for stations whose estimate has changed.

        A station appears in the ground station's list the moment perception
        confirms it - which is exactly the "targets appear as the drone scans"
        behaviour the operator sees, and it needs no extra machinery here.
        """
        deltas: List[Dict] = []
        for record in records:
            position = np.asarray(record.position, dtype=float)[:2]
            payload = str(record.qr_id)
            reached = bool(getattr(record, "reached", False))
            signature = (
                round(float(position[0]), 3),
                round(float(position[1]), 3),
                reached,
                round(float(record.confidence), 2),
                str(record.status.value if hasattr(record.status, "value") else record.status),
            )
            previous = self._targets.get(payload)
            if previous is not None and _close_enough(
                previous, signature, self.position_tolerance_m
            ):
                continue
            self._targets[payload] = signature
            delta = anchor.point(position)
            delta.update(
                {
                    "id": payload,
                    "operation": "upsert",
                    "kind": "qrcode",
                    "reached": reached,
                    "confidence": round(float(record.confidence), 3),
                    # The station shows this verbatim as the referee's evidence.
                    "decodedContent": payload,
                }
            )
            deltas.append(delta)
        return deltas

    # -- obstacles -------------------------------------------------------
    def obstacle_deltas(self, anchor: GeoAnchor, records: Iterable) -> List[Dict]:
        """Obstacle upserts, plus removals for footprints that no longer exist.

        Obstacle identity comes from connected-component *order*, so it is not
        stable while the map is still growing: two cells that were separate
        blobs become one, and every id after them shifts. In an upsert-only
        stream that leaves the operator's map accumulating obstacles that the
        robot no longer believes in - one arena scan produced thirteen where
        the world model held seven. The station supports ``remove``; anything
        that has gone is said to have gone.
        """
        records = list(records)
        deltas: List[Dict] = []
        live = {str(record.obstacle_id) for record in records}
        for stale in sorted(set(self._obstacles) - live):
            del self._obstacles[stale]
            deltas.append({"id": stale, "operation": "remove"})
        for record in records:
            centre = np.asarray(record.centre, dtype=float)[:2]
            radius = float(record.radius)
            identifier = str(record.obstacle_id)
            signature = (
                round(float(centre[0]), 3),
                round(float(centre[1]), 3),
                False,
                round(radius, 2),
                "",
            )
            previous = self._obstacles.get(identifier)
            if previous is not None and _close_enough(
                previous, signature, self.position_tolerance_m
            ):
                continue
            self._obstacles[identifier] = signature
            delta = anchor.point(centre)
            delta.update(
                {
                    "id": identifier,
                    "operation": "upsert",
                    # Everything in this arena is static; a moving obstacle
                    # would come from the rover's runtime evidence instead.
                    "kind": "static",
                    "radiusM": round(radius, 3),
                    "severity": 1.0,
                }
            )
            deltas.append(delta)
        return deltas

    # -- occupancy -------------------------------------------------------
    def voxel_deltas(self, anchor: GeoAnchor, grid, *, batch: int = 200) -> List[Dict]:
        """The change in the occupied set since the last call.

        Only cells that have *become* occupied are sent, so the station's 3D
        view fills in as the drone flies rather than being redrawn from scratch
        several times a second - and cells the mapper has since retracted are
        removed, because evidence that failed the hit-ratio test must not stay
        on an operator's map as a solid block.
        """
        if grid is None:
            return []
        from .occupancy import OCCUPIED

        occupied = {
            (int(row), int(col)) for row, col in np.argwhere(grid.data >= OCCUPIED)
        }
        deltas: List[Dict] = []
        for key in sorted(set(self._voxels) - occupied):
            del self._voxels[key]
            deltas.append({"id": _voxel_id(key), "operation": "remove"})
            if len(deltas) >= batch:
                return deltas

        for key in sorted(occupied - set(self._voxels)):
            world = grid.cell_to_world(key)
            self._voxels[key] = 1.0
            delta = anchor.point(world)
            delta.update(
                {
                    "id": _voxel_id(key),
                    "operation": "upsert",
                    "altitudeM": round(anchor.altitude_m, 3),
                    "sizeM": round(float(grid.metadata.resolution), 3),
                    "occupancy": 1.0,
                }
            )
            deltas.append(delta)
            if len(deltas) >= batch:
                break
        return deltas

    # -- route -----------------------------------------------------------
    def route_block(self, anchor: GeoAnchor, path_xy, *, altitude_m: float = 0.0) -> Optional[Dict]:
        """The rover's planned route, resent only when the plan itself changes."""
        if path_xy is None or len(path_xy) == 0:
            return None
        points = np.asarray(path_xy, dtype=float)[:, :2]
        signature = tuple(np.round(points.reshape(-1), 3))
        if signature == self._route:
            return None
        self._route = signature
        waypoints = []
        for index, point in enumerate(points):
            waypoint = anchor.point(point)
            waypoint.update(
                {
                    "index": index,
                    "operation": "upsert",
                    "altitudeM": round(float(altitude_m) + anchor.altitude_m, 3),
                }
            )
            waypoints.append(waypoint)
        return {"waypoints": waypoints}


def _voxel_id(cell: Tuple[int, int]) -> str:
    return f"vox-{cell[0]}-{cell[1]}"


def _close_enough(previous: Tuple, current: Tuple, tolerance_m: float) -> bool:
    if previous[2:] != current[2:]:
        return False
    moved = math.hypot(current[0] - previous[0], current[1] - previous[1])
    return moved <= tolerance_m


@dataclass
class TwinMessageBuilder:
    """Assembles ``DigitalTwinMessageV1`` payloads for one mission."""

    anchor: GeoAnchor
    source_id: str = "gazebo-mission"
    auth_token: str = ""
    tracker: TwinDeltaTracker = field(default_factory=TwinDeltaTracker)
    voxel_batch: int = 200
    _sequence: int = field(default=0, repr=False)

    def next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def build(
        self,
        *,
        timestamp_ms: int,
        vehicle: str,
        xy: Sequence[float],
        altitude_m: float,
        yaw_rad: float,
        speed_mps: float,
        state: MissionState,
        replanning: bool = False,
        status: str = "running",
        warning: str = "",
        note: str = "",
        waypoint_index: int = 0,
        targets: Optional[Iterable] = None,
        obstacles: Optional[Iterable] = None,
        grid=None,
        path_xy=None,
        ack_requested: bool = False,
    ) -> Dict:
        """One UDP message: this vehicle's pose plus whatever else is new."""
        phase = mission_phase(state, replanning=replanning)
        message: Dict = {
            "schemaVersion": SCHEMA_VERSION,
            "sequenceId": self.next_sequence(),
            "timestampMs": int(timestamp_ms),
            "sourceId": self.source_id,
            "vehicleType": vehicle,
            "missionPhase": phase,
            "pose": pose_block(self.anchor, xy, altitude_m, yaw_rad),
            "telemetry": {
                "altitudeM": round(float(altitude_m), 3),
                "speedMps": round(float(speed_mps), 3),
                "mode": "AUTO",
                "waypointIndex": int(waypoint_index),
                # Mesh radio and battery are not modelled. -1 is the schema's
                # own "no data" marker, and it is the honest answer: reporting
                # a cheerful 100% would be a fabricated telemetry field on an
                # operator's screen.
                "batteryPercent": -1.0,
                "batteryVoltage": -1.0,
            },
            "mission": {
                "phase": phase,
                "status": status,
                "activeVehicle": vehicle,
                "warning": warning,
                "note": note,
            },
            "routeMode": "replace",
            "ackRequested": bool(ack_requested),
        }
        if self.auth_token:
            message["authToken"] = self.auth_token

        # ``None`` means "this message carries no world-model update", which is
        # not the same as "the world model is empty". Both vehicles publish
        # poses, only one carries the map; treating the rover's silence as an
        # empty set had the two messages removing and re-adding every obstacle
        # in turn, and whichever arrived last decided what the operator saw.
        if targets is not None:
            targets_delta = self.tracker.target_deltas(self.anchor, targets)
            if targets_delta:
                message["targets"] = targets_delta
        if obstacles is not None:
            obstacles_delta = self.tracker.obstacle_deltas(self.anchor, obstacles)
            if obstacles_delta:
                message["obstacles"] = obstacles_delta
        voxels_delta = self.tracker.voxel_deltas(self.anchor, grid, batch=self.voxel_batch)
        if voxels_delta:
            message["voxelCells"] = voxels_delta
        route = self.tracker.route_block(self.anchor, path_xy)
        if route is not None:
            message["route"] = route
            message["replaceRoute"] = True
        return message


def qr_photo_message(
    *,
    timestamp_ms: int,
    source_id: str,
    target_id: str,
    image_base64: str,
    auth_token: str = "",
    label: str = "",
) -> Dict:
    """A rover close-up of a decoded code, for the station's QR gallery.

    Sent as its own message because it is large and rare: attaching a JPEG to
    the pose stream would push every pose packet past the datagram limit.
    """
    message: Dict = {
        "schemaVersion": SCHEMA_VERSION,
        "timestampMs": int(timestamp_ms),
        "sourceId": source_id,
        "vehicleType": VEHICLE_ROVER,
        "imagery": {
            "pipeline": "custom",
            "mode": "single_mosaic",
            "label": label or f"{target_id} verified by the rover",
            "imageBase64": image_base64,
            "targetId": target_id,
            "overlayAlpha": 1.0,
        },
    }
    if auth_token:
        message["authToken"] = auth_token
    return message


def iter_json_datagrams(message: Dict, *, max_bytes: int = 60000) -> Iterator[str]:
    """Serialise one message, splitting it if it would not fit in a datagram.

    Oversize payloads are always the delta arrays, so they are the only thing
    split: the pose and mission blocks are repeated in each part, which the
    station's engine handles because every entry carries its own ``upsert``.
    """
    import json

    encoded = json.dumps(message, separators=(",", ":"))
    if len(encoded.encode("utf-8")) <= max_bytes:
        yield encoded
        return

    bulky = [key for key in ("voxelCells", "obstacles", "targets") if key in message]
    if not bulky:
        # Nothing splittable: send it and let the transport complain rather
        # than silently dropping a message the operator is waiting for.
        yield encoded
        return

    key = bulky[0]
    entries = message[key]
    half = max(1, len(entries) // 2)
    for chunk in (entries[:half], entries[half:]):
        if not chunk:
            continue
        part = dict(message)
        part[key] = chunk
        yield from iter_json_datagrams(part, max_bytes=max_bytes)
