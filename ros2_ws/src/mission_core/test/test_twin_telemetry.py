"""TEST 8 - the ground-station feed matches the station's own contract.

The Simurgh digital twin parses these messages with Unity's ``JsonUtility``
into ``DigitalTwinMessageV1``, which is strict in one direction and silent in
the other: a field it does not know is ignored, and a field it *does* know but
receives in the wrong shape is dropped to a default without complaint.  A
mis-named key therefore does not raise anywhere - it just leaves the operator
looking at an empty map.

These tests pin the message against the contract as the station's own C#
declares it, and against the arithmetic that has to be right for a track to
land in the correct place on a real map.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from mission_core.mission_state import MissionState
from mission_core.occupancy import OCCUPIED, GridMetadata, OccupancyGrid
from mission_core.twin_telemetry import (
    PHASE_COMPLETE,
    PHASE_DYNAMIC_REPLAN,
    PHASE_JOINT_OPERATION,
    PHASE_SCAN,
    GeoAnchor,
    TwinDeltaTracker,
    TwinMessageBuilder,
    iter_json_datagrams,
    mission_phase,
    qr_photo_message,
)


class FakeTarget:
    def __init__(self, qr_id, xy, confidence=0.9, status="CONFIRMED", reached=False):
        self.qr_id = qr_id
        self.position = np.array([xy[0], xy[1], 0.65])
        self.confidence = confidence
        self.status = status
        self.reached = reached


class FakeObstacle:
    def __init__(self, obstacle_id, centre, radius):
        self.obstacle_id = obstacle_id
        self.centre = np.asarray(centre, dtype=float)
        self.radius = radius


@pytest.fixture
def anchor() -> GeoAnchor:
    return GeoAnchor(latitude=39.868, longitude=32.735, altitude_m=850.0)


@pytest.fixture
def builder(anchor) -> TwinMessageBuilder:
    return TwinMessageBuilder(anchor=anchor, source_id="test", auth_token="simurgh-2026")


def build(builder, **overrides):
    defaults = dict(
        timestamp_ms=1_711_804_800_123,
        vehicle="uav",
        xy=(1.0, 2.0),
        altitude_m=4.0,
        yaw_rad=0.0,
        speed_mps=1.6,
        state=MissionState.EXPLORING,
    )
    defaults.update(overrides)
    return builder.build(**defaults)


# ---------------------------------------------------------------------------
# Geography: a track in the wrong place looks exactly like a track
# ---------------------------------------------------------------------------

def test_the_map_origin_maps_to_the_anchor(anchor) -> None:
    latitude, longitude = anchor.to_geo(0.0, 0.0)
    assert latitude == pytest.approx(anchor.latitude)
    assert longitude == pytest.approx(anchor.longitude)


def test_metres_convert_to_degrees_at_the_right_scale(anchor) -> None:
    """100 m north and 100 m east must come back as 100 m, not 100 degrees.

    Longitude shrinks with latitude; at Ankara's 39.9 degrees a degree of
    longitude is about 77 % of a degree of latitude, and getting that factor
    wrong stretches the whole arena sideways on the map.
    """
    north_lat, _ = anchor.to_geo(0.0, 100.0)
    _, east_lon = anchor.to_geo(100.0, 0.0)

    metres_per_deg_lat = 111_320.0
    assert (north_lat - anchor.latitude) * metres_per_deg_lat == pytest.approx(100.0, abs=0.5)
    east_m = (
        (east_lon - anchor.longitude)
        * metres_per_deg_lat
        * math.cos(math.radians(anchor.latitude))
    )
    assert east_m == pytest.approx(100.0, abs=0.5)


def test_enu_yaw_becomes_a_compass_heading(builder) -> None:
    """The station draws a heading arrow; ENU yaw and compass bearing differ."""
    east = build(builder, yaw_rad=0.0)["pose"]["yawDeg"]
    north = build(builder, yaw_rad=math.pi / 2.0)["pose"]["yawDeg"]
    assert east == pytest.approx(90.0)  # +x is east, which is 090 on a compass
    assert north == pytest.approx(0.0)  # +y is north, which is 000


# ---------------------------------------------------------------------------
# The contract, field by field
# ---------------------------------------------------------------------------

def test_the_message_carries_every_field_the_station_reads(builder) -> None:
    message = build(builder)
    assert message["schemaVersion"] == "1.0"
    assert message["authToken"] == "simurgh-2026"
    assert message["vehicleType"] == "uav"
    assert message["missionPhase"] == PHASE_SCAN
    assert set(message["pose"]) == {
        "latitude", "longitude", "altitudeM", "yawDeg", "pitchDeg", "rollDeg"
    }
    assert message["mission"]["activeVehicle"] == "uav"
    assert message["telemetry"]["mode"] == "AUTO"
    # JsonUtility maps missing numbers to 0, so "no battery fitted" has to be
    # said explicitly or the operator sees a flat 0 % and calls an abort.
    assert message["telemetry"]["batteryPercent"] == -1.0


def test_every_mission_state_maps_onto_a_phase_the_station_knows() -> None:
    known = {PHASE_SCAN, PHASE_JOINT_OPERATION, PHASE_DYNAMIC_REPLAN, PHASE_COMPLETE}
    for state in MissionState:
        assert mission_phase(state) in known
    assert mission_phase(MissionState.EXPLORING) == PHASE_SCAN
    assert mission_phase(MissionState.ROVER_NAVIGATING) == PHASE_JOINT_OPERATION
    assert mission_phase(MissionState.MISSION_SUCCESS) == PHASE_COMPLETE
    # A replan is a reason, not a state: the station colours it differently.
    assert mission_phase(MissionState.ROVER_NAVIGATING, replanning=True) == PHASE_DYNAMIC_REPLAN
    # ...but a finished mission stays finished.
    assert mission_phase(MissionState.MISSION_SUCCESS, replanning=True) == PHASE_COMPLETE


def test_a_target_carries_the_decoded_payload_as_evidence(builder) -> None:
    message = build(builder, targets=[FakeTarget("TARGET_2", (7.0, -5.0))])
    target = message["targets"][0]
    assert target["operation"] == "upsert"
    assert target["kind"] == "qrcode"
    assert target["decodedContent"] == "TARGET_2", "the referee's evidence field"
    assert target["reached"] is False


# ---------------------------------------------------------------------------
# Deltas: the station holds the map, this only reports what changed
# ---------------------------------------------------------------------------

def test_unchanged_targets_and_obstacles_are_not_resent(builder) -> None:
    targets = [FakeTarget("TARGET_1", (6.0, 6.0))]
    obstacles = [FakeObstacle("obstacle_00", (0.0, -6.0), 3.0)]

    first = build(builder, targets=targets, obstacles=obstacles)
    assert first["targets"] and first["obstacles"]

    second = build(builder, targets=targets, obstacles=obstacles)
    assert "targets" not in second and "obstacles" not in second

    # A station whose estimate actually moved is news again.
    targets[0].position = np.array([6.4, 6.0, 0.65])
    assert build(builder, targets=targets)["targets"][0]["id"] == "TARGET_1"


def test_a_verified_target_is_resent_because_reached_changed(builder) -> None:
    """The operator's map has to show which stations are done."""
    target = FakeTarget("TARGET_3", (-6.0, 6.0))
    build(builder, targets=[target])
    target.reached = True
    resent = build(builder, targets=[target])
    assert resent["targets"][0]["reached"] is True


def test_only_newly_occupied_cells_are_sent(anchor) -> None:
    """A 110x110 grid is 12 100 cells; the wire carries the difference."""
    tracker = TwinDeltaTracker()
    grid = OccupancyGrid(GridMetadata(0.2, 20, 20, -2.0, -2.0))
    grid.data[5, 5] = OCCUPIED
    first = tracker.voxel_deltas(anchor, grid)
    assert len(first) == 1
    assert first[0]["occupancy"] == 1.0
    assert first[0]["sizeM"] == pytest.approx(0.2)

    assert tracker.voxel_deltas(anchor, grid) == [], "an unchanged grid is not news"

    grid.data[6, 6] = OCCUPIED
    assert len(tracker.voxel_deltas(anchor, grid)) == 1


def test_the_voxel_batch_is_bounded(anchor) -> None:
    tracker = TwinDeltaTracker()
    grid = OccupancyGrid(GridMetadata(0.2, 40, 40, -4.0, -4.0))
    grid.data[:] = OCCUPIED
    assert len(tracker.voxel_deltas(anchor, grid, batch=25)) == 25


def test_a_route_is_sent_once_per_plan(builder) -> None:
    path = np.array([[-8.0, -8.0], [-4.0, -6.0], [5.2, -5.0]])
    message = build(builder, path_xy=path)
    assert len(message["route"]["waypoints"]) == 3
    assert message["route"]["waypoints"][0]["index"] == 0
    assert message["replaceRoute"] is True
    assert "route" not in build(builder, path_xy=path)
    # A replan is a different plan, and the map has to follow it.
    assert "route" in build(builder, path_xy=path[::-1])


# ---------------------------------------------------------------------------
# The wire
# ---------------------------------------------------------------------------

def test_every_message_is_one_json_object(builder) -> None:
    message = build(builder, targets=[FakeTarget("TARGET_1", (6.0, 6.0))])
    datagrams = list(iter_json_datagrams(message))
    assert len(datagrams) == 1
    assert json.loads(datagrams[0])["targets"][0]["id"] == "TARGET_1"


def test_an_oversize_message_is_split_rather_than_dropped(anchor) -> None:
    """One JSON object per datagram, and a datagram has a size limit.

    A full arena scan produces thousands of newly occupied cells; sending them
    as one packet means IP fragmentation, and one lost fragment loses the lot.
    """
    builder = TwinMessageBuilder(anchor=anchor, voxel_batch=4000)
    grid = OccupancyGrid(GridMetadata(0.2, 60, 60, -6.0, -6.0))
    grid.data[:] = OCCUPIED
    message = build(builder, grid=grid)

    datagrams = list(iter_json_datagrams(message, max_bytes=8000))
    assert len(datagrams) > 1
    total_cells = 0
    for datagram in datagrams:
        assert len(datagram.encode("utf-8")) <= 8000
        parsed = json.loads(datagram)
        # Every part stands on its own: the station applies upserts, so a part
        # that arrived without its siblings is still correct, just partial.
        assert parsed["pose"]["latitude"] == pytest.approx(anchor.to_geo(1.0, 2.0)[0])
        total_cells += len(parsed.get("voxelCells", []))
    assert total_cells == len(message["voxelCells"])


def test_the_qr_photo_message_targets_the_gallery() -> None:
    message = qr_photo_message(
        timestamp_ms=1,
        source_id="test",
        target_id="TARGET_2",
        image_base64="Zm9v",
        auth_token="simurgh-2026",
    )
    assert message["imagery"]["targetId"] == "TARGET_2"
    assert message["imagery"]["imageBase64"] == "Zm9v"
    assert message["vehicleType"] == "rover"
    assert "pose" not in message, "a photo is not a pose and must not move the vehicle"


# ---------------------------------------------------------------------------
# Removal: an upsert-only stream is a map that only ever grows
# ---------------------------------------------------------------------------

def test_a_retracted_cell_is_removed_from_the_operators_map(anchor) -> None:
    """Occupancy is evidence, and evidence can be withdrawn.

    A cell that fails the hit-ratio test after further observation goes back to
    free in the robot's map. Without a removal it stays a solid block on the
    station's, and the operator is looking at an obstacle the planner does not
    believe in.
    """
    tracker = TwinDeltaTracker()
    grid = OccupancyGrid(GridMetadata(0.2, 20, 20, -2.0, -2.0))
    grid.data[5, 5] = OCCUPIED
    grid.data[6, 6] = OCCUPIED
    assert len(tracker.voxel_deltas(anchor, grid)) == 2

    grid.data[5, 5] = 0
    deltas = tracker.voxel_deltas(anchor, grid)
    assert [d["operation"] for d in deltas] == ["remove"]
    assert deltas[0]["id"] == "vox-5-5"
    assert tracker.voxel_deltas(anchor, grid) == []


def test_obstacles_that_merged_away_are_removed(anchor) -> None:
    """Connected-component ids shift as the map grows.

    Two blobs that become one leave an id behind, and every later id shifts by
    one. One arena scan sent thirteen obstacle ids for a world model that held
    seven, and an upsert-only station would have drawn all thirteen.
    """
    tracker = TwinDeltaTracker()
    before = [
        FakeObstacle("obstacle_00", (0.0, -6.0), 1.0),
        FakeObstacle("obstacle_01", (2.0, -6.0), 1.0),
    ]
    assert len(tracker.obstacle_deltas(anchor, before)) == 2

    merged = [FakeObstacle("obstacle_00", (1.0, -6.0), 3.0)]
    deltas = tracker.obstacle_deltas(anchor, merged)
    operations = {d["id"]: d["operation"] for d in deltas}
    assert operations["obstacle_01"] == "remove"
    assert operations["obstacle_00"] == "upsert"


def test_a_pose_only_message_does_not_erase_the_map(builder) -> None:
    """Both vehicles send poses; only one carries the world model.

    Regression: the rover's message defaulted to an empty target and obstacle
    list, the tracker read that as "everything is gone", and the two streams
    removed and re-added the whole map in turn at 5 Hz. Whichever message
    arrived last decided what the operator was looking at - which, the rover
    being second, was an arena with nothing in it.
    """
    targets = [FakeTarget("TARGET_1", (6.0, 6.0))]
    obstacles = [FakeObstacle("obstacle_00", (0.0, -6.0), 3.0)]
    build(builder, vehicle="uav", targets=targets, obstacles=obstacles)

    rover = build(builder, vehicle="rover")
    assert "targets" not in rover and "obstacles" not in rover

    # And a genuinely empty world model still says so.
    cleared = build(builder, vehicle="uav", obstacles=[])
    assert [d["operation"] for d in cleared["obstacles"]] == ["remove"]
