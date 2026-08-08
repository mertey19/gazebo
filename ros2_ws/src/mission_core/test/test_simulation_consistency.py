"""Cross-file consistency between the config, the SDF world and the launch file.

Four descriptions of the same physical system exist in this repository:

* ``config/mission.yaml``            - what the algorithms believe
* ``models/*/model.sdf``             - what Gazebo builds
* ``launch/mission.launch.py``       - the TF that connects them
* ``test/offline_mission.py``        - what the offline harness simulates

A silent disagreement between any two of them is the single most damaging class
of bug in this project: a camera mounted 0.1 m from where TF says it is puts
every discovered target 0.1 m off, and nothing in the logs would say so.  These
tests pin all four together.

They are also the closest thing to a Gazebo check that can run without Gazebo,
so they cover the layer the rest of the suite cannot reach.
"""

from __future__ import annotations

import ast
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
import yaml

from mission_core.config import load_mission_config
from mission_core.geometry import (
    R_BODY_TO_FORWARD_OPTICAL,
    R_BODY_TO_NADIR_OPTICAL,
    matrix_to_quaternion,
)

import offline_mission

REPO_ROOT = Path(__file__).resolve().parents[4]
BRINGUP = REPO_ROOT / "ros2_ws" / "src" / "mission_bringup"
CONFIG_PATH = BRINGUP / "config" / "mission.yaml"
WORLD_PATH = BRINGUP / "worlds" / "mission_arena.sdf"
LAUNCH_PATH = BRINGUP / "launch" / "mission.launch.py"
BRIDGE_PATH = BRINGUP / "config" / "gz_bridge.yaml"
CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"

#: Every quantity below is a length in metres; 1 mm is far tighter than any
#: physical tolerance that matters here and catches copy-paste drift.
TOLERANCE = 1e-6


def parse_pose(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """SDF ``<pose>x y z roll pitch yaw</pose>`` -> (translation, rpy)."""
    values = [float(v) for v in text.split()]
    assert len(values) == 6, f"malformed pose: {text!r}"
    return np.array(values[:3]), np.array(values[3:])


@pytest.fixture(scope="module")
def config():
    return load_mission_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def launch_constants() -> Dict[str, object]:
    """Read the module-level constants out of the launch file.

    Parsed with ``ast`` rather than imported: ``launch`` and ``launch_ros`` are
    not installed on a machine that only runs the unit tests.
    """
    tree = ast.parse(LAUNCH_PATH.read_text(encoding="utf-8"))
    constants: Dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id.isupper():
                constants[target.id] = ast.literal_eval(node.value)
    return constants


@pytest.fixture(scope="module")
def sdf_models() -> Dict[str, ET.Element]:
    return {
        name: ET.parse(BRINGUP / "models" / name / "model.sdf").getroot()
        for name in ("scout_drone", "mission_rover")
    }


def find_sensor(model_root: ET.Element, sensor_name: str) -> ET.Element:
    for sensor in model_root.iter("sensor"):
        if sensor.get("name") == sensor_name:
            return sensor
    raise AssertionError(f"sensor {sensor_name!r} not found in the model")


# ---------------------------------------------------------------------------
# Camera intrinsics: SDF vs mission.yaml
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model,sensor,section",
    [("scout_drone", "down_camera", "drone"), ("mission_rover", "front_camera", "rover")],
)
def test_sdf_camera_matches_configuration(sdf_models, config, model, sensor, section) -> None:
    """Wrong intrinsics silently scale every PnP range estimate."""
    camera = find_sensor(sdf_models[model], sensor).find("camera")
    expected = getattr(config, section).camera

    assert float(camera.find("horizontal_fov").text) == pytest.approx(
        expected.horizontal_fov_rad, abs=1e-6
    )
    image = camera.find("image")
    assert int(image.find("width").text) == expected.width
    assert int(image.find("height").text) == expected.height

    sensor = find_sensor(sdf_models[model], sensor)
    assert float(sensor.find("update_rate").text) == pytest.approx(
        expected.update_rate_hz, abs=1e-6
    ), "SDF sensor rate and the configured camera rate must agree"


def test_camera_rate_is_not_wasted_on_frames_perception_throttles_away(config) -> None:
    """Rendering faster than the detector consumes is pure cost.

    Without a GPU (CI, or any headless box) camera rendering dominates the
    simulation's real-time factor, so an over-fast sensor is the difference
    between a mission that finishes and one that times out.
    """
    for section in ("drone", "rover"):
        rate = getattr(config, section).camera.update_rate_hz
        assert rate <= config.perception.qr_detection_rate_hz + 1e-9, (
            f"{section} camera renders at {rate} Hz but perception throttles to "
            f"{config.perception.qr_detection_rate_hz} Hz"
        )


@pytest.mark.parametrize(
    "model,sensor,frame",
    [
        ("scout_drone", "down_camera", "drone/camera_optical_frame"),
        ("mission_rover", "front_camera", "rover/camera_optical_frame"),
    ],
)
def test_camera_publishes_the_frame_the_detector_looks_up(sdf_models, model, sensor, frame) -> None:
    camera_sensor = find_sensor(sdf_models[model], sensor)
    assert camera_sensor.find("gz_frame_id").text == frame
    assert camera_sensor.find("camera").find("optical_frame_id").text == frame


# ---------------------------------------------------------------------------
# Sensor mounting: SDF vs launch TF vs offline harness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model,sensor,launch_key,harness_attr",
    [
        ("scout_drone", "down_camera", "DRONE_CAMERA_XYZ", "DRONE_CAMERA_MOUNT"),
        ("scout_drone", "down_lidar", "DRONE_LIDAR_XYZ", "DRONE_LIDAR_MOUNT"),
        ("mission_rover", "front_camera", "ROVER_CAMERA_XYZ", "ROVER_CAMERA_MOUNT"),
    ],
)
def test_sensor_mount_agrees_everywhere(
    sdf_models, launch_constants, model, sensor, launch_key, harness_attr
) -> None:
    """The SDF pose, the static TF and the harness constant must be identical."""
    sdf_translation, _ = parse_pose(find_sensor(sdf_models[model], sensor).find("pose").text)
    launch_translation = np.array([float(v) for v in launch_constants[launch_key]])
    harness_translation = np.array(getattr(offline_mission, harness_attr), dtype=float)

    assert np.allclose(sdf_translation, launch_translation, atol=TOLERANCE), (
        f"{sensor}: SDF {sdf_translation} vs launch {launch_translation}"
    )
    assert np.allclose(sdf_translation, harness_translation, atol=TOLERANCE), (
        f"{sensor}: SDF {sdf_translation} vs harness {harness_translation}"
    )


def test_launch_optical_quaternions_are_the_derived_ones(launch_constants) -> None:
    """The hand-entered TF quaternions must equal the code's rotation matrices."""
    for key, matrix in (
        ("NADIR_OPTICAL_QUAT", R_BODY_TO_NADIR_OPTICAL),
        ("FORWARD_OPTICAL_QUAT", R_BODY_TO_FORWARD_OPTICAL),
    ):
        launch_quat = np.array([float(v) for v in launch_constants[key]])
        derived = matrix_to_quaternion(matrix)
        # q and -q are the same rotation, so compare up to sign.
        same = np.allclose(launch_quat, derived, atol=1e-9) or np.allclose(
            launch_quat, -derived, atol=1e-9
        )
        assert same, f"{key}: launch has {launch_quat}, geometry.py derives {derived}"


def test_drone_lidar_is_mounted_level(sdf_models) -> None:
    """A rotated lidar frame is an easy way to smear the whole occupancy grid.

    The design aims the lidar downwards through its vertical scan range so the
    base_link -> lidar_link transform stays a pure translation, matching the
    identity rotation the launch file publishes.
    """
    lidar = find_sensor(sdf_models["scout_drone"], "down_lidar")
    _, rpy = parse_pose(lidar.find("pose").text)
    assert np.allclose(rpy, 0.0, atol=TOLERANCE), f"drone lidar is rotated: rpy={rpy}"

    vertical = lidar.find("lidar").find("scan").find("vertical")
    min_angle = float(vertical.find("min_angle").text)
    max_angle = float(vertical.find("max_angle").text)
    assert min_angle < max_angle <= 0.0, "the lidar cone must point downwards"
    assert min_angle >= -np.pi / 2 - 1e-6


def test_drone_lidar_cone_covers_the_lane_spacing(sdf_models, config) -> None:
    """Consecutive lawnmower lanes must overlap in the lidar swath too."""
    lidar = find_sensor(sdf_models["scout_drone"], "down_lidar")
    max_angle = float(lidar.find("lidar").find("scan").find("vertical").find("max_angle").text)
    # Elevation is measured from horizontal, so the ground radius is
    # altitude / tan(depression angle).
    depression = abs(max_angle)
    radius = config.drone.scan_altitude_m / np.tan(depression)
    assert 2.0 * radius > config.drone.lane_spacing_m, (
        f"lidar swath {2 * radius:.2f} m does not cover "
        f"{config.drone.lane_spacing_m:.2f} m lanes"
    )


# ---------------------------------------------------------------------------
# Station models: generated SDF vs configuration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", ["TARGET_1", "TARGET_2", "TARGET_3"])
def test_station_model_exists_with_a_unique_texture(payload: str) -> None:
    import cv2

    model_dir = BRINGUP / "models" / f"target_station_{payload.lower()}"
    assert (model_dir / "model.sdf").is_file()
    assert (model_dir / "meshes" / "plate.obj").is_file()
    assert (model_dir / "meshes" / "plate.mtl").is_file()

    textures = list((model_dir / "materials" / "textures").glob("*.png"))
    assert len(textures) == 1, f"expected exactly one texture, found {textures}"
    image = cv2.imread(str(textures[0]), cv2.IMREAD_GRAYSCALE)
    ok, decoded, _, _ = cv2.QRCodeDetector().detectAndDecodeMulti(image)
    assert ok and list(decoded) == [payload], (
        f"{textures[0]} does not decode to {payload}: "
        f"{decoded if ok else 'no code found'}"
    )


def test_station_materials_and_textures_have_unique_names() -> None:
    """Ogre caches materials and textures by name, not by path.

    Regression, found only by running in Gazebo: all three stations shipped a
    material called ``qr_material`` and a texture called ``qr.png``, so
    whichever loaded first was displayed on all three. Every station showed the
    same code and the world model - correctly - refused to act on it with
    DUPLICATE_QR. Names must be unique, not merely the directories holding them.
    """
    materials: Dict[str, str] = {}
    texture_names: Dict[str, str] = {}
    object_names: Dict[str, str] = {}

    for payload in ("TARGET_1", "TARGET_2", "TARGET_3"):
        model_dir = BRINGUP / "models" / f"target_station_{payload.lower()}"
        mtl = (model_dir / "meshes" / "plate.mtl").read_text(encoding="utf-8")
        obj = (model_dir / "meshes" / "plate.obj").read_text(encoding="utf-8")

        declared = re.findall(r"^newmtl\s+(\S+)", mtl, re.MULTILINE)
        used = re.findall(r"^usemtl\s+(\S+)", obj, re.MULTILINE)
        maps = re.findall(r"^map_Kd\s+(\S+)", mtl, re.MULTILINE)
        objects = re.findall(r"^o\s+(\S+)", obj, re.MULTILINE)

        assert len(declared) == 1 and len(used) == 1 and len(maps) == 1
        assert used[0] == declared[0], (
            f"{payload}: the OBJ uses {used[0]!r} but the MTL declares {declared[0]!r}"
        )
        assert (model_dir / "meshes" / maps[0]).resolve().is_file(), (
            f"{payload}: map_Kd points at {maps[0]}, which does not exist"
        )
        materials[payload] = declared[0]
        texture_names[payload] = Path(maps[0]).name
        object_names[payload] = objects[0] if objects else ""

    for label, mapping in (
        ("material", materials),
        ("texture filename", texture_names),
        ("OBJ object", object_names),
    ):
        values = list(mapping.values())
        assert len(set(values)) == len(values), (
            f"stations share a {label}: {mapping} - Ogre will show one code on all of them"
        )


def test_plate_mesh_winding_is_not_mirrored() -> None:
    """The Gazebo plate must satisfy the same handedness rule as the harness.

    UV (0,0) is the image bottom-left in OBJ, so the vertex assigned to it must
    be the quad's bottom-left for texture-right x texture-up to equal +normal.
    """
    obj = (BRINGUP / "models" / "target_station_target_1" / "meshes" / "plate.obj").read_text(
        encoding="utf-8"
    )
    vertices = [
        [float(v) for v in line.split()[1:4]]
        for line in obj.splitlines()
        if line.startswith("v ")
    ]
    uvs = [
        [float(v) for v in line.split()[1:3]]
        for line in obj.splitlines()
        if line.startswith("vt ")
    ]
    face = re.search(r"^f\s+(.*)$", obj, re.MULTILINE)
    assert face is not None
    pairs = [tuple(int(i) - 1 for i in token.split("/")[:2]) for token in face.group(1).split()]

    corner = {tuple(uvs[uv]): np.asarray(vertices[v]) for v, uv in pairs}
    bottom_left = corner[(0.0, 0.0)]
    bottom_right = corner[(1.0, 0.0)]
    top_left = corner[(0.0, 1.0)]

    right = bottom_right - bottom_left
    up = top_left - bottom_left
    normal = np.cross(right, up)
    normal /= np.linalg.norm(normal)
    assert np.allclose(normal, [0.0, 0.0, 1.0], atol=1e-9), (
        f"plate.obj texture basis is mirrored: right x up = {normal}, expected +Z"
    )


@pytest.mark.parametrize("payload", ["TARGET_1", "TARGET_2", "TARGET_3"])
def test_station_plate_size_matches_configuration(config, payload: str) -> None:
    """The plate scale in the SDF is what PnP assumes the marker measures."""
    root = ET.parse(
        BRINGUP / "models" / f"target_station_{payload.lower()}" / "model.sdf"
    ).getroot()
    scales = [
        [float(v) for v in mesh.find("scale").text.split()]
        for mesh in root.iter("mesh")
    ]
    assert scales, "the station model has no plate meshes"
    for scale in scales:
        assert scale[0] == pytest.approx(config.mission.qr_plate_size_m, abs=1e-4)
        assert scale[1] == pytest.approx(config.mission.qr_plate_size_m, abs=1e-4)


def test_station_has_a_collision_body_so_it_is_mappable(config) -> None:
    """A station the lidar cannot see is a station the planner drives into."""
    root = ET.parse(BRINGUP / "models" / "target_station_target_1" / "model.sdf").getroot()
    boxes = [
        [float(v) for v in box.find("size").text.split()]
        for collision in root.iter("collision")
        for box in collision.iter("box")
    ]
    assert boxes, "the station has no collision geometry"
    tallest = max(box[2] for box in boxes)
    assert tallest > config.world_model.obstacle_min_height_m, (
        f"the station is only {tallest:.2f} m tall; the mapper treats anything below "
        f"{config.world_model.obstacle_min_height_m:.2f} m as ground"
    )


# ---------------------------------------------------------------------------
# World layout: SDF vs offline harness
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def world_root() -> ET.Element:
    return ET.parse(WORLD_PATH).getroot()


def world_includes(world_root: ET.Element) -> Dict[str, np.ndarray]:
    result = {}
    for include in world_root.iter("include"):
        uri = include.find("uri").text.replace("model://", "")
        pose, _ = parse_pose(include.find("pose").text)
        result[uri] = pose
    return result


def world_obstacles(world_root: ET.Element) -> List[Tuple[np.ndarray, np.ndarray]]:
    found = []
    for model in world_root.iter("model"):
        name = model.get("name") or ""
        if not name.startswith("obstacle"):
            continue
        pose, _ = parse_pose(model.find("pose").text)
        box = next(model.iter("box"))
        size = np.array([float(v) for v in box.find("size").text.split()])
        found.append((pose, size))
    return found


def test_world_station_positions_match_the_offline_harness(world_root) -> None:
    """The harness must simulate the arena the SDF actually builds."""
    sdf_positions = {
        uri.replace("target_station_", "").upper(): pose[:2]
        for uri, pose in world_includes(world_root).items()
        if uri.startswith("target_station_")
    }
    harness_positions = offline_mission.default_world().ground_truth_station_xy()

    assert set(sdf_positions) == set(harness_positions), (
        f"SDF has {sorted(sdf_positions)}, harness has {sorted(harness_positions)}"
    )
    for payload, expected in harness_positions.items():
        assert np.allclose(sdf_positions[payload], expected, atol=TOLERANCE), (
            f"{payload}: world {sdf_positions[payload]} vs harness {expected}"
        )


def test_world_obstacles_match_the_offline_harness(world_root) -> None:
    sdf = sorted(
        ((tuple(np.round(p[:2], 6)), tuple(np.round(s[:2], 6))) for p, s in world_obstacles(world_root))
    )
    harness = sorted(
        (
            (tuple(np.round(np.asarray(o.centre)[:2], 6)), tuple(np.round(np.asarray(o.size)[:2], 6)))
            for o in offline_mission.default_world().obstacles
        )
    )
    assert sdf == harness, f"world obstacles {sdf} vs harness {harness}"


def test_world_has_the_required_mission_content(world_root) -> None:
    """Three uniquely coded stations, two obstacles, both vehicles."""
    includes = world_includes(world_root)
    stations = [u for u in includes if u.startswith("target_station_")]
    assert len(stations) == 3 and len(set(stations)) == 3
    assert len(world_obstacles(world_root)) >= 2
    assert "scout_drone" in includes and "mission_rover" in includes


def test_vehicle_spawn_poses_match_the_launch_file(world_root, launch_constants) -> None:
    """map -> odom static transforms are derived from these spawn poses."""
    includes = world_includes(world_root)
    assert np.allclose(
        includes["mission_rover"][:2],
        [float(v) for v in launch_constants["ROVER_SPAWN_XY"]],
        atol=TOLERANCE,
    )
    assert np.allclose(
        includes["scout_drone"][:2],
        [float(v) for v in launch_constants["DRONE_SPAWN_XY"]],
        atol=TOLERANCE,
    )


def test_stations_and_obstacles_are_inside_the_planning_area(world_root, config) -> None:
    minimum = np.array(config.mission.area_min_xy)
    maximum = np.array(config.mission.area_max_xy)
    for uri, pose in world_includes(world_root).items():
        assert np.all(pose[:2] > minimum) and np.all(pose[:2] < maximum), (
            f"{uri} at {pose[:2]} lies outside the mapped area"
        )


def test_the_default_target_actually_requires_avoiding_an_obstacle(world_root, config) -> None:
    """If the straight line were free, the mission would prove nothing."""
    includes = world_includes(world_root)
    start = includes["mission_rover"][:2]
    goal = includes[f"target_station_{config.mission.target_qr.lower()}"][:2]

    hits = False
    for centre, size in world_obstacles(world_root):
        lower = centre[:2] - size[:2] / 2.0
        upper = centre[:2] + size[:2] / 2.0
        for t in np.linspace(0.0, 1.0, 400):
            point = start + t * (goal - start)
            if np.all(point >= lower) and np.all(point <= upper):
                hits = True
                break
    assert hits, (
        f"the straight line from the rover to {config.mission.target_qr} is already clear; "
        "the demo would not exercise obstacle avoidance"
    )


# ---------------------------------------------------------------------------
# Bridge and node wiring
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bridge_entries() -> List[dict]:
    return yaml.safe_load(BRIDGE_PATH.read_text(encoding="utf-8"))


def test_bridge_covers_every_topic_the_nodes_need(bridge_entries) -> None:
    ros_topics = {entry["ros_topic_name"] for entry in bridge_entries}
    required = {
        "/clock",
        "/drone/cmd_vel",
        "/drone/odometry",
        "/drone/scan/points",
        "/drone/camera/camera_info",
        "/rover/cmd_vel",
        "/rover/odometry",
        "/rover/scan",
        "/rover/camera/camera_info",
        "/tf",
    }
    assert required <= ros_topics, f"bridge is missing {sorted(required - ros_topics)}"


def test_bridge_directions_are_sane(bridge_entries) -> None:
    """Only the two command topics may flow into the simulator."""
    into_gz = {e["ros_topic_name"] for e in bridge_entries if e["direction"] == "ROS_TO_GZ"}
    assert into_gz == {"/drone/cmd_vel", "/rover/cmd_vel"}


#: Plugin outputs that are intentionally produced but not bridged. DiffDrive
#: always publishes odometry; the rover's is dead-reckoned and superseded by an
#: OdometryPublisher, so it is routed to a dead topic rather than consumed.
DELIBERATELY_UNBRIDGED = {
    "/model/mission_rover/wheel_odometry",
    "/model/mission_rover/wheel_tf",
}


def test_bridge_gz_topics_match_the_sdf_plugins(bridge_entries, sdf_models) -> None:
    """A renamed plugin topic would leave the bridge quietly connected to nothing."""
    gz_topics = {entry["gz_topic_name"] for entry in bridge_entries} | DELIBERATELY_UNBRIDGED
    for model_name, root in sdf_models.items():
        for plugin in root.iter("plugin"):
            for tag in ("topic", "odom_topic", "tf_topic"):
                element = plugin.find(tag)
                if element is None:
                    continue
                # The lidar/camera sensor topics get a gz suffix; plugin topics
                # are bridged verbatim.
                assert element.text in gz_topics, (
                    f"{model_name}: plugin topic {element.text} is not bridged"
                )


def test_launch_file_starts_every_mission_node() -> None:
    text = LAUNCH_PATH.read_text(encoding="utf-8")
    for executable in (
        "qr_detector_node",
        "world_model_node",
        "drone_explorer_node",
        "rover_path_follower_node",
        "mission_manager_node",
    ):
        assert executable in text, f"{executable} is never launched"
    # Two detector instances: one per camera.
    assert text.count("qr_detector_node") >= 2


def test_gazebo_ci_matrix_repeats_every_configured_target() -> None:
    """A green CI run must cover target-specific behaviour and basic flakiness."""
    workflow = yaml.safe_load(CI_WORKFLOW_PATH.read_text(encoding="utf-8"))
    job = workflow["jobs"]["gazebo-e2e"]
    strategy = job["strategy"]
    matrix = strategy["matrix"]
    config = load_mission_config(CONFIG_PATH)

    assert set(matrix["target"]) == set(config.mission.known_payloads)
    assert matrix["trial"] == [1, 2, 3]
    assert strategy["fail-fast"] is False
    assert strategy["max-parallel"] <= len(matrix["target"])


def test_every_gazebo_matrix_job_is_gated_by_the_mission_verdict() -> None:
    """Pipeline activity alone must never make an E2E matrix leg green."""
    workflow = yaml.safe_load(CI_WORKFLOW_PATH.read_text(encoding="utf-8"))
    job = workflow["jobs"]["gazebo-e2e"]
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    artifact_steps = [step for step in job["steps"] if step.get("uses") == "actions/upload-artifact@v4"]

    assert "await_mission.py --timeout" in commands
    assert "check_pipeline.py" in commands
    assert len(artifact_steps) == 1
    artifact_name = artifact_steps[0]["with"]["name"]
    assert "matrix.target" in artifact_name and "matrix.trial" in artifact_name


def test_setup_py_entry_points_resolve_to_real_modules() -> None:
    setup_text = (REPO_ROOT / "ros2_ws" / "src" / "mission_nodes" / "setup.py").read_text(
        encoding="utf-8"
    )
    nodes_dir = REPO_ROOT / "ros2_ws" / "src" / "mission_nodes" / "mission_nodes"
    for match in re.finditer(r"(\w+) = (mission_nodes\.\w+):(\w+)", setup_text):
        _, module, function = match.groups()
        source = nodes_dir / f"{module.split('.')[-1]}.py"
        assert source.is_file(), f"entry point references missing module {module}"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert function in functions, f"{module} has no {function}()"


def test_nodes_only_use_fields_that_exist_in_the_interfaces() -> None:
    """Catch a message field renamed in the .msg but not in the node."""
    interfaces = REPO_ROOT / "ros2_ws" / "src" / "mission_interfaces"
    fields: Dict[str, set] = {}
    for msg in (interfaces / "msg").glob("*.msg"):
        names = set()
        for line in msg.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if not line or "=" in line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                names.add(parts[1])
        fields[msg.stem] = names

    # Spot-check the messages the nodes populate most heavily.
    assert {"qr_id", "position_map", "normal_map", "confidence", "range_m"} <= fields[
        "QrObservation"
    ]
    assert {"state", "requested_qr", "verified_qr", "failure_reason", "trace"} <= fields[
        "MissionStatus"
    ]
    assert {"phase", "at_scan_altitude", "complete", "coverage_fraction"} <= fields[
        "ExplorationStatus"
    ]
    assert {"goal_reached", "failed", "cross_track_error_m", "detail"} <= fields[
        "TrackingStatus"
    ]
    assert {"targets_confirmed", "map_known_fraction", "qr_ids"} <= fields["WorldModelStatus"]


# ---------------------------------------------------------------------------
# Frame discipline in the nodes
# ---------------------------------------------------------------------------

NODES_DIR = REPO_ROOT / "ros2_ws" / "src" / "mission_nodes" / "mission_nodes"


def test_odometry_is_never_consumed_as_a_map_pose() -> None:
    """Every node that subscribes to Odometry must transform it into the map frame.

    Regression, and the most expensive bug in this project so far: gz's
    DiffDrive dead-reckons from zero at spawn, so a rover spawned at (-8, -8)
    reports (0, 0) while parked. Two nodes used that pose directly as a map
    coordinate, and the rover then drove a correctly planned, collision-free
    path to a point one whole spawn offset away from the station - reporting
    "Goal reached, cross-track 0.00 m" the entire time.

    The offline harness cannot catch this, because there odometry *is* the map
    pose. So it is pinned at the source level instead.
    """
    offenders = []
    for source in sorted(NODES_DIR.glob("*_node.py")):
        text = source.read_text(encoding="utf-8")
        if "Odometry" not in text:
            continue
        subscribes = re.search(r"create_subscription\(\s*\n?\s*Odometry", text) is not None
        if not subscribes:
            continue
        transforms = (
            "odometry_pose_in_frame" in text or "odometry_position_in_frame" in text
        )
        if not transforms:
            offenders.append(source.name)
        # Reading the pose straight out of the message is the actual mistake.
        if re.search(r"pose_to_xy_yaw\(\s*msg\.pose\.pose\s*\)", text):
            offenders.append(f"{source.name} (uses msg.pose.pose directly)")

    assert not offenders, (
        "these nodes consume odometry without transforming it into the planning "
        f"frame: {offenders}"
    )


def test_the_frame_checker_compares_tf_against_odometry() -> None:
    """check_frames.py must be able to tell the two apart.

    It is the only thing in the stack that would have caught the bug above at
    runtime, and it did - so keep its two-way comparison.
    """
    text = (BRINGUP / "scripts" / "check_frames.py").read_text(encoding="utf-8")
    assert "tf_matches_odom" in text
    assert "does not agree with odometry" in text


def test_tf_lookups_do_not_block_and_do_not_drop_late_data() -> None:
    """TF lookups must not wait, and must not discard a slightly-late stamp.

    Regression: the world model looked transforms up at the message stamp with
    a 0.1 s timeout inside a single-threaded executor. The TF listener is a
    callback on that same executor, so nothing could arrive during the wait and
    the lookup always timed out; a sensor stamped a few ms ahead of the newest
    transform then had its data thrown away. One Gazebo run integrated zero
    lidar sweeps and produced a completely empty occupancy grid while every
    other stage looked healthy.
    """
    common = (
        REPO_ROOT / "ros2_ws" / "src" / "mission_nodes" / "mission_nodes" / "common.py"
    ).read_text(encoding="utf-8")
    signature = re.search(r"def lookup_transform\((.*?)\) ->", common, re.DOTALL)
    assert signature is not None
    assert re.search(r"timeout_s: float = 0\.0", signature.group(1)), (
        "lookup_transform must default to a non-blocking lookup"
    )
    assert "allow_latest_fallback" in signature.group(1)
    assert "rclpy.time.Time()" in common, "there must be a latest-available fallback"


def test_safety_lidar_offset_matches_the_sdf(sdf_models, config) -> None:
    """The configured lidar offset is used in a safety inequality, so it must
    equal the real mounting pose rather than approximate it."""
    pose, _ = parse_pose(find_sensor(sdf_models["mission_rover"], "safety_lidar").find("pose").text)
    assert pose[0] == pytest.approx(config.rover.safety_lidar_forward_offset_m, abs=1e-6)


def test_both_vehicles_publish_an_absolute_pose(sdf_models) -> None:
    """Rover and drone must localise the same way.

    Regression: the rover used DiffDrive's wheel odometry, which dead-reckons
    from spawn. Over a 15 m drive its drift put the rover 0.27 m inside the
    planner's clearance while it reported zero cross-track error, and placed a
    QR observation 1.51 m from where the drone saw the same station - just past
    the association radius, so one station became two and DUPLICATE_QR fired
    after a correct verification.
    """
    for model in ("scout_drone", "mission_rover"):
        publishers = [
            plugin
            for plugin in sdf_models[model].iter("plugin")
            if "OdometryPublisher" in (plugin.get("name") or "")
        ]
        assert publishers, f"{model} has no OdometryPublisher"
        assert publishers[0].find("odom_topic").text == f"/model/{model}/odometry"

    # DiffDrive's own odometry must not reach the same topic.
    diff_drive = [
        plugin
        for plugin in sdf_models["mission_rover"].iter("plugin")
        if "DiffDrive" in (plugin.get("name") or "")
    ][0]
    assert diff_drive.find("odom_topic").text != "/model/mission_rover/odometry"


def test_map_to_odom_is_identity_for_both_vehicles(launch_constants) -> None:
    """Neither transform may re-apply a spawn offset to a world-frame pose."""
    text = LAUNCH_PATH.read_text(encoding="utf-8")
    for name in ("map_to_drone_odom", "map_to_rover_odom"):
        block = text.split(f'"{name}"', 1)[1][:220]
        assert '("0.0", "0.0", "0.0")' in block, f"{name} is not an identity translation"
