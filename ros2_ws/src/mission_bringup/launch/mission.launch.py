"""Single entry point for the complete autonomous mission.

    ros2 launch mission_bringup mission.launch.py target_qr:=TARGET_2

Brings up, in one shot: Gazebo with the arena world, the ros_gz bridges, the
static TF tree, both QR detectors, both monocular obstacle mappers, the world
model, the drone explorer, the rover path follower, the mission manager, and
optionally RViz.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# ---------------------------------------------------------------------------
# Frame geometry.
#
# These are the *only* hand-entered transforms in the system, and each mirrors
# a <pose> in the corresponding SDF. The quaternions are not guessed: they are
# printed by mission_core.geometry.matrix_to_quaternion() applied to
# R_BODY_TO_NADIR_OPTICAL and R_BODY_TO_FORWARD_OPTICAL, the same matrices the
# perception code and the offline harness use. Regenerate with:
#
#   python -c "import sys; sys.path.insert(0,'ros2_ws/src/mission_core'); \
#     from mission_core.geometry import *; \
#     print(matrix_to_quaternion(R_BODY_TO_NADIR_OPTICAL), \
#           matrix_to_quaternion(R_BODY_TO_FORWARD_OPTICAL))"
# ---------------------------------------------------------------------------

#: base_link -> camera_optical_frame for the drone camera, pitched 30 degrees
#: below horizontal. Printed by matrix_to_quaternion(optical_from_depression(
#: config.drone.camera_depression_rad)); a test re-derives it from the config.
DRONE_OPTICAL_QUAT = (
    "-0.6123724356957946",
    "0.6123724356957946",
    "-0.35355339059327373",
    "0.35355339059327384",
)
#: base_link -> camera_optical_frame for a forward camera (REP-103 optical).
FORWARD_OPTICAL_QUAT = ("0.5", "-0.5", "0.5", "-0.5")

#: Sensor mount offsets; must equal the <pose> translations in the SDF models.
#: One camera per vehicle is the complete sensor suite.
DRONE_CAMERA_XYZ = ("0.10", "0.0", "-0.08")
ROVER_CAMERA_XYZ = ("0.22", "0.0", "0.55")

#: Mirrors vision.rover_max_range_m: the rover mapper must not inherit the
#: drone's trusted range, which is measured from four metres up.
ROVER_VISION_RANGE_M = 6.0

#: Vehicle spawn poses; must equal the <pose> entries in mission_arena.sdf.
DRONE_SPAWN_XY = ("-8.0", "-6.5")
ROVER_SPAWN_XY = ("-8.0", "-8.0")


def _static_tf(name: str, parent: str, child: str, xyz, quat) -> Node:
    """A latched static transform, given translation and an (x, y, z, w) quat."""
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=name,
        output="log",
        arguments=[
            "--x", xyz[0], "--y", xyz[1], "--z", xyz[2],
            "--qx", quat[0], "--qy", quat[1], "--qz", quat[2], "--qw", quat[3],
            "--frame-id", parent, "--child-frame-id", child,
        ],
    )


def generate_launch_description() -> LaunchDescription:
    bringup_share = get_package_share_directory("mission_bringup")
    default_config = os.path.join(bringup_share, "config", "mission.yaml")
    default_world = os.path.join(bringup_share, "worlds", "mission_arena.sdf")
    models_dir = os.path.join(bringup_share, "models")

    target_qr = LaunchConfiguration("target_qr")
    config_file = LaunchConfiguration("config_file")
    world = LaunchConfiguration("world")
    use_rviz = LaunchConfiguration("rviz")
    ground_station = LaunchConfiguration("ground_station")
    headless = LaunchConfiguration("headless")
    auto_start = LaunchConfiguration("auto_start")

    arguments = [
        DeclareLaunchArgument(
            "target_qr",
            default_value="TARGET_2",
            description="QR payload the rover must physically reach and verify.",
        ),
        DeclareLaunchArgument("config_file", default_value=default_config),
        DeclareLaunchArgument("world", default_value=default_world),
        DeclareLaunchArgument(
            "rviz", default_value="false", description="Start RViz2 for debugging."
        ),
        DeclareLaunchArgument(
            "ground_station",
            default_value="false",
            description=(
                "Stream the mission to the Simurgh digital-twin ground station over "
                "UDP. Off by default: the mission must not depend on an operator "
                "console being reachable. Set the arena's real position in "
                "config/mission.yaml (ground_station.anchor_latitude/longitude) "
                "before switching it on, or every track lands somewhere plausible "
                "and wrong."
            ),
        ),
        DeclareLaunchArgument(
            "headless",
            default_value="false",
            description="Run gz-sim without its GUI (sensors still render).",
        ),
        DeclareLaunchArgument(
            "auto_start",
            default_value="true",
            description=(
                "Start the mission immediately. Set false to trigger it later via "
                "the /mission/run_mission action."
            ),
        ),
    ]

    # gz needs to resolve model:// URIs for the stations and the vehicles.
    resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=models_dir + os.pathsep + os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"]
            )
        ),
        launch_arguments={
            # -r starts the world unpaused; -v2 keeps gz's own logging readable.
            "gz_args": ["-r -v2 ", world],
            "on_exit_shutdown": "true",
        }.items(),
        condition=UnlessCondition(headless),
    )
    # `-s` runs the server only. Sensor rendering still happens, so the mission
    # is unaffected; this is the mode to use on a machine without a display.
    gz_sim_headless = ExecuteProcess(
        cmd=["gz", "sim", "-r", "-s", "-v2", world],
        output="screen",
        condition=IfCondition(headless),
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_bridge",
        output="screen",
        parameters=[
            {"config_file": os.path.join(bringup_share, "config", "gz_bridge.yaml")},
            {"use_sim_time": True},
        ],
    )
    # Images go through the dedicated image bridge: it avoids a full
    # serialise/deserialise of every frame in the generic bridge.
    image_bridge = Node(
        package="ros_gz_image",
        executable="image_bridge",
        name="gz_image_bridge",
        output="screen",
        arguments=["/drone/camera/image", "/rover/camera/image"],
        parameters=[{"use_sim_time": True}],
    )

    # ---- static TF -------------------------------------------------------
    # Both map -> */odom are identity: each vehicle publishes its *world* pose
    # through gz's OdometryPublisher. Verify after any world change with:
    #   ros2 run mission_bringup check_frames.py
    static_transforms = [
        _static_tf(
            "map_to_drone_odom", "map", "drone/odom",
            ("0.0", "0.0", "0.0"), ("0.0", "0.0", "0.0", "1.0"),
        ),
        # Identity for both vehicles: each publishes its *world* pose through
        # gz's OdometryPublisher. The rover used DiffDrive's spawn-relative
        # wheel odometry until its dead-reckoning drift started failing
        # missions; see the plugin comment in mission_rover/model.sdf.
        _static_tf(
            "map_to_rover_odom", "map", "rover/odom",
            ("0.0", "0.0", "0.0"), ("0.0", "0.0", "0.0", "1.0"),
        ),
        _static_tf(
            "drone_camera_optical", "drone/base_link", "drone/camera_optical_frame",
            DRONE_CAMERA_XYZ, DRONE_OPTICAL_QUAT,
        ),
        _static_tf(
            "rover_camera_optical", "rover/base_link", "rover/camera_optical_frame",
            ROVER_CAMERA_XYZ, FORWARD_OPTICAL_QUAT,
        ),
    ]

    common_params = [{"config_file": config_file}, {"use_sim_time": True}]

    drone_qr_detector = Node(
        package="mission_nodes",
        executable="qr_detector_node",
        name="drone_qr_detector",
        output="screen",
        parameters=common_params
        + [
            {
                "camera_name": "drone",
                "image_topic": "/drone/camera/image",
                "camera_info_topic": "/drone/camera/camera_info",
                "observation_topic": "/perception/drone/qr_observations",
                "camera_optical_frame": "drone/camera_optical_frame",
            }
        ],
    )
    rover_qr_detector = Node(
        package="mission_nodes",
        executable="qr_detector_node",
        name="rover_qr_detector",
        output="screen",
        parameters=common_params
        + [
            {
                "camera_name": "rover",
                "image_topic": "/rover/camera/image",
                "camera_info_topic": "/rover/camera/camera_info",
                "observation_topic": "/perception/rover/qr_observations",
                "camera_optical_frame": "rover/camera_optical_frame",
            }
        ],
    )
    # The obstacle mappers run on the very same image streams as the QR
    # detectors above: one camera per vehicle is the whole sensor suite, so
    # every metre of geometry in the occupancy grid comes out of these frames.
    drone_obstacle_mapper = Node(
        package="mission_nodes",
        executable="visual_obstacle_node",
        name="drone_obstacle_mapper",
        output="screen",
        parameters=common_params
        + [
            {
                "camera_name": "drone",
                "image_topic": "/drone/camera/image",
                "camera_info_topic": "/drone/camera/camera_info",
                "observation_topic": "/perception/drone/ground_observations",
                "camera_optical_frame": "drone/camera_optical_frame",
                # The drone escorts the rover from behind and would otherwise
                # map it as an obstacle standing on its own route.
                "self_filter_frames": ["rover/base_link"],
            }
        ],
    )
    rover_obstacle_mapper = Node(
        package="mission_nodes",
        executable="visual_obstacle_node",
        name="rover_obstacle_mapper",
        output="screen",
        parameters=common_params
        + [
            {
                "camera_name": "rover",
                "image_topic": "/rover/camera/image",
                "camera_info_topic": "/rover/camera/camera_info",
                "observation_topic": "/perception/rover/ground_observations",
                "camera_optical_frame": "rover/camera_optical_frame",
                # The rover only has to see far enough to stop; the drone's
                # 10 m is measured from four metres up, not from 0.55 m.
                "max_range_m": ROVER_VISION_RANGE_M,
                # Nothing to filter: the only other vehicle is above the
                # rover's horizon, where no ground contact can be measured.
                "self_filter_frames": [""],
            }
        ],
    )
    world_model = Node(
        package="mission_nodes",
        executable="world_model_node",
        name="world_model",
        output="screen",
        parameters=common_params,
    )
    drone_explorer = Node(
        package="mission_nodes",
        executable="drone_explorer_node",
        name="drone_explorer",
        output="screen",
        # The mission manager calls ~/start; auto_start here would begin the
        # scan before the manager has entered TAKEOFF and break the state trace.
        parameters=common_params + [{"auto_start": False}],
    )
    rover_follower = Node(
        package="mission_nodes",
        executable="rover_path_follower_node",
        name="rover_path_follower",
        output="screen",
        parameters=common_params,
    )
    mission_manager = Node(
        package="mission_nodes",
        executable="mission_manager_node",
        name="mission_manager",
        output="screen",
        parameters=common_params + [{"target_qr": target_qr}, {"auto_start": auto_start}],
    )

    # Read-only by construction: it subscribes and sends UDP, publishes no
    # topic and offers no service, so a station that disappears mid-mission
    # cannot affect the mission.
    twin_bridge = Node(
        package="mission_nodes",
        executable="twin_bridge_node",
        name="twin_bridge",
        output="screen",
        parameters=common_params + [{"ground_station.enabled": True}],
        condition=IfCondition(ground_station),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", os.path.join(bringup_share, "rviz", "mission.rviz")],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(
        arguments
        + [
            resource_path,
            LogInfo(msg=["[MISSION] launching with target_qr=", target_qr]),
            gz_sim,
            gz_sim_headless,
            bridge,
            image_bridge,
            *static_transforms,
            drone_qr_detector,
            rover_qr_detector,
            drone_obstacle_mapper,
            rover_obstacle_mapper,
            world_model,
            drone_explorer,
            rover_follower,
            mission_manager,
            twin_bridge,
            rviz,
        ]
    )
