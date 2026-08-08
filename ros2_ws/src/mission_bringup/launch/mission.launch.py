"""Single entry point for the complete autonomous mission.

    ros2 launch mission_bringup mission.launch.py target_qr:=TARGET_2

Brings up, in one shot: Gazebo with the arena world, the ros_gz bridges, the
static TF tree, both QR detectors, the world model, the drone explorer, the
rover path follower, the mission manager, and optionally RViz.
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

#: base_link -> camera_optical_frame for a nadir camera (x right, y down, z fwd).
NADIR_OPTICAL_QUAT = ("-0.7071067811865475", "0.7071067811865476", "0.0", "0.0")
#: base_link -> camera_optical_frame for a forward camera (REP-103 optical).
FORWARD_OPTICAL_QUAT = ("0.5", "-0.5", "0.5", "-0.5")

#: Sensor mount offsets; must equal the <pose> translations in the SDF models.
DRONE_CAMERA_XYZ = ("0.10", "0.0", "-0.08")
DRONE_LIDAR_XYZ = ("0.0", "0.0", "-0.06")
ROVER_CAMERA_XYZ = ("0.22", "0.0", "0.55")
ROVER_LIDAR_XYZ = ("0.24", "0.0", "0.30")

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
    # map -> drone/odom is identity because gz's OdometryPublisher reports the
    # model's *world* pose. map -> rover/odom is the rover spawn pose because
    # DiffDrive dead-reckons from zero at spawn. Verify both after any world
    # change with: ros2 run mission_bringup check_frames.py
    static_transforms = [
        _static_tf(
            "map_to_drone_odom", "map", "drone/odom",
            ("0.0", "0.0", "0.0"), ("0.0", "0.0", "0.0", "1.0"),
        ),
        _static_tf(
            "map_to_rover_odom", "map", "rover/odom",
            (ROVER_SPAWN_XY[0], ROVER_SPAWN_XY[1], "0.0"), ("0.0", "0.0", "0.0", "1.0"),
        ),
        _static_tf(
            "drone_camera_optical", "drone/base_link", "drone/camera_optical_frame",
            DRONE_CAMERA_XYZ, NADIR_OPTICAL_QUAT,
        ),
        _static_tf(
            "drone_lidar", "drone/base_link", "drone/lidar_link",
            DRONE_LIDAR_XYZ, ("0.0", "0.0", "0.0", "1.0"),
        ),
        _static_tf(
            "rover_camera_optical", "rover/base_link", "rover/camera_optical_frame",
            ROVER_CAMERA_XYZ, FORWARD_OPTICAL_QUAT,
        ),
        _static_tf(
            "rover_lidar", "rover/base_link", "rover/lidar_link",
            ROVER_LIDAR_XYZ, ("0.0", "0.0", "0.0", "1.0"),
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
            world_model,
            drone_explorer,
            rover_follower,
            mission_manager,
            rviz,
        ]
    )
