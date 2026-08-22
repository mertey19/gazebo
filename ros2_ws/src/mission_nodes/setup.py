from setuptools import find_packages, setup

package_name = "mission_nodes"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mert Erdogan Yildirim",
    maintainer_email="merterdoganyildirim@gmail.com",
    description="ROS 2 nodes for the autonomous UAV/UGV QR mission.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "qr_detector_node = mission_nodes.qr_detector_node:main",
            "visual_obstacle_node = mission_nodes.visual_obstacle_node:main",
            "world_model_node = mission_nodes.world_model_node:main",
            "drone_explorer_node = mission_nodes.drone_explorer_node:main",
            "mission_manager_node = mission_nodes.mission_manager_node:main",
            "rover_path_follower_node = mission_nodes.rover_path_follower_node:main",
            "twin_bridge_node = mission_nodes.twin_bridge_node:main",
        ],
    },
)
