from setuptools import find_packages, setup

package_name = "mission_core"

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
    description="ROS-free mission logic for the autonomous UAV/UGV QR mission.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": []},
)
