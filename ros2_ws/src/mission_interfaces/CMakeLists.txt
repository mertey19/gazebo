cmake_minimum_required(VERSION 3.8)
project(mission_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/QrObservation.msg"
  "msg/TargetRecord.msg"
  "msg/TargetArray.msg"
  "msg/Obstacle.msg"
  "msg/ObstacleArray.msg"
  "msg/WorldModelStatus.msg"
  "msg/MissionStatus.msg"
  "msg/TrackingStatus.msg"
  "msg/ExplorationStatus.msg"
  "srv/GetTarget.srv"
  "srv/PlanPath.srv"
  "action/RunMission.action"
  DEPENDENCIES builtin_interfaces geometry_msgs nav_msgs std_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
