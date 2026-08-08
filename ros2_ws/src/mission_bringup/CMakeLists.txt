cmake_minimum_required(VERSION 3.8)
project(mission_bringup)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

install(
  DIRECTORY launch config worlds models rviz
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/check_frames.py
    scripts/check_pipeline.py
    scripts/await_mission.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
