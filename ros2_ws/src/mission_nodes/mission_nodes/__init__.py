"""ROS 2 nodes for the autonomous UAV/UGV QR mission.

Each node is a thin adapter around ``mission_core``:

===========================  ==============================================
``qr_detector_node``         camera frames -> world-referenced QR observations
``world_model_node``         observations + lidar -> the digital twin
``drone_explorer_node``      lawnmower coverage flight for the scout UAV
``mission_manager_node``     mission state machine, planning, path transmission
``rover_path_follower_node`` follows the received nav_msgs/Path
===========================  ==============================================
"""
