# Scout flight progress, consumed by the mission manager.

std_msgs/Header header

string phase                     # GROUNDED | CLIMBING | SCANNING | COMPLETE
bool at_scan_altitude
bool complete
uint32 waypoint_index
uint32 waypoint_count
float64 coverage_fraction        # [0, 1]
float64 distance_to_waypoint_m
float64 altitude_m
