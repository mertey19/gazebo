# Mission state machine snapshot. Published on every transition and
# periodically, with transient-local durability so a late subscriber still
# learns the current state.

std_msgs/Header header

string state                # MissionState enum value, e.g. "ROVER_NAVIGATING"
string requested_qr
string verified_qr          # empty until the rover confirms a payload
string failure_reason       # FailureReason enum value, "NONE" while healthy
string failure_detail

float64 elapsed_s
uint32 path_poses
float64 path_length_m
string trace                # full state chain, e.g. "IDLE -> TAKEOFF -> ..."
