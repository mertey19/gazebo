# Rover path-following diagnostics, published every control cycle.

std_msgs/Header header

string state                     # IDLE | TRACKING | ALIGNING | GOAL_REACHED | FAILED
float64 distance_to_goal_m
float64 cross_track_error_m
float64 heading_error_rad
uint32 progress_index            # index of the path segment currently being tracked
float64 commanded_linear
float64 commanded_angular
bool goal_reached
bool failed
string failure_reason            # FailureReason enum value
string detail
