# One camera frame turned into floor geometry by visual_obstacle_node.
#
# This is the mission's only source of ranged information: neither vehicle
# carries a lidar or a depth sensor, so obstacle positions are recovered by
# intersecting image rays with the known ground plane. See
# mission_core/vision_mapping.py for the construction and its assumptions.

std_msgs/Header header             # stamp = image acquisition time, frame_id = planning frame

string source                      # which camera produced it, e.g. "drone_camera"

# Both arrays are flat [x, y, z] triples in the planning frame, rather than
# geometry_msgs/Point[]: free_space carries thousands of samples per frame and
# rclpy type-checks every element of a message array individually, which at the
# camera's rate costs more CPU than the segmentation that produced them.

# Where obstacles were seen to *stand* on the floor. x/y is the contact point;
# z is the height measured for that obstacle from the top of its silhouette,
# not the height of the point itself (which is the floor, by construction).
float32[] contacts

# Floor samples: positions observed to be empty (z is the floor plane).
# Deliberately separate from the contacts, because a cell that has both in one
# frame is a boundary cell, and the far denser floor samples must not outvote
# the measurement.
float32[] free_space

# Nearest contact inside the vehicle's forward wedge, as a horizontal distance
# from the camera, or +inf when the wedge is clear. Computed in the sensor's
# own geometry so the rover's emergency stop does not depend on how well the
# vehicle is localised in the planning frame.
float64 nearest_forward_range_m

uint32 columns_evaluated           # image columns searched for a contact
float64 non_ground_fraction        # share of the usable image that was not floor
bool usable                        # false when the frame was rejected wholesale
