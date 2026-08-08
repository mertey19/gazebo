# One QR code seen by one camera in one frame.
#
# Published by qr_detector_node. The relative observation and the world
# estimate are both carried so a consumer can re-derive the world pose with a
# different TF snapshot, or audit the transform that was applied.

std_msgs/Header header            # stamp = image acquisition time, frame_id = camera optical frame

string qr_id                      # decoded payload, e.g. "TARGET_2"
string source                     # which camera produced it, e.g. "drone_camera"

geometry_msgs/Pose pose_camera    # marker pose in the camera optical frame (relative observation)
geometry_msgs/Point position_map  # marker position transformed into the planning frame
geometry_msgs/Vector3 normal_map  # outward normal of the printed face, in the planning frame
string map_frame_id               # frame position_map/normal_map are expressed in

float64 range_m                   # camera-to-marker distance
float64 code_size_m               # physical side length used for the PnP solve
float64 apparent_size_px          # mean side length of the detected quad
float64 reprojection_error_px     # RMS residual of the accepted PnP solution
float64 ambiguity_ratio           # 0 = unambiguous, ->1 = planar pose ambiguity unresolved
float64 confidence                # fused quality metric in [0, 1]
