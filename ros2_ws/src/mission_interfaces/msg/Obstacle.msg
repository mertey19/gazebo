# An obstacle footprint extracted from the observed occupancy grid.
# Axis-aligned in the planning frame; sufficient for a 2D MVP planner and a
# natural place to add a polygon or a velocity when dynamic obstacles land.

string id
geometry_msgs/Point centre         # footprint centre, z = 0
geometry_msgs/Vector3 size         # x/y footprint extents, z unused
float64 height_m                   # tallest silhouette measured over the footprint
uint32 cell_count                  # occupied grid cells backing this obstacle
