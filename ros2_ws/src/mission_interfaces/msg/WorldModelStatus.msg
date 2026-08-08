# Summary of the digital twin, published at world_model.publish_rate_hz.
std_msgs/Header header

uint32 targets_total
uint32 targets_confirmed
uint32 targets_ambiguous
uint32 obstacles
uint32 rejected_observations
float64 map_known_fraction        # [0, 1] fraction of the grid that has been observed
float64 map_occupied_fraction
string[] qr_ids                   # payloads decoded so far
