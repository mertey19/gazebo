# Fused estimate of one physical QR-coded station.

uint8 STATUS_TENTATIVE=0
uint8 STATUS_CONFIRMED=1
uint8 STATUS_AMBIGUOUS=2   # same payload seen at two incompatible places

string qr_id
geometry_msgs/Point position       # fused position in the planning frame
geometry_msgs/Vector3 normal       # fused outward normal of the printed face
uint8 status
uint32 observation_count
float64 confidence                 # [0, 1]
float64 position_spread_m          # RMS scatter of the contributing observations
builtin_interfaces/Time first_seen
builtin_interfaces/Time last_seen
