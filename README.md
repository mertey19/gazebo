# Autonomous UAV + UGV QR Mission

[![CI](https://github.com/mertey19/gazebo/actions/workflows/ci.yml/badge.svg)](https://github.com/mertey19/gazebo/actions/workflows/ci.yml)

A scout drone maps an unknown arena, reads the QR code on each ground station,
and hands the rover a collision-free route to the one station the operator
asked for. The rover drives there and confirms with its own camera that it is
standing in front of the right code.

```
ros2 launch mission_bringup mission.launch.py target_qr:=TARGET_2
```

The mission is **not** told where `TARGET_2` is. It has to find it.

---

## 1. What runs where — read this first

Every row is backed by a job in CI, so none of it is a claim about code that
has merely been written.

| Layer | Status |
|---|---|
| `mission_core` algorithms (QR/PnP, monocular mapping, world model, occupancy, A*, pure pursuit, state machine, validator, ground-station feed) | **Executed.** 201 tests, including a full offline mission per target. |
| Offline mission harness (renders real QR textures through a real pinhole camera, segments and back-projects those same frames, integrates unicycle kinematics) | **Executed.** Drives the production pipeline with zero simulator. Reproduce with `python scripts/run_offline_mission.py`. |
| ROS 2 Jazzy layer (interfaces, nodes, launch) | **Executed.** `colcon build` + `colcon test`, interfaces introspectable, every node entry point installed, launch file expands. |
| **Gazebo Harmonic, full mission** | **Executed.** `gazebo-e2e` launches the whole stack headless and blocks on the real mission verdict. A green workflow requires all three targets to succeed in three independent trials each. |

The last row is the one that matters, and it reports `MISSION_SUCCESS`:

```
[    0.2s] state=EXPLORING          target=TARGET_2
[  192.9s] state=TARGET_FOUND       target=TARGET_2
[  193.1s] state=PLANNING           target=TARGET_2
[  193.2s] state=PATH_READY         target=TARGET_2
[  193.4s] state=ROVER_NAVIGATING   target=TARGET_2
[  218.8s] state=VERIFYING_TARGET   target=TARGET_2
[  220.2s] state=MISSION_SUCCESS    target=TARGET_2 verified=TARGET_2

path : 5 poses, 14.98 m       QR position error vs ground truth: ~2 cm
```

That recorded run is one concrete trace. The current CI gate is broader: its
matrix expands to `TARGET_1`, `TARGET_2` and `TARGET_3`, with three independent
Gazebo jobs per target. `fail-fast` is disabled, so a failure keeps the other
eight trials running and preserves a complete diagnostic set.

A green `unit-tests` says nothing about Gazebo, which is why the jobs are kept
separate. The development host itself is Windows 11 with no WSL2 and no Docker,
so ROS 2 and gz-sim cannot run on it — the Ubuntu evidence comes from CI, and
[§11](#11-known-limitations) lists what CI still does not cover.

> **The Gazebo trace quoted above predates the removal of both lidars.** The
> mission scenario forbids ranging sensors, so obstacle geometry is now
> recovered from the vehicles' cameras ([§3.1](#31-monocular-obstacle-mapping)).
> Everything in `mission_core`, including a full offline mission per target, has
> been re-run against that pipeline and passes; the Gazebo row is re-earned by
> the next green `gazebo-e2e`, not by this sentence.

The split exists precisely so that this distinction is possible: all mission
*logic* lives in a ROS-free library, and the ROS nodes are thin adapters that
move data between DDS topics and that library.

---

## 2. Stack

The repository was empty at the start, so nothing was inherited. The choices,
and why:

| Choice | Rationale |
|---|---|
| **ROS 2 Jazzy** | Current LTS; matches Gazebo Harmonic's default pairing. |
| **Gazebo Harmonic (gz-sim 8)** + `ros_gz_bridge` | The supported simulator for Jazzy. |
| **No PX4 / ArduPilot** | The scout needs to hold an altitude and fly a lawnmower. gz-sim's `VelocityControl` does that deterministically. A SITL stack would add a second autopilot, a MAVLink bridge and an attitude-tuning loop to an MVP whose subject is perception and planning. Swapping it in later touches one SDF plugin block and one node's output topic. |
| **No lidar, and no depth camera either** | Required by the mission scenario. One RGB camera per vehicle is the entire sensor suite, so obstacle geometry has to be *derived* — see [§3.1](#31-monocular-obstacle-mapping). A depth or RGBD sensor would have kept the old point-cloud pipeline almost unchanged, which is exactly why it is not used: it would satisfy the letter of "no lidar" and none of its intent. `test_no_vehicle_carries_a_range_sensor` fails if any is added back. |
| **No Nav2** | Nav2 was not present, and it brings a behaviour-tree stack, costmap plugins and lifecycle management for a 22 × 22 m arena. A* + pure pursuit is complete, deterministic, and directly verifiable — the planner has a unit test that asserts no path ever intersects an obstacle. |
| **OpenCV `QRCodeDetector` + `solvePnP`** | Payload and 6-DoF pose both come out of the image. |
| **A\*** on an observed occupancy grid | Explicitly preferred over RRT* for this scale. |

---

## 3. Architecture

```
                         ros2 launch mission_bringup mission.launch.py target_qr:=TARGET_2
                                                    |
                                                    v
   +--------------------------------------------------------------------------------------+
   |                                  GAZEBO  (mission_arena.sdf)                          |
   |  scout_drone      mission_rover      3x target_station_*      2x static obstacle       |
   |  one RGB camera   one RGB camera     unique QR plate                                   |
   |  (30 deg down)    (forward)          (top + 4 sides)                                   |
   +--------------------------------------------------------------------------------------+
        |  image/odom                                              cmd_vel  ^
        v                                                                   |
   +----------------------------- ros_gz_bridge + ros_gz_image -----------------------------+
        |                                         |                         |
        v                                         v                         |
 /drone/camera/image                       /rover/camera/image               |
        |          \                              |          \              |
        v           v                             v           v             |
 +----------------+ +------------------+  +----------------+ +------------------+
 | qr_detector    | | visual_obstacle  |  | qr_detector    | | visual_obstacle  |
 |    (drone)     | |     (drone)      |  |    (rover)     | |     (rover)      |
 | decode -> PnP  | | floor/not-floor  |  | decode -> PnP  | | floor/not-floor  |
 | -> TF -> map   | | -> ground plane  |  | -> TF -> map   | | -> ground plane  |
 +-------+--------+ +---------+--------+  +--------+-------+ +---------+--------+
         |                    |                    |                   |
   QrObservation      GroundObservation      QrObservation      GroundObservation
         |                    |                    |                   |
         |                    |                    |                   +--> rover_path_follower
         |                    |                    |                   |    (safety stop)
         v                    v                    |                   |
   +-----------------------------------+           |                   |
   |         world_model_node          | <---------|-------------------+
   |  fuse sightings -> TargetRecord   |           |    (runtime obstacles)
   |  contacts + floor -> OccupancyGrid|           |
   |  connected components -> Obstacle |           |
   +------+---------------+------------+           |
          |               |                        |                          |
  /world_model/targets   /world_model/             |                          |
  /world_model/obstacles  occupancy_grid           |                          |
          |               |                        |                          |
          v               v                        v                          |
   +-------------------------------------------------------------+          |
   |                    mission_manager_node                     |          |
   |   MissionOrchestrator: explicit state machine               |          |
   |   IDLE -> TAKEOFF -> EXPLORING -> TARGET_FOUND -> PLANNING   |          |
   |        -> PATH_READY -> SENDING_PATH -> ROVER_NAVIGATING     |          |
   |        -> VERIFYING_TARGET -> MISSION_SUCCESS / FAILED       |          |
   |                                                             |          |
   |   A* on the inflated observed grid  +  MissionValidator      |          |
   +---------+-----------------------------------+---------------+          |
             |                                   |                          |
   ~/start   |                    /mission/rover_path (nav_msgs/Path)        |
             v                                   v                          |
   +-------------------+              +--------------------------+          |
   | drone_explorer    |              | rover_path_follower       |---------+
   | lawnmower flight  |              | pure pursuit + safety stop|
   +-------------------+              +--------------------------+
             |                                   |
   /drone/exploration_status          /rover/tracking_status
             |                                   |
             +--------------> mission_manager <--+
```

The planner lives in the mission manager, **not** in the rover. The rover only
ever receives a `nav_msgs/Path` on `/mission/rover_path` and follows it; it has
no access to the world model, the target list or the obstacle map.

### Package layout

```
ros2_ws/src/
├── mission_core/          ROS-free. All mission logic + the whole test suite.
│   ├── mission_core/
│   │   ├── camera.py          pinhole model, CameraInfo -> intrinsics
│   │   ├── config.py          typed YAML config + startup sanity checks
│   │   ├── errors.py          FailureReason taxonomy
│   │   ├── exploration.py     lawnmower pattern + flight controller
│   │   ├── geometry.py        SE(3), quaternions, frame conventions
│   │   ├── mission_state.py   state machine + legal transition table
│   │   ├── occupancy.py       OccupancyGrid, inflation, mapper
│   │   ├── orchestrator.py    the mission itself
│   │   ├── path_following.py  pure pursuit + unicycle model
│   │   ├── planner.py         A*, shortcutting, approach-pose selection
│   │   ├── qr.py              decode + PnP marker pose
│   │   ├── twin_telemetry.py  ground-station message building
│   │   ├── validation.py      the success criteria
│   │   ├── vision_mapping.py  floor segmentation + ground-plane projection
│   │   └── world_model.py     the digital twin
│   └── test/                  201 tests, incl. sim_harness/offline_mission
├── mission_interfaces/    10 msgs, 2 srvs, 1 action
├── mission_nodes/         7 rclpy nodes (thin adapters over mission_core)
└── mission_bringup/       world, models, config, launch, rviz, frame checker
scripts/generate_qr_targets.py   generates the station models from mission.yaml
```

### 3.1 Monocular obstacle mapping

Neither vehicle carries a lidar, a depth camera or any other ranging device —
the scenario does not allow one — so the occupancy grid is built from the same
RGB frames the QR detector reads. Two facts make that possible without depth:
the arena floor is a **plane of known height**, and the **pose of the camera
above it is known** from TF. A pixel plus a plane is an intersection, and an
intersection is a position; the construction is exact, not fitted.

The geometry is then applied only where it is valid:

* an obstacle's **ground-contact pixel** — the bottom of its silhouette — lies
  on the floor by definition, so its intersection is the obstacle's true base;
* pixels **above** that contact are on a vertical face and are deliberately not
  intersected with the floor. Doing so is the classic inverse-perspective
  smear: it paints a fake footprint stretching away from the camera. The top of
  the silhouette is instead intersected with the *vertical line through the
  contact*, which **measures the obstacle's height** from a single image;
* pixels classified as floor are intersected and reported as free space;
* whatever is hidden behind an obstacle yields no pixels at all, so it stays
  `UNKNOWN`. That matters more here than it did with a lidar: what a camera
  cannot see behind an obstacle *is* the obstacle's own interior, and with
  `planner.allow_unknown = false` an unseen interior is never planned through
  even though only the visible rim is ever marked occupied.

Which pixels are floor is decided by the one assumption that remains: the floor
is the **dominant surface**, not a hardcoded colour. The robust median of the
Lab chroma channels is the floor's colour and the MAD is its natural spread, so
a pixel is not-floor when its chroma sits `vision.chroma_sigma` robust
deviations away. Three properties follow, and each is pinned by a test in
[`test_vision_mapping.py`](ros2_ws/src/mission_core/test/test_vision_mapping.py):

* **shadows are not walls.** Sun and ambient are both white, so shadowed floor
  keeps the floor's hue and only loses lightness. A brightness-based segmenter
  maps a phantom obstacle beside every real one; a chroma-based one does not.
  A second, deliberately one-sided lightness test catches achromatic obstacles
  such as a white QR plate, and cannot fire on a shadow, which is darker.
* **the sky is not a wall standing at the horizon.** The vanishing line of the
  ground plane follows from the camera pose alone, and everything above it is
  excluded from both the answer and the statistics. On the rover — whose camera
  looks straight ahead — that is half of every frame.
* **the floor reference is remembered between frames.** Re-deriving it per
  frame assumes the floor is the majority of every frame, and the moment that
  stops being true is the moment it matters most: a vehicle close enough to a
  wall for the wall to fill its view. Learned from frames the floor does
  dominate and carried forward, the reference stays the floor.

The occupancy mapper then counts contacts and floor samples per cell
separately, so a cell only becomes occupied when the contact is seen from
enough viewpoints — a mis-segmented frame cannot carve a hole in a wall, and a
one-off contact never becomes one.

---

## 4. Mission flow

1. **IDLE** — validate the requested payload against `known_payloads`. An
   unknown payload fails here, before takeoff.
2. **TAKEOFF** — climb to `drone.scan_altitude_m` (6.0 m).
3. **EXPLORING** — fly a boustrophedon pattern with 5.0 m lanes. Each frame is
   decoded; each decode is turned into a marker pose by `solvePnP` and lifted
   into `map` through TF, and the *same* frame is segmented into floor and
   not-floor to fill the occupancy grid. The mapped strip is a trapezoid,
   narrowest (5.75 m) at its near edge, so 5.0 m lanes still overlap; the
   pattern is also shifted back by that near edge, because a camera pitched
   down sees ground *ahead* of the aircraft rather than beneath it, and the
   pattern has to cover what the sensor sweeps rather than what the vehicle
   overflies.
4. **TARGET_FOUND** — the requested payload has `CONFIRMED` status
   (≥3 consistent observations) *and* the sweep has finished, so the digital
   twin is complete rather than merely sufficient.
5. **PLANNING** — pick a collision-free standoff pose on a ring around the
   discovered position that also has line of sight to it, then A* from the
   rover's live odometry to that pose on the grid inflated by
   `rover_radius + safety_margin`.
6. **PATH_READY → SENDING_PATH** — publish `nav_msgs/Path` on
   `/mission/rover_path` (transient-local, so a late subscriber still gets it).
7. **ROVER_NAVIGATING** — pure pursuit, with a cross-track watchdog and a
   navigation timeout. A tracking failure stops the rover and replans once
   from its live pose; a repeated failure aborts instead of looping forever.
8. **VERIFYING_TARGET** — the rover's own camera must decode the same payload,
   `required_consecutive_reads` times, within `verification.max_range_m`.
   An unreadable code gets one fresh heading sweep; a wrong decoded payload
   still fails immediately.
9. **MISSION_SUCCESS** — only after every mandatory validator check passes.

Any step can transition to **MISSION_FAILED** with a named
[`FailureReason`](ros2_ws/src/mission_core/mission_core/errors.py). Nothing
continues silently past a critical failure.

### Example log

```
[MISSION] Requested QR: TARGET_2
[DRONE] Taking off
[DRONE] Reached scan altitude 6.0 m; exploration started
[QR] TARGET_2 detected at (7.00, -5.00) range 5.16 m
[QR] TARGET_1 detected at (6.00, 6.01) range 5.48 m
[QR] TARGET_3 detected at (-6.01, 6.00) range 5.37 m
[WORLD_MODEL] 3 targets confirmed ['TARGET_1','TARGET_2','TARGET_3'], 5 obstacles mapped
[MISSION] Requested target TARGET_2 found at (7.00, -5.00) confidence 0.95
[PLANNER] Planning rover path
[PLANNER] Path contains 3 poses, 14.02 m, 311 nodes expanded, clearance 0.55 m
[MISSION] Path sent to rover
[ROVER] Navigation started
[ROVER] Goal reached
[QR] Rover detected TARGET_2
[MISSION] QR verification successful
[MISSION] SUCCESS
```

(That is real output from the offline harness, not an illustration.)

---

## 5. Interfaces

### Topics

| Topic | Type | Publisher | QoS |
|---|---|---|---|
| `/drone/camera/image` | `sensor_msgs/Image` | gz bridge | sensor |
| `/drone/camera/camera_info` | `sensor_msgs/CameraInfo` | gz bridge | sensor |
| `/drone/odometry` | `nav_msgs/Odometry` | gz bridge | sensor |
| `/drone/cmd_vel` | `geometry_msgs/Twist` | `drone_explorer` | reliable |
| `/drone/exploration_status` | `mission_interfaces/ExplorationStatus` | `drone_explorer` | latched |
| `/rover/camera/image` | `sensor_msgs/Image` | gz bridge | sensor |
| `/rover/camera/camera_info` | `sensor_msgs/CameraInfo` | gz bridge | sensor |
| `/rover/odometry` | `nav_msgs/Odometry` | gz bridge | sensor |
| `/rover/cmd_vel` | `geometry_msgs/Twist` | `rover_path_follower` | reliable |
| `/rover/tracking_status` | `mission_interfaces/TrackingStatus` | `rover_path_follower` | latched |
| `/perception/drone/qr_observations` | `mission_interfaces/QrObservation` | `drone_qr_detector` | reliable |
| `/perception/rover/qr_observations` | `mission_interfaces/QrObservation` | `rover_qr_detector` | reliable |
| `/perception/drone/ground_observations` | `mission_interfaces/GroundObservation` | `drone_obstacle_mapper` | reliable |
| `/perception/rover/ground_observations` | `mission_interfaces/GroundObservation` | `rover_obstacle_mapper` | reliable |
| `/world_model/targets` | `mission_interfaces/TargetArray` | `world_model` | latched |
| `/world_model/obstacles` | `mission_interfaces/ObstacleArray` | `world_model` | latched |
| `/world_model/occupancy_grid` | `nav_msgs/OccupancyGrid` | `world_model` | latched |
| `/world_model/status` | `mission_interfaces/WorldModelStatus` | `world_model` | latched |
| `/world_model/markers` | `visualization_msgs/MarkerArray` | `world_model` | latched |
| **`/mission/rover_path`** | **`nav_msgs/Path`** | `mission_manager` | latched |
| `/mission/status` | `mission_interfaces/MissionStatus` | `mission_manager` | latched |

Nothing publishes a "travelled path" topic. The ground station lengthens each
vehicle's trail from successive pose messages itself, and RViz keeps its own
history, so a trail topic would only duplicate state its consumers already hold.

Latched = `TRANSIENT_LOCAL` + `RELIABLE`, so a tool started mid-mission still
receives the current map, path and state.

### Services

| Service | Type | Purpose |
|---|---|---|
| `/world_model/get_target` | `mission_interfaces/GetTarget` | Look up one payload; reports `found` and `usable` separately. |
| `/mission/plan_path` | `mission_interfaces/PlanPath` | Plan to a *discovered* target on demand. Refuses unconfirmed payloads rather than inventing coordinates. |
| `/mission/abort` | `std_srvs/Trigger` | Stop the mission and the rover. |
| `/drone_explorer/start`, `/drone_explorer/hold` | `std_srvs/Trigger` | Flight control. |
| `/rover_path_follower/stop` | `std_srvs/Trigger` | Clear the path and halt. |

### Action

| Action | Type |
|---|---|
| `/mission/run_mission` | `mission_interfaces/RunMission` |

Goal: `target_qr`. Feedback: state, elapsed time, targets confirmed, obstacles
mapped, coverage fraction. Result: success flag, final state, failure reason,
verified payload, path length, and the full validation report flattened into
parallel `check_names` / `check_passed` / `check_details` arrays.

### TF frames

```
map
├── drone/odom ── drone/base_link ── drone/camera_optical_frame
└── rover/odom ── rover/base_link ── rover/camera_optical_frame
```

* `map → */odom` — static, published by the launch file (see §11).
* `*/odom → */base_link` — from gz (`OdometryPublisher` / `DiffDrive`), bridged
  onto `/tf`.
* `base_link → camera_optical_frame` — static. The quaternions are **derived**,
  not typed by hand: they are `matrix_to_quaternion()` applied to
  `R_BODY_TO_NADIR_OPTICAL` and `R_BODY_TO_FORWARD_OPTICAL` from
  [`geometry.py`](ros2_ws/src/mission_core/mission_core/geometry.py) — the same
  matrices the perception code uses. The launch file documents the one-liner
  that regenerates them.
* Four frames, not six: with the lidars gone there is one sensor per vehicle,
  so the only mounting transform that can be wrong is the camera's — and that
  one is checked against the SDF, the launch file and the offline harness by
  `test_sensor_mount_agrees_everywhere`.

Camera optical frames follow REP-103: x right, y down, z forward. QR poses come
out of `solvePnP` in the optical frame and are transformed to `map` with a
single TF lookup — there are no corrective offsets anywhere in the codebase.

---

## 5.1 Ground station

The mission streams to the [Simurgh digital-twin ground
station](https://github.com/mertey19/groundstation) — Unity + Mapbox, with a
mission engine that already understands this exact scenario. **Nothing on the
station side changes.** It speaks a JSON dialect over UDP whose vocabulary maps
onto the mission one-to-one:

| Station field | Source |
|---|---|
| `pose` (`uav` / `rover`) | `/drone/odometry`, `/rover/odometry`, through TF into `map` |
| `missionPhase` | `/mission/status` → `scan` / `joint_operation` / `dynamic_replan` / `complete` |
| `route.waypoints` | `/mission/rover_path` |
| `targets[]` | `/world_model/targets` — `decodedContent` is the payload the rover read, `reached` whether it is verified |
| `obstacles[]` | `/world_model/obstacles` |
| `voxelCells[]` | `/world_model/occupancy_grid`, occupied cells only |
| `imagery.imageBase64` | the rover's own frame of the code it verified, for the station's QR gallery |

`twin_bridge_node` is the translator, and it is read-only by construction: it
subscribes and sends UDP, publishes no topic, offers no service and touches no
vehicle, so a ground station that disappears mid-mission cannot affect the
mission.

```bash
ros2 launch mission_bringup mission.launch.py ground_station:=true
```

Two properties are worth stating because they are what make the display
*correct* rather than merely populated:

* **Everything is a delta.** A 110 × 110 grid is 12 100 cells; sending it whole
  at the world model's rate would saturate a field link. Only what changed is
  sent, as `upsert` — which is also why targets and obstacles appear on the
  operator's map exactly as the drone discovers them, with no extra machinery.
* **Retractions are sent too.** Occupancy is evidence, and evidence can be
  withdrawn: a cell that later fails the hit-ratio test, or an obstacle whose
  connected component merged into another, is `remove`d. Without that the map
  only ever grows — one arena scan put thirteen obstacles on the station's map
  for a world model that held seven.

### Without ROS, without Gazebo

The same messages can be produced from the offline harness, which makes the
ground station demonstrable on a machine that has neither:

```bash
python scripts/stream_mission_to_ground_station.py --host 127.0.0.1
```

It flies a complete mission and sends the same `DigitalTwinMessageV1` stream
`twin_bridge_node` sends — same builder, same schema, same deltas. `--speed 2`
plays at twice real time, `--speed 0` as fast as the mission computes.

> **Set the anchor before a demo.** The mission plans in local ENU metres and
> the station plots on a globe, so `ground_station.anchor_latitude/longitude`
> place the `map` origin on the Earth. The shipped value is a **placeholder**
> (Hacettepe Beytepe); leave it wrong and every track lands somewhere entirely
> plausible and entirely incorrect.

---

## 6. How to run

### Prerequisites

Ubuntu 24.04 with ROS 2 Jazzy and Gazebo Harmonic:

```bash
sudo apt install ros-jazzy-desktop ros-jazzy-ros-gz ros-jazzy-cv-bridge ros-jazzy-tf2-ros python3-opencv python3-numpy python3-yaml
```

### Build

```bash
cd ros2_ws && colcon build --symlink-install && source install/setup.bash
```

### Launch

```bash
ros2 launch mission_bringup mission.launch.py target_qr:=TARGET_2
```

Options: `rviz:=true`, `headless:=true` (no gz GUI; sensors still render),
`auto_start:=false` (wait for the action), `config_file:=/path/to/mission.yaml`.

### Change the target

Three equivalent ways:

```bash
# 1 - launch argument
ros2 launch mission_bringup mission.launch.py target_qr:=TARGET_3

# 2 - action goal (with auto_start:=false)
ros2 action send_goal /mission/run_mission mission_interfaces/action/RunMission \
  "{target_qr: 'TARGET_1'}" --feedback

# 3 - edit mission.target_qr in config/mission.yaml
```

### Watch it

```bash
ros2 topic echo /mission/status
ros2 topic echo /world_model/status
ros2 topic echo /mission/rover_path --once
ros2 run mission_bringup check_frames.py     # verify the TF assumptions
```

### Regenerate the QR station models

Only needed after changing `qr_plate_size_m`, `qr_quiet_zone_modules` or
`known_payloads`:

```bash
python scripts/generate_qr_targets.py
```

This writes the texture, the UV-mapped plate mesh and the SDF for each station
from the same YAML the code reads, so the physical plate size and the size used
for PnP cannot drift apart.

---

## 7. How to run the tests

No ROS installation required — `mission_core` is deliberately ROS-free:

```bash
python -m pytest                          # everything (201 tests, ~7 min)
python -m pytest -m "not integration"     # fast unit tests only (~3 s)
python -m pytest -m integration -v        # full end-to-end mission runs
```

Under a built workspace:

```bash
cd ros2_ws && colcon test --packages-select mission_core && colcon test-result --verbose
```

| File | Covers |
|---|---|
| `test_qr_perception.py` | TEST 1 — decode, PnP, TF, degenerate views, bad frames |
| `test_world_model.py` | TEST 2 — fusion, duplicates, occupancy mapping, obstacle extraction |
| `test_vision_mapping.py` | TEST 7 — monocular obstacle mapping: measured contacts and heights, shadows, sky, close obstacles |
| `test_planner.py` | TEST 3 — no path ever intersects an obstacle |
| `test_mission_failures.py` | TEST 4 & 5 — every `FailureReason`, verification rejection |
| `test_mission_integration.py` | TEST 6 — full mission per target, plus end-to-end failures |
| `test_path_following.py` | controller limits, watchdog, goal capture |
| `test_config.py` | YAML/code agreement, startup sanity checks |
| `test_twin_telemetry.py` | TEST 8 — the ground-station feed against the station's own C# contract: geography, phases, deltas, retractions, datagram limits |
| `test_simulation_consistency.py` | SDF vs YAML vs launch TF vs harness — the closest thing to a Gazebo check that runs without Gazebo |

To watch one mission run end to end with a full log, a perception-accuracy
table and the validation report:

```bash
python scripts/run_offline_mission.py --target TARGET_2 --save-frames out/
```

`--save-frames` also writes the drone and rover camera images, so the imagery
the decoder actually worked from can be inspected.

The integration tests drive the production code through
`test/offline_mission.py` + `test/sim_harness.py`: real QR textures rendered
onto real 3D quads through a real pinhole camera model, decoded by real OpenCV,
posed by real `solvePnP`, and mapped by segmenting and back-projecting those
same rendered frames. No Gazebo, and no shortcuts through the logic.

The harness's surface colours are copied from `mission_arena.sdf` rather than
invented, because the obstacle mapper now separates floor from not-floor by
colour: an offline scene with an easier separation than the real arena would
make every mapping test meaningless.

---

## 8. Configuration

Everything tunable lives in
[`config/mission.yaml`](ros2_ws/src/mission_bringup/config/mission.yaml). Each
node re-declares the values it uses as ROS parameters whose defaults come from
that file, so `ros2 param set` and launch arguments override it without the two
ever disagreeing.

Notable parameters: `drone.scan_altitude_m`, `drone.scan_speed_mps`,
`drone.lane_spacing_m`, `perception.qr_detection_rate_hz`,
`vision.drone_max_range_m`, `vision.chroma_sigma`,
`vision.min_obstacle_height_m`, `planner.obstacle_safety_margin_m`,
`planner.rover_radius_m`, `planner.planning_resolution_m`,
`planner.approach_distance_m`, `rover.goal_tolerance_m`,
`rover.max_linear_velocity`, `mission.mission_timeout_s`.

The whole `vision:` section is the sensor model: with no ranging device on
either vehicle, those values *are* how a frame becomes geometry.

`MissionConfig.validate()` runs at every node's startup and **refuses to
start** on an incoherent configuration. It catches, among others:

* a scan altitude at which the QR codes cannot be resolved (it computes
  pixels-per-module for every payload);
* a lane spacing wider than the mapped ground swath — measured at the near
  edge of the camera's view, where the strip it maps is narrowest, not at the
  station plates where it is widest (which would leave unmapped strips);
* a trusted vision range shorter than the near edge of that view, which would
  discard every contact and leave the grid empty;
* a goal tolerance larger than the approach distance (the rover would stop
  inside the station);
* a planning resolution coarser than the rover radius;
* any unknown key in the YAML — a typo fails loudly instead of silently
  reverting to a default.

---

## 9. No hardcoding — how it is enforced

| Rule | Enforcement |
|---|---|
| No hardcoded target coordinates | The only place a station position enters the system is `QrDetector.detect()` → `solvePnP` → TF. Grep for `TARGET_` outside tests, config and the world file: it appears only as a *payload string*. |
| QR recognition is real | `test_qr_perception.py` decodes rendered textures with `cv2.QRCodeDetector`; the encoder and decoder are exercised against each other. |
| No Gazebo model names | No node subscribes to any pose or model-list topic from the simulator. |
| No teleporting | The rover only ever receives `geometry_msgs/Twist`. |
| Obstacle avoidance is real | `test_planner.py` densely resamples every planned path and asserts it never enters the inflated grid; `test_mission_integration.py` additionally asserts the *driven trajectory* stays clear, and that the straight line would have collided. |
| Position alone is never success | `MissionValidator` requires `rover_qr_verification`; `test_position_alone_never_produces_success` pins it. |
| Ground truth is test-only | `SyntheticWorld.ground_truth_station_xy()` is called from exactly two assertions, both after the mission has finished. |
| No ranging sensor smuggled back in | `test_no_vehicle_carries_a_range_sensor` fails if any SDF gains a `gpu_lidar`, `ray`, `depth_camera` or `rgbd_camera`; a second check fails if the gz bridge ever carries `LaserScan`, `PointCloud2` or `Range`. |
| Obstacle geometry is measured, not assumed | `test_vision_mapping.py` puts a wall of known size at a known place and requires the contact within 5 cm and the height within 10 cm — from one synthetic mask and the camera pose alone. |

---

## 10. Things found and fixed while building this

Recorded because each was a genuine defect caught by *running* the pipeline,
not by reading it. The second group came only from Gazebo — no amount of
offline testing would have surfaced them.

**Found by running in Gazebo (Ubuntu, ROS 2 Jazzy, OpenCV 4.6):**

A. **All three stations displayed the same code.** Every MTL declared a
   material called `qr_material` and every texture was `qr.png`; Ogre caches
   both by *name*, so whichever loaded first was painted on all three. The
   world model spotted one payload at three places and refused to act on it
   with `DUPLICATE_QR` — the safety logic worked, which is why nothing worse
   happened.
B. **The offline harness rendered every texture mirrored.** Texture-right was
   derived from the inward view direction, giving a left-handed basis. On a QR
   code that moves the third finder pattern from bottom-left to bottom-right.
   OpenCV ≥ 4.7 decodes mirrored codes and hid it; OpenCV 4.6 — what Ubuntu
   24.04 and ROS Jazzy ship — decoded *nothing* in 349 frames.
C. **The mission manager crashed on the failure path.** `level = ...error if
   ... else ...info; level(msg)` — rclpy keys its logger cache by caller
   location and rejects a severity change from one line, so the node died the
   first time a mission failed.
D. **Odometry was consumed as a map pose.** gz's `DiffDrive` dead-reckons from
   zero at spawn, so a rover spawned at (-8, -8) reports (0, 0) while parked.
   The rover drove a correctly planned, genuinely collision-free path to a
   point one whole spawn offset from the station, reporting "Goal reached,
   cross-track 0.00 m" throughout. Every vehicle pose now goes through TF.
E. **Odometry yaw drift left the station just outside the camera.** The rover
   arrived in the right place facing ~0.6 rad off. Fixed by looking around
   (`VerificationSweep`) rather than by widening a tolerance.
F. **TF lookups blocked and then discarded data.** Waiting on a transform
   inside a single-threaded executor cannot work — the TF listener is a
   callback on that same executor — and a sensor stamped 20 ms ahead of the
   newest transform had its data dropped. One run mapped zero camera frames
   and produced a completely empty map.
G. **The validator judged a path against a later map.** Runtime obstacle
   mapping grows the grid while the rover drives, so a completed, verified
   mission was failed with `PATH_INTERSECTS_OBSTACLE`. Paths are now checked
   against the grid they were planned on.

**Found by running the offline pipeline:**

1. **PnP range biased by 1.56×.** `QRCodeDetector` returns the corners of the
   *code*, but the plate is what is 0.80 m wide — and `cv2.QRCodeEncoder`
   emits its own quiet zone on top of the configured one. Every target was
   being placed too far along its viewing ray. Fixed by cropping the encoder
   output to the bounding box of its dark modules, which is the code boundary
   by construction, and deriving `code_side_length_m()` from the module counts.
2. **`SOLVEPNP_IPPE_SQUARE` fails on a nadir view.** For a fronto-parallel
   marker — exactly what a downward camera sees — it returns either NaN or, far
   worse, a *finite* pose with an identity rotation and a 129 px residual. Both
   solvers are now always run and every candidate is scored by a reprojection
   error this code computes itself.
3. **Obstacle inflation delivered 0.40 m when 0.55 m was configured.** The
   circular kernel measured centre-to-centre, ignoring that an occupied cell is
   a square. Now widened by half a cell diagonal.
4. **The approach-pose search tested line of sight *through* the station.** The
   target centre is inside a mapped obstacle, so the check could never pass and
   every mission failed with `NO_VALID_PATH`.
5. **A snapped start point produced a path whose first segment was inside the
   inflation zone**, defeating the planner's own collision guarantee.

**Found while replacing the lidars with the cameras:**

6. **The lawnmower covered the ground under the drone, not the ground it can
   see.** A lidar pointing down maps what the aircraft overflies; a camera
   pitched 30° down cannot see anything closer than 2.97 m ahead of itself. The
   unshifted pattern left the first stretch of every lane permanently
   unobserved — including the corner the rover is parked in — and the very
   first plan failed with `NO_VALID_PATH` because the rover's own start cell
   had never been observed. The pattern is now shifted back by exactly that
   near edge.
7. **The offline renderer deleted obstacles the moment they got close.** Any
   face with a single corner behind the image plane was skipped whole, because
   there was no near-plane clipping — so a wall the drone was flying past
   vanished from the frame, the floor was drawn where it should have been, and
   the mapper marked the wall's own footprint free. Harmless when a lidar did
   the mapping; fatal once the mapping is done from these pixels. The renderer
   now clips faces (Sutherland–Hodgman) and builds the texture homography from
   the quad's geometry instead of its four projected corners, which no longer
   all exist.
8. **A per-frame floor-colour model inverts exactly when it matters.** The
   floor is identified as the dominant surface, so a frame filled by a wall
   makes the *wall* the reference and the strip of real floor beside it the
   "obstacle" — at the moment the vehicle is closest to something solid. The
   reference is now learned from frames the floor demonstrably owns and carried
   forward; a frame that cannot establish one is not mapped at all.
9. **A failed validation crashed the mission manager.** `report.failure_reason
   or <default>` — but `FailureReason` is a `str` enum, so `NONE` is *truthy*
   and was passed straight through to a state machine that refuses to enter
   `MISSION_FAILED` without a reason. Not every check carries one (a rover that
   never reached its goal is not a planner fault), so a clean, explainable
   mission failure became a traceback. Found by a mission that legitimately
   failed validation while the sensor change was being brought up.

---

## 11. Known limitations

**The Gazebo evidence predates the sensor change.** The `gazebo-e2e` job runs
the real mission to `MISSION_SUCCESS` on Ubuntu 24.04 with ROS 2 Jazzy and
Gazebo Harmonic — frame tree, sensor plumbing, OBJ/MTL texturing, plugin names
and the whole state machine, for real. That evidence was collected *before*
both lidars were removed. The mission logic, the new mapper and a full offline
mission per target have all been re-run since and pass, but the first run of
the camera-only pipeline inside Gazebo is the next `gazebo-e2e`, and until it
is green this row is a claim about code that has not met the renderer.
Specifically unproven offline: how Ogre's lighting and shadows affect the
floor/not-floor separation, which is the one part of the new pipeline whose
input the offline renderer only approximates. The remaining caveats:

1. **CI flies at 0.5 m/s instead of the shipped 1.6.** GitHub runners have no
   GPU, so Gazebo renders the drone camera through llvmpipe at roughly 0.7
   frames per *simulated* second instead of 6. What governs target
   confirmation is frames per metre flown, so the drone has to fly slower to
   see each station the three times `min_observations` requires.
   `config/mission_ci.yaml` changes that one value and nothing else — two
   tests enforce that the overlay may not touch camera resolution, scan
   altitude, perception thresholds, planner clearances or any verification
   rule, so a green run still means the shipped mission works.
   (Capping the simulator's real-time factor does *not* help: rendering is
   wall-clock bound, so Gazebo drops more frames per simulated second —
   measured 0.7 at RTF 0.15, 0.33 at 0.05.)
2. **Nine runs are still not a statistical success-rate study.** Every workflow
   runs all three targets three times (`3 targets × 3 trials`) and requires all
   nine verdicts to be `MISSION_SUCCESS`. That catches target-specific defects
   and obvious flakiness, but `n=3` per target is too small for a meaningful
   reliability percentage.
3. **The CI world is deterministic.** The trials exercise renderer and callback
   timing variation, but they do not randomise station layouts, obstacle
   geometry, sensor noise or initial pose.
4. **gz-sim plugin filenames** (`gz-sim-velocity-control-system`, etc.) are the
   Harmonic spellings and differ in Garden/Fortress.
5. **No GUI run.** CI runs `headless:=true` (server only). The GUI path and the
   RViz config are unexercised.

`ros2 run mission_bringup check_frames.py` (before the mission starts) and
`ros2 run mission_bringup check_pipeline.py` (any time) remain the two commands
that localise a problem to a stage on a new machine.

**Design limitations, deliberate for the MVP:**

* **The drone is kinematic.** `VelocityControl` tracks a commanded twist; there
  are no rotor dynamics, and gravity is disabled on the drone link. Perception,
  mapping, planning and verification are unaffected, but this is not a
  flight-dynamics demonstration.
* **Localisation is simulator-provided odometry.** There is no VIO, GPS model
  or SLAM. `map → odom` is static, so odometry drift would accumulate
  uncorrected. This is the standard MVP simplification; note that it concerns
  *self*-localisation only — target positions are always perceived.
* **The Gazebo scenario's obstacles are static.** At runtime, a newly appeared
  obstacle is added to the shared map from the rover's own camera; a sustained
  safety stop triggers a bounded replan from the live rover pose. Runtime
  obstacle evidence is currently sticky, so a moved-away obstacle is not
  cleared until the map is reset; evidence decay remains future work.
* **Monocular mapping assumes a flat floor of one appearance.** Every position
  it reports comes from intersecting a ray with the plane `vision.ground_z_m`,
  so a slope, a step or a ramp would be read as a range error, and an obstacle
  whose colour matches the floor's is invisible to it. Both are properties of
  the arena, not of the algorithm, and both would need a real ground model or a
  second cue to lift.
* **A camera maps rims, not volumes.** Only the line where an obstacle meets
  the floor is ever measured; the interior is occluded and stays `UNKNOWN`,
  which is what keeps the planner out of it (`allow_unknown: false`). An
  obstacle much thicker than twice the inflation radius therefore needs to be
  seen from more than one side before its far edge exists in the map at all.
  A handful of rays grazing a silhouette's lateral edge can also mark an
  interior cell free; inflation from the mapped rim covers that for the 1 m
  walls in this arena, and would not for a much thicker one.
* **Measured obstacle height is an upper bound.** Seen from above, the top of a
  silhouette is the object's *far* top edge, and one view cannot separate that
  from a taller object standing at the contact point. The overestimate is about
  `depth x (altitude - height) / range` — 0.4 m on this arena's walls. Height
  feeds the `ObstacleArray` topic, RViz and the "is this tall enough to be an
  obstacle" gate, none of which are harmed by erring high.
* **2D planning.** The rover is a ground vehicle and the arena is flat.
* **Obstacle footprints are axis-aligned bounding boxes** from connected
  components. Fine for the boxes in this world; an L-shaped obstacle would be
  over-approximated. The *planner* uses the grid itself, not these boxes, so
  this only affects the `ObstacleArray` topic and RViz.
* **One mission at a time.** The manager rejects a new action goal while a
  mission is running.
* **Bounded recovery only.** A tracking failure gets one fresh plan from the
  live rover pose, and an unreadable QR gets one fresh heading sweep. A second
  failure or a decoded wrong payload remains terminal. There is no general
  behaviour-tree recovery or alternate-target search.

---

## 12. Troubleshooting

| Symptom | Check |
|---|---|
| No QR observations | `ros2 topic hz /drone/camera/image`; then `ros2 param set /drone_qr_detector publish_annotated_image true` and view `~/annotated_image`. |
| `TF unavailable` warnings | `ros2 run tf2_tools view_frames`; confirm `/tf` is being bridged from `/model/*/tf`. |
| Everything offset by a constant | `ros2 run mission_bringup check_frames.py` (limitation 1 above). |
| Empty occupancy grid | `ros2 topic echo /perception/drone/ground_observations --once` — are there contacts and free samples? If `usable` is false, the floor was not the dominant surface in the frame. Also confirm the `Sensors` system is in the world file. |
| Phantom obstacles across the map | The floor/not-floor split is failing. `ros2 param set /drone_obstacle_mapper vision.chroma_sigma 5.0` widens the tolerance; a mapper that reports a high `non_ground_fraction` on open floor has lost its floor reference. |
| `NO_VALID_PATH` | `ros2 topic echo /world_model/occupancy_grid --once` in RViz; the map may be too sparse, or `obstacle_safety_margin_m` too large for the gap. |
| Mission never leaves `EXPLORING` | `ros2 topic echo /world_model/status` — is the payload listed, and is it `CONFIRMED` rather than `TENTATIVE`/`AMBIGUOUS`? |
| Rover reaches the goal but never verifies | `verification.max_range_m` versus the actual standoff; `ros2 topic echo /perception/rover/qr_observations`. |
