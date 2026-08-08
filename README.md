# Autonomous UAV + UGV QR Mission

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

| Layer | Status |
|---|---|
| `mission_core` algorithms (QR/PnP, world model, occupancy, A*, pure pursuit, state machine, validator) | **Executed and tested.** 128 tests, including a full end-to-end mission per target. |
| Offline mission harness (renders real QR textures through a real pinhole camera, ray-casts a lidar, integrates unicycle kinematics) | **Executed.** Drives the production pipeline with zero simulator. Reproduce with `python scripts/run_offline_mission.py`. |
| Gazebo world, SDF models, `ros_gz` bridge, launch file, ROS 2 nodes | **Written, not executed.** 29 of the 128 tests are static cross-file consistency checks over exactly these files (§7). |

The development host is Windows 11 with no WSL2 and no Docker, so ROS 2 and
gz-sim cannot be installed on it. Every claim in this README about the
`mission_core` pipeline is backed by a test run; nothing here claims the
Gazebo layer has been run. See [§11 Known limitations](#11-known-limitations)
for exactly what that leaves unverified and how to check it in one command.

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
   |  nadir camera     front camera       unique QR plate                                   |
   |  nadir 3D lidar   planar lidar       (top + 4 sides)                                   |
   +--------------------------------------------------------------------------------------+
        |  image/points/odom                                       cmd_vel  ^
        v                                                                   |
   +----------------------------- ros_gz_bridge + ros_gz_image -----------------------------+
        |                    |                    |                         |
        v                    v                    v                         |
 /drone/camera/image   /drone/scan/points   /rover/camera/image              |
        |                    |                    |                         |
        v                    |                    v                         |
 +----------------+          |          +----------------+                  |
 | qr_detector    |          |          | qr_detector    |                  |
 |    (drone)     |          |          |    (rover)     |                  |
 | decode -> PnP  |          |          | decode -> PnP  |                  |
 | -> TF -> map   |          |          | -> TF -> map   |                  |
 +-------+--------+          |          +--------+-------+                  |
         |                   |                   |                          |
   QrObservation             |             QrObservation                    |
         |                   |                   |                          |
         v                   v                   |                          |
   +-----------------------------------+         |                          |
   |         world_model_node          |         |                          |
   |  fuse sightings -> TargetRecord   |         |                          |
   |  lidar -> OccupancyGrid           |         |                          |
   |  connected components -> Obstacle |         |                          |
   +------+---------------+------------+         |                          |
          |               |                      |                          |
  /world_model/targets   /world_model/           |                          |
  /world_model/obstacles  occupancy_grid         |                          |
          |               |                      |                          |
          v               v                      v                          |
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
│   │   ├── validation.py      the success criteria
│   │   └── world_model.py     the digital twin
│   └── test/                  128 tests, incl. sim_harness/offline_mission
├── mission_interfaces/    9 msgs, 2 srvs, 1 action
├── mission_nodes/         5 rclpy nodes (thin adapters over mission_core)
└── mission_bringup/       world, models, config, launch, rviz, frame checker
scripts/generate_qr_targets.py   generates the station models from mission.yaml
```

---

## 4. Mission flow

1. **IDLE** — validate the requested payload against `known_payloads`. An
   unknown payload fails here, before takeoff.
2. **TAKEOFF** — climb to `drone.scan_altitude_m` (6.0 m).
3. **EXPLORING** — fly a boustrophedon pattern with 5.0 m lanes. The camera
   footprint is 6.93 m at that altitude, so lanes overlap and nothing is
   missed. Each frame is decoded; each decode is turned into a marker pose by
   `solvePnP` and lifted into `map` through TF. The downward lidar fills an
   occupancy grid in parallel.
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
   navigation timeout.
8. **VERIFYING_TARGET** — the rover's own camera must decode the same payload,
   `required_consecutive_reads` times, within `verification.max_range_m`.
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
| `/drone/scan/points` | `sensor_msgs/PointCloud2` | gz bridge | sensor |
| `/drone/odometry` | `nav_msgs/Odometry` | gz bridge | sensor |
| `/drone/cmd_vel` | `geometry_msgs/Twist` | `drone_explorer` | reliable |
| `/drone/exploration_status` | `mission_interfaces/ExplorationStatus` | `drone_explorer` | latched |
| `/rover/camera/image` | `sensor_msgs/Image` | gz bridge | sensor |
| `/rover/camera/camera_info` | `sensor_msgs/CameraInfo` | gz bridge | sensor |
| `/rover/scan` | `sensor_msgs/LaserScan` | gz bridge | sensor |
| `/rover/odometry` | `nav_msgs/Odometry` | gz bridge | sensor |
| `/rover/cmd_vel` | `geometry_msgs/Twist` | `rover_path_follower` | reliable |
| `/rover/tracking_status` | `mission_interfaces/TrackingStatus` | `rover_path_follower` | latched |
| `/perception/drone/qr_observations` | `mission_interfaces/QrObservation` | `drone_qr_detector` | reliable |
| `/perception/rover/qr_observations` | `mission_interfaces/QrObservation` | `rover_qr_detector` | reliable |
| `/world_model/targets` | `mission_interfaces/TargetArray` | `world_model` | latched |
| `/world_model/obstacles` | `mission_interfaces/ObstacleArray` | `world_model` | latched |
| `/world_model/occupancy_grid` | `nav_msgs/OccupancyGrid` | `world_model` | latched |
| `/world_model/status` | `mission_interfaces/WorldModelStatus` | `world_model` | latched |
| `/world_model/markers` | `visualization_msgs/MarkerArray` | `world_model` | latched |
| **`/mission/rover_path`** | **`nav_msgs/Path`** | `mission_manager` | latched |
| `/mission/status` | `mission_interfaces/MissionStatus` | `mission_manager` | latched |

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
│                                 └─ drone/lidar_link
└── rover/odom ── rover/base_link ── rover/camera_optical_frame
                                  └─ rover/lidar_link
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
* The drone lidar is mounted **level** and aimed downward through its vertical
  scan range instead of by rotating the sensor, so `base_link → lidar_link` is
  a pure translation. One less rotation to get wrong.

Camera optical frames follow REP-103: x right, y down, z forward. QR poses come
out of `solvePnP` in the optical frame and are transformed to `map` with a
single TF lookup — there are no corrective offsets anywhere in the codebase.

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
python -m pytest                          # everything (128 tests, ~2 min)
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
| `test_planner.py` | TEST 3 — no path ever intersects an obstacle |
| `test_mission_failures.py` | TEST 4 & 5 — every `FailureReason`, verification rejection |
| `test_mission_integration.py` | TEST 6 — full mission per target, plus end-to-end failures |
| `test_path_following.py` | controller limits, watchdog, goal capture |
| `test_config.py` | YAML/code agreement, startup sanity checks |
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
posed by real `solvePnP`, mapped by a real ray-cast lidar. No Gazebo, and no
shortcuts through the logic.

---

## 8. Configuration

Everything tunable lives in
[`config/mission.yaml`](ros2_ws/src/mission_bringup/config/mission.yaml). Each
node re-declares the values it uses as ROS parameters whose defaults come from
that file, so `ros2 param set` and launch arguments override it without the two
ever disagreeing.

Notable parameters: `drone.scan_altitude_m`, `drone.scan_speed_mps`,
`drone.lane_spacing_m`, `perception.qr_detection_rate_hz`,
`planner.obstacle_safety_margin_m`, `planner.rover_radius_m`,
`planner.planning_resolution_m`, `planner.approach_distance_m`,
`rover.goal_tolerance_m`, `rover.max_linear_velocity`,
`mission.mission_timeout_s`.

`MissionConfig.validate()` runs at every node's startup and **refuses to
start** on an incoherent configuration. It catches, among others:

* a scan altitude at which the QR codes cannot be resolved (it computes
  pixels-per-module for every payload);
* a lane spacing wider than the camera's ground footprint (which would leave
  unscanned strips);
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

---

## 10. Things found and fixed while building this

Recorded because each was a genuine defect caught by running the pipeline, not
by reading it:

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

---

## 11. Known limitations

**Not executed on this host.** ROS 2 and gz-sim cannot be installed on the
Windows 11 development machine (no WSL2, no Docker). The Gazebo world, the SDF
models, the bridge configuration, the launch file and the five ROS nodes are
written but have never been run. Concretely, these are the assumptions a first
Ubuntu run should check:

1. **`map → */odom` static transforms.** The launch file assumes gz's
   `OdometryPublisher` reports a *world* pose (so `map → drone/odom` is
   identity) while `DiffDrive` dead-reckons from zero at spawn (so
   `map → rover/odom` is the rover's spawn pose). If either changed, every
   position shifts by a constant offset.
   → `ros2 run mission_bringup check_frames.py` prints the residual and names
   the transform to fix.
2. **`<gz_frame_id>` support** on the sensors, and `<optical_frame_id>` on the
   cameras. The QR node warns (and continues, using the configured frame) if an
   image arrives with an unexpected `frame_id`.
3. **OBJ/MTL texturing** of the station plates in ogre2. Explicit UV-mapped
   quads were chosen over textured primitive boxes specifically to avoid
   depending on how a renderer UV-maps a cube, but the material path itself is
   untested. → `ros2 topic echo /perception/drone/qr_observations` after
   takeoff; nothing arriving means the plates are not rendering.
4. **gz-sim plugin filenames** (`gz-sim-velocity-control-system`, etc.) are the
   Harmonic spellings and differ in Garden/Fortress.

**Design limitations, deliberate for the MVP:**

* **The drone is kinematic.** `VelocityControl` tracks a commanded twist; there
  are no rotor dynamics, and gravity is disabled on the drone link. Perception,
  mapping, planning and verification are unaffected, but this is not a
  flight-dynamics demonstration.
* **Localisation is simulator-provided odometry.** There is no VIO, GPS model
  or SLAM. `map → odom` is static, so odometry drift would accumulate
  uncorrected. This is the standard MVP simplification; note that it concerns
  *self*-localisation only — target positions are always perceived.
* **Obstacles are static.** The architecture does not preclude dynamic ones:
  the mapper accumulates per-cell evidence and the planner re-reads the grid on
  every plan. What is missing is evidence *decay* and replanning-on-change.
* **2D planning.** The rover is a ground vehicle and the arena is flat.
* **Obstacle footprints are axis-aligned bounding boxes** from connected
  components. Fine for the boxes in this world; an L-shaped obstacle would be
  over-approximated. The *planner* uses the grid itself, not these boxes, so
  this only affects the `ObstacleArray` topic and RViz.
* **One mission at a time.** The manager rejects a new action goal while a
  mission is running.
* **No recovery behaviours.** A tracking failure or a verification mismatch
  ends the mission. The state machine has a `VERIFYING_TARGET → PLANNING` edge
  reserved for retrying against another candidate, but nothing drives it yet.

---

## 12. Troubleshooting

| Symptom | Check |
|---|---|
| No QR observations | `ros2 topic hz /drone/camera/image`; then `ros2 param set /drone_qr_detector publish_annotated_image true` and view `~/annotated_image`. |
| `TF unavailable` warnings | `ros2 run tf2_tools view_frames`; confirm `/tf` is being bridged from `/model/*/tf`. |
| Everything offset by a constant | `ros2 run mission_bringup check_frames.py` (limitation 1 above). |
| Empty occupancy grid | `ros2 topic hz /drone/scan/points`, and confirm the `Sensors` system is in the world file. |
| `NO_VALID_PATH` | `ros2 topic echo /world_model/occupancy_grid --once` in RViz; the map may be too sparse, or `obstacle_safety_margin_m` too large for the gap. |
| Mission never leaves `EXPLORING` | `ros2 topic echo /world_model/status` — is the payload listed, and is it `CONFIRMED` rather than `TENTATIVE`/`AMBIGUOUS`? |
| Rover reaches the goal but never verifies | `verification.max_range_m` versus the actual standoff; `ros2 topic echo /perception/rover/qr_observations`. |
