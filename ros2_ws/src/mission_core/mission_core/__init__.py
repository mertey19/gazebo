"""ROS-free implementation of the UAV/UGV mission logic.

Every algorithm that decides *what the robots do* lives here: QR perception,
world-model fusion, occupancy mapping, path planning, path following, the
mission state machine and the success validator.  The ROS 2 nodes in
``mission_nodes`` are thin adapters that move data between DDS topics and these
classes.

The split exists for two reasons: the logic can be unit tested without a ROS
installation, and a defect can be reproduced from a saved image or grid instead
of from a full simulation run.
"""

from .camera import PinholeCamera
from .config import MissionConfig, load_mission_config
from .errors import FailureReason, MissionError, PerceptionError, PlanningError
from .exploration import FlightPhase, WaypointFlightController, lawnmower_waypoints
from .geometry import Transform
from .mission_state import MissionState, StateMachine
from .occupancy import OccupancyGrid, OccupancyMapper
from .orchestrator import (
    MissionCommand,
    MissionInputs,
    MissionOrchestrator,
    MissionOutputs,
)
from .path_following import PurePursuitController, TrackingStatus
from .planner import AStarPlanner, PlannedPath
from .qr import QrDetection, QrDetector, render_qr_image
from .validation import MissionValidator, ValidationReport
from .vision_mapping import (
    GroundObservation,
    MonocularObstacleDetector,
    segment_non_ground,
)
from .world_model import ObstacleRecord, TargetObservation, TargetRecord, TargetStatus, WorldModel

__all__ = [
    "AStarPlanner",
    "FailureReason",
    "FlightPhase",
    "GroundObservation",
    "MonocularObstacleDetector",
    "MissionCommand",
    "MissionInputs",
    "MissionOrchestrator",
    "MissionOutputs",
    "MissionConfig",
    "MissionError",
    "MissionState",
    "MissionValidator",
    "ObstacleRecord",
    "OccupancyGrid",
    "OccupancyMapper",
    "PerceptionError",
    "PinholeCamera",
    "PlannedPath",
    "PlanningError",
    "PurePursuitController",
    "QrDetection",
    "QrDetector",
    "StateMachine",
    "TargetObservation",
    "TargetRecord",
    "TargetStatus",
    "TrackingStatus",
    "Transform",
    "ValidationReport",
    "WaypointFlightController",
    "WorldModel",
    "lawnmower_waypoints",
    "load_mission_config",
    "render_qr_image",
    "segment_non_ground",
]
