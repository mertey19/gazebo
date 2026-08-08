"""Mission manager - the node that runs the state machine.

Owns a :class:`mission_core.orchestrator.MissionOrchestrator` and does nothing
clever itself: it collects topic data into a ``MissionInputs`` snapshot, ticks
the orchestrator, and performs whatever side effect comes back.  All mission
policy lives in ``mission_core`` where it is unit tested.

Responsibilities:

* expose ``/mission/run_mission`` (action) and start one automatically when
  ``auto_start`` is set, so a single launch produces a complete mission;
* transmit the planned path on ``/mission/rover_path`` as ``nav_msgs/Path``;
* judge rover-side QR verification;
* publish ``/mission/status`` and log every state transition.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from mission_core.errors import FailureReason, PlanningError
from mission_core.mission_state import MissionState
from mission_core.orchestrator import MissionCommand, MissionInputs, MissionOrchestrator
from mission_core.world_model import WorldModel

from mission_interfaces.action import RunMission
from mission_interfaces.msg import (
    ExplorationStatus,
    MissionStatus,
    QrObservation,
    TargetArray,
    TrackingStatus,
)
from mission_interfaces.srv import PlanPath

from .common import (
    DEFAULT_QOS,
    LATCHED_QOS,
    SENSOR_QOS,
    declare_mission_config,
    make_header,
    node_time_seconds,
    occupancy_from_msg,
    odometry_pose_in_frame,
    planned_path_to_msg,
    target_record_from_msg,
)


class MissionManagerNode(Node):
    """Drives one mission at a time from IDLE to a terminal state."""

    def __init__(self) -> None:
        super().__init__("mission_manager")
        self.config = declare_mission_config(self)

        self.declare_parameter("auto_start", True)
        self.declare_parameter("target_qr", self.config.mission.target_qr)
        self.declare_parameter("rover_path_topic", "/mission/rover_path")
        self.declare_parameter("rover_odometry_topic", "/rover/odometry")
        self.declare_parameter("drone_start_service", "/drone_explorer/start")
        self.declare_parameter("rover_stop_service", "/rover_path_follower/stop")
        self.declare_parameter("rover_search_service", "/rover_path_follower/search")

        self.map_frame = self.config.frames.map_frame
        # Local mirror of the digital twin. The world_model node is the single
        # authority on fusion; this copy is read-only input to planning.
        self.world_model = WorldModel(
            association_radius_m=self.config.world_model.association_radius_m,
            min_observations=self.config.world_model.min_observations,
            min_confidence=self.config.world_model.min_confidence,
        )
        self.orchestrator: Optional[MissionOrchestrator] = None
        self._lock = threading.Lock()

        # -- latest inputs
        from tf2_ros import Buffer, TransformListener

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._exploration: Optional[ExplorationStatus] = None
        self._rover_pose: Optional[np.ndarray] = None
        self._tracking: Optional[TrackingStatus] = None
        self._path_published = False
        self._verified_qr: Optional[str] = None
        self._verification_reads: List[str] = []
        self._goal_handle = None
        self._logged_messages = 0

        # The action's execute callback blocks for the whole mission. It must
        # NOT share a mutually-exclusive group with the tick timer, or the
        # timer that advances the mission could never run while the action is
        # waiting for it to finish - a guaranteed deadlock.
        sensors = ReentrantCallbackGroup()
        action_group = ReentrantCallbackGroup()
        control = MutuallyExclusiveCallbackGroup()

        self.path_pub = self.create_publisher(
            Path, str(self.get_parameter("rover_path_topic").value), LATCHED_QOS
        )
        self.status_pub = self.create_publisher(MissionStatus, "/mission/status", LATCHED_QOS)

        self.create_subscription(
            TargetArray, "/world_model/targets", self._on_targets, LATCHED_QOS,
            callback_group=sensors,
        )
        self.create_subscription(
            OccupancyGridMsg, "/world_model/occupancy_grid", self._on_grid, LATCHED_QOS,
            callback_group=sensors,
        )
        self.create_subscription(
            ExplorationStatus, "/drone/exploration_status", self._on_exploration, LATCHED_QOS,
            callback_group=sensors,
        )
        self.create_subscription(
            Odometry, str(self.get_parameter("rover_odometry_topic").value),
            self._on_rover_odometry, SENSOR_QOS, callback_group=sensors,
        )
        self.create_subscription(
            TrackingStatus, "/rover/tracking_status", self._on_tracking, LATCHED_QOS,
            callback_group=sensors,
        )
        self.create_subscription(
            QrObservation, "/perception/rover/qr_observations",
            self._on_rover_observation, DEFAULT_QOS, callback_group=sensors,
        )

        self.drone_start_client = self.create_client(
            Trigger, str(self.get_parameter("drone_start_service").value)
        )
        self.rover_stop_client = self.create_client(
            Trigger, str(self.get_parameter("rover_stop_service").value)
        )
        self.rover_search_client = self.create_client(
            Trigger, str(self.get_parameter("rover_search_service").value)
        )

        self.create_service(PlanPath, "/mission/plan_path", self._on_plan_path)
        self.create_service(Trigger, "/mission/abort", self._on_abort)
        self.action_server = ActionServer(
            self,
            RunMission,
            "/mission/run_mission",
            execute_callback=self._execute_mission,
            goal_callback=self._accept_goal,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=action_group,
        )

        self.create_timer(0.1, self._tick, callback_group=control)
        self.create_timer(1.0, self._publish_status, callback_group=control)

        if bool(self.get_parameter("auto_start").value):
            # One-shot: create the orchestrator once the graph has settled, so
            # the latched world-model topics have been delivered first.
            self._autostart_timer = self.create_timer(
                2.0, self._autostart, callback_group=control
            )
        self.get_logger().info("[MISSION] manager ready")

    # -- input callbacks ---------------------------------------------------
    def _on_targets(self, msg: TargetArray) -> None:
        with self._lock:
            self.world_model.replace_targets(
                [target_record_from_msg(record) for record in msg.targets]
            )

    def _on_grid(self, msg: OccupancyGridMsg) -> None:
        with self._lock:
            self.world_model.set_occupancy(occupancy_from_msg(msg))

    def _on_exploration(self, msg: ExplorationStatus) -> None:
        self._exploration = msg

    def _on_rover_odometry(self, msg: Odometry) -> None:
        # Planning happens in the map frame, so the start pose must be in the
        # map frame. Raw odometry is spawn-relative; see odometry_pose_in_frame.
        pose, error = odometry_pose_in_frame(self.tf_buffer, msg, self.map_frame)
        if pose is None:
            self.get_logger().warn(
                f"[MISSION] cannot express rover odometry in {self.map_frame}: {error}",
                throttle_duration_sec=5.0,
            )
            return
        self._rover_pose = pose

    def _on_tracking(self, msg: TrackingStatus) -> None:
        self._tracking = msg

    def _on_rover_observation(self, msg: QrObservation) -> None:
        """Accumulate rover-side reads; only consistent close-range ones count."""
        if self.orchestrator is None:
            return
        if self.orchestrator.state is not MissionState.VERIFYING_TARGET:
            return
        if msg.range_m > self.config.verification.max_range_m:
            # A code readable from further away belongs to a different station
            # seen past the one the rover is parked at.
            self.get_logger().debug(
                f"[QR] ignoring {msg.qr_id} at {msg.range_m:.2f} m "
                f"(> {self.config.verification.max_range_m:.2f} m)"
            )
            return
        self._verification_reads.append(msg.qr_id)
        required = self.config.verification.required_consecutive_reads
        recent = self._verification_reads[-required:]
        if len(recent) >= required and len(set(recent)) == 1:
            self._verified_qr = recent[0]

    # -- mission lifecycle -------------------------------------------------
    def _autostart(self) -> None:
        self._autostart_timer.cancel()
        target = str(self.get_parameter("target_qr").value)
        self.get_logger().info(f"[MISSION] auto-starting mission for {target}")
        self._begin(target)

    def _begin(self, target_qr: str) -> None:
        with self._lock:
            self.orchestrator = MissionOrchestrator(
                self.config, self.world_model, requested_qr=target_qr
            )
            self._path_published = False
            self._verified_qr = None
            self._verification_reads.clear()
            self._logged_messages = 0

    def _accept_goal(self, goal_request) -> GoalResponse:
        if self.orchestrator is not None and not self.orchestrator.machine.is_terminal:
            self.get_logger().warn(
                f"[MISSION] rejecting goal {goal_request.target_qr!r}: mission "
                f"{self.orchestrator.requested_qr!r} is still running in state "
                f"{self.orchestrator.state.value}"
            )
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute_mission(self, goal_handle):
        target = goal_handle.request.target_qr or self.config.mission.target_qr
        self.get_logger().info(f"[MISSION] action goal accepted: {target}")
        self._begin(target)
        self._goal_handle = goal_handle

        # A plain sleep rather than rclpy Rate: this callback runs on its own
        # executor thread and only needs a poll interval, and Rate's internal
        # timer would add another object competing for executor threads.
        poll_interval_s = 0.2
        while rclpy.ok():
            orchestrator = self.orchestrator
            if orchestrator is None:
                break
            if goal_handle.is_cancel_requested:
                self._stop_rover()
                goal_handle.canceled()
                self.get_logger().warn("[MISSION] goal cancelled by client")
                return self._build_result(orchestrator, cancelled=True)
            if orchestrator.machine.is_terminal:
                break
            feedback = RunMission.Feedback()
            feedback.state = orchestrator.state.value
            feedback.detail = orchestrator.machine.failure_detail
            feedback.elapsed_s = float(
                orchestrator.machine.elapsed(node_time_seconds(self))
            )
            summary = self.world_model.summary()
            feedback.targets_confirmed = int(summary["targets_confirmed"])
            feedback.obstacles_mapped = int(summary["obstacles"])
            feedback.coverage_fraction = float(
                self._exploration.coverage_fraction if self._exploration else 0.0
            )
            goal_handle.publish_feedback(feedback)
            time.sleep(poll_interval_s)

        orchestrator = self.orchestrator
        if orchestrator is None:  # pragma: no cover - shutdown race
            goal_handle.abort()
            return RunMission.Result()
        if orchestrator.machine.succeeded:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return self._build_result(orchestrator)

    def _build_result(self, orchestrator, cancelled: bool = False):
        result = RunMission.Result()
        result.success = bool(orchestrator.machine.succeeded and not cancelled)
        result.final_state = orchestrator.state.value
        result.failure_reason = orchestrator.machine.failure_reason.value
        result.verified_qr = orchestrator.verified_qr or ""
        result.path_length_m = float(
            orchestrator.path.length_m if orchestrator.path is not None else 0.0
        )
        result.path_poses = int(len(orchestrator.path) if orchestrator.path is not None else 0)
        result.duration_s = float(orchestrator.machine.elapsed(node_time_seconds(self)))
        report = orchestrator.report
        if report is not None:
            result.check_names = [c.name for c in report.checks]
            result.check_passed = [bool(c.passed) for c in report.checks]
            result.check_details = [c.detail for c in report.checks]
        return result

    # -- main tick ---------------------------------------------------------
    def _tick(self) -> None:
        orchestrator = self.orchestrator
        if orchestrator is None or orchestrator.machine.is_terminal:
            return

        with self._lock:
            inputs = MissionInputs(
                now=node_time_seconds(self),
                # The drone is "ready" once it is publishing odometry-derived
                # exploration status; starting before that would command a
                # takeoff nobody is listening to.
                drone_ready=self._exploration is not None,
                drone_at_scan_altitude=bool(
                    self._exploration.at_scan_altitude if self._exploration else False
                ),
                exploration_complete=bool(
                    self._exploration.complete if self._exploration else False
                ),
                rover_pose=self._rover_pose,
                rover_goal_reached=bool(self._tracking.goal_reached if self._tracking else False),
                rover_tracking_failed=bool(self._tracking.failed if self._tracking else False),
                rover_failure_detail=self._tracking.detail if self._tracking else "",
                verified_qr=self._verified_qr,
                path_published=self._path_published,
            )
            outputs = orchestrator.update(inputs)

        for message in outputs.messages:
            # Separate call sites on purpose: rclpy keys its logger cache by
            # caller location and raises "Logger severity cannot be changed
            # between calls" if one line logs at two severities. Picking a
            # bound method dynamically crashed the node the first time a
            # mission failed - i.e. exactly on the path that must stay alive.
            if "FAILED" in message:
                self.get_logger().error(message)
            else:
                self.get_logger().info(message)
        if outputs.transition is not None:
            self.get_logger().info(f"[MISSION] state {outputs.transition}")
            self._publish_status()

        self._handle_command(outputs)

    def _handle_command(self, outputs) -> None:
        if outputs.command is MissionCommand.START_TAKEOFF:
            self._call_trigger(self.drone_start_client, "drone explorer start")
        elif outputs.command is MissionCommand.PUBLISH_PATH and outputs.path is not None:
            msg = planned_path_to_msg(outputs.path, self, self.map_frame)
            self.path_pub.publish(msg)
            self._path_published = True
            self.get_logger().info(
                f"[MISSION] published {len(msg.poses)} poses on "
                f"{self.path_pub.topic_name} in frame {self.map_frame}"
            )
        elif outputs.command is MissionCommand.START_VERIFICATION:
            # Each attempt must earn its own consecutive-read quorum; stale
            # reads from an earlier sweep cannot complete a later attempt.
            self._verified_qr = None
            self._verification_reads.clear()
            self._call_trigger(self.rover_search_client, "rover verification sweep")
        elif outputs.command is MissionCommand.PREPARE_REPLAN:
            # The old TrackingStatus is transient-local and remains FAILED
            # until the follower publishes for the new path. Discard it here
            # so it cannot consume the recovery budget a second time.
            self._tracking = None
            self._path_published = False
            self._stop_rover()
        elif outputs.command is MissionCommand.STOP_ROVER:
            self._stop_rover()

        if outputs.report is not None and outputs.is_terminal:
            for line in outputs.report.render().splitlines():
                self.get_logger().info(line)
            self.get_logger().info(f"[MISSION] trace: {self.orchestrator.describe()}")

    def _stop_rover(self) -> None:
        self._call_trigger(self.rover_stop_client, "rover stop")

    def _call_trigger(self, client, label: str) -> None:
        if not client.service_is_ready():
            self.get_logger().warn(
                f"[MISSION] {label} service not available at {client.srv_name}",
                throttle_duration_sec=5.0,
            )
            return
        future = client.call_async(Trigger.Request())
        future.add_done_callback(
            lambda f: self.get_logger().debug(f"[MISSION] {label} -> {f.result()}")
        )

    # -- services ----------------------------------------------------------
    def _on_plan_path(self, request, response):
        """Standalone planning query, independent of the running mission."""
        with self._lock:
            record = self.world_model.get_target(request.target_qr)
            grid = self.world_model.occupancy
        if grid is None:
            response.success = False
            response.failure_reason = FailureReason.NO_VALID_PATH.value
            response.message = "no occupancy grid has been published yet"
            return response
        if record is None or not record.is_usable:
            response.success = False
            response.failure_reason = FailureReason.TARGET_NOT_DISCOVERED.value
            response.message = (
                f"{request.target_qr} is not a confirmed target "
                f"(status={record.status.value if record else 'NEVER_OBSERVED'}); "
                "the drone must discover it first"
            )
            return response

        if request.use_current_rover_pose:
            if self._rover_pose is None:
                response.success = False
                response.failure_reason = FailureReason.LOCALIZATION_UNAVAILABLE.value
                response.message = "rover odometry is unavailable"
                return response
            start = self._rover_pose[:2]
        else:
            start = np.array([request.start.x, request.start.y])

        planner = (
            self.orchestrator.planner
            if self.orchestrator is not None
            else MissionOrchestrator(self.config, self.world_model).planner
        )
        try:
            inflated = planner.inflate(grid)
            goal = planner.select_approach_pose(
                grid,
                record.position[:2],
                start,
                approach_distance_m=self.config.planner.approach_distance_m,
                target_clearance_m=self.config.planner.target_footprint_radius_m,
                inflated_grid=inflated,
                distance_tolerance_m=self.config.planner.approach_distance_tolerance_m,
                samples=self.config.planner.approach_samples,
            )
            path = planner.plan(
                inflated, start, goal[:2], goal_yaw=float(goal[2]), pre_inflated=True
            )
        except PlanningError as exc:
            response.success = False
            response.failure_reason = exc.reason.value
            response.message = exc.detail or str(exc)
            return response

        response.success = True
        response.path = planned_path_to_msg(path, self, self.map_frame)
        response.length_m = float(path.length_m)
        response.expanded_nodes = int(path.expanded_nodes)
        response.clearance_m = float(path.inflation_radius_m)
        response.failure_reason = FailureReason.NONE.value
        response.message = f"planned {len(path)} poses"
        return response

    def _on_abort(self, _request, response):
        orchestrator = self.orchestrator
        if orchestrator is None or orchestrator.machine.is_terminal:
            response.success = False
            response.message = "no mission is running"
            return response
        orchestrator.machine.fail(
            FailureReason.MISSION_TIMEOUT, "aborted by operator", node_time_seconds(self)
        )
        self._stop_rover()
        response.success = True
        response.message = f"mission aborted in state {orchestrator.state.value}"
        self.get_logger().error("[MISSION] aborted by operator request")
        return response

    # -- status ------------------------------------------------------------
    def _publish_status(self) -> None:
        orchestrator = self.orchestrator
        msg = MissionStatus()
        msg.header = make_header(self, self.map_frame)
        if orchestrator is None:
            msg.state = MissionState.IDLE.value
            msg.requested_qr = str(self.get_parameter("target_qr").value)
            msg.failure_reason = FailureReason.NONE.value
            msg.trace = MissionState.IDLE.value
            self.status_pub.publish(msg)
            return
        msg.state = orchestrator.state.value
        msg.requested_qr = orchestrator.requested_qr
        msg.verified_qr = orchestrator.verified_qr or ""
        msg.failure_reason = orchestrator.machine.failure_reason.value
        msg.failure_detail = orchestrator.machine.failure_detail
        msg.elapsed_s = float(orchestrator.machine.elapsed(node_time_seconds(self)))
        msg.path_poses = int(len(orchestrator.path) if orchestrator.path is not None else 0)
        msg.path_length_m = float(
            orchestrator.path.length_m if orchestrator.path is not None else 0.0
        )
        msg.trace = orchestrator.describe()
        self.status_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionManagerNode()
    # The action server blocks its own callback while a mission runs, so the
    # sensor callbacks and the tick timer need their own threads.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
