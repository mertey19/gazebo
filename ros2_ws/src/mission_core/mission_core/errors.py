"""Mission failure taxonomy.

Every abort path in the stack maps onto exactly one :class:`FailureReason` so
that logs, the ``/mission/status`` topic and the automated validator all speak
the same language.  Silent continuation after any of these is a bug.
"""

from __future__ import annotations

from enum import Enum


class FailureReason(str, Enum):
    """Reason a mission stopped short of success."""

    NONE = "NONE"
    #: The mission ran out of exploration time without seeing the requested QR.
    TARGET_NOT_DISCOVERED = "TARGET_NOT_DISCOVERED"
    #: The requested payload was never decoded by any camera.
    QR_NOT_DETECTED = "QR_NOT_DETECTED"
    #: The same payload was observed at two spatially incompatible places.
    DUPLICATE_QR = "DUPLICATE_QR"
    #: A camera frame was unusable (empty, wrong encoding, degenerate).
    INVALID_CAMERA_FRAME = "INVALID_CAMERA_FRAME"
    #: The planner could not connect start and goal through free space.
    NO_VALID_PATH = "NO_VALID_PATH"
    #: The produced path crosses an obstacle - a planner defect, never shipped.
    PATH_INTERSECTS_OBSTACLE = "PATH_INTERSECTS_OBSTACLE"
    #: The rover drifted further from the path than the controller allows.
    PATH_TRACKING_FAILURE = "PATH_TRACKING_FAILURE"
    #: The rover did not reach the goal within the allotted time.
    NAVIGATION_TIMEOUT = "NAVIGATION_TIMEOUT"
    #: The rover arrived but read a different payload than requested.
    QR_VERIFICATION_MISMATCH = "QR_VERIFICATION_MISMATCH"
    #: The rover arrived but could not read any payload at all.
    QR_VERIFICATION_TIMEOUT = "QR_VERIFICATION_TIMEOUT"
    #: The whole mission exceeded ``mission_timeout_s``.
    MISSION_TIMEOUT = "MISSION_TIMEOUT"
    #: Localisation / TF was unavailable when it was needed.
    LOCALIZATION_UNAVAILABLE = "LOCALIZATION_UNAVAILABLE"
    #: Operator asked for a payload that is not a configured mission target.
    UNKNOWN_TARGET_REQUESTED = "UNKNOWN_TARGET_REQUESTED"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class MissionError(RuntimeError):
    """Raised by core algorithms when a mission-fatal condition is detected."""

    def __init__(self, reason: FailureReason, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason.value}: {detail}" if detail else reason.value)


class PlanningError(MissionError):
    """No collision-free path could be produced."""

    def __init__(self, detail: str = "", reason: FailureReason = FailureReason.NO_VALID_PATH) -> None:
        super().__init__(reason, detail)


class PerceptionError(MissionError):
    """A camera frame could not be processed."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(FailureReason.INVALID_CAMERA_FRAME, detail)
