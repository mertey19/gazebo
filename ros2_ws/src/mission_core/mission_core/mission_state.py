"""Explicit mission state machine.

The set of legal transitions is data, not control flow scattered through the
mission manager.  An illegal transition raises instead of quietly happening,
which is what makes the mission deterministic and reviewable after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, FrozenSet, List, Optional

from .errors import FailureReason


class MissionState(str, Enum):
    """Every state the mission can occupy."""

    IDLE = "IDLE"
    TAKEOFF = "TAKEOFF"
    EXPLORING = "EXPLORING"
    TARGET_FOUND = "TARGET_FOUND"
    PLANNING = "PLANNING"
    PATH_READY = "PATH_READY"
    SENDING_PATH = "SENDING_PATH"
    ROVER_NAVIGATING = "ROVER_NAVIGATING"
    VERIFYING_TARGET = "VERIFYING_TARGET"
    MISSION_SUCCESS = "MISSION_SUCCESS"
    MISSION_FAILED = "MISSION_FAILED"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: Terminal states: the mission manager stops commanding after reaching one.
TERMINAL_STATES: FrozenSet[MissionState] = frozenset(
    {MissionState.MISSION_SUCCESS, MissionState.MISSION_FAILED}
)

#: The nominal forward progression.  ``MISSION_FAILED`` is reachable from every
#: non-terminal state and is added programmatically below.
_NOMINAL_TRANSITIONS: Dict[MissionState, FrozenSet[MissionState]] = {
    MissionState.IDLE: frozenset({MissionState.TAKEOFF}),
    MissionState.TAKEOFF: frozenset({MissionState.EXPLORING}),
    # EXPLORING -> EXPLORING is not a transition; re-scanning is internal to
    # the explorer.  The mission leaves EXPLORING only once the requested QR
    # has a CONFIRMED record.
    MissionState.EXPLORING: frozenset({MissionState.TARGET_FOUND}),
    MissionState.TARGET_FOUND: frozenset({MissionState.PLANNING}),
    MissionState.PLANNING: frozenset({MissionState.PATH_READY}),
    MissionState.PATH_READY: frozenset({MissionState.SENDING_PATH}),
    MissionState.SENDING_PATH: frozenset({MissionState.ROVER_NAVIGATING}),
    MissionState.ROVER_NAVIGATING: frozenset({MissionState.VERIFYING_TARGET}),
    # Verification may bounce back to planning if the rover reached a station
    # whose code does not match and another candidate is still available.
    MissionState.VERIFYING_TARGET: frozenset(
        {MissionState.MISSION_SUCCESS, MissionState.PLANNING}
    ),
    MissionState.MISSION_SUCCESS: frozenset(),
    MissionState.MISSION_FAILED: frozenset(),
}

ALLOWED_TRANSITIONS: Dict[MissionState, FrozenSet[MissionState]] = {
    state: (targets | {MissionState.MISSION_FAILED} if state not in TERMINAL_STATES else targets)
    for state, targets in _NOMINAL_TRANSITIONS.items()
}


class InvalidTransition(RuntimeError):
    """Raised when the mission manager attempts an undeclared transition."""


@dataclass(frozen=True)
class StateTransition:
    """One recorded state change."""

    previous: MissionState
    current: MissionState
    stamp: float
    detail: str = ""
    failure_reason: FailureReason = FailureReason.NONE

    def __str__(self) -> str:
        suffix = f" ({self.detail})" if self.detail else ""
        return f"{self.previous.value} -> {self.current.value}{suffix}"


@dataclass
class StateMachine:
    """Tracks the current mission state and its history."""

    state: MissionState = MissionState.IDLE
    started_at: float = 0.0
    history: List[StateTransition] = field(default_factory=list)
    failure_reason: FailureReason = FailureReason.NONE
    failure_detail: str = ""
    #: Invoked after every accepted transition; used by the node for logging
    #: and for publishing ``/mission/status``.
    on_transition: Optional[Callable[[StateTransition], None]] = None

    def can_transition(self, target: MissionState) -> bool:
        return target in ALLOWED_TRANSITIONS[self.state]

    def transition(
        self,
        target: MissionState,
        stamp: float,
        detail: str = "",
        failure_reason: FailureReason = FailureReason.NONE,
    ) -> StateTransition:
        """Move to ``target``; raises :class:`InvalidTransition` if illegal."""
        if target is self.state:
            raise InvalidTransition(f"already in state {target.value}")
        if not self.can_transition(target):
            raise InvalidTransition(
                f"illegal transition {self.state.value} -> {target.value}; "
                f"allowed: {sorted(s.value for s in ALLOWED_TRANSITIONS[self.state])}"
            )
        if target is MissionState.MISSION_FAILED and failure_reason is FailureReason.NONE:
            raise ValueError("MISSION_FAILED requires a FailureReason")

        record = StateTransition(self.state, target, float(stamp), detail, failure_reason)
        self.state = target
        self.history.append(record)
        if target is MissionState.MISSION_FAILED:
            self.failure_reason = failure_reason
            self.failure_detail = detail
        if self.on_transition is not None:
            self.on_transition(record)
        return record

    def fail(self, reason: FailureReason, detail: str, stamp: float) -> StateTransition:
        """Convenience abort that is safe to call from any non-terminal state."""
        if self.is_terminal:
            raise InvalidTransition(
                f"cannot fail from terminal state {self.state.value}"
            )
        return self.transition(MissionState.MISSION_FAILED, stamp, detail, reason)

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def succeeded(self) -> bool:
        return self.state is MissionState.MISSION_SUCCESS

    def elapsed(self, now: float) -> float:
        return float(now) - float(self.started_at)

    def time_in_state(self, now: float) -> float:
        if not self.history:
            return self.elapsed(now)
        return float(now) - self.history[-1].stamp

    def visited(self, state: MissionState) -> bool:
        return state is MissionState.IDLE or any(t.current is state for t in self.history)

    def describe(self) -> str:
        """Single-line trace of the whole mission, for the final log line."""
        chain = [MissionState.IDLE.value] + [t.current.value for t in self.history]
        return " -> ".join(chain)
