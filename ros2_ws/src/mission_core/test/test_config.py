"""Configuration loading and the startup sanity checks it performs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from mission_core.config import (
    ConfigError,
    DroneConfig,
    MissionConfig,
    MissionConfigSection,
    PlannerConfig,
    config_to_dict,
    load_mission_config,
    mission_config_from_dict,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
SHIPPED_CONFIG = REPO_ROOT / "ros2_ws" / "src" / "mission_bringup" / "config" / "mission.yaml"


def test_shipped_configuration_loads_and_validates() -> None:
    assert SHIPPED_CONFIG.is_file(), f"missing {SHIPPED_CONFIG}"
    config = load_mission_config(SHIPPED_CONFIG)
    assert config.validate() == []
    assert config.mission.target_qr in config.mission.known_payloads


def test_shipped_configuration_matches_the_code_defaults() -> None:
    """A drift between YAML and dataclass defaults is a silent trap."""
    assert config_to_dict(load_mission_config(SHIPPED_CONFIG)) == config_to_dict(MissionConfig())


def test_round_trip_through_dict_is_lossless() -> None:
    config = MissionConfig()
    assert config_to_dict(mission_config_from_dict(config_to_dict(config))) == config_to_dict(
        config
    )


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    """A typo must fail at startup rather than silently use a default."""
    path = tmp_path / "typo.yaml"
    path.write_text(
        yaml.safe_dump({"mission": {"target_qr": "TARGET_1", "targt_qr": "TARGET_2"}}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unknown key"):
        load_mission_config(path)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_mission_config(tmp_path / "nope.yaml")


def test_scan_altitude_that_cannot_be_decoded_is_rejected() -> None:
    """The single most useful startup check: can the drone read the codes?"""
    config = MissionConfig()
    too_high = replace(config, drone=replace(config.drone, scan_altitude_m=30.0))
    problems = too_high.validate()
    assert any("px per QR module" in p for p in problems)
    with pytest.raises(ConfigError):
        too_high.require_valid()


def test_lane_spacing_wider_than_the_camera_footprint_is_rejected() -> None:
    config = MissionConfig()
    gappy = replace(config, drone=replace(config.drone, lane_spacing_m=20.0))
    assert any("footprint" in p for p in gappy.validate())


def test_goal_tolerance_must_stay_inside_the_approach_distance() -> None:
    config = MissionConfig()
    bad = replace(config, planner=replace(config.planner, approach_distance_m=0.2))
    assert any("approach_distance_m" in p for p in bad.validate())


def test_recovery_budgets_are_bounded() -> None:
    config = MissionConfig()
    negative_replans = replace(config, rover=replace(config.rover, max_replans=-1))
    no_verification_attempts = replace(
        config, verification=replace(config.verification, max_attempts=0)
    )
    no_safety_delay = replace(
        config, rover=replace(config.rover, safety_stop_replan_delay_s=0.0)
    )
    assert any("max_replans" in p for p in negative_replans.validate())
    assert any("max_attempts" in p for p in no_verification_attempts.validate())
    assert any("safety_stop_replan_delay_s" in p for p in no_safety_delay.validate())


def test_unknown_requested_payload_is_rejected() -> None:
    config = MissionConfig()
    bad = replace(config, mission=replace(config.mission, target_qr="TARGET_42"))
    assert any("known_payloads" in p for p in bad.validate())


def test_derived_quantities_are_consistent() -> None:
    config = MissionConfig()
    assert config.clearance_m == pytest.approx(
        config.planner.rover_radius_m + config.planner.obstacle_safety_margin_m
    )
    assert config.camera_ground_footprint_m() > config.drone.lane_spacing_m
    assert 0.0 < config.code_size_m("TARGET_1") < config.mission.qr_plate_size_m
    assert config.planner.allow_unknown is False


def test_planner_resolution_coarser_than_the_rover_is_rejected() -> None:
    config = MissionConfig()
    bad = replace(config, planner=PlannerConfig(planning_resolution_m=1.0))
    assert any("planning_resolution_m" in p for p in bad.validate())


def test_nested_sections_can_be_overridden_individually(tmp_path: Path) -> None:
    """Partial YAML must merge onto defaults, not blank out whole sections."""
    path = tmp_path / "partial.yaml"
    path.write_text(yaml.safe_dump({"drone": {"scan_altitude_m": 5.0}}), encoding="utf-8")
    config = load_mission_config(path)
    assert config.drone.scan_altitude_m == 5.0
    # Untouched values inside the same section keep their defaults.
    assert config.drone.scan_speed_mps == DroneConfig().scan_speed_mps
    assert config.mission.target_qr == MissionConfigSection().target_qr


# ---------------------------------------------------------------------------
# The CI overlay
# ---------------------------------------------------------------------------

CI_CONFIG = REPO_ROOT / "ros2_ws" / "src" / "mission_bringup" / "config" / "mission_ci.yaml"

#: The only value the CI overlay may change. A GPU-less runner renders too few
#: frames per metre flown, and flying slower is the honest compensation.
#: Anything else would mean a green CI stopped implying the shipped mission works.
CI_ALLOWED_OVERRIDES = {"drone.scan_speed_mps"}


def _flatten_config(data: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_config(value, name))
        else:
            flat[name] = value
    return flat


def test_ci_overlay_loads_and_validates() -> None:
    assert CI_CONFIG.is_file(), f"missing {CI_CONFIG}"
    assert load_mission_config(CI_CONFIG).validate() == []


def test_ci_overlay_only_relaxes_scan_speed() -> None:
    """CI must not be made to pass by weakening what the mission checks."""
    shipped = _flatten_config(config_to_dict(load_mission_config(SHIPPED_CONFIG)))
    ci = _flatten_config(config_to_dict(load_mission_config(CI_CONFIG)))

    differing = {k for k in shipped if shipped[k] != ci[k]}
    assert differing, "the CI overlay no longer overrides anything"
    unexpected = differing - CI_ALLOWED_OVERRIDES
    assert not unexpected, (
        f"the CI overlay changes {sorted(unexpected)}, which is not allowed. "
        "If that is intentional, justify it in CI_ALLOWED_OVERRIDES."
    )
    # Slower, never faster: the whole point is more frames per metre.
    assert ci["drone.scan_speed_mps"] < shipped["drone.scan_speed_mps"]


def test_ci_overlay_keeps_perception_and_safety_identical() -> None:
    shipped = load_mission_config(SHIPPED_CONFIG)
    ci = load_mission_config(CI_CONFIG)

    assert config_to_dict(ci.perception) == config_to_dict(shipped.perception)
    assert config_to_dict(ci.planner) == config_to_dict(shipped.planner)
    assert config_to_dict(ci.verification) == config_to_dict(shipped.verification)
    assert config_to_dict(ci.world_model) == config_to_dict(shipped.world_model)
    assert config_to_dict(ci.drone.camera) == config_to_dict(shipped.drone.camera)
    assert ci.drone.scan_altitude_m == shipped.drone.scan_altitude_m
    assert ci.drone.finish_scan_after_target_found is True


def test_emergency_stop_cannot_fire_on_a_planned_path() -> None:
    """The safety lidar must not see the clearance the planner guarantees.

    Regression from the Gazebo matrix: TARGET_1 and TARGET_3 failed 0/3 with
    PATH_TRACKING_FAILURE while TARGET_2 passed 3/3. The stop threshold was
    0.45 m measured from a lidar mounted 0.24 m ahead of base_link, but the
    planner only guaranteed 0.55 m from base_link - so on any route that
    passed near an obstacle the lidar necessarily read ~0.31 m and stopped the
    rover on its own correctly planned path. TARGET_2's route detoured widely
    and never came close, which is exactly why it looked fine.
    """
    config = load_mission_config(SHIPPED_CONFIG)
    visible = config.clearance_m - config.rover.safety_lidar_forward_offset_m
    assert config.rover.obstacle_stop_distance_m < visible, (
        f"stop distance {config.rover.obstacle_stop_distance_m:.2f} m >= clearance "
        f"visible to the lidar {visible:.2f} m"
    )
    # Pure pursuit cuts corners; leave room for that too.
    assert visible - config.rover.obstacle_stop_distance_m >= 0.10


def test_stop_distance_still_allows_the_rover_to_stop() -> None:
    """A threshold below the braking distance would be decorative."""
    config = load_mission_config(SHIPPED_CONFIG)
    braking_m = config.rover.max_linear_velocity**2 / (2.0 * 1.5)  # SDF max accel
    assert config.rover.obstacle_stop_distance_m >= braking_m * 0.9, (
        f"stop distance {config.rover.obstacle_stop_distance_m:.2f} m is below the "
        f"{braking_m:.2f} m the rover needs to halt from full speed"
    )


def test_inconsistent_stop_distance_is_rejected() -> None:
    from dataclasses import replace as dc_replace

    config = load_mission_config(SHIPPED_CONFIG)
    bad = dc_replace(config, rover=dc_replace(config.rover, obstacle_stop_distance_m=0.9))
    assert any("obstacle_stop_distance_m" in p for p in bad.validate())
