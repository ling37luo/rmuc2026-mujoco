from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rmuc2026_mujoco import (
    FieldAsset,
    MuJoCoScenario,
    get_scenario,
    list_scenarios,
    load_isaac_heightfield,
    scenario_descriptor,
)


def test_scenario_registry_has_shared_baseline_and_training_stages() -> None:
    ids = [item["scenario_id"] for item in list_scenarios()]
    assert ids == [
        "full_eval",
        "turn_basic",
        "stairs_basic",
        "slope_basic",
        "fly_ramp_north",
        "fly_ramp_south",
        "boundary_contact",
    ]
    assert get_scenario("fly_ramp_north").routes[0]["route_id"] == "fly_ramp_north_to_gap"


def test_scenario_descriptor_binds_pack_hash(field_asset_dir: Path) -> None:
    descriptor = scenario_descriptor(
        FieldAsset.open(field_asset_dir),
        "turn_basic",
    )
    assert descriptor["source_manifest_sha256"]
    assert descriptor["profile"] == "collision_only"
    assert len(descriptor["profile_hash"]) == 64


def test_mujoco_scenario_runs_without_rl_framework(field_asset_dir: Path) -> None:
    robot = field_asset_dir.parent / "robot.xml"
    robot.write_text(
        """<mujoco model="robot"><worldbody><body name="robot" pos="0 0 1">
<freejoint/><geom name="robot_geom" type="sphere" size=".1"/>
</body></worldbody></mujoco>""",
        encoding="utf-8",
    )
    env = MuJoCoScenario(field_asset_dir, robot, scenario="turn_basic")
    before = env.reset()
    result = env.step()
    assert before.shape == result.observation.shape
    report = env.run(2)
    assert report["steps"] == 2
    assert report["warnings"] == {}
    assert env.metadata()["robot_mjcf_sha256"]


def test_isaac_input_is_only_verified_field_data(field_asset_dir: Path) -> None:
    data = load_isaac_heightfield(field_asset_dir)
    assert data.height_m.dtype == np.float32
    assert data.to_dict()["source"] == "verified_runtime_pack_heightfield"


@pytest.mark.parametrize("scenario_id", ["fly_ramp_north", "fly_ramp_south"])
def test_isaac_fly_ramp_descriptor_keeps_pack_identity_and_spawn(
    field_asset_dir: Path, monkeypatch: pytest.MonkeyPatch, scenario_id: str
) -> None:
    field = FieldAsset.open(field_asset_dir, verify=True)
    expected = scenario_descriptor(field, scenario_id, profile="collision_only")
    route_id = f"{scenario_id}_to_gap"
    route = {
        "route_id": route_id,
        "source_manifest_sha256": field.manifest_sha256,
        "gap_m": 0.65,
        "takeoff_xyz_m": [0.0, 0.0, 0.5],
        "landing_edge_xyz_m": [0.65, 0.0, 0.2],
        "spawn": {"xyz_m": [-0.5, 0.0, 0.0], "heading_yaw_rad": 0.0},
    }

    def fake_route(asset: FieldAsset, selected: str) -> dict:
        assert asset is field
        assert selected == scenario_id
        return route

    monkeypatch.setattr("rmuc2026_mujoco.fly_routes.fly_route_descriptor", fake_route)
    data = load_isaac_heightfield(field, scenario=scenario_id)
    payload = data.to_dict()
    assert payload["scenario"]["profile_hash"] == expected["profile_hash"]
    assert payload["scenario"]["source_manifest_sha256"] == field.manifest_sha256
    assert (
        payload["scenario"]["heightfield_samples_sha256"] == expected["heightfield_samples_sha256"]
    )
    assert payload["scenario"]["fly_ramp_route"] == route
    assert payload["spawn_points"] == [
        {
            "route_id": route_id,
            "position_xyz_m": [-0.5, 0.0, 0.0],
            "heading_yaw_rad": 0.0,
            "position_reference": "terrain_surface",
            "topology_verified": False,
        }
    ]
    assert payload["bounds_xy_m"] == expected["field_bounds_xy_m"]
