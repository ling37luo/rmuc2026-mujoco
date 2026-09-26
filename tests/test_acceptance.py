from __future__ import annotations

import json
from pathlib import Path

import mujoco
import pytest

from rmuc2026_mujoco import FieldAsset, compose_with_robot
from rmuc2026_mujoco.acceptance import _pair_name, _validate_routes, run_acceptance


def test_headless_profile_benchmark_keeps_contacts_and_state_parity(
    field_asset_dir: Path, tmp_path: Path
) -> None:
    output = tmp_path / "acceptance.json"
    result = run_acceptance(
        field_asset_dir,
        profiles=("full", "collision_only"),
        steps=20,
        env_counts=(1, 2),
        output_path=output,
    )

    assert result["status"] == "PASS"
    assert result["profile_parity"]["collision_only"]["status"] == "PASS"
    assert result["profiles"]["full"]["env_counts"][0]["total_steps_completed"] == 20
    assert result["profiles"]["full"]["env_counts"][1]["total_steps_completed"] == 40
    assert result["profiles"]["full"]["env_counts"][0]["finite"]
    assert result["profiles"]["full"]["env_counts"][0]["max_penetration_m"] >= 0
    assert json.loads(output.read_text())["manifest_sha256"] == result["manifest_sha256"]

    with pytest.raises(FileExistsError):
        run_acceptance(field_asset_dir, steps=1, env_counts=(1,), output_path=output)


def test_route_contract_and_bidirectional_probe(field_asset_dir: Path, tmp_path: Path) -> None:
    routes = tmp_path / "routes.json"
    routes.write_text(
        json.dumps(
            {
                "routes": [
                    {
                        "id": "small_slope",
                        "start_xy_m": [0.0, 0.0],
                        "end_xy_m": [0.08, 0.0],
                        "expect": "traverse",
                        "directions": ["forward"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    selected, digest = _validate_routes(routes)
    assert selected[0]["directions"] == ["forward"]
    assert len(digest) == 64

    result = run_acceptance(
        field_asset_dir,
        profiles=("collision_only",),
        steps=5,
        env_counts=(1,),
        routes_json=routes,
    )
    assert len(result["route_trials"]) == 3
    assert result["status"] == "PASS"
    assert all(item["status"] == "PASS" for item in result["route_trials"])
    assert all(item["direction"] == "forward" for item in result["route_trials"])
    assert all(item["finite"] and item["warnings"] == {} for item in result["route_trials"])
    assert all("max_penetration_m" in item for item in result["route_trials"])


@pytest.mark.parametrize(
    "route",
    [
        {"id": "bad", "start_xy_m": [0, 0], "end_xy_m": [1, 0], "expect": "block"},
        {
            "id": "bad",
            "start_xy_m": [0, 0],
            "end_xy_m": [1, 0],
            "expect": "traverse",
            "directions": ["forward", "forward"],
        },
    ],
)
def test_invalid_route_contract_fails(tmp_path: Path, route: dict) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"routes": [route]}), encoding="utf-8")
    with pytest.raises(ValueError):
        _validate_routes(path)


def test_route_preset_rejects_another_pack_hash(field_asset_dir: Path, tmp_path: Path) -> None:
    path = tmp_path / "wrong_pack_routes.json"
    path.write_text(
        json.dumps(
            {
                "source_manifest_sha256": "0" * 64,
                "routes": [
                    {
                        "id": "test",
                        "start_xy_m": [0, 0],
                        "end_xy_m": [0.1, 0],
                        "expect": "traverse",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source manifest"):
        run_acceptance(field_asset_dir, routes_json=path, steps=1, env_counts=(1,))


def test_route_selection_rejects_unknown_id(field_asset_dir: Path, tmp_path: Path) -> None:
    path = tmp_path / "routes.json"
    path.write_text(
        json.dumps(
            {
                "routes": [
                    {
                        "id": "known",
                        "start_xy_m": [0, 0],
                        "end_xy_m": [0.1, 0],
                        "expect": "traverse",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="route selection"):
        run_acceptance(
            field_asset_dir,
            routes_json=path,
            route_ids=("missing",),
            steps=1,
            env_counts=(1,),
        )


def test_unnamed_robot_geom_contact_identity_is_profile_independent(
    field_asset_dir: Path, tmp_path: Path
) -> None:
    robot = tmp_path / "robot.xml"
    robot.write_text(
        '<mujoco><worldbody><body name="robot" pos="0 0 1">'
        '<freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    names = []
    for profile in ("full", "collision_only"):
        model, _ = compose_with_robot(FieldAsset.open(field_asset_dir), robot, profile=profile)
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        field = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "rmuc2026_field/rmuc2026_field_collision"
        )
        names.append(_pair_name(model, int(model.body_geomadr[body]), field))
    assert names[0] == names[1]
    assert "robot/unnamed_geom[0]" in names[0]
