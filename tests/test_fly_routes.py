from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from rmuc2026_mujoco.cli import _configure_fly_camera, main
from rmuc2026_mujoco.fly_routes import fly_route_descriptor
from rmuc2026_mujoco.jump_probe import LandingProfile
from rmuc2026_mujoco.ramp_audit import FIXED_FLY_RAMPS


@pytest.mark.parametrize("scenario_index", [0, 1])
def test_fly_route_uses_pack_translation_and_measured_landing(monkeypatch, scenario_index):
    ramp = FIXED_FLY_RAMPS[scenario_index]
    asset = SimpleNamespace(
        recommended_spawn={"x_before_translation_m": 11.0, "y_before_translation_m": 7.0},
        manifest_sha256="pack-hash",
        manifest={"validation_status": "DRAFT_BLOCKED"},
    )
    landing = LandingProfile(1.18, 1.83, 0.65, -0.01, 0.17)
    monkeypatch.setattr(
        "rmuc2026_mujoco.fly_routes.measure_runtime_landing_profile",
        lambda _asset, _ramp: landing,
    )
    monkeypatch.setattr(
        "rmuc2026_mujoco.fly_routes.surface_at",
        lambda *_args, **_kwargs: SimpleNamespace(
            height_m=0.16,
            maximum_window_slope_deg=2.0,
            local_relief_upper_bound_m=0.01,
        ),
    )
    monkeypatch.setattr(
        "rmuc2026_mujoco.fly_routes.height_at",
        lambda _asset, x, y, **_kwargs: 0.2,
    )
    route = fly_route_descriptor(asset, f"fly_ramp_{'north' if scenario_index == 0 else 'south'}")

    along = ramp.cad_low_seam_along_m - 0.60
    assert route["approach_xyz_m"] == pytest.approx(
        [
            ramp.low_edge_center_xyz_m[0] - 11.0 + along * ramp.uphill_unit_xy[0],
            ramp.low_edge_center_xyz_m[1] - 7.0 + along * ramp.uphill_unit_xy[1],
            0.16,
        ]
    )
    assert route["spawn"]["xyz_m"] == route["approach_xyz_m"]
    assert route["spawn"]["heading_yaw_rad"] == pytest.approx(
        math.atan2(*reversed(ramp.uphill_unit_xy))
    )
    assert route["landing_edge_xyz_m"][2] == pytest.approx(0.17)
    assert route["landing_target_along_m"] == pytest.approx(2.03)
    assert route["gap_m"] == pytest.approx(0.65)
    assert route["source_manifest_sha256"] == "pack-hash"
    assert route["topology_verified"] is False
    assert route["approach_distance_m"] == pytest.approx(0.60)
    assert route["approach_surface"]["maximum_window_slope_deg"] == pytest.approx(2.0)

    farther = fly_route_descriptor(
        asset, f"fly_ramp_{'north' if scenario_index == 0 else 'south'}", 0.90
    )
    assert farther["approach_distance_m"] == pytest.approx(0.90)
    assert farther["approach_xyz_m"][:2] == pytest.approx(
        [
            route["approach_xyz_m"][0] - 0.30 * ramp.uphill_unit_xy[0],
            route["approach_xyz_m"][1] - 0.30 * ramp.uphill_unit_xy[1],
        ]
    )


def test_fly_route_rejects_wrong_scenario():
    with pytest.raises(ValueError, match="not a fly ramp"):
        fly_route_descriptor(SimpleNamespace(), "slope_basic")


@pytest.mark.parametrize("distance", [0.0, -0.1, float("nan")])
def test_fly_route_rejects_invalid_approach_distance(distance):
    with pytest.raises(ValueError, match="approach distance"):
        fly_route_descriptor(SimpleNamespace(), "fly_ramp_north", distance)


def test_view_places_robot_at_selected_fly_approach(
    field_asset_dir: Path, tmp_path: Path, monkeypatch, capsys
):
    route = {
        "approach_xyz_m": [0.2, 0.1, 0.2],
        "landing_target_xyz_m": [1.0, 0.1, 0.2],
        "heading_yaw_rad": 0.5,
        "uphill_unit_xy": [math.cos(0.5), math.sin(0.5)],
    }
    monkeypatch.setattr("rmuc2026_mujoco.cli.fly_route_descriptor", lambda *_args: route)
    robot = tmp_path / "robot.xml"
    robot.write_text(
        '<mujoco model="robot"><worldbody><body pos="0 0 1"><freejoint/>'
        '<geom type="sphere" size=".1"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    telemetry = tmp_path / "trace.json"
    assert (
        main(
            [
                "view",
                str(field_asset_dir),
                "--robot",
                str(robot),
                "--scenario",
                "fly_ramp_north",
                "--headless",
                "--steps",
                "1",
                "--telemetry",
                str(telemetry),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["scenario_id"] == "fly_ramp_north"
    qpos = json.loads(telemetry.read_text())[0]["qpos"]
    assert qpos[:2] == pytest.approx(route["approach_xyz_m"][:2])
    assert qpos[3:7] == pytest.approx([math.cos(0.25), 0.0, 0.0, math.sin(0.25)])


def test_fly_spawn_camera_frames_approach_and_landing():
    viewer = SimpleNamespace(
        cam=SimpleNamespace(lookat=[0.0, 0.0, 0.0], distance=0.0, azimuth=0.0, elevation=0.0),
        viewport=SimpleNamespace(width=800, height=600),
    )
    route = {
        "approach_xyz_m": [0.0, 0.0, 0.2],
        "landing_target_xyz_m": [2.0, 0.0, 0.2],
        "heading_yaw_rad": 0.0,
    }
    assert _configure_fly_camera(viewer, route) == (800, 600)
    assert viewer.cam.lookat == pytest.approx([1.0, 0.0, 0.2])
    assert viewer.cam.distance == pytest.approx(4.5)
    assert viewer.cam.azimuth == pytest.approx(90.0)
