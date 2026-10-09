"""Synthetic geometry and real MuJoCo coverage; no official assets downloaded."""

import json

import mujoco
import numpy as np
import pytest

from rmuc2026_mujoco import FieldAsset
from rmuc2026_mujoco.cli import main
from rmuc2026_mujoco.mjcf import FIELD_COLLISION_GEOM_NAME
from rmuc2026_mujoco.query import height_at
from rmuc2026_mujoco.slope_batch import run_slope_batch
from rmuc2026_mujoco.stairs import _reference_step, load_stair_catalog, screen_step


def step_samples(edge=0.0):
    along = np.linspace(-0.7, 0.7, 561)
    across = np.linspace(-0.3, 0.3, 7)
    s, t = np.meshgrid(along, across, indexing="ij")
    return along, across, 0.02 * t + 0.2 * (s >= edge)


def test_source_step_checks_treads_and_seam_separately():
    s, t, source = step_samples()
    _, _, runtime = step_samples(0.01)
    result = screen_step(s, t, source, runtime)
    assert result["step_height_m"] == pytest.approx(0.2)
    assert result["max_tread_error_m"] == 0
    assert result["max_seam_location_error_m"] == pytest.approx(0.01, abs=0.0025)
    _, _, moved = step_samples(0.0175)
    with pytest.raises(ValueError, match="seam_mismatch"):
        screen_step(s, t, source, moved)
    with pytest.raises(ValueError, match="tread_mismatch"):
        screen_step(s, t, source, runtime + 0.005)


def test_source_obstacle_and_unmeasured_floor_are_not_accepted_as_steps():
    s, t, source = step_samples()
    obstacle = source.copy()
    obstacle[50:100] += 1
    with pytest.raises(ValueError, match="tread_not_clear"):
        screen_step(s, t, obstacle, source)
    source[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing_source_support"):
        screen_step(s, t, source, source)


def test_two_source_treads_preserve_planes_and_have_no_second_contact_owner():
    spec = mujoco.MjSpec.from_string(f'''<mujoco><worldbody>
      <geom name="{FIELD_COLLISION_GEOM_NAME}" type="box" size="1 1 .1"
            friction=".7 .003 .0002" solref=".02 1"/>
    </worldbody></mujoco>''')
    route = {
        "riser_xy_m": [0, 0],
        "uphill_unit_xy": [0, 1],
        "source_audit": {"source_planes_stz": [[0, 0.02, 0.1], [0, 0.02, 0.3]]},
    }
    _reference_step(spec, route)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    original = model.geom(FIELD_COLLISION_GEOM_NAME).id
    assert model.geom_contype[original] == model.geom_conaffinity[original] == 0
    for side, expected_z in ((-0.3, 0.1), (0.3, 0.3)):
        # The original visual geom remains, so use an explicit single-geom ray.
        name = "stairs_reference_0" if side < 0 else "stairs_reference_1"
        gid = model.geom(name).id
        distance = mujoco.mj_rayMesh(
            model, data, gid, np.array([0.0, side, 1.0]), np.array([0.0, 0.0, -1.0])
        )
        assert 1 - distance == pytest.approx(expected_z, abs=1e-6)
        np.testing.assert_allclose(model.geom_friction[gid], [0.7, 0.003, 0.0002])


def synthetic_catalog(field):
    route = {
        "route_id": "synthetic_step_route",
        "length_m": 0.6,
        "width_m": 0.6,
        "low_xyz_m": [0, 0, height_at(field, 0, 0)],
        "high_xyz_m": [0.6, 0, height_at(field, 0.6, 0)],
        "uphill_unit_xy": [1, 0],
        "heading_yaw_rad": 0,
        "waypoints_xyz_m": [[0, 0, height_at(field, 0, 0)], [0.6, 0, height_at(field, 0.6, 0)]],
    }
    return {
        "scenario_id": "stairs_basic",
        "source_manifest_sha256": field.manifest_sha256,
        "heightfield_samples_sha256": field.collision["samples_sha256"],
        "patches": [{"patch_id": route["route_id"], "route": route}],
    }


def test_catalog_is_bound_to_pack_not_private_paths(field_asset_dir, tmp_path):
    field = FieldAsset.open(field_asset_dir)
    catalog = synthetic_catalog(field)
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"catalog": catalog}))
    assert load_stair_catalog(field, path) == catalog
    catalog["source_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        load_stair_catalog(field, catalog)
    with pytest.raises(ValueError, match="requires --route-catalog"):
        load_stair_catalog(field, None)


def test_stair_runner_spawn_workers_and_resets_use_the_supplied_routes(field_asset_dir):
    field = FieldAsset.open(field_asset_dir)
    report = run_slope_batch(
        field,
        route_catalog=synthetic_catalog(field),
        directions=["uphill", "downhill"],
        repeats=2,
        speeds=[0.3],
        duration_s=0.04,
        workers=2,
    )
    assert report["scenario_id"] == "stairs_basic"
    assert report["workers"] == 2
    assert report["summary"]["episodes"] == 4
    assert report["summary"]["errors"] == 0
    assert report["warnings"] == {} and report["nonfinite_episodes"] == 0
    episodes = report["episodes"]
    assert all(e["scenario_id"] == "stairs_basic" for e in episodes)
    assert all(e["case_id"].startswith("stairs_") for e in episodes)
    np.testing.assert_array_equal(episodes[0]["initial_pose"], episodes[1]["initial_pose"])
    assert len({e["seed"] for e in episodes}) == 4


def test_stair_view_uses_same_session_and_records_scenario(field_asset_dir, tmp_path, capsys):
    field = FieldAsset.open(field_asset_dir)
    path, report = tmp_path / "catalog.json", tmp_path / "view.json"
    path.write_text(json.dumps(synthetic_catalog(field)))
    assert (
        main(
            [
                "view",
                str(field_asset_dir),
                "--scenario",
                "stairs_basic",
                "--route-catalog",
                str(path),
                "--headless",
                "--steps",
                "20",
                "--profile",
                "collision_only",
                "--telemetry",
                str(report),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["scenario_id"] == "stairs_basic"
    assert json.loads(report.read_text())["summary"]["steps"] == 20


def test_small_source_edge_bevel_is_measured_across_the_width():
    s, t, source = step_samples()
    _, _, bevel = step_samples(0.01)
    source[:, -1] = bevel[:, -1]
    result = screen_step(s, t, source, source)
    assert result["status"] == "PASS_STATIC"
    assert result["max_seam_location_error_m"] == 0


def test_stair_view_does_not_silently_fall_back_to_a_slope(field_asset_dir, capsys):
    assert main(["view", str(field_asset_dir), "--scenario", "stairs_basic", "--headless"]) == 2
    assert "requires --route-catalog" in capsys.readouterr().err


def test_source_ray_miss_returns_unmeasured_not_fabricated_ground():
    import trimesh
    from rmuc2026_mujoco.stairs import _source_top

    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False
    )
    # Inside AABB, outside the triangle: trimesh returns an empty 1-D hit array.
    top = _source_top([mesh], np.array([[0.9, 0.9]]), np.zeros(3))
    assert not np.isfinite(top[0])


@pytest.mark.parametrize("changed", ["progress", "bounce", "warning"])
def test_paired_checks_do_not_pass_just_because_both_wheels_were_blocked(
    field_asset_dir, monkeypatch, changed
):
    from rmuc2026_mujoco.stairs import run_stair_checks

    field = FieldAsset.open(field_asset_dir)
    catalog = synthetic_catalog(field)
    monkeypatch.setattr("rmuc2026_mujoco.stairs.build_stair_catalog", lambda *args: catalog)
    baseline = dict(
        finite=True,
        warnings={},
        reached_finish=False,
        max_penetration_m=0.001,
        maximum_progress_m=0.34,
        max_center_height_m=0.06,
        final_center_height_m=0.06,
        max_vertical_speed_m_s=0.1,
    )

    def run(*args, prepare_spec=None, **kwargs):
        result = dict(baseline)
        if prepare_spec is None:
            if changed == "progress":
                result["maximum_progress_m"] = 0.1
            elif changed == "bounce":
                result["max_center_height_m"] = 0.5
            else:
                result["warnings"] = {"mjWARN_BADQACC": 1}
        return result

    monkeypatch.setattr("rmuc2026_mujoco.acceptance._run_route", run)
    result = run_stair_checks(field, "unused", speeds=[0.3])
    assert result["status"] == "FAIL"
    assert result["summary"]["passed"] == 0
