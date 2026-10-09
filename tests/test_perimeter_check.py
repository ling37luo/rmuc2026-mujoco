"""Boundary checks must exercise motion/escape, not merely register a wall hit."""

from types import SimpleNamespace

import numpy as np
import pytest

from rmuc2026_mujoco.cli import main
from rmuc2026_mujoco.perimeter_check import (
    _wheel_model,
    build_perimeter_catalog,
    fence_corners,
    run_perimeter_checks,
    run_wheel_route,
    screen_support,
)


def test_support_screen_detects_deck_to_fence_gaps():
    d, t = np.meshgrid(np.linspace(0.005, 0.8, 81), np.linspace(-0.65, 0.65, 131))
    height = 0.02 * d + 0.01 * t
    assert screen_support(d, t, height)["slope_deg"] < 3
    broken = height.copy()
    broken[:, d[0] < 0.04] -= 0.08
    with pytest.raises(ValueError, match="support_not_continuous"):
        screen_support(d, t, broken)
    height[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing_support"):
        screen_support(d, t, height)


def test_corner_boxes_touch_but_a_ground_route_is_not_certified():
    panels = [
        {"name": "fence_left", "pos": [-2, 0, 1], "size": [0.025, 0.975, 1]},
        {"name": "fence_right", "pos": [2, 0, 1], "size": [0.025, 0.975, 1]},
        {"name": "fence_bottom", "pos": [0, -1, 1], "size": [2.025, 0.025, 1]},
        {"name": "fence_top", "pos": [0, 1, 1], "size": [2.025, 0.025, 1]},
    ]
    rows = fence_corners({"panels": panels})
    assert all(r["panel_join_status"] == "PASS_STATIC" for r in rows)
    assert all(r["driving_status"] == "UNVERIFIED" for r in rows)
    panels[3]["pos"][1] += 0.02
    rows = fence_corners({"panels": panels})
    assert sum(r["panel_join_status"] == "FAIL_GAP" for r in rows) == 2


def _flat_probe(tmp_path, monkeypatch, *, wall=True, gap=False):
    floor = '<geom name="rmuc2026_field_collision" type="plane" size="2 2 .1"/>'
    if gap:
        floor = '<geom name="rmuc2026_field_collision" type="box" pos="1.1 0 -.1" size="1 2 .1"/>'
    fence = '<geom name="fence" type="box" pos="-.025 0 .5" size=".025 2 .5"/>' if wall else ""
    xml = tmp_path / "probe.xml"
    xml.write_text(
        f'<mujoco><option timestep=".002"/><worldbody>{floor}{fence}</worldbody></mujoco>'
    )
    monkeypatch.setattr(
        "rmuc2026_mujoco.perimeter_check.inject_exact_heightfield", lambda *a, **k: None
    )
    route = {
        "id": "synthetic",
        "axis": 0,
        "inward_sign": 1,
        "face_m": 0,
        "transverse_center_m": 0,
        "fence_geom": "fence",
        "support": {"plane_dtz": [0, 0, 0]},
    }
    return _wheel_model(SimpleNamespace(entrypoint_for=lambda _: xml), "full", route), route


def test_probe_can_slide_both_ways_retreat_and_repeat_reset(tmp_path, monkeypatch):
    model, route = _flat_probe(tmp_path, monkeypatch)
    first, trace1 = run_wheel_route(model, route, 0.5, 0)
    second, trace2 = run_wheel_route(model, route, 0.5, 0)
    assert first == second
    assert first["status"] == "PASS"
    assert first["phases"][-1]["final_depth_transverse_m"][0] > 0.54
    np.testing.assert_array_equal(trace1["qpos"], trace2["qpos"])
    np.testing.assert_array_equal(trace1["qvel"], trace2["qvel"])
    assert trace1["contacts"] == trace2["contacts"]


@pytest.mark.parametrize("wall,gap", [(False, False), (True, True)])
def test_missing_wall_or_support_does_not_pass(tmp_path, monkeypatch, wall, gap):
    model, route = _flat_probe(tmp_path, monkeypatch, wall=wall, gap=gap)
    result, _ = run_wheel_route(model, route, 0.5, 0)
    assert result["status"] == "FAIL"
    assert result["failure_reasons"]


def test_pack_without_fence_is_not_silently_accepted(field_asset_dir):
    with pytest.raises(ValueError, match="perimeter fence"):
        build_perimeter_catalog(field_asset_dir, "unused.json")
    with pytest.raises(ValueError, match="finite and positive"):
        run_perimeter_checks(field_asset_dir, "unused.json", speeds=[float("nan")])


@pytest.mark.parametrize("kind", ["perimeter", "wall"])
def test_cli_forwards_profiles_and_preserves_existing_report(tmp_path, monkeypatch, kind):
    from rmuc2026_mujoco import perimeter_check, wall_check

    received = {}

    def run(*args, **kwargs):
        received.update(kwargs)
        return {"status": "PASS", "summary": {"trials": 12}}

    module = perimeter_check if kind == "perimeter" else wall_check
    monkeypatch.setattr(module, f"run_{kind}_checks", run)
    output = tmp_path / "report.json"
    argv = [
        f"{kind}-check",
        "pack",
        "--source-manifest",
        "source.json",
        "--output",
        str(output),
        "--profiles",
        "full",
        "collision_only",
        "--speeds",
        "0.5",
    ]
    assert main(argv) == 0
    assert received == {"profiles": ["full", "collision_only"], "speeds": [0.5]}
    before = output.read_bytes()
    assert main(argv) != 0
    assert output.read_bytes() == before
