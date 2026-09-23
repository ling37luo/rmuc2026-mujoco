import json
from types import SimpleNamespace

import pytest

from rmuc2026_mujoco import FieldAsset
from rmuc2026_mujoco.cli import build_parser, main
from rmuc2026_mujoco.fly_batch import fly_cases, run_fly_batch, run_fly_episode
from rmuc2026_mujoco.query import height_at


def test_case_grid_and_seed_order_do_not_depend_on_workers():
    cases = fly_cases(
        speeds=[1.8, 2.2],
        lateral_offsets=[-0.05, 0.0, 0.05],
        heading_offsets_deg=[-2.0, 2.0],
        repeats=2,
        seed=100,
    )
    assert len(cases) == 48
    assert cases == fly_cases(
        speeds=[1.8, 2.2],
        lateral_offsets=[-0.05, 0.0, 0.05],
        heading_offsets_deg=[-2.0, 2.0],
        repeats=2,
        seed=100,
    )
    assert [case["seed"] for case in cases] == list(range(100, 148))
    assert {case["scenario_id"] for case in cases} == {"fly_ramp_north", "fly_ramp_south"}
    assert {case["approach_distance_m"] for case in cases} == {0.6}


def test_case_grid_can_compare_approach_distances_on_the_same_field():
    cases = fly_cases(
        scenarios=["fly_ramp_north"],
        speeds=[2.5],
        approach_distances=[0.6, 0.9],
        seed=100,
    )
    assert [case["approach_distance_m"] for case in cases] == [0.6, 0.9]
    assert [case["seed"] for case in cases] == [100, 101]


@pytest.mark.parametrize(
    "options",
    [
        {"scenarios": []},
        {"speeds": [0.0]},
        {"speeds": [float("nan")]},
        {"approach_distances": [0.0]},
        {"approach_distances": [float("nan")]},
        {"lateral_offsets": [float("inf")]},
        {"heading_offsets_deg": []},
        {"repeats": 0},
        {"duration_s": 0},
    ],
)
def test_invalid_cases_are_rejected_before_workers(options):
    with pytest.raises(ValueError):
        fly_cases(**options)


class EpisodeSession:
    model = SimpleNamespace(opt=SimpleNamespace(timestep=0.1))

    def __init__(self, behavior):
        self.behavior = behavior

    def reset(self, **kwargs):
        self.reset_args = kwargs
        self.steps = 0
        self.failure = None
        self.completed = False
        self.progress = SimpleNamespace(timeout_reason=lambda: "did_not_reach_ramp")

    def step(self):
        self.steps += 1
        if self.behavior == "error":
            raise RuntimeError("controller")
        if self.behavior == "fail":
            self.failure = {"reason": "short_or_lip_impact", "time_s": 0.1}
        if self.behavior == "pass" and self.steps == 2:
            self.completed = True
        return True

    def report(self):
        return {"steps": self.steps, "physics_status": "PASS"}


@pytest.mark.parametrize(
    "behavior,outcome,reason",
    [
        ("pass", "PASS", None),
        ("fail", "FAIL", "short_or_lip_impact"),
        ("timeout", "TIMEOUT", "did_not_reach_ramp"),
        ("error", "ERROR", "controller_or_step_exception"),
    ],
)
def test_episode_classifies_task_outcome_without_conflating_physics(behavior, outcome, reason):
    session = EpisodeSession(behavior)
    result = run_fly_episode(session, fly_cases(scenarios=["fly_ramp_north"], duration_s=0.3)[0])
    assert result["outcome"] == outcome
    assert result["failure_reason"] == reason
    assert result["physics_status"] == ("ERROR" if outcome == "ERROR" else "PASS")
    assert session.reset_args["seed"] == result["seed"]


def test_cli_dispatches_fly_matrix_and_saves_full_report(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    calls = []

    def run(asset, **kwargs):
        calls.append(kwargs)
        return {"status": "PASS", "task_status": "INCOMPLETE", "episodes": [{"outcome": "FAIL"}]}

    monkeypatch.setattr("rmuc2026_mujoco.fly_batch.run_fly_batch", run)
    assert main(["run", "pack", "--scenario", "fly_ramp", "--workers", "2"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert "episodes" not in shown
    assert json.loads(open(shown["telemetry"]).read())["episodes"] == [{"outcome": "FAIL"}]
    assert calls[0]["scenarios"] == ("fly_ramp_north", "fly_ramp_south")
    assert calls[0]["speeds"] == (1.5, 1.8, 2.0, 2.2, 2.5)
    assert calls[0]["approach_distances"] is None
    args = build_parser().parse_args(["run", "pack", "--scenario", "fly_ramp_north"])
    assert args.lateral_offsets == [0.0]


def test_real_mujoco_session_reuses_model_and_independent_resets(field_asset_dir, monkeypatch):
    field = FieldAsset.open(field_asset_dir)
    z = height_at(field, 0.0, 0.0)
    route = {
        "scenario_id": "fly_ramp_north",
        "route_id": "synthetic",
        "source_manifest_sha256": field.manifest_sha256,
        "approach_xyz_m": [0.0, 0.0, z],
        "low_seam_xyz_m": [0.3, 0.0, z],
        "takeoff_xyz_m": [0.6, 0.0, z],
        "landing_edge_xyz_m": [1.2, 0.0, z],
        "landing_target_xyz_m": [1.4, 0.0, z],
        "uphill_unit_xy": [1.0, 0.0],
        "heading_yaw_rad": 0.0,
        "surface_width_m": 0.86,
    }
    monkeypatch.setattr("rmuc2026_mujoco.fly_runtime.fly_route_descriptor", lambda *_: route)
    result = run_fly_batch(
        field,
        scenarios=["fly_ramp_north"],
        speeds=[0.3],
        repeats=2,
        duration_s=0.04,
    )
    assert result["status"] == "PASS"
    assert result["summary"]["timed_out"] == 2
    assert result["warnings"] == {} and result["nonfinite_episodes"] == 0
    assert result["environment_count"] == 1
    assert result["by_approach_distance_m"] == {"0.6": {"episodes": 2, "landed_stably": 0}}
    assert result["episodes"][0]["initial_pose"] == result["episodes"][1]["initial_pose"]


def test_example_rover_speed_limit_is_rejected_before_loading_pack():
    with pytest.raises(ValueError, match="supports up to"):
        run_fly_batch("nonexistent-pack", speeds=[3.5])


def test_headless_fly_view_uses_example_session(field_asset_dir, tmp_path, monkeypatch, capsys):
    field = FieldAsset.open(field_asset_dir)
    z = height_at(field, 0.0, 0.0)
    route = {
        "scenario_id": "fly_ramp_north",
        "route_id": "synthetic",
        "source_manifest_sha256": field.manifest_sha256,
        "approach_xyz_m": [0.0, 0.0, z],
        "low_seam_xyz_m": [0.3, 0.0, z],
        "takeoff_xyz_m": [0.6, 0.0, z],
        "landing_edge_xyz_m": [1.2, 0.0, z],
        "landing_target_xyz_m": [1.4, 0.0, z],
        "uphill_unit_xy": [1.0, 0.0],
        "heading_yaw_rad": 0.0,
        "surface_width_m": 0.86,
    }
    monkeypatch.setattr("rmuc2026_mujoco.fly_runtime.fly_route_descriptor", lambda *_: route)
    telemetry = tmp_path / "fly-view.json"
    assert (
        main(
            [
                "view",
                str(field_asset_dir),
                "--scenario",
                "fly_ramp_north",
                "--control",
                "policy",
                "--headless",
                "--steps",
                "20",
                "--telemetry",
                str(telemetry),
            ]
        )
        == 0
    )
    shown = json.loads(capsys.readouterr().out)
    assert shown["physics_status"] == "PASS"
    assert shown["flight_status"] == "INCOMPLETE"
    assert len(json.loads(telemetry.read_text())["rows"]) > 0
