import json
from types import SimpleNamespace

import numpy as np
import pytest

from rmuc2026_mujoco import FieldAsset, run_slope_batch, run_slope_episode
from rmuc2026_mujoco.cli import build_parser, main
from rmuc2026_mujoco.query import height_at
from rmuc2026_mujoco.slope_batch import slope_cases


def catalog():
    return {
        "patches": [
            {"patch_id": "a", "route": {"route_id": "a", "length_m": 1.5}},
            {"patch_id": "rejected", "route": None, "route_rejection": "cliff"},
            {"patch_id": "b", "route": {"route_id": "b", "length_m": 1.0}},
        ]
    }


def test_matrix_has_all_directions_speeds_repeats_and_stable_seeds():
    cases = slope_cases(catalog(), repeats=2, seed=10)
    assert len(cases) == 24
    assert cases == slope_cases(catalog(), repeats=2, seed=10)
    assert [c["seed"] for c in cases] == list(range(10, 34))
    assert {c["patch"] for c in cases} == {"a", "b"}
    assert len({(c["patch"], c["direction"], c["speed_mps"], c["repeat"]) for c in cases}) == 24
    up = next(c for c in cases if c["patch"] == "a" and c["direction"] == "uphill")
    trip = next(c for c in cases if c["patch"] == "a" and c["direction"] == "roundtrip")
    assert trip["timeout_s"] > up["timeout_s"]
    with pytest.raises(ValueError, match="cliff"):
        slope_cases(catalog(), patches=["rejected"])


@pytest.mark.parametrize(
    "options",
    [
        {"speeds": [0]},
        {"speeds": [float("nan")]},
        {"repeats": 0},
        {"directions": []},
        {"duration_s": -1},
    ],
)
def test_invalid_matrix_does_not_start_workers(options):
    with pytest.raises(ValueError):
        slope_cases(catalog(), **options)


class EpisodeSession:
    """A controller that can fail, time out or finish on a known physical step."""

    model = SimpleNamespace(opt=SimpleNamespace(timestep=0.1))

    def __init__(self, behavior):
        self.behavior = behavior

    def reset(self, **kwargs):
        self.reset_args = kwargs
        self.steps = 0
        self.failure = None
        self.completed = False

    def step(self):
        self.steps += 1
        if self.behavior == "error":
            raise RuntimeError("policy error")
        if self.behavior == "fail":
            self.failure = {"reason": "outside_route", "time_s": 0.1}
        if self.behavior == "pass" and self.steps == 2:
            self.completed = True
        return True

    def report(self):
        return {"steps": self.steps, "first_failure": self.failure}


@pytest.mark.parametrize(
    "behavior,outcome,steps",
    [("pass", "PASS", 2), ("fail", "FAIL", 1), ("timeout", "TIMEOUT", 3), ("error", "ERROR", 1)],
)
def test_automatic_episode_stops_at_outcome_and_reset_clears_previous_attempt(
    behavior, outcome, steps
):
    case = slope_cases(catalog(), patches=["a"], duration_s=0.3)[0]
    session = EpisodeSession(behavior)
    previous = run_slope_episode(session, case)
    assert previous["outcome"] == outcome and previous["steps"] == steps
    assert previous["truncated"] == (outcome == "TIMEOUT")
    assert session.reset_args["seed"] == case["seed"]
    assert session.mode == "policy"
    session.behavior = "pass"
    result = run_slope_episode(session, {**case, "direction": "downhill"})
    assert result["outcome"] == "PASS" and result["steps"] == 2
    assert previous["outcome"] == outcome


def test_real_mujoco_batch_reuses_model_with_independent_resets(
    field_asset_dir, monkeypatch, tmp_path
):
    field = FieldAsset.open(field_asset_dir)
    route = {
        "route_id": "synthetic",
        "length_m": 0.6,
        "width_m": 0.6,
        "low_xyz_m": [0, 0, height_at(field, 0, 0)],
        "high_xyz_m": [0.6, 0, height_at(field, 0.6, 0)],
        "uphill_unit_xy": [1, 0],
        "heading_yaw_rad": 0,
    }
    tiny_catalog = {"patches": [{"patch_id": "synthetic", "route": route}]}
    monkeypatch.setattr("rmuc2026_mujoco.slope_runtime.slope_catalog", lambda _: tiny_catalog)
    monkeypatch.setattr("rmuc2026_mujoco.slope_batch.slope_catalog", lambda _: tiny_catalog)
    result = run_slope_batch(
        field, speeds=[0.3], repeats=2, duration_s=0.04, trajectory_dir=tmp_path / "trajectories"
    )
    assert result["summary"]["timed_out"] == 6
    assert result["summary"]["errors"] == 0
    assert result["environment_count"] == 1
    assert result["warnings"] == {} and result["nonfinite_episodes"] == 0
    assert result["episodes_per_worker"] == [6]
    episodes = result["episodes"]
    assert all(e["sim_time_s"] == pytest.approx(0.04) for e in episodes)
    np.testing.assert_array_equal(episodes[0]["initial_pose"], episodes[1]["initial_pose"])
    np.testing.assert_allclose(episodes[2]["initial_pose"][:2], [0.6, 0])
    np.testing.assert_allclose(episodes[2]["initial_pose"][3:7], [0, 0, 0, 1], atol=1e-10)
    assert all(e["leg_results"]["uphill"]["completed_at_s"] is None for e in episodes)
    trajectories = sorted((tmp_path / "trajectories").glob("*.json"))
    assert len(trajectories) == 6
    assert all(json.loads(p.read_text())["rows"] for p in trajectories)


def test_cli_dispatches_automatic_matrix_and_saves_full_report(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    calls = []

    def run(asset, **kwargs):
        calls.append(kwargs)
        return {"status": "PASS", "episodes": [{"outcome": "PASS"}]}

    monkeypatch.setattr("rmuc2026_mujoco.slope_batch.run_slope_batch", run)
    assert (
        main(["run", "pack", "--scenario", "slope_basic", "--workers", "2", "--repeats", "3"]) == 0
    )
    shown = json.loads(capsys.readouterr().out)
    assert "episodes" not in shown
    assert json.loads(open(shown["telemetry"]).read())["episodes"] == [{"outcome": "PASS"}]
    assert calls[0]["robot"] is None and calls[0]["workers"] == 2 and calls[0]["repeats"] == 3
    assert calls[0]["directions"] == ["uphill", "downhill", "roundtrip"]
    args = build_parser().parse_args(["run", "pack", "--scenario", "slope_basic"])
    assert args.duration is None
