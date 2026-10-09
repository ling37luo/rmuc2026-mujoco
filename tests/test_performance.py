import json

import numpy as np
import pytest

from rmuc2026_mujoco.performance import run_benchmark, summarize, wheel_command
from test_query import _replace_heightfield


def test_parallel_workers_and_profiles_keep_identical_state(field_asset_dir, tmp_path):
    axis = np.linspace(-3, 3, 31)
    _replace_heightfield(
        field_asset_dir, world_x=axis, world_y=axis, world_height=np.zeros((31, 31))
    )
    output = tmp_path / "bench.json"
    result = run_benchmark(
        field_asset_dir,
        output=output,
        profiles=["full", "collision_only"],
        env_counts=[1, 2],
        workers=[1, 2],
        steps=30,
        warmup=3,
        repeats=1,
    )
    assert result["status"] == "PASS"
    assert result["state_contact_parity"] == "PASS"
    assert len(result["trials"]) == 6
    for row in result["trials"]:
        assert row["steps_completed"] == row["env_count"] * 30
        assert len({w["pid"] for w in row["worker_results"]}) == row["workers"]
        assert sorted(e for w in row["worker_results"] for e in w["env_ids"]) == list(
            range(row["env_count"])
        )
        assert row["steps_per_second"] > 0
    assert json.loads(output.read_text())["status"] == "PASS"
    result["trials"][-1]["worker_results"][0]["traces"][0]["state_contact_sha256"] = "mismatch"
    assert summarize(result["trials"])["status"] == "FAIL"
    with pytest.raises(ValueError, match="already exists"):
        run_benchmark(field_asset_dir, output=output)


def test_command_has_stop_drive_turn_reverse():
    assert np.all(wheel_command(0) == 0)
    assert np.all(wheel_command(0.5) > 0)
    turn = wheel_command(1.5)
    assert turn[0] < 0 < turn[1]
    assert np.all(wheel_command(2.5) < 0)
    assert np.all(wheel_command(3.5) == 0)
    assert np.array_equal(wheel_command(4.5), wheel_command(0.5))


def test_benchmark_cli_reports_parity(tmp_path, monkeypatch, capsys):
    from rmuc2026_mujoco import performance
    from rmuc2026_mujoco.cli import main

    received = {}

    def run(pack, **kwargs):
        received.update(kwargs)
        return dict(status="PASS", state_contact_parity="PASS", medians=[])

    monkeypatch.setattr(performance, "run_benchmark", run)
    assert (
        main(
            [
                "benchmark",
                "pack",
                "--output",
                str(tmp_path / "result.json"),
                "--env-counts",
                "1",
                "16",
                "--workers",
                "1",
                "4",
            ]
        )
        == 0
    )
    assert received["env_counts"] == [1, 16]
    assert received["workers"] == [1, 4]
    assert '"state_contact_parity": "PASS"' in capsys.readouterr().out


def test_png_capture_is_readable(tmp_path):
    from rmuc2026_mujoco.performance_view import _save_png
    from PIL import Image

    pixels = np.zeros((4, 7, 3), dtype=np.uint8)
    pixels[0, 0] = [255, 123, 19]
    path = tmp_path / "capture.png"
    _save_png(path, pixels)
    with Image.open(path) as image:
        assert np.array_equal(np.asarray(image), pixels)
