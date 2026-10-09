import json

import pytest

from rmuc2026_mujoco import regression
from rmuc2026_mujoco.cli import main


def test_suite_reports_failure_and_continues_remaining_sections(
    field_asset_dir, tmp_path, monkeypatch
):
    def broken(*args):
        raise RuntimeError("controller failed")

    monkeypatch.setattr(regression, "check_public_interfaces", broken)
    monkeypatch.setattr(
        regression, "check_surface_routes", lambda *a: {"status": "PASS", "ordinary_patches": []}
    )
    monkeypatch.setattr(
        regression,
        "run_stair_checks",
        lambda *a, **k: {"status": "PASS", "catalog": {"patches": []}},
    )
    monkeypatch.setattr(
        regression,
        "check_example_episodes",
        lambda *a: {"status": "PASS", "summary": {"episodes": 2, "completed": 0, "failed": 2}},
    )
    output = tmp_path / "suite.json"
    result = regression.run_regression(field_asset_dir, "source.json", output=output)
    assert result["status"] == "FAIL"
    assert result["failed_sections"] == ["interfaces"]
    assert set(result["unavailable_sections"]) == {"walls", "perimeter", "corners"}
    assert result["sections"]["interfaces"]["error"] == "RuntimeError: controller failed"
    assert result["robot_task_outcomes"]["failed"] == 2
    assert json.loads(output.read_text()) == result
    before = output.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        regression.run_regression(field_asset_dir, "source.json", output=output)
    assert output.read_bytes() == before


def test_missing_sections_and_contact_differences_cannot_be_passes():
    result = regression.summarize_sections({"corners": {"status": "NOT_AVAILABLE"}})
    assert result["status"] == "PARTIAL"
    with pytest.raises(ValueError, match="no regression sections"):
        regression.summarize_sections({})
    rows = [
        dict(case_id="route", profile=p, steps=10, state_contact_sha256=d)
        for p, d in (("full", "a"), ("collision_only", "b"))
    ]
    assert regression._parity(rows)[0]["status"] == "FAIL"


def test_check_cli_exposes_robot_outcomes_and_failure_exit_code(tmp_path, monkeypatch, capsys):
    received = {}

    def run(pack, source, **kwargs):
        received.update(kwargs)
        kwargs["progress"]("interfaces", "PASS")
        return dict(
            status="FAIL",
            suite_id=regression.SUITE_ID,
            passed_sections=["interfaces"],
            failed_sections=["corners"],
            unavailable_sections=[],
            robot_task_outcomes={"failed": 2},
        )

    monkeypatch.setattr(regression, "run_regression", run)
    assert (
        main(
            [
                "check",
                "pack",
                "--source-manifest",
                "source.json",
                "--output",
                str(tmp_path / "r.json"),
                "--profiles",
                "collision_only",
                "--speeds",
                "0.5",
            ]
        )
        == 2
    )
    assert received["profiles"] == ["collision_only"]
    assert received["speeds"] == [0.5]
    assert '"robot_task_outcomes"' in capsys.readouterr().out
