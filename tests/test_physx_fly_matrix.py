"""Matrix orchestration tests with a fake Isaac python.sh; no PhysX is started."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_RUNNER_PATH = Path(__file__).resolve().parents[1] / "adapters/isaac/run_physx_fly_matrix.py"
_SPEC = importlib.util.spec_from_file_location("run_physx_fly_matrix", _RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)


@pytest.fixture
def matrix_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, list]:
    root = tmp_path / "regions"
    root.mkdir()
    calls = []
    for side in ("north", "south"):
        (root / side).mkdir()
        export = root / f"isaac_export_schema3_{side}"
        export.mkdir()
        descriptor = {
            "schema_version": 3,
            "scenario_id": f"fly_ramp_{side}",
            "grid_file_sha256": f"grid-{side}",
            "identity": {
                "source_manifest_sha256": "source-manifest",
                "source_profile_hash": "source-profile",
                "region_manifest_sha256": f"region-{side}",
                "region_profile_hash": f"profile-{side}",
            },
        }
        (export / "descriptor.json").write_text(json.dumps(descriptor), encoding="utf-8")

    def verify(export: Path, *, source_region: Path) -> None:
        calls.append((export, source_region))

    monkeypatch.setattr(_RUNNER, "load_isaac_training_region_export", verify)
    fake_python = tmp_path / "python.sh"
    fake_python.write_text(
        """#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('smoke')
p.add_argument('export', type=Path)
p.add_argument('--envs', type=int, required=True)
p.add_argument('--steps', type=int, required=True)
p.add_argument('--device', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
side = a.export.name.rsplit('_', 1)[-1]
case = f'{side}-{a.envs}'
if os.environ.get('FAKE_WARN_CASE') == case:
    print('PhysX solver warning: contact buffer overflow', file=sys.stderr)
if os.environ.get('FAKE_FAIL_CASE') == case:
    print('fake Isaac launch failed', file=sys.stderr)
    sys.exit(7)
d = json.loads((a.export / 'descriptor.json').read_text())
report = {
    'runtime_status': 'PHYSX_CONTACT_PROBE_PASS',
    'backend': 'isaac_sim_6_1_physx',
    'scenario_id': d['scenario_id'],
    'environment_count': a.envs,
    'steps_per_environment': a.steps + 1,
    'device': a.device,
    'grid_file_sha256': d['grid_file_sha256'],
    'source_manifest_sha256': d['identity']['source_manifest_sha256'],
    'source_profile_hash': d['identity']['source_profile_hash'],
    'region_manifest_sha256': d['identity']['region_manifest_sha256'],
    'region_profile_hash': d['identity']['region_profile_hash'],
    'ray_max_abs_error_m': 0.0001,
    'contacted_probe_count': a.envs * 5,
    'expected_probe_count': a.envs * 5,
    'environment_steps_per_second': 1234.0,
    'process_max_rss_bytes': 1000000,
}
a.output.write_text(json.dumps(report))
print(f'finished {case}')
time.sleep(0.05)
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    return root, fake_python, calls


def _args(root: Path, fake_python: Path, output: Path, *, device: str = "cuda:0") -> list[str]:
    return [
        "--isaac-python",
        str(fake_python),
        "--regions-root",
        str(root),
        "--output-new-dir",
        str(output),
        "--device",
        device,
        "--steps",
        "3",
    ]


def test_fake_isaac_matrix_success_records_all_six_and_source_identity(
    tmp_path: Path, matrix_inputs: tuple[Path, Path, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, fake_python, calls = matrix_inputs
    monkeypatch.setattr(_RUNNER, "_read_gpu_memory_mib", lambda _target: (512.0, None))
    output = tmp_path / "success"
    assert _RUNNER.main(_args(root, fake_python, output)) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "PHYSX_MATRIX_PASS"
    assert summary["acceptance_complete"] is True
    assert [(row["side"], row["envs"]) for row in summary["cases"]] == [
        (side, envs) for side in ("north", "south") for envs in (1, 4, 16)
    ]
    assert calls == [
        (root / f"isaac_export_schema3_{side}", root / side) for side in ("north", "south")
    ]
    for row in summary["cases"]:
        assert row["case_status"] == "PASS"
        assert row["exit_code"] == 0
        assert row["probe_pass"] is True
        assert row["descriptor_identity"]["region_manifest_sha256"] == f"region-{row['side']}"
        assert row["gpu_memory"]["complete"] is True
        assert "device_global" in row["gpu_memory"]["scope"]
        assert (output / row["report"]).is_file()
        assert (output / row["stdout_log"]).read_text().startswith("finished")
        assert (output / row["stderr_log"]).is_file()


def test_fake_isaac_failure_keeps_logs_missing_report_and_incomplete_gpu_metrics(
    tmp_path: Path, matrix_inputs: tuple[Path, Path, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, fake_python, _calls = matrix_inputs
    monkeypatch.setenv("FAKE_FAIL_CASE", "south-4")
    monkeypatch.setenv("FAKE_WARN_CASE", "north-16")
    monkeypatch.setattr(
        _RUNNER, "_read_gpu_memory_mib", lambda _target: (None, "nvidia-smi unavailable")
    )
    output = tmp_path / "failed"
    assert _RUNNER.main(_args(root, fake_python, output)) == 2
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "PHYSX_MATRIX_FAIL"
    assert len(summary["cases"]) == 6
    failed = next(row for row in summary["cases"] if row["case_id"] == "south-4")
    assert failed["exit_code"] == 7
    assert failed["report"] is None
    assert "did not produce" in failed["report_error"]
    assert "fake Isaac launch failed" in (output / failed["stderr_log"]).read_text()
    warned = next(row for row in summary["cases"] if row["case_id"] == "north-16")
    assert warned["warning_match_count"] == 1
    assert warned["case_status"] == "REVIEW_REQUIRED"
    assert all(row["gpu_memory"]["status"] == "metrics_incomplete" for row in summary["cases"])
    assert summary["acceptance_complete"] is False


def test_gpu_metrics_unreadable_does_not_claim_matrix_pass(
    tmp_path: Path, matrix_inputs: tuple[Path, Path, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, fake_python, _calls = matrix_inputs
    monkeypatch.setattr(_RUNNER, "_read_gpu_memory_mib", lambda _target: (None, "unavailable"))
    output = tmp_path / "incomplete"
    assert _RUNNER.main(_args(root, fake_python, output)) == 2
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "PHYSX_MATRIX_METRICS_INCOMPLETE"
    assert all(row["probe_pass"] for row in summary["cases"])
    assert summary["acceptance_complete"] is False


def test_solver_warning_requires_review_even_when_probes_pass(
    tmp_path: Path, matrix_inputs: tuple[Path, Path, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, fake_python, _calls = matrix_inputs
    monkeypatch.setenv("FAKE_WARN_CASE", "north-4")
    monkeypatch.setattr(_RUNNER, "_read_gpu_memory_mib", lambda _target: (512.0, None))
    output = tmp_path / "warning"
    assert _RUNNER.main(_args(root, fake_python, output)) == 2
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "PHYSX_MATRIX_REVIEW_REQUIRED"
    assert all(row["probe_pass"] for row in summary["cases"])
    assert summary["acceptance_complete"] is False


def test_missing_source_region_fails_before_any_isaac_process(
    tmp_path: Path, matrix_inputs: tuple[Path, Path, list]
) -> None:
    root, fake_python, calls = matrix_inputs
    (root / "south").rmdir()
    output = tmp_path / "missing-source"
    assert _RUNNER.main(_args(root, fake_python, output, device="cpu")) == 2
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "SOURCE_VERIFICATION_FAILED"
    assert summary["cases"] == []
    assert "source training region is missing" in summary["source_verification"]["south"]["error"]
    assert calls == [(root / "isaac_export_schema3_north", root / "north")]


def test_isaac_python_symlink_is_not_resolved(
    tmp_path: Path, matrix_inputs: tuple[Path, Path, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, fake_python, _calls = matrix_inputs
    python_link = tmp_path / "venv-python"
    python_link.symlink_to(fake_python)
    monkeypatch.setattr(_RUNNER, "_read_gpu_memory_mib", lambda _target: (512.0, None))
    output = tmp_path / "symlink"
    assert _RUNNER.main(_args(root, python_link, output)) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["isaac_python"] == str(python_link)
    assert all(row["command"][0] == str(python_link) for row in summary["cases"])
