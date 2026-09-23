"""Run the two fly-ramp PhysX smoke exports at 1, 4 and 16 environments.

Launch this runner with the repository's Python environment; it launches each
Isaac smoke case with a fresh Isaac Python process. The runner itself does not
import Isaac Sim or claim that an offline check is a PhysX result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any

from rmuc2026_mujoco import load_isaac_training_region_export


_SIDES = ("north", "south")
_ENV_COUNTS = (1, 4, 16)
_SMOKE = Path(__file__).with_name("physx_fly_contact_smoke.py")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GPU_POLL_SECONDS = 0.25
_WARNING = re.compile(
    r"(?:\b(?:physx|solver|physics|contact)\b.*\b(?:warning|warn|error|failed|failure|overflow)\b"
    r"|\b(?:warning|warn|error|failed|failure|overflow)\b.*\b(?:physx|solver|physics|contact)\b)",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_source(regions_root: Path, side: str) -> dict[str, Any]:
    """Verify one exported grid and descriptor against its source region."""
    export = regions_root / f"isaac_export_schema3_{side}"
    source = regions_root / side
    result: dict[str, Any] = {
        "export": str(export),
        "source_region": str(source),
        "verified": False,
    }
    try:
        if not source.is_dir():
            raise FileNotFoundError(f"source training region is missing: {source}")
        if not export.is_dir():
            raise FileNotFoundError(f"schema-3 export is missing: {export}")
        descriptor_path = export / "descriptor.json"
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        if descriptor.get("schema_version") != 3:
            raise ValueError("PhysX matrix requires a schema-3 export")
        if descriptor.get("scenario_id") != f"fly_ramp_{side}":
            raise ValueError(f"export scenario does not match {side}")
        # This checks the complete grid, collision records and identity against
        # the original region, rather than trusting the descriptor's claims.
        load_isaac_training_region_export(export, source_region=source)
        result.update(
            verified=True,
            descriptor_sha256=_sha256(descriptor_path),
            grid_file_sha256=descriptor["grid_file_sha256"],
            descriptor_identity=descriptor["identity"],
            scenario_id=descriptor["scenario_id"],
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _gpu_query_target(device: str) -> str | None:
    if device == "cpu":
        return None
    match = re.fullmatch(r"cuda(?::(\d+))?", device)
    if match is None:
        raise ValueError("--device must be cpu, cuda, or cuda:N")
    ordinal = int(match.group(1) or 0)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "-1":
        mapped = [item.strip() for item in visible.split(",")]
        if ordinal >= len(mapped) or not mapped[ordinal]:
            raise ValueError("CUDA_VISIBLE_DEVICES does not contain the selected device")
        return mapped[ordinal]
    return str(ordinal)


def _read_gpu_memory_mib(target: str) -> tuple[float | None, str | None]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None, "nvidia-smi is unavailable"
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                target,
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        return None, f"nvidia-smi exited {completed.returncode}: {completed.stderr.strip()}"
    try:
        # A single -i target is expected. Refuse ambiguous or N/A output.
        lines = completed.stdout.strip().splitlines()
        if len(lines) != 1:
            raise ValueError("expected one GPU memory value")
        value = float(lines[0].strip())
        if not math.isfinite(value) or value < 0:
            raise ValueError("GPU memory value must be finite and nonnegative")
        return value, None
    except ValueError as exc:
        return None, f"invalid nvidia-smi memory result: {exc}"


def _new_gpu_record(device: str, target: str | None) -> dict[str, Any]:
    if target is None:
        return {"status": "not_applicable_cpu_device", "complete": True}
    return {
        "status": "sampling",
        "complete": False,
        "scope": "device_global_memory_used_mib_includes_other_processes",
        "device": device,
        "nvidia_smi_target": target,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "poll_interval_s": _GPU_POLL_SECONDS,
        "baseline_mib": None,
        "max_observed_mib": None,
        "max_observed_delta_from_baseline_mib": None,
        "valid_in_process_sample_count": 0,
        "read_errors": [],
    }


def _sample_gpu(record: dict[str, Any], phase: str) -> None:
    if "nvidia_smi_target" not in record:
        return
    value, error = _read_gpu_memory_mib(record["nvidia_smi_target"])
    if error is not None:
        if error not in record["read_errors"] and len(record["read_errors"]) < 5:
            record["read_errors"].append(error)
        return
    if phase == "before":
        record["baseline_mib"] = value
    elif phase == "running":
        record["valid_in_process_sample_count"] += 1
    if record["max_observed_mib"] is None or value > record["max_observed_mib"]:
        record["max_observed_mib"] = value


def _finish_gpu(record: dict[str, Any]) -> None:
    if "nvidia_smi_target" not in record:
        return
    baseline = record["baseline_mib"]
    maximum = record["max_observed_mib"]
    record["complete"] = baseline is not None and record["valid_in_process_sample_count"] > 0
    record["status"] = "sampled_global_memory" if record["complete"] else "metrics_incomplete"
    if baseline is not None and maximum is not None:
        record["max_observed_delta_from_baseline_mib"] = maximum - baseline


def _scan_warnings(stdout_path: Path, stderr_path: Path) -> tuple[int, list[dict[str, Any]]]:
    count = 0
    examples: list[dict[str, Any]] = []
    for stream, path in (("stdout", stdout_path), ("stderr", stderr_path)):
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                if _WARNING.search(line):
                    count += 1
                    if len(examples) < 20:
                        examples.append(
                            {"stream": stream, "line": line_number, "text": line.rstrip()[:500]}
                        )
    return count, examples


def _check_report(
    report: dict[str, Any],
    side: str,
    envs: int,
    steps: int,
    device: str,
    source: dict[str, Any],
) -> list[str]:
    expected = {
        "runtime_status": "PHYSX_CONTACT_PROBE_PASS",
        "backend": "isaac_sim_6_1_physx",
        "scenario_id": f"fly_ramp_{side}",
        "environment_count": envs,
        "steps_per_environment": steps + 1,
        "device": device,
        "grid_file_sha256": source["grid_file_sha256"],
        "source_manifest_sha256": source["descriptor_identity"]["source_manifest_sha256"],
        "source_profile_hash": source["descriptor_identity"]["source_profile_hash"],
        "region_manifest_sha256": source["descriptor_identity"]["region_manifest_sha256"],
        "region_profile_hash": source["descriptor_identity"]["region_profile_hash"],
    }
    return [key for key, value in expected.items() if report.get(key) != value]


def _run_case(
    isaac_python: Path,
    output_dir: Path,
    side: str,
    envs: int,
    steps: int,
    device: str,
    gpu_target: str | None,
    source: dict[str, Any],
) -> dict[str, Any]:
    case_id = f"{side}-{envs}"
    stdout_path = output_dir / f"{case_id}.stdout.log"
    stderr_path = output_dir / f"{case_id}.stderr.log"
    report_path = output_dir / f"{case_id}.json"
    command = [
        str(isaac_python),
        str(_SMOKE),
        source["export"],
        "--envs",
        str(envs),
        "--steps",
        str(steps),
        "--device",
        device,
        "--output",
        str(report_path),
    ]
    result: dict[str, Any] = {
        "case_id": case_id,
        "side": side,
        "envs": envs,
        "command": command,
        "export": source["export"],
        "source_region": source["source_region"],
        "descriptor_sha256": source["descriptor_sha256"],
        "descriptor_identity": source["descriptor_identity"],
        "stdout_log": stdout_path.name,
        "stderr_log": stderr_path.name,
        "report_expected_path": report_path.name,
        "report": None,
        "exit_code": None,
        "runtime_status": None,
        "probe_pass": False,
        "gpu_memory": _new_gpu_record(device, gpu_target),
    }
    started = time.perf_counter()
    _sample_gpu(result["gpu_memory"], "before")
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            process = subprocess.Popen(command, cwd=_REPO_ROOT, stdout=stdout, stderr=stderr)
        except OSError as exc:
            result["launch_error"] = f"{type(exc).__name__}: {exc}"
            stderr.write((result["launch_error"] + "\n").encode("utf-8"))
        else:
            while process.poll() is None:
                _sample_gpu(result["gpu_memory"], "running")
                try:
                    process.wait(timeout=_GPU_POLL_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
            result["exit_code"] = process.returncode
    _sample_gpu(result["gpu_memory"], "after")
    _finish_gpu(result["gpu_memory"])
    result["elapsed_wall_seconds"] = time.perf_counter() - started
    count, examples = _scan_warnings(stdout_path, stderr_path)
    result["warning_match_count"] = count
    result["warning_examples"] = examples
    if report_path.is_file():
        result["report"] = report_path.name
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            result["runtime_status"] = report.get("runtime_status")
            result["report_mismatches"] = _check_report(report, side, envs, steps, device, source)
            for key in (
                "ray_max_abs_error_m",
                "contacted_probe_count",
                "expected_probe_count",
                "environment_steps_per_second",
                "process_max_rss_bytes",
            ):
                result[key] = report.get(key)
            result["probe_pass"] = result["exit_code"] == 0 and not result["report_mismatches"]
        except (OSError, json.JSONDecodeError, TypeError, AttributeError) as exc:
            result["report_error"] = f"{type(exc).__name__}: {exc}"
    else:
        result["report_error"] = "smoke process did not produce a JSON report"
    if not result["probe_pass"]:
        result["case_status"] = "FAIL"
    elif count:
        result["case_status"] = "REVIEW_REQUIRED"
    elif not result["gpu_memory"]["complete"]:
        result["case_status"] = "METRICS_INCOMPLETE"
    else:
        result["case_status"] = "PASS"
    return result


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaac-python",
        type=Path,
        required=True,
        help="Isaac Sim 6.1 python.sh or isolated pip environment's python",
    )
    parser.add_argument("--regions-root", type=Path, required=True)
    parser.add_argument("--output-new-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    try:
        gpu_target = _gpu_query_target(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    output_dir = args.output_new_dir.expanduser().absolute()
    if output_dir.exists():
        parser.error(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    regions_root = args.regions_root.expanduser().resolve()
    # Keep a venv/bin/python symlink intact: resolving it would run system Python.
    isaac_python = args.isaac_python.expanduser().absolute()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PHYSX_MATRIX_RUNNING",
        "scope": "two_fly_ramps_static_mesh_and_fence_sphere_contact_only",
        "isaac_python": str(isaac_python),
        "regions_root": str(regions_root),
        "device": args.device,
        "steps": args.steps,
        "environment_counts": list(_ENV_COUNTS),
        "gpu_memory_note": "nvidia-smi memory.used is device-global and includes unrelated processes; sampled peak is not process peak VRAM",
        "source_verification": {},
        "cases": [],
    }
    for side in _SIDES:
        summary["source_verification"][side] = _verify_source(regions_root, side)
    if not all(row["verified"] for row in summary["source_verification"].values()):
        summary["status"] = "SOURCE_VERIFICATION_FAILED"
        summary["acceptance_complete"] = False
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_summary(output_dir / "summary.json", summary)
        print(output_dir / "summary.json")
        return 2
    _write_summary(output_dir / "summary.json", summary)
    for side in _SIDES:
        for envs in _ENV_COUNTS:
            result = _run_case(
                isaac_python,
                output_dir,
                side,
                envs,
                args.steps,
                args.device,
                gpu_target,
                summary["source_verification"][side],
            )
            summary["cases"].append(result)
            _write_summary(output_dir / "summary.json", summary)
    cases = summary["cases"]
    if any(not case["probe_pass"] for case in cases):
        summary["status"] = "PHYSX_MATRIX_FAIL"
    elif any(case["warning_match_count"] for case in cases):
        summary["status"] = "PHYSX_MATRIX_REVIEW_REQUIRED"
    elif any(not case["gpu_memory"]["complete"] for case in cases):
        summary["status"] = "PHYSX_MATRIX_METRICS_INCOMPLETE"
    else:
        summary["status"] = "PHYSX_MATRIX_PASS"
    summary["acceptance_complete"] = summary["status"] == "PHYSX_MATRIX_PASS"
    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_summary(output_dir / "summary.json", summary)
    print(output_dir / "summary.json")
    return 0 if summary["acceptance_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
