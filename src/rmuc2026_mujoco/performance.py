"""Fixed public-rover workloads with fresh, genuinely parallel MuJoCo workers."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time
import traceback

import mujoco
import numpy as np

from .acceptance import PROFILE_ORDER, _contact_sample, _rss_bytes, _warning_counts
from .corner_check import TraceDigest, solver_config
from .manifest import FieldAsset
from .mjcf import compose_with_robot
from .query import find_spawn_candidates
from .scenarios import scenario_descriptor
from .slope_demo import rover_xml
from .slope_runtime import reset_slope_spawn

WORKLOAD_ID = "public_rover_drive_turn_reverse_v1"
DIAGNOSTIC_STRIDE = 10


def machine_snapshot():
    result = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        cpu_count=os.cpu_count(),
        mujoco_version=mujoco.__version__,
    )
    result["thread_environment"] = {
        k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
    if hasattr(os, "sched_getaffinity"):
        result["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    if hasattr(os, "getloadavg"):
        result["load_average"] = list(os.getloadavg())
    try:
        result["memory_kib"] = {
            line.split(":")[0]: int(line.split()[1])
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.split(":")[0] in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
        }
        result["cpu_model"] = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        )
    except (OSError, StopIteration):
        pass
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        result["gpu_device_global_snapshot"] = gpu.stdout.strip() or gpu.stderr.strip()
    except (OSError, subprocess.TimeoutExpired):
        result["gpu_device_global_snapshot"] = None
    return result


def select_spawn(asset):
    candidates = find_spawn_candidates(
        asset,
        count=1,
        footprint_radius_m=0.5,
        boundary_margin_m=0.5,
        max_slope_deg=3,
        max_relief_m=0.02,
    )
    if not candidates:
        raise ValueError("benchmark needs a screened 1 m-wide flat patch")
    return candidates[0].to_dict()


def load_rover(asset, profile, spawn):
    with tempfile.TemporaryDirectory(prefix="rmuc-bench-") as temp:
        path = Path(temp) / "rover.xml"
        path.write_text(rover_xml())
        model, data = compose_with_robot(asset, path, profile=profile)
    route = dict(
        low_xyz_m=[spawn["x_m"], spawn["y_m"], spawn["terrain_height_m"]],
        heading_yaw_rad=0.0,
        uphill_unit_xy=[1, 0],
    )
    reset_slope_spawn(model, data, route)
    return model, data


def wheel_command(t):
    """One four-second deterministic cycle, in wheel rad/s; no policy network."""
    t %= 4.0
    vx, yaw = (0.0, 0.0)
    if 0.3 <= t < 1.3:
        vx = 0.1
    elif 1.3 <= t < 2.3:
        yaw = 0.3
    elif 2.3 <= t < 3.3:
        vx = -0.1
    return np.array([(vx - 0.21 * yaw) / 0.08, (vx + 0.21 * yaw) / 0.08] * 2)


def _worker(conn, start, pack, profile, spawn, env_ids, steps, warmup):
    try:
        before = _rss_bytes()
        started = time.perf_counter()
        asset = FieldAsset.open(pack, verify=False)  # Parent verifies once before spawning.
        model, initial = load_rover(asset, profile, spawn)
        load_s = time.perf_counter() - started
        started = time.perf_counter()
        # Warm caches, then start every measured environment from the same reset.
        for step in range(warmup):
            initial.ctrl[:] = wheel_command(step * model.opt.timestep)
            mujoco.mj_step(model, initial)
        reset_slope_spawn(
            model,
            initial,
            dict(
                low_xyz_m=[spawn["x_m"], spawn["y_m"], spawn["terrain_height_m"]],
                heading_yaw_rad=0.0,
            ),
        )
        data_items = []
        for _ in env_ids:
            data = mujoco.MjData(model)
            data.qpos[:] = initial.qpos
            mujoco.mj_forward(model, data)
            data_items.append(data)
        init_s = time.perf_counter() - started
        actions = [wheel_command(step * model.opt.timestep) for step in range(steps)]
        # Full-state traces are measured for every environment; named contacts at 50 Hz.
        digests = [TraceDigest() for _ in env_ids]
        warnings, max_contacts, penetration = {}, 0, 0.0
        finite, completed = True, 0
        max_displacement = 0.0
        rss_ready = _rss_bytes()
        conn.send(dict(kind="ready", pid=os.getpid(), load_s=load_s, init_s=init_s))
        if not start.wait(300):
            raise TimeoutError("benchmark start signal timed out")
        begin = time.perf_counter()
        for step, action in enumerate(actions):
            for i, data in enumerate(data_items):
                data.ctrl[:] = action
                mujoco.mj_step(model, data)
                pairs = Counter()
                if step % DIAGNOSTIC_STRIDE == 0 or step == steps - 1:
                    pairs, depth, contacts_finite = _contact_sample(model, data)
                    finite &= contacts_finite and all(
                        np.isfinite(x).all() for x in (data.qpos, data.qvel, data.qacc)
                    )
                    warnings.update(_warning_counts(data))
                    max_contacts = max(max_contacts, int(data.ncon))
                    penetration = max(penetration, depth)
                    max_displacement = max(
                        max_displacement, float(np.max(np.abs(data.qpos[:2] - initial.qpos[:2])))
                    )
                digests[i].record(data, pairs)
                completed += 1
            if not finite or warnings:
                break
        end = time.perf_counter()
        conn.send(
            dict(
                kind="result",
                pid=os.getpid(),
                env_ids=env_ids,
                steps_completed=completed,
                begin_s=begin,
                end_s=end,
                steps_per_second=completed / (end - begin),
                load_s=load_s,
                init_s=init_s,
                rss_before_load_bytes=before,
                rss_ready_bytes=rss_ready,
                rss_after_run_bytes=_rss_bytes(),
                solver_config=solver_config(model),
                model_nmesh=int(model.nmesh),
                model_mesh_faces=int(model.nmeshface),
                model_ngeom=int(model.ngeom),
                finite=bool(finite),
                warnings=warnings,
                max_contacts=max_contacts,
                max_penetration_m=penetration,
                max_displacement_xy_m=max_displacement,
                traces=[dict(env_id=e, **d.report()) for e, d in zip(env_ids, digests)],
                status="PASS"
                if finite
                and not warnings
                and completed == steps * len(env_ids)
                and max_contacts > 0
                and max_displacement + 0.35 < spawn["footprint_radius_m"]
                else "FAIL",
            )
        )
    except BaseException:
        conn.send(dict(kind="error", error=traceback.format_exc()))
    finally:
        conn.close()


def run_trial(pack, profile, spawn, *, envs, workers, steps, warmup):
    """One model per process; disjoint MjData objects, synchronized timed start."""
    context = mp.get_context("spawn")
    start = context.Event()
    processes, pipes = [], []
    began = time.perf_counter()
    try:
        for worker in range(workers):
            parent, child = context.Pipe()
            proc = context.Process(
                target=_worker,
                args=(
                    child,
                    start,
                    str(pack),
                    profile,
                    spawn,
                    list(range(worker * (envs // workers), (worker + 1) * (envs // workers))),
                    steps,
                    warmup,
                ),
            )
            proc.start()
            child.close()
            pipes.append(parent)
            processes.append(proc)
        for pipe in pipes:
            if not pipe.poll(300):
                raise TimeoutError("worker model loading timed out")
            message = pipe.recv()
            if message["kind"] != "ready":
                raise RuntimeError(message.get("error", str(message)))
        ready = time.perf_counter()
        start.set()
        results = []
        for pipe in pipes:
            if not pipe.poll(300):
                raise TimeoutError("worker simulation timed out")
            message = pipe.recv()
            if message["kind"] != "result":
                raise RuntimeError(message.get("error", str(message)))
            results.append(message)
        for proc in processes:
            proc.join(timeout=10)
            if proc.exitcode != 0:
                raise RuntimeError(f"benchmark worker exited {proc.exitcode}")
    finally:
        for pipe in pipes:
            pipe.close()
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join()
    end = time.perf_counter()
    elapsed = max(w["end_s"] for w in results) - min(w["begin_s"] for w in results)
    completed = sum(w["steps_completed"] for w in results)
    rss = [w["rss_ready_bytes"] for w in results]
    return dict(
        profile=profile,
        env_count=envs,
        workers=workers,
        envs_per_worker=envs // workers,
        status="PASS" if all(w["status"] == "PASS" for w in results) else "FAIL",
        steps_completed=completed,
        stepping_wall_s=elapsed,
        steps_per_second=completed / elapsed,
        startup_wall_s=ready - began,
        end_to_end_wall_s=end - began,
        end_to_end_steps_per_second=completed / (end - began),
        worker_rss_sum_bytes=sum(rss) if all(x is not None for x in rss) else None,
        worker_results=results,
    )


def summarize(trials):
    reference = trials[0]["worker_results"][0]["traces"][0]
    identity = (reference["steps"], reference["state_contact_sha256"])
    parity = all(
        (t["steps"], t["state_contact_sha256"]) == identity
        for r in trials
        for w in r["worker_results"]
        for t in w["traces"]
    )
    groups = {}
    for row in trials:
        key = (row["profile"], row["env_count"], row["workers"])
        groups.setdefault(key, []).append(row)
    medians = []
    for (profile, envs, workers), rows in groups.items():
        summary = dict(
            profile=profile,
            env_count=envs,
            workers=workers,
            envs_per_worker=envs // workers,
            repeats=len(rows),
        )
        for key in (
            "steps_per_second",
            "end_to_end_steps_per_second",
            "startup_wall_s",
            "worker_rss_sum_bytes",
        ):
            values = [r[key] for r in rows if r[key] is not None]
            summary[key] = statistics.median(values) if values else None
        summary["steps_per_second_range"] = [
            min(r["steps_per_second"] for r in rows),
            max(r["steps_per_second"] for r in rows),
        ]
        summary["max_worker_load_s"] = statistics.median(
            max(w["load_s"] for w in r["worker_results"]) for r in rows
        )
        medians.append(summary)
    return dict(
        status="PASS" if parity and all(r["status"] == "PASS" for r in trials) else "FAIL",
        state_contact_parity="PASS" if parity else "FAIL",
        medians=medians,
    )


def run_benchmark(
    pack,
    *,
    output,
    profiles=None,
    env_counts=(1, 4, 16),
    workers=(1, 4),
    steps=2000,
    warmup=100,
    repeats=3,
    progress=None,
):
    if steps < 1 or warmup < 0 or repeats < 1 or not env_counts or not workers:
        raise ValueError("positive steps/repeats/env counts/workers are required")
    if any(n < 1 for n in (*env_counts, *workers)):
        raise ValueError("env counts and workers must be positive")
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise ValueError(f"output already exists: {target}")
    asset = FieldAsset.open(pack, verify=True)
    profiles = list(profiles or [p for p in PROFILE_ORDER if p in asset.available_runtime_profiles])
    for profile in profiles:
        asset.entrypoint_for(profile)
    layouts = [
        (n, w)
        for n in dict.fromkeys(env_counts)
        for w in dict.fromkeys(workers)
        if w <= n and n % w == 0
    ]
    if not layouts:
        raise ValueError("no evenly divisible worker/environment layout")
    spawn = select_spawn(asset)
    report = dict(
        schema_version=1,
        workload_id=WORKLOAD_ID,
        status="RUNNING",
        source_manifest_sha256=asset.manifest_sha256,
        heightfield_samples_sha256=asset.collision["samples_sha256"],
        validation_status=asset.manifest.get("validation_status"),
        profiles={
            p: scenario_descriptor(asset, "full_eval", profile=p)["profile_hash"] for p in profiles
        },
        robot_mjcf_sha256=hashlib.sha256(rover_xml().encode()).hexdigest(),
        seed=0,
        spawn=spawn,
        steps_per_env=steps,
        warmup_steps=warmup,
        repetitions=repeats,
        diagnostic_stride=DIAGNOSTIC_STRIDE,
        isolation="fresh spawn workers per trial; one model per worker, independent data",
        timing="control + physics + per-step state hash + sampled contact/health; no IO",
        memory_scope="sum of worker RSS after allocation; shared pages may be counted twice",
        cache_scope="OS file caches retained; load results are not cold-disk measurements",
        machine_before=machine_snapshot(),
        trials=[],
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    def save():
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    # Rotate order to reduce first-profile and changing-load bias.
    try:
        for repeat in range(repeats):
            for n, w in layouts:
                for profile in (
                    profiles[repeat % len(profiles) :] + profiles[: repeat % len(profiles)]
                ):
                    if progress:
                        progress(
                            f"{profile}: {n} envs / {w} workers, repeat {repeat + 1}/{repeats}"
                        )
                    result = run_trial(
                        asset.root, profile, spawn, envs=n, workers=w, steps=steps, warmup=warmup
                    )
                    report["trials"].append(dict(repeat=repeat, **result))
                    save()
        report.update(summarize(report["trials"]))
    except Exception as exc:
        report.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["machine_after"] = machine_snapshot()
        save()
    return report
