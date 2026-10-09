"""Visible, uncapped GLFW rendering/physics workload; no native viewer UI panels."""

from __future__ import annotations

import json
import math
import multiprocessing as mp
from pathlib import Path
import statistics
import struct
import time
import traceback
import zlib

import mujoco
import numpy as np

from .acceptance import PROFILE_ORDER, _rss_bytes, _warning_counts
from .display import FieldDisplayController
from .fly_routes import fly_route_descriptor
from .manifest import FieldAsset
from .performance import load_rover, machine_snapshot, select_spawn, wheel_command
from .slope_runtime import reset_slope_spawn
from .visual_benchmark import _camera


def _save_png(path, pixels):
    def chunk(kind, content):
        return (
            struct.pack(">I", len(content))
            + kind
            + content
            + struct.pack(">I", zlib.crc32(kind + content) & 0xFFFFFFFF)
        )

    h, w = pixels.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(b"\0" + row.tobytes() for row in pixels)))
        + chunk(b"IEND", b"")
    )


def _view_worker(conn, pack, profile, spawn, frames, width, height, screenshot_dir, repeat):
    import glfw
    from OpenGL import GL

    window, context = None, None
    try:
        if not glfw.init():
            raise RuntimeError("GLFW requires an available desktop display")
        glfw.window_hint(glfw.FOCUSED, glfw.FALSE)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        window = glfw.create_window(width, height, f"RMUC benchmark: {profile}", None, None)
        if window is None:
            raise RuntimeError("could not create the benchmark window")
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        asset = FieldAsset.open(pack, verify=False)
        load_started = time.perf_counter()
        model, data = load_rover(asset, profile, spawn)
        load_s = time.perf_counter() - load_started
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0  # Debug collision overlay stays hidden in the visual profiles.
        if profile == "collision_only":
            option.geomgroup[3] = 1
        FieldDisplayController(asset, lighting="flat", livery="off").apply(model, option.geomgroup)
        scene = mujoco.MjvScene(model, maxgeom=max(10000, model.ngeom + 100))
        context_started = time.perf_counter()
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
        context_s = time.perf_counter() - context_started
        fw, fh = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, fw, fh)
        center = asset.collision["geom_center_after_translation_m"]
        views = {
            "overview": _camera(mujoco, (center[0], center[1], 0.4), distance=32, azimuth=90),
            "rover": _camera(
                mujoco,
                (spawn["x_m"], spawn["y_m"], spawn["terrain_height_m"]),
                distance=3.0,
                azimuth=120,
            ),
        }
        for side, azimuth in [("north", 105), ("south", 285)]:
            route = fly_route_descriptor(asset, f"fly_ramp_{side}")
            views[f"fly_{side}"] = _camera(
                mujoco, tuple(route["takeoff_xyz_m"]), distance=5, azimuth=azimuth
            )
        route = dict(
            low_xyz_m=[spawn["x_m"], spawn["y_m"], spawn["terrain_height_m"]], heading_yaw_rad=0.0
        )
        records = {}
        for name, camera in views.items():

            def render():
                mujoco.mjv_updateScene(
                    model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene
                )
                mujoco.mjr_render(viewport, scene, context)

            reset_slope_spawn(model, data, route)
            for _ in range(10):
                render()
                glfw.swap_buffers(window)
                glfw.poll_events()
            GL.glFinish()
            times = []
            completed_steps = 0
            for frame in range(frames):
                if glfw.window_should_close(window):
                    raise RuntimeError("benchmark window closed before completion")
                begin = time.perf_counter()
                target_steps = int((frame + 1) / (60 * model.opt.timestep))
                while completed_steps < target_steps:
                    data.ctrl[:] = wheel_command(completed_steps * model.opt.timestep)
                    mujoco.mj_step(model, data)
                    completed_steps += 1
                render()
                glfw.swap_buffers(window)
                GL.glFinish()  # Include completed GPU work, not just command submission.
                glfw.poll_events()
                times.append(time.perf_counter() - begin)
            warnings = _warning_counts(data)
            finite = all(np.isfinite(x).all() for x in (data.qpos, data.qvel, data.qacc))
            records[name] = dict(
                frames=frames,
                physics_steps=completed_steps,
                frames_per_second=frames / sum(times),
                frame_ms_p50=float(np.percentile(times, 50) * 1000),
                frame_ms_p95=float(np.percentile(times, 95) * 1000),
                warnings=warnings,
                finite=bool(finite),
            )
            if screenshot_dir and repeat == 0:
                render()
                pixels = np.empty((fh, fw, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(pixels, None, viewport, context)
                path = Path(screenshot_dir) / f"{profile}_{name}.png"
                _save_png(path, np.flipud(pixels))
                records[name]["screenshot"] = str(path.resolve())
        conn.send(
            dict(
                kind="result",
                profile=profile,
                repeat=repeat,
                views=records,
                framebuffer_px=[fw, fh],
                model_load_s=load_s,
                render_context_s=context_s,
                rss_with_renderer_bytes=_rss_bytes(),
                gl_renderer=GL.glGetString(GL.GL_RENDERER).decode(),
                gl_version=GL.glGetString(GL.GL_VERSION).decode(),
                geometric_mean_fps=math.exp(
                    statistics.mean(math.log(v["frames_per_second"]) for v in records.values())
                ),
                status="PASS"
                if all(v["finite"] and not v["warnings"] for v in records.values())
                else "FAIL",
            )
        )
    except BaseException:
        conn.send(dict(kind="error", error=traceback.format_exc()))
    finally:
        if context is not None:
            context.free()
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()
        conn.close()


def run_view_benchmark(
    pack,
    *,
    output,
    profiles=None,
    frames=120,
    repeats=3,
    width=1280,
    height=720,
    screenshots=None,
    progress=None,
):
    if frames < 1 or repeats < 1 or min(width, height) < 64:
        raise ValueError("positive frame/repeat counts and viewport >=64 px are required")
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise ValueError(f"output already exists: {target}")
    asset = FieldAsset.open(pack, verify=True)
    profiles = list(profiles or [p for p in PROFILE_ORDER if p in asset.available_runtime_profiles])
    for profile in profiles:
        asset.entrypoint_for(profile)
    spawn = select_spawn(asset)
    report = dict(
        schema_version=1,
        status="RUNNING",
        source_manifest_sha256=asset.manifest_sha256,
        spawn=spawn,
        lighting="flat",
        livery="off",
        frames_per_view=frames,
        measurement="visible GLFW, swap interval 0, glFinish, physics + scene + render + swap",
        claim_boundary="Uncapped frame workload; excludes native passive-viewer UI panels. "
        "Device load includes other applications; not display refresh rate.",
        machine_before=machine_snapshot(),
        trials=[],
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    def save():
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    context = mp.get_context("spawn")
    save()
    try:
        for repeat in range(repeats):
            for profile in profiles[repeat % len(profiles) :] + profiles[: repeat % len(profiles)]:
                if progress:
                    progress(f"Visible {profile}, repeat {repeat + 1}/{repeats}")
                parent, child = context.Pipe(False)
                process = context.Process(
                    target=_view_worker,
                    args=(
                        child,
                        str(asset.root),
                        profile,
                        spawn,
                        frames,
                        width,
                        height,
                        str(screenshots) if screenshots else None,
                        repeat,
                    ),
                )
                process.start()
                child.close()
                try:
                    if not parent.poll(300):
                        raise TimeoutError("render benchmark timed out")
                    row = parent.recv()
                    process.join(timeout=10)
                    if row["kind"] != "result" or process.exitcode != 0:
                        raise RuntimeError(
                            row.get("error", f"render worker exited {process.exitcode}")
                        )
                    report["trials"].append(row)
                    save()
                finally:
                    parent.close()
                    if process.is_alive():
                        process.terminate()
                        process.join()
        report["medians"] = {}
        for profile in profiles:
            rows = [r for r in report["trials"] if r["profile"] == profile]
            rss = [
                r["rss_with_renderer_bytes"]
                for r in rows
                if r["rss_with_renderer_bytes"] is not None
            ]
            report["medians"][profile] = dict(
                geometric_mean_fps=statistics.median(r["geometric_mean_fps"] for r in rows),
                geometric_mean_fps_range=[
                    min(r["geometric_mean_fps"] for r in rows),
                    max(r["geometric_mean_fps"] for r in rows),
                ],
                rss_with_renderer_bytes=statistics.median(rss) if rss else None,
                views={
                    v: statistics.median(r["views"][v]["frames_per_second"] for r in rows)
                    for v in rows[0]["views"]
                },
            )
        report["status"] = (
            "PASS" if all(r["status"] == "PASS" for r in report["trials"]) else "FAIL"
        )
    except Exception as exc:
        report.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["machine_after"] = machine_snapshot()
        save()
    return report
