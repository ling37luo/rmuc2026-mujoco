"""Measure fixed-camera field rendering without changing any field physics."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

from .fly_routes import fly_route_descriptor
from .manifest import FieldAsset
from .mjcf import load_model


def _camera(mujoco, lookat: tuple[float, float, float], *, distance: float, azimuth: float):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = -55.0
    return camera


def benchmark_render(
    pack: str | Path,
    *,
    profile: str,
    frames: int = 60,
    width: int = 1280,
    height: int = 720,
) -> dict:
    """Time the same three field-only views for one profile in this process."""

    if frames < 1 or min(width, height) < 64:
        raise ValueError("frames and viewport dimensions must be positive")
    import mujoco

    asset = FieldAsset.open(pack, verify=True)
    model, data = load_model(asset, profile=profile)
    bounds = asset.collision["geom_center_after_translation_m"]
    overview = (float(bounds[0]), float(bounds[1]), 0.4)
    routes = {side: fly_route_descriptor(asset, f"fly_ramp_{side}") for side in ("north", "south")}
    views = {
        "overview": _camera(mujoco, overview, distance=32.0, azimuth=90.0),
        "fly_north": _camera(
            mujoco,
            tuple(float(x) for x in routes["north"]["takeoff_xyz_m"]),
            distance=5.0,
            azimuth=105.0,
        ),
        "fly_south": _camera(
            mujoco,
            tuple(float(x) for x in routes["south"]["takeoff_xyz_m"]),
            distance=5.0,
            azimuth=285.0,
        ),
    }
    results = {}
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for name, camera in views.items():
            for _ in range(5):
                renderer.update_scene(data, camera=camera)
                renderer.render()
            start = time.perf_counter()
            for _ in range(frames):
                renderer.update_scene(data, camera=camera)
                renderer.render()
            elapsed = time.perf_counter() - start
            results[name] = {
                "frames": frames,
                "wall_time_s": elapsed,
                "frames_per_second": frames / elapsed,
            }
    fps = [record["frames_per_second"] for record in results.values()]
    return {
        "artifact_type": "rmuc2026_visual_benchmark",
        "manifest_sha256": asset.manifest_sha256,
        "profile": profile,
        "resolution_px": [width, height],
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "views": results,
        "geometric_mean_fps": math.exp(sum(math.log(value) for value in fps) / len(fps)),
        "claim_boundary": "fixed-camera field-only rendering throughput, not interactive GUI FPS",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = benchmark_render(args.pack, profile=args.profile, frames=args.frames)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
