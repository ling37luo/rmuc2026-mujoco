"""Check an exported fly-ramp region with Isaac Lab's kit-less Newton backend.

Run with a Python environment containing optional ``newton`` and ``warp``::

    python examples/isaac_newton_fly_probe.py REGION \
      --source-pack PACK --output fly_newton_probe.json

This runs 120 mm sphere contacts at the approach, ramp face and landing top.
It is a collision smoke, not a robot jump or Isaac Sim/PhysX validation.
Newton XPBD is used because the installed MJWarp heightfield kernel can
overflow its 50-prism pair limit on this exact 1 cm field.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np

from rmuc2026_mujoco.manifest import FieldAsset, sha256_file
from rmuc2026_mujoco.query import HeightFieldData, load_heightfield
from rmuc2026_mujoco.scenarios import scenario_descriptor
from rmuc2026_mujoco.training_region import TrainingRegion


PROBE_RADIUS_M = 0.06
NORMAL_SUPPORT_TOLERANCE_M = 0.01


def _triangle_surface(grid: HeightFieldData, x_m: float, y_m: float) -> tuple[float, float, float]:
    """Return height and XY derivatives from the region's two-triangle split."""

    x_axis, y_axis = grid.x_m, grid.y_m
    if not (x_axis[0] <= x_m <= x_axis[-1] and y_axis[0] <= y_m <= y_axis[-1]):
        raise ValueError("probe left the region heightfield")
    col = int(np.clip(np.searchsorted(x_axis, x_m, side="right") - 1, 0, len(x_axis) - 2))
    row = int(np.clip(np.searchsorted(y_axis, y_m, side="right") - 1, 0, len(y_axis) - 2))
    dx = x_axis[col + 1] - x_axis[col]
    dy = y_axis[row + 1] - y_axis[row]
    u = (x_m - x_axis[col]) / dx
    v = (y_m - y_axis[row]) / dy
    z00, z10 = grid.height_m[row, col : col + 2]
    z01, z11 = grid.height_m[row + 1, col : col + 2]
    if v <= u:
        dz_dx = (z10 - z00) / dx
        dz_dy = (z11 - z10) / dy
        height = z00 + (z10 - z00) * u + (z11 - z10) * v
    else:
        dz_dx = (z11 - z01) / dx
        dz_dy = (z01 - z00) / dy
        height = z00 + (z11 - z01) * u + (z01 - z00) * v
    return float(height), float(dz_dx), float(dz_dy)


def _triangle_height(grid: HeightFieldData, x_m: float, y_m: float) -> float:
    return _triangle_surface(grid, x_m, y_m)[0]


def _source_parity(region: TrainingRegion, field: FieldAsset, grid: HeightFieldData) -> dict:
    source = region.manifest["source"]
    scenario = region.manifest["scenario_id"]
    if (
        source["source_manifest_sha256"] != field.manifest_sha256
        or source["source_collision_samples_sha256"] != field.collision["samples_sha256"]
        or source["source_profile_hash"]
        != scenario_descriptor(field, scenario, profile="collision_only")["profile_hash"]
    ):
        raise ValueError("region identity disagrees with the verified source pack")
    original = load_heightfield(field)
    (row0, row1), (col0, col1) = source["source_grid_slice_yx"]
    exact = (
        np.array_equal(grid.x_m, original.x_m[col0:col1])
        and np.array_equal(grid.y_m, original.y_m[row0:row1])
        and np.array_equal(grid.height_m, original.height_m[row0:row1, col0:col1])
    )
    if not exact:
        raise ValueError("region samples differ from the declared source grid slice")
    return {"exact_source_grid_slice": True, "source_grid_slice_yx": source["source_grid_slice_yx"]}


def run(
    region_path: Path, source_pack: Path | None, *, device: str, steps: int, dt_s: float
) -> dict:
    try:
        import newton
        import warp as wp
    except ImportError as exc:
        raise RuntimeError(
            "this optional example needs the kit-less Isaac Lab Newton environment"
        ) from exc

    region = TrainingRegion.open(region_path, verify=True)
    grid = region.heightfield()
    manifest = region.manifest
    route = manifest["route"]
    if manifest["grid"]["interpolation"] != "mujoco_hfield_triangle":
        raise ValueError("unsupported training-region interpolation")
    field = FieldAsset.open(source_pack, verify=True) if source_pack else None
    parity = _source_parity(region, field, grid) if field else {"exact_source_grid_slice": None}
    if device.startswith("cuda") and not wp.is_cuda_available():
        raise RuntimeError("Warp CUDA device is unavailable")

    mid_x = (route["low_seam_xyz_m"][0] + route["takeoff_xyz_m"][0]) / 2
    mid_y = (route["low_seam_xyz_m"][1] + route["takeoff_xyz_m"][1]) / 2
    points = {
        "approach": route["approach_xyz_m"],
        "ramp_mid": [mid_x, mid_y, _triangle_height(grid, mid_x, mid_y)],
        "landing": route["landing_target_xyz_m"],
    }
    hx = float((grid.x_m[-1] - grid.x_m[0]) / 2)
    hy = float((grid.y_m[-1] - grid.y_m[0]) / 2)
    center_x = float((grid.x_m[-1] + grid.x_m[0]) / 2)
    center_y = float((grid.y_m[-1] + grid.y_m[0]) / 2)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        started = time.perf_counter()
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.margin = 0.0
        cfg.gap = 0.001
        hfield = newton.Heightfield(
            data=np.asarray(grid.height_m, dtype=np.float32),
            nrow=len(grid.y_m),
            ncol=len(grid.x_m),
            hx=hx,
            hy=hy,
        )
        builder.add_shape_heightfield(
            xform=wp.transform(p=(center_x, center_y, 0.0), q=wp.quat_identity()),
            heightfield=hfield,
            cfg=cfg,
        )
        bodies = {}
        for name, (x_m, y_m, z_m) in points.items():
            body = builder.add_body(
                xform=wp.transform(
                    p=(x_m, y_m, z_m + PROBE_RADIUS_M + 0.025), q=wp.quat_identity()
                ),
                label=name,
            )
            builder.add_shape_sphere(body=body, radius=PROBE_RADIUS_M, cfg=cfg)
            bodies[name] = body
        model = builder.finalize(device=device)
        model.set_gravity((0.0, 0.0, -9.81))
        solver = newton.solvers.SolverXPBD(model, iterations=10)
        state_a, state_b = model.state(), model.state()
        control, contacts = model.control(), model.contacts()
        build_wall_s = time.perf_counter() - started
        started = time.perf_counter()
        max_contacts = 0
        for step in range(steps):
            model.collide(state_a, contacts)
            state_a.clear_forces()
            solver.step(state_a, state_b, control, contacts, dt_s)
            state_a, state_b = state_b, state_a
            if step % 10 == 0 or step == steps - 1:
                max_contacts = max(max_contacts, int(contacts.rigid_contact_count.numpy()[0]))
        wp.synchronize()
        simulation_wall_s = time.perf_counter() - started
        poses = state_a.body_q.numpy()

    probes = {}
    for name, body in bodies.items():
        x_m, y_m, z_m = (float(value) for value in poses[body, :3])
        terrain_z, dz_dx, dz_dy = _triangle_surface(grid, x_m, y_m)
        vertical_error = z_m - PROBE_RADIUS_M - terrain_z
        normal_error = (z_m - terrain_z) / math.sqrt(1 + dz_dx**2 + dz_dy**2)
        normal_error -= PROBE_RADIUS_M
        probes[name] = {
            "body_xyz_m": [x_m, y_m, z_m],
            "terrain_height_at_final_xy_m": terrain_z,
            "terrain_slope_deg": math.degrees(math.atan(math.hypot(dz_dx, dz_dy))),
            "sphere_bottom_minus_field_m": vertical_error,
            "normal_clearance_minus_radius_m": normal_error,
            "finite": bool(np.isfinite(poses[body]).all()),
            "normal_supported_within_1cm": abs(normal_error) <= NORMAL_SUPPORT_TOLERANCE_M,
        }
    passed = max_contacts > 0 and all(
        probe["finite"] and probe["normal_supported_within_1cm"] for probe in probes.values()
    )
    return {
        "status": "PASS_CONTACT_SMOKE" if passed else "FAIL_CONTACT_SMOKE",
        "scope": "Newton XPBD heightfield collision with 120 mm spheres; no robot flight or Isaac Sim/PhysX",
        "backend": "isaaclab_kitless_newton_xpbd",
        "newton_version": version("newton"),
        "warp_version": version("warp-lang"),
        "device": device,
        "region_manifest_sha256": region.manifest_sha256,
        "region_profile_hash": manifest["profile_hash"],
        "source_manifest_sha256": manifest["source"]["source_manifest_sha256"],
        "scenario_id": manifest["scenario_id"],
        "heightfield_shape_yx": list(grid.height_m.shape),
        "heightfield_npz_sha256": sha256_file(region.root / "collision/heightfield.npz"),
        "heightfield_float32_sha256": hashlib.sha256(
            np.asarray(grid.height_m, dtype=np.float32).tobytes()
        ).hexdigest(),
        "source_parity": parity,
        "steps": steps,
        "dt_s": dt_s,
        "build_wall_s": build_wall_s,
        "simulation_wall_s": simulation_wall_s,
        "max_contacts_sampled": max_contacts,
        "probes": probes,
        "warnings": {
            "python": [str(item.message) for item in captured],
            "solver_warning_counter": "not_exposed_by_newton_xpbd",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("region", type=Path, help="exported fly-ramp training-region directory")
    parser.add_argument(
        "--source-pack", type=Path, help="verify exact slice against source runtime pack"
    )
    parser.add_argument("--device", default="cuda:0", help="Newton/Warp device (default: cuda:0)")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument(
        "--output", type=Path, help="write JSON result without overwriting an existing file"
    )
    args = parser.parse_args()
    if args.steps <= 0 or not 0 < args.dt <= 0.01:
        parser.error("--steps must be positive and --dt must be in (0, 0.01]")
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    result = run(args.region, args.source_pack, device=args.device, steps=args.steps, dt_s=args.dt)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] == "PASS_CONTACT_SMOKE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
