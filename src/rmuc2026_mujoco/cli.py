"""Command-line interface for source acquisition, local build, and runtime loading."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time

from .builder import BuilderUnavailable, build_runtime_asset_pack
from .download import (
    LAST_REVIEWED_RULEBOOK_PUBLICATION_DATE,
    LAST_REVIEWED_RULEBOOK_SHA256,
    LAST_REVIEWED_RULEBOOK_SIZE,
    LAST_REVIEWED_RULEBOOK_URL,
    LAST_REVIEWED_RULEBOOK_VERSION,
    LAST_RULE_REVIEW_DATE,
    OFFICIAL_RULE_CENTRE,
    OFFICIAL_SOURCE_PAGE,
    OFFICIAL_STEP_SHA256,
    OFFICIAL_STEP_SIZE,
    OFFICIAL_STEP_URL,
    DownloadError,
    download_official_rulebook_v2_0_0,
    download_official_step,
)
from .display import FieldDisplayController
from .energy_unit import load_field_with_energy_unit
from .errors import Rmuc2026Error
from .manifest import DEFAULT_RUNTIME_PROFILE, RUNTIME_PROFILE_NAMES, FieldAsset, verify_asset
from .mjcf import UNOFFICIAL_FRICTION_PRESETS, load_model
from .pack import ExportBlocked
from .perimeter_fence import export_fenced_pack
from .query import (
    HEIGHTFIELD_CLAIM_BOUNDARY,
    field_bounds,
    find_spawn_candidates,
    surface_at,
)
from .ramp_source_audit import audit_fly_ramp_source_overlap
from .viewer import (
    FocusScopedKeyboardListener,
    SafePassiveViewerSession,
    create_viewer_key_interceptor,
    stop_keyboard_listener,
    update_viewer_status_overlay,
    viewer_viewport_size,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rmuc2026-field")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("source", help="print the pinned official source identity")
    download = commands.add_parser("download", help="download directly from the official URL")
    download.add_argument("destination", type=Path)
    download.add_argument("--acknowledge-reference-only", action="store_true")
    build = commands.add_parser("build", help="build a local runtime pack from a verified STEP")
    build.add_argument("--step", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--target-visual-faces", type=int, default=2_300_000)
    build.add_argument("--heightfield-resolution", type=float, default=0.02)
    build.add_argument(
        "--experimental-edge-void",
        action="store_true",
        help="1 cm official-source outer void candidate; robot edge validation is blocked",
    )
    build.add_argument("--include-surface-guide", action="store_true")
    build.add_argument(
        "--rulebook",
        type=Path,
        help="local official V2.0.0 rulebook PDF; required with --include-surface-guide",
    )
    setup = commands.add_parser(
        "setup",
        help="download the pinned official STEP and build a local runtime pack",
    )
    setup.add_argument("output", type=Path)
    setup.add_argument(
        "--step-cache",
        type=Path,
        default=Path.home() / ".cache/rmuc2026-mujoco/RMUC2026_V2.0.0.stp",
    )
    # ``setup`` itself is an explicit request to fetch the official reference
    # into a local cache.  Keep accepting the older acknowledgement flag so
    # existing scripts continue to work, but do not require the redundant
    # spelling in the documented one-command setup interface.
    setup.add_argument(
        "--acknowledge-reference-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    setup.add_argument("--target-visual-faces", type=int, default=2_300_000)
    setup.add_argument("--heightfield-resolution", type=float, default=0.02)
    setup.add_argument(
        "--experimental-edge-void",
        action="store_true",
        help="1 cm official-source outer void candidate; robot edge validation is blocked",
    )
    setup.add_argument("--include-surface-guide", action="store_true")
    setup.add_argument(
        "--rulebook-cache",
        type=Path,
        default=Path.home() / ".cache/rmuc2026-mujoco/RMUC2026_rulebook_V2.0.0.pdf",
        help="verified local cache used only when --include-surface-guide is selected",
    )
    for name in ("verify", "info"):
        command = commands.add_parser(name)
        command.add_argument("asset", type=Path)
    fence = commands.add_parser(
        "fence-pack", help="make a new local pack with a physical perimeter proxy"
    )
    fence.add_argument("source", type=Path)
    fence.add_argument("output", type=Path)
    ramp_source = commands.add_parser(
        "ramp-source-audit", help="verify fly-ramp overlaps against the pinned local CAD"
    )
    ramp_source.add_argument("asset", type=Path)
    ramp_source.add_argument("source_manifest", type=Path)
    ramp_source.add_argument("--output", type=Path, required=True)
    surface = commands.add_parser("surface", help="query local heightfield geometry")
    surface.add_argument("asset", type=Path)
    surface.add_argument("x", type=float)
    surface.add_argument("y", type=float)
    surface.add_argument("--window-radius", type=float, default=0.15)
    spawns = commands.add_parser(
        "spawns",
        help="runtime-proxy screen terrain-only spawn candidates from the heightfield",
    )
    spawns.add_argument("asset", type=Path)
    spawns.add_argument("--count", type=int, default=8)
    spawns.add_argument("--footprint-radius", type=float, default=0.35)
    spawns.add_argument("--boundary-margin", type=float, default=0.25)
    spawns.add_argument("--minimum-separation", type=float, default=1.0)
    spawns.add_argument("--max-slope-deg", type=float, default=5.0)
    spawns.add_argument("--max-relief", type=float, default=0.03)
    spawns.add_argument("--ground-height-min", type=float, default=-0.05)
    spawns.add_argument("--ground-height-max", type=float, default=0.05)
    energy_unit = commands.add_parser(
        "energy-unit-probe", help="drop an optional rulebook-sized movable prop on verified terrain"
    )
    energy_unit.add_argument("asset", type=Path)
    energy_unit.add_argument("x", type=float)
    energy_unit.add_argument("y", type=float)
    energy_unit.add_argument("--steps", type=int, default=1000)
    view = commands.add_parser("view")
    view.add_argument("asset", type=Path)
    view.add_argument("--duration", type=float, default=0.0, help="seconds; 0 waits until close")
    view.add_argument(
        "--profile",
        choices=RUNTIME_PROFILE_NAMES,
        default=DEFAULT_RUNTIME_PROFILE,
        help="runtime profile; collision_only skips CAD visual meshes",
    )
    view.add_argument(
        "--friction",
        "--friction-preset",
        dest="friction_preset",
        choices=tuple(UNOFFICIAL_FRICTION_PRESETS),
        help="unofficial field-friction sensitivity preset",
    )
    view.add_argument(
        "--camera",
        choices=("overview", "spawn", "native"),
        default="overview",
        help="initial passive-viewer camera",
    )
    view.add_argument(
        "--lighting",
        choices=("flat", "shadow"),
        default="flat",
        help="initial field lighting mode; press L to toggle when viewer controls are installed",
    )
    view.add_argument(
        "--livery",
        choices=("off", "on"),
        default="off",
        help="initial optional surface-guide visibility; press G to toggle when available",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "source":
            print(
                json.dumps(
                    {
                        "download_started": False,
                        "included_in_package": False,
                        "official_source_page": OFFICIAL_SOURCE_PAGE,
                        "official_step_sha256": OFFICIAL_STEP_SHA256,
                        "official_step_size_bytes": OFFICIAL_STEP_SIZE,
                        "official_step_url": OFFICIAL_STEP_URL,
                        "license_review": "NO_EXPLICIT_GRANT_FOUND",
                        "redistribution_authorized": False,
                        "redistribution_permission": "UNVERIFIED",
                        "rulebook_review": {
                            "reviewed_on": LAST_RULE_REVIEW_DATE,
                            "official_rule_centre": OFFICIAL_RULE_CENTRE,
                            "latest_reviewed_version": LAST_REVIEWED_RULEBOOK_VERSION,
                            "latest_reviewed_publication_date": (
                                LAST_REVIEWED_RULEBOOK_PUBLICATION_DATE
                            ),
                            "latest_reviewed_url": LAST_REVIEWED_RULEBOOK_URL,
                            "latest_reviewed_size_bytes": LAST_REVIEWED_RULEBOOK_SIZE,
                            "latest_reviewed_sha256": LAST_REVIEWED_RULEBOOK_SHA256,
                            "comparison_scope": "rulebook_chapter_4_only",
                            "static_field_chapter_vs_v2_0_0": "NO_CHANGE_DETECTED",
                            "comparison_method": "normalised_text_and_embedded_image_hashes",
                            "mechanical_certification": False,
                            "future_updates_require_new_review": True,
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "download":
            result = download_official_step(
                args.destination,
                acknowledge_reference_only=args.acknowledge_reference_only,
                progress=_terminal_progress,
            )
            print(
                json.dumps(
                    {
                        "path": str(result.path),
                        "reused": result.reused,
                        "sha256": result.sha256,
                        "size_bytes": result.size_bytes,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "build":
            _validate_build_options(args.target_visual_faces, args.heightfield_resolution)
            if args.experimental_edge_void and args.heightfield_resolution != 0.01:
                raise ValueError("--experimental-edge-void requires --heightfield-resolution 0.01")
            if args.include_surface_guide and args.rulebook is None:
                raise ValueError("--rulebook is required with --include-surface-guide for build")
            result = build_runtime_asset_pack(
                args.step,
                args.output,
                target_visual_faces=args.target_visual_faces,
                heightfield_resolution_m=args.heightfield_resolution,
                include_edge_void=args.experimental_edge_void,
                include_surface_guide=args.include_surface_guide,
                rulebook_pdf=args.rulebook,
            )
            print(
                json.dumps(
                    {
                        "artifact_type": result["artifact_type"],
                        "output": str(args.output.expanduser().resolve()),
                        "status": result["status"],
                        "validation_status": result["validation_status"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "setup":
            _validate_build_options(args.target_visual_faces, args.heightfield_resolution)
            if args.experimental_edge_void and args.heightfield_resolution != 0.01:
                raise ValueError("--experimental-edge-void requires --heightfield-resolution 0.01")
            downloaded = download_official_step(
                args.step_cache,
                acknowledge_reference_only=True,
                progress=_terminal_progress,
            )
            downloaded_rulebook = None
            if args.include_surface_guide:
                downloaded_rulebook = download_official_rulebook_v2_0_0(
                    args.rulebook_cache,
                    acknowledge_reference_only=True,
                    progress=_terminal_progress,
                )
            result = build_runtime_asset_pack(
                downloaded.path,
                args.output,
                target_visual_faces=args.target_visual_faces,
                heightfield_resolution_m=args.heightfield_resolution,
                include_edge_void=args.experimental_edge_void,
                include_surface_guide=args.include_surface_guide,
                rulebook_pdf=(
                    downloaded_rulebook.path if downloaded_rulebook is not None else None
                ),
            )
            print(
                json.dumps(
                    {
                        "artifact_type": result["artifact_type"],
                        "official_step_reused": downloaded.reused,
                        "official_rulebook_reused": (
                            downloaded_rulebook.reused if downloaded_rulebook is not None else None
                        ),
                        "output": str(args.output.expanduser().resolve()),
                        "status": result["status"],
                        "validation_status": result["validation_status"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "verify":
            print(json.dumps(verify_asset(args.asset).to_dict(), indent=2, sort_keys=True))
            return 0
        if args.command == "fence-pack":
            contract = export_fenced_pack(args.source, args.output)
            print(
                json.dumps(
                    {
                        "output": str(args.output.expanduser().resolve()),
                        "perimeter_fence": contract,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "ramp-source-audit":
            report = audit_fly_ramp_source_overlap(
                FieldAsset.open(args.asset, verify=True), args.source_manifest
            )
            output = args.output.expanduser().resolve()
            if output.exists():
                raise ValueError(f"output already exists: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps({"status": report["status"], "output": str(output)}, sort_keys=True))
            return 0 if report["status"] == "PASS_STATIC" else 2
        asset = FieldAsset.open(args.asset, verify=True)
        if args.command == "info":
            report = asset.report().to_dict()
            report["field_bounds_xy_m"] = field_bounds(asset)
            report["validation_boundary"] = asset.manifest.get("validation_boundary")
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "surface":
            report = surface_at(
                asset,
                args.x,
                args.y,
                window_radius_m=args.window_radius,
            ).to_dict()
            report.update(
                {
                    "query_schema_version": 1,
                    "query_algorithm": "mujoco_hfield_triangle_v1",
                    "manifest_sha256": asset.manifest_sha256,
                    "heightfield_samples_sha256": asset.collision["samples_sha256"],
                }
            )
            print(
                json.dumps(
                    report,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "spawns":
            candidates = find_spawn_candidates(
                asset,
                count=args.count,
                footprint_radius_m=args.footprint_radius,
                boundary_margin_m=args.boundary_margin,
                minimum_separation_m=args.minimum_separation,
                max_slope_deg=args.max_slope_deg,
                max_relief_m=args.max_relief,
                ground_height_range_m=(args.ground_height_min, args.ground_height_max),
            )
            print(
                json.dumps(
                    {
                        "query_schema_version": 1,
                        "query_algorithm": "mujoco_hfield_triangle_spawn_screen_v1",
                        "manifest_sha256": asset.manifest_sha256,
                        "heightfield_samples_sha256": asset.collision["samples_sha256"],
                        "status": "RUNTIME_PROXY_SCREENED_NOT_TOPOLOGY_VERIFIED",
                        "requested_count": args.count,
                        "candidate_count": len(candidates),
                        "candidates": [candidate.to_dict() for candidate in candidates],
                        "selection_contract": {
                            "footprint": "axis_aligned_square_enclosing_requested_radius",
                            "slope_interpolation": "mujoco_hfield_triangle",
                            "footprint_radius_m": args.footprint_radius,
                            "boundary_margin_m": args.boundary_margin,
                            "minimum_separation_m": args.minimum_separation,
                            "max_slope_deg": args.max_slope_deg,
                            "max_relief_m": args.max_relief,
                            "ground_height_range_m": [
                                args.ground_height_min,
                                args.ground_height_max,
                            ],
                            "ground_height_range_applies_to": ("entire_screening_window"),
                        },
                        "topology_verified": False,
                        "claim_boundary": HEIGHTFIELD_CLAIM_BOUNDARY,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "energy-unit-probe":
            if args.steps <= 0:
                raise ValueError("--steps must be positive")
            import mujoco

            model, data, unit = load_field_with_energy_unit(asset, x_m=args.x, y_m=args.y)
            for _ in range(args.steps):
                mujoco.mj_step(model, data)
            warnings = {
                mujoco.mjtWarning(index).name: warning.number
                for index, warning in enumerate(data.warning)
                if warning.number
            }
            status = (
                "PASS"
                if not warnings
                and all(math.isfinite(float(value)) for value in data.qpos)
                and data.ncon > 0
                else "FAIL"
            )
            print(
                json.dumps(
                    {
                        "status": status,
                        "unit": unit,
                        "physics_steps": args.steps,
                        "sim_time_s": float(data.time),
                        "final_center_z_m": float(data.qpos[2]),
                        "final_contact_count": int(data.ncon),
                        "warnings": warnings,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0 if status == "PASS" else 2
        if args.duration < 0.0:
            raise ValueError("--duration must be non-negative")
        model, data = load_model(
            asset,
            profile=args.profile,
            friction_preset=args.friction_preset,
        )
        import mujoco.viewer

        display = FieldDisplayController(
            asset,
            lighting=args.lighting,
            livery=args.livery,
        )
        close_requested = threading.Event()
        started = time.monotonic()
        viewer_context = mujoco.viewer.launch_passive(model, data)
        viewer_session = SafePassiveViewerSession(viewer_context)
        with viewer_session as viewer:
            listener = _start_display_key_listener(
                display,
                close_requested,
                on_change=lambda: _show_display_status(
                    viewer,
                    display,
                    camera=args.camera,
                    shortcuts=True,
                ),
            )
            viewer_session.listener = listener
            with viewer.lock():
                fitted_viewport = _configure_camera(viewer, asset, args.camera)
                display.apply(model, viewer.opt.geomgroup)
            _show_display_status(
                viewer,
                display,
                camera=args.camera,
                shortcuts=listener is not None,
            )
            while (
                viewer.is_running()
                and not close_requested.is_set()
                and (args.duration == 0.0 or time.monotonic() - started < args.duration)
            ):
                current_viewport = viewer_viewport_size(viewer)
                needs_refit = (
                    args.camera == "overview"
                    and current_viewport is not None
                    and current_viewport != fitted_viewport
                )
                with viewer.lock():
                    if needs_refit:
                        fitted_viewport = _configure_camera(viewer, asset, args.camera)
                    display.apply(model, viewer.opt.geomgroup)
                viewer.sync()
                time.sleep(1.0 / 60.0)
        return 0
    except (
        Rmuc2026Error,
        DownloadError,
        BuilderUnavailable,
        ExportBlocked,
        OSError,
        ValueError,
    ) as exc:
        print(f"rmuc2026-field: {exc}", file=sys.stderr)
        return 2


def _validate_build_options(target_visual_faces: int, heightfield_resolution: float) -> None:
    if target_visual_faces <= 0:
        raise ValueError("--target-visual-faces must be positive")
    if not 0.01 <= heightfield_resolution <= 0.10:
        raise ValueError("--heightfield-resolution must be within [0.01, 0.10] metres")


def _start_display_key_listener(
    display: FieldDisplayController,
    close_requested: threading.Event,
    *,
    on_change=None,
    focus_check=None,
    keyboard_module=None,
    key_interceptor=None,
):
    """Start focus-scoped, edge-triggered L/G controls when pynput is usable.

    MuJoCo also assigns plain L/G to Additive/Fog.  A target-window passive
    grab reserves only those two unmodified keys while that viewer has focus.
    Keeping pynput suppression disabled means no other application is affected.
    """

    if keyboard_module is None:
        try:
            from pynput import keyboard as keyboard_module
        except ImportError:
            return None
    if focus_check is None or key_interceptor is None:
        if focus_check is not None or key_interceptor is not None:
            raise ValueError("focus_check and key_interceptor must be supplied together")
        input_scope = create_viewer_key_interceptor()
        if input_scope is None:
            return None
        key_interceptor, focus_check = input_scope

    pressed: set[str] = set()
    released_at: dict[str, float] = {}
    modifiers: set[str] = set()

    def modifier_name(key) -> str | None:
        groups = {
            "ALT": ("alt", "alt_l", "alt_r", "alt_gr"),
            "CTRL": ("ctrl", "ctrl_l", "ctrl_r"),
            "SHIFT": ("shift", "shift_l", "shift_r"),
            "CMD": ("cmd", "cmd_l", "cmd_r"),
        }
        for group, attributes in groups.items():
            for attribute in attributes:
                candidate = getattr(keyboard_module.Key, attribute, None)
                if candidate is not None and key == candidate:
                    return group
        return None

    def key_name(key) -> str | None:
        if key == keyboard_module.Key.esc:
            return "ESC"
        character = getattr(key, "char", None)
        if isinstance(character, str) and character:
            return character.upper()
        return None

    def on_press(key) -> bool | None:
        modifier = modifier_name(key)
        if modifier is not None:
            modifiers.add(modifier)
            return None
        name = key_name(key)
        if name is None or name in pressed:
            return None
        now = time.monotonic()
        if name in {"L", "G"} and now - released_at.get(name, float("-inf")) < 0.05:
            pressed.add(name)
            return None
        pressed.add(name)
        try:
            focused = bool(focus_check())
        except Exception:
            focused = False
        if not focused:
            return None
        if modifiers:
            return None
        if name == "ESC":
            close_requested.set()
            return False
        if display.press_name(name) and on_change is not None:
            on_change()
        return None

    def on_release(key) -> None:
        modifier = modifier_name(key)
        if modifier is not None:
            modifiers.discard(modifier)
            return
        name = key_name(key)
        if name is not None:
            pressed.discard(name)
            released_at[name] = time.monotonic()

    listener = keyboard_module.Listener(
        on_press=on_press,
        on_release=on_release,
        suppress=False,
    )
    try:
        listener.start()
    except Exception:
        stop_keyboard_listener(listener)
        key_interceptor.close()
        raise
    return FocusScopedKeyboardListener(listener, key_interceptor)


def _print_display_status(display: FieldDisplayController, *, shortcuts: bool) -> None:
    livery = "on" if display.livery_visible else "off"
    if not display.livery_available:
        livery = "unavailable"
    shortcut_text = "L/G enabled" if shortcuts else "L/G unavailable; use launch options"
    print(
        f"[display] lighting={display.lighting_mode} livery={livery}; {shortcut_text}",
        flush=True,
    )


def _show_display_status(
    viewer,
    display: FieldDisplayController,
    *,
    camera: str,
    shortcuts: bool,
) -> None:
    livery = "on" if display.livery_visible else "off"
    if not display.livery_available:
        livery = "unavailable"
    update_viewer_status_overlay(
        viewer,
        lighting=display.lighting_mode,
        livery=livery,
        camera=camera,
        shortcuts=shortcuts,
    )
    _print_display_status(display, shortcuts=shortcuts)


def _configure_camera(viewer, asset: FieldAsset, mode: str) -> tuple[int, int] | None:
    viewport = viewer_viewport_size(viewer)
    if mode == "native":
        return viewport
    (minimum_x, minimum_y), (maximum_x, maximum_y) = field_bounds(asset)
    if mode == "overview":
        minimum_z, maximum_z = _field_height_bounds(asset)
        viewer.cam.lookat[:] = (
            0.5 * (minimum_x + maximum_x),
            0.5 * (minimum_y + maximum_y),
            0.5 * (minimum_z + maximum_z),
        )
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -50.0
        viewer.cam.distance = _overview_camera_distance(
            viewer,
            half_width_x=0.5 * (maximum_x - minimum_x),
            half_width_y=0.5 * (maximum_y - minimum_y),
            half_height_z=0.5 * (maximum_z - minimum_z),
            viewport=viewport,
        )
        return viewport
    viewer.cam.lookat[:] = (0.0, 0.0, 0.0)
    viewer.cam.distance = 5.0
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -25.0
    return viewport


def _field_height_bounds(asset: FieldAsset) -> tuple[float, float]:
    collision = asset.collision
    minimum = float(collision.get("minimum_height_m", 0.0))
    maximum = float(collision.get("maximum_height_m", minimum))
    spawn = asset.manifest.get("coordinate_frame", {}).get("recommended_spawn", {})
    translation = float(spawn.get("terrain_height_m", 0.0))
    return minimum - translation, maximum - translation


def _overview_camera_distance(
    viewer,
    *,
    half_width_x: float,
    half_width_y: float,
    half_height_z: float,
    viewport: tuple[int, int] | None,
) -> float:
    """Fit the full 3D field box with 12% breathing room in the live viewport."""

    aspect = 16.0 / 9.0 if viewport is None else viewport[0] / viewport[1]
    fovy = 45.0
    try:
        fovy = float(viewer.m.vis.global_.fovy)
    except Exception:
        pass
    fovy = min(175.0, max(1.0, fovy))
    vertical_tangent = math.tan(math.radians(0.5 * fovy))
    horizontal_tangent = vertical_tangent * max(aspect, 1.0e-6)

    azimuth = math.radians(float(viewer.cam.azimuth))
    elevation = math.radians(float(viewer.cam.elevation))
    right_extent = abs(math.cos(azimuth)) * half_width_x + abs(math.sin(azimuth)) * half_width_y
    forward_extent = abs(math.sin(azimuth)) * half_width_x + abs(math.cos(azimuth)) * half_width_y
    vertical_extent = (
        abs(math.sin(elevation)) * forward_extent + abs(math.cos(elevation)) * half_height_z
    )
    depth_extent = (
        abs(math.cos(elevation)) * forward_extent + abs(math.sin(elevation)) * half_height_z
    )
    perspective_fit = max(
        right_extent / horizontal_tangent,
        vertical_extent / vertical_tangent,
    )
    return depth_extent + 1.12 * perspective_fit


_LAST_PROGRESS_BUCKET = -1


def _terminal_progress(downloaded: int, total: int) -> None:
    """Print bounded download progress without flooding redirected logs."""

    global _LAST_PROGRESS_BUCKET
    if total <= 0:
        return
    percent = min(100, int(downloaded * 100 / total))
    bucket = percent // 5
    if bucket == _LAST_PROGRESS_BUCKET and downloaded < total:
        return
    _LAST_PROGRESS_BUCKET = bucket
    print(
        f"official STEP download: {percent:3d}% "
        f"({downloaded / 1024**2:.1f}/{total / 1024**2:.1f} MiB)",
        file=sys.stderr,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
