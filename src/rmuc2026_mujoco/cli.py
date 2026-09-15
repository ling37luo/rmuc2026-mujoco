"""Command-line interface for source acquisition, local build, and runtime loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from .builder import BuilderUnavailable, build_runtime_asset_pack
from .download import (
    OFFICIAL_SOURCE_PAGE,
    OFFICIAL_STEP_SHA256,
    OFFICIAL_STEP_SIZE,
    OFFICIAL_STEP_URL,
    DownloadError,
    download_official_step,
)
from .errors import Rmuc2026Error
from .manifest import DEFAULT_RUNTIME_PROFILE, RUNTIME_PROFILE_NAMES, FieldAsset, verify_asset
from .mjcf import UNOFFICIAL_FRICTION_PRESETS, load_model
from .pack import ExportBlocked
from .query import field_bounds


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
    build.add_argument("--target-visual-faces", type=int, default=450_000)
    build.add_argument("--heightfield-resolution", type=float, default=0.02)
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
    setup.add_argument("--acknowledge-reference-only", action="store_true")
    setup.add_argument("--target-visual-faces", type=int, default=450_000)
    setup.add_argument("--heightfield-resolution", type=float, default=0.02)
    for name in ("verify", "info"):
        command = commands.add_parser(name)
        command.add_argument("asset", type=Path)
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
                        "redistribution_permission": "UNVERIFIED",
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
            result = build_runtime_asset_pack(
                args.step,
                args.output,
                target_visual_faces=args.target_visual_faces,
                heightfield_resolution_m=args.heightfield_resolution,
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
            downloaded = download_official_step(
                args.step_cache,
                acknowledge_reference_only=args.acknowledge_reference_only,
                progress=_terminal_progress,
            )
            result = build_runtime_asset_pack(
                downloaded.path,
                args.output,
                target_visual_faces=args.target_visual_faces,
                heightfield_resolution_m=args.heightfield_resolution,
            )
            print(
                json.dumps(
                    {
                        "artifact_type": result["artifact_type"],
                        "official_step_reused": downloaded.reused,
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
        asset = FieldAsset.open(args.asset, verify=True)
        if args.command == "info":
            report = asset.report().to_dict()
            report["field_bounds_xy_m"] = field_bounds(asset)
            report["validation_boundary"] = asset.manifest.get("validation_boundary")
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.duration < 0.0:
            raise ValueError("--duration must be non-negative")
        model, data = load_model(
            asset,
            profile=args.profile,
            friction_preset=args.friction_preset,
        )
        import mujoco.viewer

        started = time.monotonic()
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and (
                args.duration == 0.0 or time.monotonic() - started < args.duration
            ):
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
    if not 0.02 <= heightfield_resolution <= 0.10:
        raise ValueError("--heightfield-resolution must be within [0.02, 0.10] metres")


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
