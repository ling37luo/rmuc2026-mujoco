"""Local-only conversion of an official RMUC STEP into runtime assets."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Any


class BuilderUnavailable(RuntimeError):
    """Raised when optional CAD conversion dependencies are unavailable."""


def build_from_official_step(
    step_path: Path,
    output_dir: Path,
    *,
    target_visual_faces: int = 2_300_000,
    heightfield_resolution_m: float = 0.02,
    include_edge_void: bool = False,
) -> dict[str, Any]:
    """Build the field locally; no official or derived asset is uploaded.

    Install the ``build`` extra before calling this function.  Heavy imports
    stay lazy so users who only load an existing local asset pack need just
    NumPy and MuJoCo.
    """

    try:
        from ._conversion import build_field
    except ImportError as exc:  # pragma: no cover - depends on optional wheels
        raise BuilderUnavailable(
            "CAD builder dependencies are missing; install rmuc2026-mujoco[build]"
        ) from exc

    return build_field(
        Path(step_path),
        Path(output_dir),
        target_visual_faces=target_visual_faces,
        heightfield_resolution_m=heightfield_resolution_m,
        preserve_cad_colors=True,
        include_edge_void=include_edge_void,
    )


def build_runtime_asset_pack(
    step_path: Path,
    output_dir: Path,
    *,
    target_visual_faces: int = 2_300_000,
    heightfield_resolution_m: float = 0.02,
    include_edge_void: bool = False,
    include_surface_guide: bool = False,
    rulebook_pdf: Path | None = None,
    minimum_free_bytes: int = 5 * 1024**3,
    keep_source_build: Path | None = None,
) -> dict[str, Any]:
    """Create a runtime pack, optionally retaining source geometry for audits.

    The official input and every derivative remain on the caller's machine.
    The temporary full conversion is removed after the compact, hash-bound
    runtime pack has been produced, unless ``keep_source_build`` is supplied.
    """

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    source_destination = (
        None if keep_source_build is None else Path(keep_source_build).expanduser().resolve()
    )
    if source_destination is not None:
        if source_destination.exists():
            raise FileExistsError(f"source build output already exists: {source_destination}")
        if (
            source_destination == destination
            or destination in source_destination.parents
            or source_destination in destination.parents
        ):
            raise ValueError("source build and runtime pack must be separate directories")
    destination.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination.parent).free
    if free < minimum_free_bytes:
        raise OSError(f"insufficient free disk: required={minimum_free_bytes}, available={free}")

    from .pack import export_runtime_asset_pack

    temporary_root = Path(tempfile.mkdtemp(prefix=".rmuc2026-conversion-", dir=destination.parent))
    full_build = temporary_root / "full-build"
    try:
        build_from_official_step(
            step_path,
            full_build,
            target_visual_faces=target_visual_faces,
            heightfield_resolution_m=heightfield_resolution_m,
            include_edge_void=include_edge_void,
        )
        export_source = full_build
        if include_surface_guide:
            if rulebook_pdf is None:
                raise ValueError("rulebook_pdf is required when include_surface_guide is enabled")
            from ._conversion import decorate_field_build

            decorated_build = temporary_root / "decorated-build"
            decorate_field_build(full_build, Path(rulebook_pdf), decorated_build)
            export_source = decorated_build
        result = export_runtime_asset_pack(
            export_source,
            destination,
            include_surface_guide=include_surface_guide,
        )
        if source_destination is not None:
            source_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(full_build), source_destination)
        return result
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


__all__ = [
    "BuilderUnavailable",
    "build_from_official_step",
    "build_runtime_asset_pack",
]
