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
    target_visual_faces: int = 450_000,
    heightfield_resolution_m: float = 0.02,
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
    )


def build_runtime_asset_pack(
    step_path: Path,
    output_dir: Path,
    *,
    target_visual_faces: int = 450_000,
    heightfield_resolution_m: float = 0.02,
    minimum_free_bytes: int = 5 * 1024**3,
) -> dict[str, Any]:
    """Create the relocatable runtime pack without retaining the heavy GLB.

    The official input and every derivative remain on the caller's machine.
    The temporary full conversion is removed after the compact, hash-bound
    runtime pack has been produced.
    """

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
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
        )
        return export_runtime_asset_pack(full_build, destination)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


__all__ = [
    "BuilderUnavailable",
    "build_from_official_step",
    "build_runtime_asset_pack",
]
