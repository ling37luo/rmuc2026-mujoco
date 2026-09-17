"""Fail-closed loading for a relocatable RMUC 2026 runtime asset pack."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET
import zlib

import numpy as np

from .download import OFFICIAL_STEP_SHA256, OFFICIAL_STEP_SIZE
from .errors import AssetIntegrityError, ManifestError
from .livery import (
    GROUND_MARKING_KNOWN_LIMITATIONS,
    GROUND_MARKING_OVERLAY_KIND,
    GROUND_MARKING_PROCESSING_ALGORITHM,
    GROUND_MARKING_REMOVED_BY_DESIGN,
    GROUND_MARKING_RETAINED_CONTENT,
    GROUND_MARKING_RGB_BLEED_RADIUS_PX,
    SOURCE_BAKED_SCENE_CONTENT,
    SOURCE_SURFACE_GUIDE_KIND,
)


RUNTIME_ARTIFACT_TYPE = "rmuc2026_mujoco_runtime_asset_pack"
SUPPORTED_SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, SUPPORTED_SCHEMA_VERSION})
SUPPORTED_VALIDATION_STATUS = "DRAFT_BLOCKED"
DEFAULT_RUNTIME_PROFILE = "full"
RUNTIME_PROFILE_NAMES = ("full", "collision_only")
CURRENT_COLLISION_KIND = "top_surface_heightfield_proxy"
LEGACY_COLLISION_KIND = "conservative_top_surface_heightfield"
SUPPORTED_COLLISION_KINDS = frozenset({CURRENT_COLLISION_KIND, LEGACY_COLLISION_KIND})
SUPPORTED_VISUAL_DISPLAY_PROFILE = "cad_source_contrast_v2"
KEY_LIGHT_NAME = "rmuc2026_key_light"
FILL_LIGHT_NAME = "rmuc2026_fill_light"
SURFACE_GUIDE_KIND = SOURCE_SURFACE_GUIDE_KIND
SURFACE_GUIDE_GEOM_NAME = "rmuc2026_surface_guide"
SURFACE_GUIDE_GEOM_GROUP = 4
SURFACE_GUIDE_BAKED_CONTENT = SOURCE_BAKED_SCENE_CONTENT
# Compatibility constants for the short-lived filtered-marking schema-2 pack.
# New packs use the complete SOURCE_SURFACE_GUIDE_KIND contract above.
GROUND_MARKING_SURFACE_SAMPLING_M = 0.05
GROUND_MARKING_SURFACE_CLEARANCE_M = 0.018
GROUND_MARKING_SURFACE_MAXIMUM_HEIGHT_M = 0.70
GROUND_MARKING_SURFACE_MAXIMUM_CELL_HEIGHT_DELTA_M = 0.20
XML_FLOAT_REL_TOLERANCE = 1e-8
XML_FLOAT_ABS_TOLERANCE = 1e-10
MAX_HEIGHTFIELD_SAMPLES = 10_000_000
MAX_HEIGHTFIELD_PNG_BYTES = 64 * 1024 * 1024
# The bootstrap PNG is a 16-bit rendering of the verified float samples.  A
# producer and this validator can only differ through the last-bit rounding of
# the same expression, so one LSB is the whole legitimate disagreement.
HEIGHTFIELD_PNG_LSB_TOLERANCE = 1
HEIGHTFIELD_PNG_SAMPLE_MAXIMUM = 65535.0
PNG_FILTER_NONE = 0
PNG_FILTER_SUB = 1
PNG_FILTER_UP = 2
PNG_FILTER_AVERAGE = 3
PNG_FILTER_PAETH = 4
PNG_GRAYSCALE16_BYTES_PER_PIXEL = 2
_GEOM_TRANSFORM_ATTRIBUTES = frozenset({"axisangle", "euler", "fromto", "quat", "xyaxes", "zaxis"})


def sha256_file(path: Path, *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    """Return a lowercase SHA-256 digest without reading a whole file at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: object, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ManifestError(f"{label} must be {qualifier}")
    return result


def _integer(value: object, *, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ManifestError(f"{label} must be an integer >= {minimum}")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be an object")
    return value


def _nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{label} must be a non-empty string")
    return value


def _sha256(value: object, *, label: str) -> str:
    result = _nonempty_string(value, label=label)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ManifestError(f"{label} must be a lowercase SHA-256")
    return result


def _number_list(
    value: object,
    *,
    label: str,
    length: int,
    positive: bool = False,
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ManifestError(f"{label} must contain {length} numbers")
    return tuple(_finite_number(item, label=label, positive=positive) for item in value)


@dataclass(frozen=True)
class ValidationReport:
    """A compact, JSON-safe result returned after complete pack verification."""

    root: Path
    manifest_sha256: str
    entrypoint: Path
    visual_mesh_count: int
    verified_file_count: int
    verified_bytes: int
    collision_shape: tuple[int, int]
    validation_status: str
    hashes_verified: bool

    def to_dict(self) -> dict[str, object]:
        return {
            # A size-only pass must not read as an integrity claim, so it gets
            # its own status instead of sharing PASS with the verified path.
            "status": "PASS" if self.hashes_verified else "PASS_SIZE_ONLY",
            "hashes_verified": self.hashes_verified,
            "root": str(self.root),
            "manifest_sha256": self.manifest_sha256,
            "entrypoint": str(self.entrypoint),
            "visual_mesh_count": self.visual_mesh_count,
            "verified_file_count": self.verified_file_count,
            "verified_bytes": self.verified_bytes,
            "collision_shape": list(self.collision_shape),
            "validation_status": self.validation_status,
        }


@dataclass(frozen=True)
class _RuntimeCollisionContract:
    """Validated manifest values that every runtime MJCF must preserve."""

    rows: int
    columns: int
    half_size_xy_m: tuple[float, float]
    maximum_height_m: float
    base_depth_m: float
    geom_center_m: tuple[float, float, float]
    image_relative: str
    image_path: Path


@dataclass(frozen=True)
class _RuntimeLiveryContract:
    """Validated schema-2 livery values bound to the full runtime profile."""

    mesh_relative: str
    texture_relative: str


@dataclass(frozen=True)
class FieldAsset:
    """A fully validated runtime pack and its parsed manifest."""

    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    verified_files: tuple[Path, ...]
    # False means only the declared sizes were compared, so a consumer that
    # treats this pack as verified must check the flag instead of the report
    # status alone.
    hashes_verified: bool
    _files_by_relative: Mapping[str, Path]

    @classmethod
    def open(cls, root: str | Path, *, verify: bool = True) -> FieldAsset:
        """Open the runtime-pack schema and optionally verify every bound file.

        Verification covers the declared sizes, every SHA-256, and the heightfield
        bootstrap PNG against the float samples it is supposed to quantize.
        ``verify=False`` keeps the structural and cross-reference checks only.
        """

        pack_root = Path(root).expanduser().resolve()
        manifest_path = pack_root / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ManifestError(f"manifest.json is missing or is a symlink: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestError(f"manifest.json could not be read: {exc}") from exc
        if not isinstance(manifest, dict):
            raise ManifestError("manifest root must be an object")
        _validate_identity(manifest)
        file_paths = _validate_file_table(pack_root, manifest, verify=verify)
        _validate_cross_references(manifest, file_paths, verify=verify)
        return cls(
            root=pack_root,
            manifest=manifest,
            manifest_sha256=sha256_file(manifest_path),
            verified_files=tuple(file_paths.values()),
            hashes_verified=verify,
            _files_by_relative=file_paths,
        )

    @property
    def entrypoint(self) -> Path:
        """Return the default/full entrypoint retained by the schema-1 API."""

        return self.entrypoint_for(DEFAULT_RUNTIME_PROFILE)

    @property
    def runtime_profiles(self) -> Mapping[str, Mapping[str, Any]]:
        """Return validated runtime profiles, synthesizing ``full`` for legacy packs."""

        runtime = self.manifest.get("runtime_profiles")
        if runtime is None:
            return {
                DEFAULT_RUNTIME_PROFILE: {
                    "entrypoint": str(self.manifest["contents"]["entrypoint"]),
                    "includes_visual_meshes": True,
                    "visual_mesh_count": len(self.visual_meshes),
                    "intended_use": "legacy_default",
                }
            }
        return runtime["profiles"]

    @property
    def available_runtime_profiles(self) -> tuple[str, ...]:
        """Return selectable profile names in stable public order."""

        return tuple(name for name in RUNTIME_PROFILE_NAMES if name in self.runtime_profiles)

    def entrypoint_for(self, profile: str = DEFAULT_RUNTIME_PROFILE) -> Path:
        """Resolve a declared profile entrypoint without accepting arbitrary paths."""

        try:
            record = self.runtime_profiles[profile]
        except KeyError as exc:
            available = ", ".join(self.available_runtime_profiles)
            raise ManifestError(
                f"runtime profile {profile!r} is unavailable; available: {available}"
            ) from exc
        relative = str(record["entrypoint"])
        return self._files_by_relative[relative]

    @property
    def visual_meshes(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.manifest["visual_meshes"])

    @property
    def collision(self) -> Mapping[str, Any]:
        return self.manifest["collision"]

    @property
    def coordinate_frame(self) -> Mapping[str, Any]:
        return self.manifest["coordinate_frame"]

    @property
    def recommended_spawn(self) -> Mapping[str, Any]:
        return self.coordinate_frame["recommended_spawn"]

    def file(self, relative: str, *, label: str = "asset") -> Path:
        """Resolve a file declared by the signed table; undeclared files are rejected."""

        try:
            return self._files_by_relative[relative]
        except KeyError as exc:
            raise AssetIntegrityError(
                f"{label} is not declared in contents.files: {relative}"
            ) from exc

    def report(self) -> ValidationReport:
        collision = self.collision
        return ValidationReport(
            root=self.root,
            manifest_sha256=self.manifest_sha256,
            entrypoint=self.entrypoint,
            visual_mesh_count=len(self.visual_meshes),
            verified_file_count=len(self.verified_files),
            verified_bytes=sum(path.stat().st_size for path in self.verified_files),
            collision_shape=(int(collision["rows_y"]), int(collision["columns_x"])),
            validation_status=str(self.manifest["validation_status"]),
            hashes_verified=self.hashes_verified,
        )


def verify_asset(root: str | Path) -> ValidationReport:
    """Fully verify a runtime asset pack and return its compact report."""

    return FieldAsset.open(root, verify=True).report()


def _validate_identity(manifest: Mapping[str, Any]) -> None:
    schema_version = manifest.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in SUPPORTED_SCHEMA_VERSIONS
    ):
        raise ManifestError("unsupported runtime-pack schema_version")
    if manifest.get("artifact_type") != RUNTIME_ARTIFACT_TYPE:
        raise ManifestError("unsupported artifact_type; pass a runtime pack, not a build directory")
    if manifest.get("status") != "PASS":
        raise ManifestError("runtime-pack integrity status is not PASS")
    if manifest.get("validation_status") != SUPPORTED_VALIDATION_STATUS:
        raise ManifestError("unsupported or overstated validation_status")
    source = _mapping(manifest.get("source_identity"), label="source_identity")
    if source.get("official_step_sha256") != OFFICIAL_STEP_SHA256:
        raise ManifestError("runtime pack is not bound to the expected official STEP hash")
    if source.get("official_step_size_bytes") != OFFICIAL_STEP_SIZE:
        raise ManifestError("runtime pack has the wrong official STEP size identity")
    if source.get("license_status") != "UNSPECIFIED":
        raise ManifestError("runtime pack must preserve the unspecified upstream asset license")
    if source.get("redistribution_authorized") is not False:
        raise ManifestError("runtime pack must preserve the unresolved redistribution boundary")
    if source.get("asset_included_in_source_repository") is not False:
        raise ManifestError("runtime pack claims that third-party-derived assets are in source")
    distribution = _mapping(manifest.get("distribution"), label="distribution")
    expected_distribution = {
        "generated_locally": True,
        "third_party_geometry": True,
        "safe_to_publish_without_rightsholder_permission": False,
        "code_license_applies_to_asset_pack": False,
    }
    if distribution != expected_distribution:
        raise ManifestError("runtime pack must preserve the local-only distribution boundary")
    boundary = _mapping(manifest.get("validation_boundary"), label="validation_boundary")
    if boundary.get("status") != SUPPORTED_VALIDATION_STATUS:
        raise ManifestError("validation_boundary.status disagrees with validation_status")
    if boundary.get("whole_field_topology_ready") is not False:
        raise ManifestError("runtime pack overstates whole-field collision topology")
    if boundary.get("final_policy_validation_ready") is not False:
        raise ManifestError("runtime pack overstates final policy-validation readiness")


def _validate_file_table(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    verify: bool,
) -> dict[str, Path]:
    contents = _mapping(manifest.get("contents"), label="contents")
    records = contents.get("files")
    if not isinstance(records, list) or not records:
        raise ManifestError("contents.files must be a non-empty list")
    if contents.get("file_count_excluding_manifest") != len(records):
        raise ManifestError("contents.file_count_excluding_manifest is inconsistent")
    result: dict[str, Path] = {}
    for index, value in enumerate(records):
        record = _mapping(value, label=f"contents.files[{index}]")
        relative = _nonempty_string(record.get("file"), label=f"contents.files[{index}].file")
        if relative in result:
            raise ManifestError(f"contents.files contains duplicate path: {relative}")
        expected_size = _integer(
            record.get("size_bytes"), label=f"contents.files[{index}].size_bytes", minimum=0
        )
        expected_sha = _sha256(record.get("sha256"), label=f"contents.files[{index}].sha256")
        path = _contained_regular_file(root, relative, label=f"contents.files[{index}]")
        if path.stat().st_size != expected_size:
            raise AssetIntegrityError(
                f"size mismatch for {relative}: expected={expected_size}, actual={path.stat().st_size}"
            )
        if verify:
            actual_sha = sha256_file(path)
            if actual_sha != expected_sha:
                raise AssetIntegrityError(
                    f"SHA-256 mismatch for {relative}: expected={expected_sha}, actual={actual_sha}"
                )
        result[relative] = path
    return result


def _validate_cross_references(
    manifest: Mapping[str, Any],
    files: Mapping[str, Path],
    *,
    verify: bool,
) -> None:
    contents = _mapping(manifest.get("contents"), label="contents")
    entrypoint = _nonempty_string(contents.get("entrypoint"), label="contents.entrypoint")
    if entrypoint not in files or Path(entrypoint).suffix.lower() != ".xml":
        raise ManifestError("contents.entrypoint must name a declared XML file")
    visual = manifest.get("visual_meshes")
    if not isinstance(visual, list) or not visual:
        raise ManifestError("visual_meshes must be a non-empty list")
    if contents.get("visual_obj_count") != len(visual):
        raise ManifestError("contents.visual_obj_count is inconsistent")
    schema_version = int(manifest["schema_version"])
    display_rgba_count = 0
    for index, value in enumerate(visual):
        record = _mapping(value, label=f"visual_meshes[{index}]")
        relative = _nonempty_string(record.get("file"), label=f"visual_meshes[{index}].file")
        declared_sha = _sha256(record.get("sha256"), label=f"visual_meshes[{index}].sha256")
        _require_cross_file(
            files, manifest, relative, declared_sha, label=f"visual_meshes[{index}]"
        )
        rgba = _number_list(record.get("rgba"), label=f"visual_meshes[{index}].rgba", length=4)
        if any(value < 0.0 or value > 1.0 for value in rgba):
            raise ManifestError(f"visual_meshes[{index}].rgba must be within [0, 1]")
        if record.get("display_rgba") is not None:
            display_rgba = _number_list(
                record.get("display_rgba"),
                label=f"visual_meshes[{index}].display_rgba",
                length=4,
            )
            if any(value < 0.0 or value > 1.0 for value in display_rgba):
                raise ManifestError(f"visual_meshes[{index}].display_rgba must be within [0, 1]")
            display_rgba_count += 1
    if schema_version == 1:
        if display_rgba_count not in {0, len(visual)}:
            raise ManifestError("display_rgba must be recorded for every visual mesh or none")
        if display_rgba_count:
            _validate_visual_display(manifest, require_lighting=False)
        if manifest.get("visual_layers") is not None:
            raise ManifestError("visual_layers requires runtime-pack schema_version 2")
        livery_contract = None
    else:
        if display_rgba_count != len(visual):
            raise ManifestError("schema 2 requires display_rgba for every visual mesh")
        _validate_visual_display(manifest, require_lighting=True)
        livery_contract = _validate_visual_layers(manifest, files)

    collision = _mapping(manifest.get("collision"), label="collision")
    if collision.get("kind") not in SUPPORTED_COLLISION_KINDS:
        raise ManifestError("unsupported collision.kind")
    for path_key, hash_key in (
        ("image_file", "image_sha256"),
        ("samples_file", "samples_sha256"),
    ):
        relative = _nonempty_string(collision.get(path_key), label=f"collision.{path_key}")
        digest = _sha256(collision.get(hash_key), label=f"collision.{hash_key}")
        _require_cross_file(files, manifest, relative, digest, label=f"collision.{path_key}")
    rows = _integer(collision.get("rows_y"), label="collision.rows_y", minimum=2)
    columns = _integer(collision.get("columns_x"), label="collision.columns_x", minimum=2)
    if rows * columns > MAX_HEIGHTFIELD_SAMPLES:
        raise ManifestError(
            "collision heightfield exceeds the runtime sample limit: "
            f"{rows}x{columns} > {MAX_HEIGHTFIELD_SAMPLES} samples"
        )
    half_size = _number_list(
        collision.get("half_size_xy_m"),
        label="collision.half_size_xy_m",
        length=2,
        positive=True,
    )
    geom_center = _number_list(
        collision.get("geom_center_after_translation_m"),
        label="collision.geom_center_after_translation_m",
        length=3,
    )
    maximum = _finite_number(
        collision.get("maximum_height_m"), label="collision.maximum_height_m", positive=True
    )
    minimum = _finite_number(collision.get("minimum_height_m"), label="collision.minimum_height_m")
    if minimum < 0.0 or minimum > maximum:
        raise ManifestError("collision minimum/maximum heights are inconsistent")
    base_depth = _finite_number(
        collision.get("base_depth_m"), label="collision.base_depth_m", positive=True
    )
    if collision.get("png_rows") != "flipped_y_for_mujoco_hfield_loader":
        raise ManifestError("unsupported collision PNG row orientation")
    # Preview-era packs predate these sample-provenance records, so they stay
    # optional; when a pack does record them they must be usable as evidence.
    for counter in ("ray_misses_filled_with_ground", "isolated_spikes_replaced"):
        recorded = collision.get(counter)
        if recorded is None:
            continue
        if isinstance(recorded, bool) or not isinstance(recorded, int) or recorded < 0:
            raise ManifestError(f"collision.{counter} must be a non-negative integer")
    structural_audit = collision.get("structural_audit")
    if structural_audit is not None:
        audit = _mapping(structural_audit, label="collision.structural_audit")
        for flag in ("underpasses_and_overhangs_preserved", "sealed_underpass_risk"):
            if flag in audit and not isinstance(audit[flag], bool):
                raise ManifestError(f"collision.structural_audit.{flag} must be a boolean")

    frame = _mapping(manifest.get("coordinate_frame"), label="coordinate_frame")
    if frame.get("world_units") != "metre-radian-kilogram-second" or frame.get("z_up") is not True:
        raise ManifestError("unsupported coordinate frame")
    if frame.get("npz_axes_before_world_translation") is not True:
        raise ManifestError("coordinate frame must declare pre-translation NPZ axes")
    expected_formulas = {
        "world_x_m": "x_m - recommended_spawn.x_before_translation_m",
        "world_y_m": "y_m - recommended_spawn.y_before_translation_m",
        "world_z_m": "height_m - recommended_spawn.terrain_height_m",
    }
    if any(frame.get(key) != value for key, value in expected_formulas.items()):
        raise ManifestError("coordinate-frame translation formulas are unsupported")
    spawn = _mapping(frame.get("recommended_spawn"), label="coordinate_frame.recommended_spawn")
    for key in ("x_before_translation_m", "y_before_translation_m", "terrain_height_m"):
        _finite_number(spawn.get(key), label=f"coordinate_frame.recommended_spawn.{key}")

    precision = _mapping(manifest.get("heightfield_precision"), label="heightfield_precision")
    if precision.get("float_npz_injection_required_before_validated_physics") is not True:
        raise ManifestError("exact NPZ heightfield injection must be required")
    if precision.get("rows_y") != rows or precision.get("columns_x") != columns:
        raise ManifestError("heightfield_precision shape disagrees with collision")
    if precision.get("float_samples_file") != collision.get("samples_file"):
        raise ManifestError("heightfield_precision sample file disagrees with collision")
    if verify:
        _verify_collision_bootstrap(
            collision,
            files,
            rows=rows,
            columns=columns,
            maximum_height_m=maximum,
        )

    _validate_runtime_profiles(
        manifest,
        files,
        visual_mesh_count=len(visual),
        visual_mesh_files=tuple(str(record["file"]) for record in visual),
        collision_contract=_RuntimeCollisionContract(
            rows=rows,
            columns=columns,
            half_size_xy_m=(half_size[0], half_size[1]),
            maximum_height_m=maximum,
            base_depth_m=base_depth,
            geom_center_m=(geom_center[0], geom_center[1], geom_center[2]),
            image_relative=str(collision["image_file"]),
            image_path=files[str(collision["image_file"])],
        ),
        require_named_lighting=schema_version == 2,
        livery_contract=livery_contract,
    )


def _validate_visual_display(
    manifest: Mapping[str, Any],
    *,
    require_lighting: bool,
) -> None:
    display = _mapping(manifest.get("visual_display"), label="visual_display")
    if display.get("profile") != SUPPORTED_VISUAL_DISPLAY_PROFILE:
        raise ManifestError("unsupported visual_display.profile")
    if display.get("source_rgba_preserved_in_visual_meshes_rgba") is not True:
        raise ManifestError("visual_display must preserve source rgba provenance")
    if display.get("display_rgba_recorded_per_visual_mesh") is not True:
        raise ManifestError("visual_display must bind every rendered colour")
    if display.get("physics_changed") is not False:
        raise ManifestError("visual_display must not claim a physics change")
    if not require_lighting:
        return
    lighting = _mapping(display.get("lighting"), label="visual_display.lighting")
    expected = {
        "key_light_name": KEY_LIGHT_NAME,
        "fill_light_name": FILL_LIGHT_NAME,
        "default_mode": "flat",
        "modes": ["flat", "shadow"],
        "toggle_key": "L",
        "physics_changed": False,
    }
    if lighting != expected:
        raise ManifestError("visual_display.lighting does not match the schema-2 contract")


def _validate_visual_layers(
    manifest: Mapping[str, Any],
    files: Mapping[str, Path],
) -> _RuntimeLiveryContract | None:
    layers = _mapping(manifest.get("visual_layers"), label="visual_layers")
    if not set(layers).issubset({"livery"}):
        raise ManifestError("visual_layers contains unsupported layers")
    if "livery" not in layers:
        return None
    livery = _mapping(layers.get("livery"), label="visual_layers.livery")
    common_keys = {
        "kind",
        "geom_name",
        "geom_group",
        "default_visible",
        "toggle_key",
        "physics",
        "mesh_file",
        "mesh_sha256",
        "texture_file",
        "texture_sha256",
    }
    common_values = {
        "geom_name": SURFACE_GUIDE_GEOM_NAME,
        "geom_group": SURFACE_GUIDE_GEOM_GROUP,
        "default_visible": False,
        "toggle_key": "G",
        "physics": False,
    }
    kind = livery.get("kind")
    if any(livery.get(key) != value for key, value in common_values.items()):
        raise ManifestError("visual_layers.livery does not match the schema-2 contract")
    if kind == SURFACE_GUIDE_KIND:
        if set(livery) != common_keys | {"contains_baked_scene_content"}:
            raise ManifestError("visual_layers.livery contains unsupported or missing fields")
        if livery.get("contains_baked_scene_content") != list(SURFACE_GUIDE_BAKED_CONTENT):
            raise ManifestError("visual_layers.livery does not match the full-guide contract")
        texture_role = "official_rulebook_surface_texture"
    elif kind == GROUND_MARKING_OVERLAY_KIND:
        expected_keys = common_keys | {
            "source_kind",
            "source_contains_baked_scene_content",
            "processing",
            "surface_sampling",
            "known_limitations",
        }
        if set(livery) != expected_keys:
            raise ManifestError("visual_layers.livery contains unsupported or missing fields")
        if (
            livery.get("source_kind") != SURFACE_GUIDE_KIND
            or livery.get("source_contains_baked_scene_content")
            != list(SURFACE_GUIDE_BAKED_CONTENT)
            or livery.get("known_limitations") != list(GROUND_MARKING_KNOWN_LIMITATIONS)
        ):
            raise ManifestError("visual_layers.livery source or limitation contract is invalid")
        _validate_ground_marking_processing(livery.get("processing"))
        _validate_surface_sampling(livery.get("surface_sampling"))
        texture_role = "rulebook_derived_ground_marking_texture"
    else:
        raise ManifestError("visual_layers.livery kind is unsupported")
    mesh_relative = _nonempty_string(
        livery.get("mesh_file"), label="visual_layers.livery.mesh_file"
    )
    texture_relative = _nonempty_string(
        livery.get("texture_file"), label="visual_layers.livery.texture_file"
    )
    if Path(mesh_relative).suffix.lower() != ".obj":
        raise ManifestError("visual_layers.livery.mesh_file must be an OBJ")
    if Path(texture_relative).suffix.lower() != ".png":
        raise ManifestError("visual_layers.livery.texture_file must be a PNG")
    mesh_sha = _sha256(livery.get("mesh_sha256"), label="visual_layers.livery.mesh_sha256")
    texture_sha = _sha256(livery.get("texture_sha256"), label="visual_layers.livery.texture_sha256")
    _require_cross_file(files, manifest, mesh_relative, mesh_sha, label="visual_layers.livery.mesh")
    _require_cross_file(
        files,
        manifest,
        texture_relative,
        texture_sha,
        label="visual_layers.livery.texture",
    )
    records = manifest["contents"]["files"]
    roles = {record.get("file"): record.get("role") for record in records}
    if roles.get(mesh_relative) != "surface_guide_visual_mesh":
        raise ManifestError("visual_layers.livery.mesh_file has the wrong file role")
    if roles.get(texture_relative) != texture_role:
        raise ManifestError("visual_layers.livery.texture_file has the wrong file role")
    return _RuntimeLiveryContract(
        mesh_relative=mesh_relative,
        texture_relative=texture_relative,
    )


def _validate_ground_marking_processing(value: object) -> None:
    processing = _mapping(value, label="visual_layers.livery.processing")
    expected_keys = {
        "algorithm",
        "output_mode",
        "retained_content",
        "removed_by_design",
        "minimum_component_span_m",
        "minimum_component_area_m2",
        "input_size_px",
        "input_pixels",
        "strong_candidate_pixels",
        "retained_pixels",
        "transparent_pixels",
        "retained_fraction",
        "transparent_rgb_edge_bleed_radius_px",
        "transparent_rgb_edge_bleed_pixels",
        "components",
    }
    if set(processing) != expected_keys:
        raise ManifestError("visual_layers.livery.processing contains unsupported fields")
    expected_values = {
        "algorithm": GROUND_MARKING_PROCESSING_ALGORITHM,
        "output_mode": "rgba_with_transparent_non_markings",
        "retained_content": list(GROUND_MARKING_RETAINED_CONTENT),
        "removed_by_design": list(GROUND_MARKING_REMOVED_BY_DESIGN),
    }
    if any(processing.get(key) != wanted for key, wanted in expected_values.items()):
        raise ManifestError("visual_layers.livery.processing method contract is invalid")
    minimum_span = _finite_number(
        processing.get("minimum_component_span_m"),
        label="visual_layers.livery.processing.minimum_component_span_m",
        positive=True,
    )
    minimum_area = _finite_number(
        processing.get("minimum_component_area_m2"),
        label="visual_layers.livery.processing.minimum_component_area_m2",
        positive=True,
    )
    if not math.isclose(minimum_span, 0.45) or not math.isclose(minimum_area, 0.002):
        raise ManifestError("visual_layers.livery.processing component gate is unsupported")
    size = processing.get("input_size_px")
    if not isinstance(size, list) or len(size) != 2:
        raise ManifestError("visual_layers.livery.processing.input_size_px must contain two values")
    width, height = (
        _integer(item, label="visual_layers.livery.processing.input_size_px", minimum=2)
        for item in size
    )
    input_pixels = _integer(
        processing.get("input_pixels"),
        label="visual_layers.livery.processing.input_pixels",
        minimum=4,
    )
    candidates = _integer(
        processing.get("strong_candidate_pixels"),
        label="visual_layers.livery.processing.strong_candidate_pixels",
        minimum=1,
    )
    retained = _integer(
        processing.get("retained_pixels"),
        label="visual_layers.livery.processing.retained_pixels",
        minimum=1,
    )
    transparent = _integer(
        processing.get("transparent_pixels"),
        label="visual_layers.livery.processing.transparent_pixels",
        minimum=0,
    )
    fraction = _finite_number(
        processing.get("retained_fraction"),
        label="visual_layers.livery.processing.retained_fraction",
    )
    if (
        input_pixels != width * height
        or retained + transparent != input_pixels
        or candidates > input_pixels
        or not 0.0 < fraction <= 1.0
        or not math.isclose(fraction, retained / input_pixels, rel_tol=1.0e-12)
    ):
        raise ManifestError("visual_layers.livery.processing pixel statistics are inconsistent")
    if processing.get("transparent_rgb_edge_bleed_radius_px") != GROUND_MARKING_RGB_BLEED_RADIUS_PX:
        raise ManifestError("visual_layers.livery.processing RGB edge bleed is unsupported")
    edge_bleed = _integer(
        processing.get("transparent_rgb_edge_bleed_pixels"),
        label="visual_layers.livery.processing.transparent_rgb_edge_bleed_pixels",
        minimum=0,
    )
    if edge_bleed > transparent:
        raise ManifestError("visual_layers.livery.processing RGB edge bleed is inconsistent")
    components = _mapping(
        processing.get("components"), label="visual_layers.livery.processing.components"
    )
    if set(components) != {"detected", "retained", "rejected"}:
        raise ManifestError("visual_layers.livery.processing.components is invalid")
    detected = _integer(
        components.get("detected"),
        label="visual_layers.livery.processing.components.detected",
        minimum=1,
    )
    kept = _integer(
        components.get("retained"),
        label="visual_layers.livery.processing.components.retained",
        minimum=1,
    )
    rejected = _integer(
        components.get("rejected"),
        label="visual_layers.livery.processing.components.rejected",
        minimum=0,
    )
    if kept + rejected != detected:
        raise ManifestError("visual_layers.livery.processing component counts are inconsistent")


def _validate_surface_sampling(value: object) -> None:
    sampling = _mapping(value, label="visual_layers.livery.surface_sampling")
    expected_keys = {
        "method",
        "target_spacing_m",
        "actual_max_spacing_xy_m",
        "clearance_m",
        "maximum_surface_height_m",
        "maximum_cell_height_delta_m",
        "vertices",
        "faces",
        "omitted_cells",
        "cell_count",
        "marking_cells",
        "transparent_cells",
        "height_filtered_marking_cells",
        "mesh_size_bytes",
        "boundary_xy_height_interpolation",
        "source_texture_alpha_drives_topology",
    }
    if set(sampling) != expected_keys:
        raise ManifestError("visual_layers.livery.surface_sampling contains unsupported fields")
    expected_values = {
        "method": "sparse_marking_driven_5cm_mujoco_triangle_height_sampling",
        "target_spacing_m": GROUND_MARKING_SURFACE_SAMPLING_M,
        "clearance_m": GROUND_MARKING_SURFACE_CLEARANCE_M,
        "maximum_surface_height_m": GROUND_MARKING_SURFACE_MAXIMUM_HEIGHT_M,
        "maximum_cell_height_delta_m": GROUND_MARKING_SURFACE_MAXIMUM_CELL_HEIGHT_DELTA_M,
        "boundary_xy_height_interpolation": True,
        "source_texture_alpha_drives_topology": True,
    }
    if any(sampling.get(key) != wanted for key, wanted in expected_values.items()):
        raise ManifestError("visual_layers.livery.surface_sampling method contract is invalid")
    actual = sampling.get("actual_max_spacing_xy_m")
    if not isinstance(actual, list) or len(actual) != 2:
        raise ManifestError(
            "visual_layers.livery.surface_sampling.actual_max_spacing_xy_m is invalid"
        )
    actual_spacing = [
        _finite_number(
            item,
            label="visual_layers.livery.surface_sampling.actual_max_spacing_xy_m",
            positive=True,
        )
        for item in actual
    ]
    if any(item > GROUND_MARKING_SURFACE_SAMPLING_M + 1.0e-9 for item in actual_spacing):
        raise ManifestError("visual_layers.livery.surface_sampling exceeds its 5 cm target")
    vertices = _integer(
        sampling.get("vertices"),
        label="visual_layers.livery.surface_sampling.vertices",
        minimum=4,
    )
    faces = _integer(
        sampling.get("faces"),
        label="visual_layers.livery.surface_sampling.faces",
        minimum=2,
    )
    omitted = _integer(
        sampling.get("omitted_cells"),
        label="visual_layers.livery.surface_sampling.omitted_cells",
        minimum=0,
    )
    cells = _integer(
        sampling.get("cell_count"),
        label="visual_layers.livery.surface_sampling.cell_count",
        minimum=1,
    )
    marking_cells = _integer(
        sampling.get("marking_cells"),
        label="visual_layers.livery.surface_sampling.marking_cells",
        minimum=1,
    )
    transparent_cells = _integer(
        sampling.get("transparent_cells"),
        label="visual_layers.livery.surface_sampling.transparent_cells",
        minimum=0,
    )
    height_filtered = _integer(
        sampling.get("height_filtered_marking_cells"),
        label="visual_layers.livery.surface_sampling.height_filtered_marking_cells",
        minimum=0,
    )
    _integer(
        sampling.get("mesh_size_bytes"),
        label="visual_layers.livery.surface_sampling.mesh_size_bytes",
        minimum=1,
    )
    visible_cells = marking_cells - height_filtered
    if (
        marking_cells + transparent_cells != cells
        or not 0 <= height_filtered < marking_cells
        or omitted != transparent_cells + height_filtered
        or faces != 2 * visible_cells
        or vertices > 4 * visible_cells
    ):
        raise ManifestError("visual_layers.livery.surface_sampling mesh counts are inconsistent")


def _validate_runtime_profiles(
    manifest: Mapping[str, Any],
    files: Mapping[str, Path],
    *,
    visual_mesh_count: int,
    visual_mesh_files: tuple[str, ...],
    collision_contract: _RuntimeCollisionContract,
    require_named_lighting: bool,
    livery_contract: _RuntimeLiveryContract | None,
) -> None:
    """Validate the optional schema-1 profile extension and its MJCF semantics.

    Packs produced before v0.1.0 did not contain this extension and remain valid as
    full-only packs. Once the extension is present, it is fail-closed: both named
    profiles and all profile metadata must match the declared MJCF payload.
    """

    value = manifest.get("runtime_profiles")
    if value is None:
        return
    runtime = _mapping(value, label="runtime_profiles")
    if set(runtime) != {"default", "profiles"}:
        raise ManifestError("runtime_profiles must contain exactly default and profiles")
    if runtime.get("default") != DEFAULT_RUNTIME_PROFILE:
        raise ManifestError("runtime_profiles.default must be 'full'")
    profiles = _mapping(runtime.get("profiles"), label="runtime_profiles.profiles")
    if set(profiles) != set(RUNTIME_PROFILE_NAMES):
        raise ManifestError("runtime_profiles must declare exactly full and collision_only")

    expected = {
        "full": {
            "includes_visual_meshes": True,
            "visual_mesh_count": visual_mesh_count,
            "intended_use": "interactive_visualization",
            "file_role": "field_mjcf_full",
        },
        "collision_only": {
            "includes_visual_meshes": False,
            "visual_mesh_count": 0,
            "intended_use": "headless_physics",
            "file_role": "field_mjcf_collision_only",
        },
    }
    entrypoints: set[str] = set()
    for name in RUNTIME_PROFILE_NAMES:
        label = f"runtime_profiles.profiles.{name}"
        record = _mapping(profiles.get(name), label=label)
        if set(record) != {
            "entrypoint",
            "includes_visual_meshes",
            "visual_mesh_count",
            "intended_use",
        }:
            raise ManifestError(f"{label} contains unsupported or missing fields")
        entrypoint = _nonempty_string(record.get("entrypoint"), label=f"{label}.entrypoint")
        if entrypoint in entrypoints:
            raise ManifestError("runtime profile entrypoints must be distinct")
        entrypoints.add(entrypoint)
        if entrypoint not in files or Path(entrypoint).suffix.lower() != ".xml":
            raise ManifestError(f"{label}.entrypoint must name a declared XML file")
        wanted = expected[name]
        if record.get("includes_visual_meshes") is not wanted["includes_visual_meshes"]:
            raise ManifestError(f"{label}.includes_visual_meshes is inconsistent")
        if record.get("visual_mesh_count") != wanted["visual_mesh_count"]:
            raise ManifestError(f"{label}.visual_mesh_count is inconsistent")
        if record.get("intended_use") != wanted["intended_use"]:
            raise ManifestError(f"{label}.intended_use is unsupported")
        file_record = next(
            item for item in manifest["contents"]["files"] if item.get("file") == entrypoint
        )
        if file_record.get("role") != wanted["file_role"]:
            raise ManifestError(f"{label}.entrypoint has the wrong file role")
        _validate_runtime_profile_xml(
            files[entrypoint],
            profile=name,
            expected_visual_mesh_count=int(wanted["visual_mesh_count"]),
            expected_visual_mesh_files=(visual_mesh_files if name == "full" else ()),
            collision_contract=collision_contract,
            require_named_lighting=require_named_lighting,
            livery_contract=(livery_contract if name == "full" else None),
        )

    if manifest["contents"]["entrypoint"] != profiles["full"]["entrypoint"]:
        raise ManifestError("contents.entrypoint must remain the full runtime profile")


def _validate_runtime_profile_xml(
    path: Path,
    *,
    profile: str,
    expected_visual_mesh_count: int,
    expected_visual_mesh_files: tuple[str, ...],
    collision_contract: _RuntimeCollisionContract,
    require_named_lighting: bool,
    livery_contract: _RuntimeLiveryContract | None,
) -> None:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ManifestError(f"runtime profile {profile!r} MJCF is invalid XML: {exc}") from exc
    if root.findall(".//include"):
        raise ManifestError(f"runtime profile {profile!r} must not load MJCF include files")
    for compiler in root.findall("./compiler"):
        path_attributes = {"assetdir", "meshdir", "strippath", "texturedir"}.intersection(
            compiler.attrib
        )
        if path_attributes:
            raise ManifestError(
                f"runtime profile {profile!r} compiler must not redirect asset paths: "
                f"{sorted(path_attributes)}"
            )

    worldbodies = root.findall("./worldbody")
    if len(worldbodies) != 1:
        raise ManifestError(f"runtime profile {profile!r} must contain exactly one worldbody")
    worldbody = worldbodies[0]
    if require_named_lighting:
        key_lights = worldbody.findall(f"./light[@name='{KEY_LIGHT_NAME}']")
        fill_lights = worldbody.findall(f"./light[@name='{FILL_LIGHT_NAME}']")
        if len(key_lights) != 1 or len(fill_lights) != 1:
            raise ManifestError(f"runtime profile {profile!r} lacks the named field lights")
        if any(
            light.get("directional") != "true" or light.get("castshadow") != "false"
            for light in (*key_lights, *fill_lights)
        ):
            raise ManifestError(f"runtime profile {profile!r} default lighting must be flat")

    mesh_assets = root.findall("./asset/mesh")
    mesh_geoms = [geom for geom in root.iter("geom") if geom.get("mesh")]
    livery_mesh_count = 1 if livery_contract is not None else 0
    expected_total_mesh_count = expected_visual_mesh_count + livery_mesh_count
    if len(mesh_assets) != expected_total_mesh_count:
        raise ManifestError(
            f"runtime profile {profile!r} declares {len(mesh_assets)} mesh assets; "
            f"expected {expected_total_mesh_count}"
        )
    if len(mesh_geoms) != expected_total_mesh_count:
        raise ManifestError(
            f"runtime profile {profile!r} declares {len(mesh_geoms)} mesh geoms; "
            f"expected {expected_total_mesh_count}"
        )
    mesh_files = tuple(mesh.get("file") for mesh in mesh_assets)
    expected_mesh_files = (
        *expected_visual_mesh_files,
        *((livery_contract.mesh_relative,) if livery_contract is not None else ()),
    )
    if None in mesh_files or sorted(mesh_files) != sorted(expected_mesh_files):
        raise ManifestError(
            f"runtime profile {profile!r} visual mesh files disagree with the manifest"
        )
    mesh_names = tuple(mesh.get("name") for mesh in mesh_assets)
    if None in mesh_names or len(set(mesh_names)) != len(mesh_names):
        raise ManifestError(f"runtime profile {profile!r} visual mesh names must be unique")
    if sorted(geom.get("mesh") for geom in mesh_geoms) != sorted(mesh_names):
        raise ManifestError(
            f"runtime profile {profile!r} visual geoms must reference every declared mesh once"
        )
    for mesh in mesh_assets:
        altered = {"refpos", "refquat", "scale"}.intersection(mesh.attrib)
        if altered:
            raise ManifestError(
                f"runtime profile {profile!r} visual mesh must retain manifest coordinates: "
                f"{sorted(altered)}"
            )

    file_assets = [
        asset
        for asset in root.findall("./asset/*")
        if asset.get("file") is not None and asset.tag not in {"hfield", "mesh"}
    ]
    expected_texture_files = (
        (livery_contract.texture_relative,) if livery_contract is not None else ()
    )
    if tuple(asset.get("file") for asset in file_assets) != expected_texture_files:
        raise ManifestError(f"runtime profile {profile!r} contains unsupported file-backed assets")
    if livery_contract is not None:
        livery_meshes = [
            mesh for mesh in mesh_assets if mesh.get("file") == livery_contract.mesh_relative
        ]
        livery_geoms = worldbody.findall(f"./geom[@name='{SURFACE_GUIDE_GEOM_NAME}']")
        if len(livery_meshes) != 1 or len(livery_geoms) != 1:
            raise ManifestError(f"runtime profile {profile!r} lacks the declared livery mesh geom")
        livery_geom = livery_geoms[0]
        livery_mesh_name = livery_meshes[0].get("name")
        if livery_geom.get("type") != "mesh" or livery_geom.get("mesh") != livery_mesh_name:
            raise ManifestError(
                f"runtime profile {profile!r} livery geom references the wrong mesh"
            )
        if (
            livery_geom.get("group") != str(SURFACE_GUIDE_GEOM_GROUP)
            or livery_geom.get("contype") != "0"
            or livery_geom.get("conaffinity") != "0"
        ):
            raise ManifestError(
                f"runtime profile {profile!r} livery geom is not visual-only group 4"
            )
        textures = [
            texture
            for texture in root.findall("./asset/texture")
            if texture.get("file") == livery_contract.texture_relative
        ]
        if len(textures) != 1 or textures[0].get("type") != "2d":
            raise ManifestError(f"runtime profile {profile!r} lacks the declared livery texture")
        texture_name = textures[0].get("name")
        materials = [
            material
            for material in root.findall("./asset/material")
            if material.get("texture") == texture_name
        ]
        if len(materials) != 1 or livery_geom.get("material") != materials[0].get("name"):
            raise ManifestError(f"runtime profile {profile!r} livery material is inconsistent")
    hfields = [
        item for item in root.findall("./asset/hfield") if item.get("name") == "rmuc2026_collision"
    ]
    collision_geoms = [
        item
        for item in root.iter("geom")
        if item.get("name") == "rmuc2026_field_collision"
        and item.get("type") == "hfield"
        and item.get("hfield") == "rmuc2026_collision"
    ]
    if len(root.findall("./asset/hfield")) != 1 or len(hfields) != 1 or len(collision_geoms) != 1:
        raise ManifestError(f"runtime profile {profile!r} lacks the canonical collision hfield")

    if collision_geoms[0] not in worldbody.findall("./geom"):
        raise ManifestError(
            f"runtime profile {profile!r} canonical collision geom must be a direct worldbody child"
        )
    collision_geom = collision_geoms[0]
    for mesh_geom in mesh_geoms:
        if mesh_geom not in worldbody.findall("./geom"):
            raise ManifestError(
                f"runtime profile {profile!r} visual mesh geoms must be direct worldbody children"
            )
        visual_transforms = {"pos", *_GEOM_TRANSFORM_ATTRIBUTES}.intersection(mesh_geom.attrib)
        if visual_transforms:
            raise ManifestError(
                f"runtime profile {profile!r} visual mesh geom must retain manifest coordinates: "
                f"{sorted(visual_transforms)}"
            )
    if collision_geom.get("class") is not None:
        raise ManifestError(
            f"runtime profile {profile!r} canonical collision geom must not use a default class"
        )
    transform_attributes = sorted(_GEOM_TRANSFORM_ATTRIBUTES.intersection(collision_geom.attrib))
    if transform_attributes:
        raise ManifestError(
            f"runtime profile {profile!r} canonical collision geom must be axis-aligned; "
            f"unsupported transform attributes: {transform_attributes}"
        )
    inherited_transforms = [
        attribute
        for default_geom in root.findall(".//default/geom")
        for attribute in ("pos", *_GEOM_TRANSFORM_ATTRIBUTES)
        if attribute in default_geom.attrib
    ]
    if inherited_transforms:
        raise ManifestError(
            f"runtime profile {profile!r} default geom may transform the canonical collision "
            f"geom: {sorted(set(inherited_transforms))}"
        )

    hfield = hfields[0]
    rows, columns = _runtime_hfield_shape(
        hfield,
        profile=profile,
        collision_contract=collision_contract,
    )
    if rows != collision_contract.rows or columns != collision_contract.columns:
        raise ManifestError(
            f"runtime profile {profile!r} collision hfield shape "
            f"({rows}, {columns}) disagrees with manifest "
            f"({collision_contract.rows}, {collision_contract.columns})"
        )

    size = _xml_number_list_attribute(
        hfield,
        "size",
        label=f"runtime profile {profile!r} collision hfield size",
        length=4,
    )
    expected_size = (
        *collision_contract.half_size_xy_m,
        collision_contract.maximum_height_m,
        collision_contract.base_depth_m,
    )
    _require_xml_numbers_close(
        size,
        expected_size,
        label=f"runtime profile {profile!r} collision hfield size",
    )

    geom_center = _xml_number_list_attribute(
        collision_geom,
        "pos",
        label=f"runtime profile {profile!r} canonical collision geom pos",
        length=3,
    )
    _require_xml_numbers_close(
        geom_center,
        collision_contract.geom_center_m,
        label=f"runtime profile {profile!r} canonical collision geom pos",
    )


def _runtime_hfield_shape(
    hfield: ET.Element,
    *,
    profile: str,
    collision_contract: _RuntimeCollisionContract,
) -> tuple[int, int]:
    raw_rows = hfield.get("nrow")
    raw_columns = hfield.get("ncol")
    image_file = hfield.get("file")
    if image_file is not None:
        if Path(image_file).as_posix() != collision_contract.image_relative:
            raise ManifestError(
                f"runtime profile {profile!r} collision hfield must reference the "
                "declared collision image"
            )
        columns, rows = _png_ihdr_dimensions(
            collision_contract.image_path,
            label=f"runtime profile {profile!r} collision hfield image",
            expected_rows=collision_contract.rows,
            expected_columns=collision_contract.columns,
        )
        # MuJoCo derives file-backed hfield dimensions from the image and ignores
        # nrow/ncol. If those redundant attributes are present, require them to
        # agree as well so the XML never advertises a different grid shape.
        if raw_rows is not None or raw_columns is not None:
            if raw_rows is None or raw_columns is None:
                raise ManifestError(
                    f"runtime profile {profile!r} collision hfield must declare both nrow and ncol"
                )
            declared_rows = _xml_integer_attribute(
                hfield,
                "nrow",
                label=f"runtime profile {profile!r} collision hfield nrow",
                minimum=2,
            )
            declared_columns = _xml_integer_attribute(
                hfield,
                "ncol",
                label=f"runtime profile {profile!r} collision hfield ncol",
                minimum=2,
            )
            if (declared_rows, declared_columns) != (rows, columns):
                raise ManifestError(
                    f"runtime profile {profile!r} collision hfield XML shape "
                    "disagrees with its PNG dimensions"
                )
        return rows, columns
    if raw_rows is None and raw_columns is None:
        raise ManifestError(
            f"runtime profile {profile!r} collision hfield has neither file nor shape"
        )
    if raw_rows is None or raw_columns is None:
        raise ManifestError(
            f"runtime profile {profile!r} collision hfield must declare both nrow and ncol"
        )
    return (
        _xml_integer_attribute(
            hfield,
            "nrow",
            label=f"runtime profile {profile!r} collision hfield nrow",
            minimum=2,
        ),
        _xml_integer_attribute(
            hfield,
            "ncol",
            label=f"runtime profile {profile!r} collision hfield ncol",
            minimum=2,
        ),
    )


def _png_ihdr_dimensions(
    path: Path,
    *,
    label: str,
    expected_rows: int,
    expected_columns: int,
) -> tuple[int, int]:
    """Validate a generated grayscale16 PNG and return its effective dimensions."""

    _decoded, width, height = _png_scanline_payload(
        path,
        label=label,
        expected_rows=expected_rows,
        expected_columns=expected_columns,
    )
    return width, height


def _png_scanline_payload(
    path: Path,
    *,
    label: str,
    expected_rows: int,
    expected_columns: int,
) -> tuple[bytes, int, int]:
    """Return the still-filtered PNG scanlines plus the validated dimensions."""

    try:
        encoded_size = path.stat().st_size
        if encoded_size > MAX_HEIGHTFIELD_PNG_BYTES:
            raise ManifestError(
                f"{label} exceeds the encoded PNG limit: "
                f"{encoded_size} > {MAX_HEIGHTFIELD_PNG_BYTES} bytes"
            )
        payload = path.read_bytes()
    except OSError as exc:
        raise ManifestError(f"{label} could not be read: {exc}") from exc
    if len(payload) < 8 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ManifestError(f"{label} lacks a valid PNG signature")

    offset = 8
    width = 0
    height = 0
    idat_parts: list[bytes] = []
    saw_ihdr = False
    saw_idat = False
    idat_closed = False
    saw_iend = False
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise ManifestError(f"{label} contains a truncated PNG chunk")
        length = int.from_bytes(payload[offset : offset + 4], "big")
        chunk_type = payload[offset + 4 : offset + 8]
        data_start = offset + 8
        data_stop = data_start + length
        crc_stop = data_stop + 4
        if crc_stop > len(payload):
            raise ManifestError(f"{label} contains a truncated PNG chunk")
        chunk_data = payload[data_start:data_stop]
        expected_crc = int.from_bytes(payload[data_stop:crc_stop], "big")
        actual_crc = zlib.crc32(chunk_data, zlib.crc32(chunk_type)) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ManifestError(f"{label} contains a PNG chunk with an invalid CRC")
        offset = crc_stop

        if not saw_ihdr:
            if chunk_type != b"IHDR" or length != 13:
                raise ManifestError(f"{label} lacks a valid first PNG IHDR chunk")
            width = int.from_bytes(chunk_data[0:4], "big")
            height = int.from_bytes(chunk_data[4:8], "big")
            if (height, width) != (expected_rows, expected_columns):
                raise ManifestError(
                    f"{label} PNG dimensions ({height}, {width}) disagree with manifest "
                    f"({expected_rows}, {expected_columns})"
                )
            bit_depth, colour_type, compression, filtering, interlace = chunk_data[8:13]
            if (bit_depth, colour_type, compression, filtering, interlace) != (16, 0, 0, 0, 0):
                raise ManifestError(f"{label} must be a non-interlaced 16-bit grayscale PNG")
            saw_ihdr = True
            continue
        if chunk_type == b"IHDR":
            raise ManifestError(f"{label} contains more than one PNG IHDR chunk")
        if chunk_type == b"IDAT":
            if idat_closed:
                raise ManifestError(f"{label} contains non-consecutive PNG IDAT chunks")
            saw_idat = True
            idat_parts.append(chunk_data)
            continue
        if saw_idat:
            idat_closed = True
        if chunk_type == b"IEND":
            if length != 0 or not saw_idat:
                raise ManifestError(f"{label} contains an invalid PNG IEND chunk")
            saw_iend = True
            if offset != len(payload):
                raise ManifestError(f"{label} contains trailing bytes after PNG IEND")
            break
        if chunk_type and chunk_type[0] & 0x20 == 0:
            raise ManifestError(f"{label} contains unsupported critical PNG chunk {chunk_type!r}")

    if not saw_iend:
        raise ManifestError(f"{label} lacks a complete PNG IEND chunk")
    if width < 2 or height < 2:
        raise ManifestError(f"{label} dimensions must both be >= 2")

    expected_scanline_bytes = 1 + 2 * width
    expected_payload_bytes = expected_scanline_bytes * height
    try:
        decompressor = zlib.decompressobj()
        decoded = decompressor.decompress(b"".join(idat_parts), expected_payload_bytes + 1)
    except zlib.error as exc:
        raise ManifestError(f"{label} contains invalid PNG IDAT zlib data: {exc}") from exc
    if (
        len(decoded) != expected_payload_bytes
        or not decompressor.eof
        or decompressor.unconsumed_tail
        or decompressor.unused_data
    ):
        raise ManifestError(f"{label} PNG IDAT does not decode to the declared grayscale image")
    if any(decoded[row * expected_scanline_bytes] > 4 for row in range(height)):
        raise ManifestError(f"{label} contains an invalid PNG scanline filter")
    return decoded, width, height


def _paeth_predictor(left: int, above: int, upper_left: int) -> int:
    """Return the PNG Paeth predictor for one byte pair."""

    estimate = left + above - upper_left
    distance_left = abs(estimate - left)
    distance_above = abs(estimate - above)
    distance_upper_left = abs(estimate - upper_left)
    if distance_left <= distance_above and distance_left <= distance_upper_left:
        return left
    if distance_above <= distance_upper_left:
        return above
    return upper_left


def _unfilter_png_scanlines(
    decoded: bytes,
    *,
    width: int,
    height: int,
    label: str,
) -> bytes:
    """Undo the per-scanline PNG filters of a validated grayscale16 image.

    A producer picks a filter per row, so the sample comparison cannot assume
    filter 0.  Filtering is byte-oriented with a two-byte pixel stride.
    """

    stride = PNG_GRAYSCALE16_BYTES_PER_PIXEL * width
    line_bytes = 1 + stride
    result = bytearray(stride * height)
    previous = bytearray(stride)
    for row in range(height):
        start = row * line_bytes
        filter_type = decoded[start]
        raw = decoded[start + 1 : start + 1 + stride]
        current = bytearray(stride)
        if filter_type == PNG_FILTER_NONE:
            current[:] = raw
        elif filter_type == PNG_FILTER_SUB:
            for index in range(stride):
                left = (
                    current[index - PNG_GRAYSCALE16_BYTES_PER_PIXEL]
                    if index >= PNG_GRAYSCALE16_BYTES_PER_PIXEL
                    else 0
                )
                current[index] = (raw[index] + left) & 0xFF
        elif filter_type == PNG_FILTER_UP:
            for index in range(stride):
                current[index] = (raw[index] + previous[index]) & 0xFF
        elif filter_type == PNG_FILTER_AVERAGE:
            for index in range(stride):
                left = (
                    current[index - PNG_GRAYSCALE16_BYTES_PER_PIXEL]
                    if index >= PNG_GRAYSCALE16_BYTES_PER_PIXEL
                    else 0
                )
                current[index] = (raw[index] + ((left + previous[index]) >> 1)) & 0xFF
        elif filter_type == PNG_FILTER_PAETH:
            for index in range(stride):
                if index >= PNG_GRAYSCALE16_BYTES_PER_PIXEL:
                    left = current[index - PNG_GRAYSCALE16_BYTES_PER_PIXEL]
                    upper_left = previous[index - PNG_GRAYSCALE16_BYTES_PER_PIXEL]
                else:
                    left = 0
                    upper_left = 0
                current[index] = (
                    raw[index] + _paeth_predictor(left, previous[index], upper_left)
                ) & 0xFF
        else:
            raise ManifestError(f"{label} contains an invalid PNG scanline filter")
        result[row * stride : (row + 1) * stride] = current
        previous = current
    return bytes(result)


def _verify_collision_bootstrap(
    collision: Mapping[str, Any],
    files: Mapping[str, Path],
    *,
    rows: int,
    columns: int,
    maximum_height_m: float,
) -> None:
    """Confirm the bootstrap PNG encodes the same surface as the float samples.

    MuJoCo derives a file-backed heightfield's dimensions from the image, so the
    PNG must exist.  Its samples are only a bootstrap: the loader replaces them
    with the verified NPZ floats before the first step.  A consumer that opens
    the entrypoint XML directly with MuJoCo never reaches that replacement, so a
    PNG that disagrees with the samples would hand that consumer a different
    field.  Bind the two here rather than trusting that one producer wrote both
    from the same array.
    """

    image_relative = str(collision["image_file"])
    samples_relative = str(collision["samples_file"])
    label = f"collision.{image_relative}"
    try:
        with np.load(files[samples_relative], allow_pickle=False) as samples:
            height = np.asarray(samples["height_m"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        raise ManifestError(f"collision.samples_file could not be read: {exc}") from exc
    if height.shape != (rows, columns):
        raise ManifestError("collision.samples_file shape disagrees with the manifest")
    if not np.isfinite(height).all():
        raise ManifestError("collision.samples_file contains non-finite samples")

    decoded, width, decoded_height = _png_scanline_payload(
        files[image_relative],
        label=label,
        expected_rows=rows,
        expected_columns=columns,
    )
    unfiltered = _unfilter_png_scanlines(
        decoded,
        width=width,
        height=decoded_height,
        label=label,
    )
    expected = np.rint(
        np.clip(height / maximum_height_m, 0.0, 1.0) * HEIGHTFIELD_PNG_SAMPLE_MAXIMUM
    ).astype(np.uint16)
    # MuJoCo reads the first PNG row as positive local Y while the samples are
    # indexed from y_min, so the producer stores the image flipped.
    observed = np.frombuffer(unfiltered, dtype=">u2").reshape(rows, columns)[::-1]
    deviation = np.abs(observed.astype(np.int32) - expected.astype(np.int32))
    worst = int(np.max(deviation))
    if worst > HEIGHTFIELD_PNG_LSB_TOLERANCE:
        row, column = (
            int(value) for value in np.unravel_index(int(np.argmax(deviation)), deviation.shape)
        )
        raise AssetIntegrityError(
            "collision bootstrap PNG disagrees with the verified float samples: "
            f"row={row}, column={column}, deviation={worst} LSB, "
            f"tolerance={HEIGHTFIELD_PNG_LSB_TOLERANCE} LSB"
        )


def _xml_integer_attribute(
    element: ET.Element,
    attribute: str,
    *,
    label: str,
    minimum: int,
) -> int:
    raw = element.get(attribute)
    if raw is None:
        raise ManifestError(f"{label} is missing")
    try:
        result = int(raw.strip(), 10)
    except ValueError as exc:
        raise ManifestError(f"{label} must be an integer") from exc
    if result < minimum:
        raise ManifestError(f"{label} must be an integer >= {minimum}")
    return result


def _xml_number_list_attribute(
    element: ET.Element,
    attribute: str,
    *,
    label: str,
    length: int,
) -> tuple[float, ...]:
    raw = element.get(attribute)
    if raw is None:
        raise ManifestError(f"{label} is missing")
    parts = raw.split()
    if len(parts) != length:
        raise ManifestError(f"{label} must contain {length} finite numbers")
    result: list[float] = []
    for part in parts:
        try:
            value = float(part)
        except ValueError as exc:
            raise ManifestError(f"{label} must contain {length} finite numbers") from exc
        if not math.isfinite(value):
            raise ManifestError(f"{label} must contain {length} finite numbers")
        result.append(value)
    return tuple(result)


def _require_xml_numbers_close(
    actual: tuple[float, ...],
    expected: tuple[float, ...],
    *,
    label: str,
) -> None:
    if len(actual) != len(expected) or any(
        not math.isclose(
            observed,
            wanted,
            rel_tol=XML_FLOAT_REL_TOLERANCE,
            abs_tol=XML_FLOAT_ABS_TOLERANCE,
        )
        # Runtime packs are also consumed by the frozen Fudan Python 3.9
        # environment. Length equality is checked above, so plain zip retains
        # the strict pairing contract without requiring Python 3.10's keyword.
        for observed, wanted in zip(actual, expected)
    ):
        raise ManifestError(
            f"{label} disagrees with manifest: actual={actual}, expected={expected}"
        )


def _require_cross_file(
    files: Mapping[str, Path],
    manifest: Mapping[str, Any],
    relative: str,
    digest: str,
    *,
    label: str,
) -> None:
    if relative not in files:
        raise ManifestError(f"{label} is not declared in contents.files")
    records = manifest["contents"]["files"]
    matches = [record for record in records if record.get("file") == relative]
    if len(matches) != 1 or matches[0].get("sha256") != digest:
        raise ManifestError(f"{label} SHA-256 disagrees with contents.files")


def _contained_regular_file(root: Path, relative: str, *, label: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute():
        raise AssetIntegrityError(f"{label} must use a relative path")
    unresolved = root / raw
    if unresolved.is_symlink():
        raise AssetIntegrityError(f"{label} must not be a symlink: {relative}")
    candidate = unresolved.resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise AssetIntegrityError(f"{label} is missing or escapes the pack root: {relative}")
    return candidate
