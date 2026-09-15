"""Fail-closed loading for a relocatable RMUC 2026 runtime asset pack."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET

from .download import OFFICIAL_STEP_SHA256, OFFICIAL_STEP_SIZE
from .errors import AssetIntegrityError, ManifestError


RUNTIME_ARTIFACT_TYPE = "rmuc2026_mujoco_runtime_asset_pack"
SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_VALIDATION_STATUS = "DRAFT_BLOCKED"
DEFAULT_RUNTIME_PROFILE = "full"
RUNTIME_PROFILE_NAMES = ("full", "collision_only")


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

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "PASS",
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
class FieldAsset:
    """A fully validated runtime pack and its parsed manifest."""

    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    verified_files: tuple[Path, ...]
    _files_by_relative: Mapping[str, Path]

    @classmethod
    def open(cls, root: str | Path, *, verify: bool = True) -> FieldAsset:
        """Open the sole public runtime-pack schema and optionally verify all hashes."""

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
        _validate_cross_references(manifest, file_paths)
        return cls(
            root=pack_root,
            manifest=manifest,
            manifest_sha256=sha256_file(manifest_path),
            verified_files=tuple(file_paths.values()),
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
        )


def verify_asset(root: str | Path) -> ValidationReport:
    """Fully verify a runtime asset pack and return its compact report."""

    return FieldAsset.open(root, verify=True).report()


def _validate_identity(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
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

    _validate_runtime_profiles(manifest, files, visual_mesh_count=len(visual))

    collision = _mapping(manifest.get("collision"), label="collision")
    if collision.get("kind") != "conservative_top_surface_heightfield":
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
    _number_list(
        collision.get("half_size_xy_m"),
        label="collision.half_size_xy_m",
        length=2,
        positive=True,
    )
    _number_list(
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
    _finite_number(collision.get("base_depth_m"), label="collision.base_depth_m", positive=True)
    if collision.get("png_rows") != "flipped_y_for_mujoco_hfield_loader":
        raise ManifestError("unsupported collision PNG row orientation")

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


def _validate_runtime_profiles(
    manifest: Mapping[str, Any],
    files: Mapping[str, Path],
    *,
    visual_mesh_count: int,
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
        )

    if manifest["contents"]["entrypoint"] != profiles["full"]["entrypoint"]:
        raise ManifestError("contents.entrypoint must remain the full runtime profile")


def _validate_runtime_profile_xml(
    path: Path,
    *,
    profile: str,
    expected_visual_mesh_count: int,
) -> None:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ManifestError(f"runtime profile {profile!r} MJCF is invalid XML: {exc}") from exc
    mesh_assets = root.findall("./asset/mesh")
    mesh_geoms = [geom for geom in root.iter("geom") if geom.get("mesh")]
    if len(mesh_assets) != expected_visual_mesh_count:
        raise ManifestError(
            f"runtime profile {profile!r} declares {len(mesh_assets)} mesh assets; "
            f"expected {expected_visual_mesh_count}"
        )
    if len(mesh_geoms) != expected_visual_mesh_count:
        raise ManifestError(
            f"runtime profile {profile!r} declares {len(mesh_geoms)} mesh geoms; "
            f"expected {expected_visual_mesh_count}"
        )
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
    if len(hfields) != 1 or len(collision_geoms) != 1:
        raise ManifestError(f"runtime profile {profile!r} lacks the canonical collision hfield")


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
