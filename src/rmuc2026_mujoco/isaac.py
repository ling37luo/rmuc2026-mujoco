"""Framework-neutral input for an optional Isaac/Isaac Lab adapter.

This module deliberately imports no Isaac package.  It exposes the verified
heightfield and identity metadata an external Isaac consumer needs to build a
high-parallel environment without changing the field geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from .errors import AssetIntegrityError, ManifestError
from .manifest import FieldAsset
from .manifest import sha256_file
from .query import field_bounds, load_heightfield
from .scenarios import scenario_descriptor
from .slope_catalog import slope_catalog
from .turning import screen_turn_spawns, turn_phase_registry
from .training_region import TrainingRegion


_OFFLINE_REGION_TYPE = "rmuc2026_isaac_heightfield_input"
_OFFLINE_REGION_SCHEMA = 3
_SUPPORTED_OFFLINE_REGION_SCHEMAS = (1, 2, 3)
_OFFLINE_GRID_FILE = "heightfield.npz"


@dataclass(frozen=True)
class IsaacHeightfieldInput:
    height_m: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    bounds_xy_m: tuple[tuple[float, float], tuple[float, float]]
    scenario: dict[str, Any]
    spawn_points: tuple[dict[str, Any], ...] = ()
    static_boxes: tuple[dict[str, Any], ...] = ()
    source: str = "verified_runtime_pack_heightfield"
    heightfield_contact: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.height_m.shape),
            "bounds_xy_m": [list(row) for row in self.bounds_xy_m],
            "scenario": self.scenario,
            "spawn_points": [dict(point) for point in self.spawn_points],
            "static_boxes": [dict(box) for box in self.static_boxes],
            "heightfield_contact": (
                None if self.heightfield_contact is None else dict(self.heightfield_contact)
            ),
            "source": self.source,
        }


def _cropped_heightfield_contact(region: TrainingRegion) -> dict[str, Any]:
    """Read the authored MuJoCo contact contract for the cropped heightfield."""

    try:
        root = ET.parse(region.root / "field.xml").getroot()
    except (OSError, ET.ParseError) as exc:
        raise ManifestError(f"training-region field XML is invalid: {exc}") from exc
    geom = root.find("./worldbody/geom[@name='rmuc2026_field_collision']")
    if geom is None or geom.get("type") != "hfield" or geom.get("hfield") != "rmuc2026_collision":
        raise ManifestError("training-region field XML has no expected heightfield geom")
    try:
        friction = [float(value) for value in geom.attrib["friction"].split()]
        solref = [float(value) for value in geom.attrib["solref"].split()]
        contype = int(geom.attrib["contype"])
        conaffinity = int(geom.attrib["conaffinity"])
        if (
            len(friction) != 3
            or len(solref) != 2
            or not np.isfinite(friction).all()
            or not np.isfinite(solref).all()
            or min(friction) < 0
            or contype < 0
            or conaffinity < 0
        ):
            raise ValueError("invalid heightfield contact parameters")
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"training-region heightfield contact is invalid: {exc}") from exc
    return {
        "name": "rmuc2026_field_collision",
        "contype": contype,
        "conaffinity": conaffinity,
        "friction": friction,
        "solref": solref,
    }


def _validate_heightfield_contact(contact: Any) -> dict[str, Any]:
    if not isinstance(contact, dict) or set(contact) != {
        "name",
        "contype",
        "conaffinity",
        "friction",
        "solref",
    }:
        raise ManifestError("Isaac region heightfield contact is invalid")
    try:
        friction = np.asarray(contact["friction"], dtype=np.float64)
        solref = np.asarray(contact["solref"], dtype=np.float64)
        if (
            contact["name"] != "rmuc2026_field_collision"
            or type(contact["contype"]) is not int
            or type(contact["conaffinity"]) is not int
            or not isinstance(contact["friction"], list)
            or not isinstance(contact["solref"], list)
            or any(type(value) not in (int, float) for value in contact["friction"])
            or any(type(value) not in (int, float) for value in contact["solref"])
            or contact["contype"] < 0
            or contact["conaffinity"] < 0
            or friction.shape != (3,)
            or solref.shape != (2,)
            or not np.isfinite(friction).all()
            or not np.isfinite(solref).all()
            or np.any(friction < 0)
        ):
            raise ValueError("invalid heightfield contact parameters")
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"Isaac region heightfield contact is invalid: {exc}") from exc
    return contact


def _cropped_static_boxes(region: TrainingRegion) -> tuple[dict[str, Any], ...]:
    """Preserve the source fence within the local grid, without cross-env overlap."""

    try:
        root = ET.parse(region.root / "field.xml").getroot()
    except (OSError, ET.ParseError) as exc:
        raise ManifestError(f"training-region field XML is invalid: {exc}") from exc
    (xmin, ymin), (xmax, ymax) = region.manifest["grid"]["bounds_xy_m"]
    boxes: list[dict[str, Any]] = []
    for geom in root.findall("./worldbody/geom"):
        name = geom.get("name", "")
        if not name.startswith("rmuc2026_perimeter_"):
            continue
        if geom.get("type") != "box" or any(
            key in geom.attrib for key in ("quat", "euler", "axisangle", "xyaxes", "zaxis")
        ):
            raise ManifestError(f"perimeter geom {name} is not an axis-aligned box")
        try:

            def vector(key: str, length: int) -> list[float]:
                values = [float(value) for value in geom.attrib[key].split()]
                if len(values) != length or not np.isfinite(values).all():
                    raise ValueError(f"invalid {key}")
                return values

            pos = vector("pos", 3)
            size = vector("size", 3)
            friction = vector("friction", 3)
            solref = vector("solref", 2)
            contype = int(geom.attrib["contype"])
            conaffinity = int(geom.attrib["conaffinity"])
            if min(size) <= 0 or min(friction) < 0 or contype < 0 or conaffinity < 0:
                raise ValueError("invalid box size or contact parameters")
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(
                f"perimeter geom {name} has invalid contact geometry: {exc}"
            ) from exc
        x0 = max(float(xmin), pos[0] - size[0])
        x1 = min(float(xmax), pos[0] + size[0])
        y0 = max(float(ymin), pos[1] - size[1])
        y1 = min(float(ymax), pos[1] + size[1])
        if x0 >= x1 or y0 >= y1:
            continue
        boxes.append(
            {
                "name": name,
                "pos_m": [(x0 + x1) / 2, (y0 + y1) / 2, pos[2]],
                "size_m": [(x1 - x0) / 2, (y1 - y0) / 2, size[2]],
                "contype": contype,
                "conaffinity": conaffinity,
                "friction": friction,
                "solref": solref,
            }
        )
    return tuple(boxes)


def _validate_static_boxes(boxes: Any, bounds: np.ndarray) -> tuple[dict[str, Any], ...]:
    if not isinstance(boxes, list):
        raise ManifestError("Isaac region static box list is invalid")
    names: set[str] = set()
    for box in boxes:
        if not isinstance(box, dict) or set(box) != {
            "name",
            "pos_m",
            "size_m",
            "contype",
            "conaffinity",
            "friction",
            "solref",
        }:
            raise ManifestError("Isaac region static box is invalid")
        try:
            pos = np.asarray(box["pos_m"], dtype=np.float64)
            size = np.asarray(box["size_m"], dtype=np.float64)
            friction = np.asarray(box["friction"], dtype=np.float64)
            solref = np.asarray(box["solref"], dtype=np.float64)
            if (
                not isinstance(box["name"], str)
                or not box["name"].startswith("rmuc2026_perimeter_")
                or box["name"] in names
                or pos.shape != (3,)
                or size.shape != (3,)
                or friction.shape != (3,)
                or solref.shape != (2,)
                or not np.isfinite(pos).all()
                or not np.isfinite(size).all()
                or not np.isfinite(friction).all()
                or not np.isfinite(solref).all()
                or np.any(size <= 0)
                or np.any(friction < 0)
                or not isinstance(box["contype"], int)
                or not isinstance(box["conaffinity"], int)
                or box["contype"] < 0
                or box["conaffinity"] < 0
                or np.any(pos[:2] - size[:2] < bounds[0] - 1e-9)
                or np.any(pos[:2] + size[:2] > bounds[1] + 1e-9)
            ):
                raise ValueError("invalid geometry or contact")
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"Isaac region static box is invalid: {exc}") from exc
        names.add(box["name"])
    return tuple(boxes)


def load_isaac_heightfield(
    asset: FieldAsset | str | Path,
    *,
    scenario: str = "turn_basic",
    profile: str = "collision_only",
) -> IsaacHeightfieldInput:
    """Return data for an Isaac adapter; no Isaac dependency is required."""

    field = asset if isinstance(asset, FieldAsset) else FieldAsset.open(asset, verify=True)
    descriptor = scenario_descriptor(field, scenario, profile=profile)
    samples = load_heightfield(field)
    spawn_points: tuple[dict[str, Any], ...] = ()
    if scenario == "turn_basic":
        spawn_points = tuple(spawn.to_dict() for spawn in screen_turn_spawns(field))
        descriptor["turn_phase_registry"] = turn_phase_registry()
        descriptor["turn_spawn_points"] = [dict(point) for point in spawn_points]
    elif scenario == "slope_basic":
        descriptor["slope_catalog"] = slope_catalog(field)
        spawn_points = tuple(
            {
                "route_id": patch["patch_id"],
                "direction": direction,
                "position_xyz_m": patch["route"][point],
                "heading_yaw_rad": patch["route"]["heading_yaw_rad"] + offset,
            }
            for patch in descriptor["slope_catalog"]["patches"]
            if patch["route"] is not None
            for direction, point, offset in (
                ("uphill", "low_xyz_m", 0.0),
                ("downhill", "high_xyz_m", float(np.pi)),
            )
        )
    elif scenario in {"fly_ramp_north", "fly_ramp_south"}:
        from .fly_routes import fly_route_descriptor

        route = fly_route_descriptor(field, scenario)
        descriptor["fly_ramp_route"] = route
        spawn_points = (
            {
                "route_id": route["route_id"],
                "position_xyz_m": list(route["spawn"]["xyz_m"]),
                "heading_yaw_rad": float(route["spawn"]["heading_yaw_rad"]),
                "position_reference": "terrain_surface",
                "topology_verified": False,
            },
        )
    return IsaacHeightfieldInput(
        height_m=np.asarray(samples.height_m, dtype=np.float32),
        x_m=np.asarray(samples.x_m, dtype=np.float32),
        y_m=np.asarray(samples.y_m, dtype=np.float32),
        bounds_xy_m=field_bounds(field),
        scenario=descriptor,
        spawn_points=spawn_points,
    )


def load_isaac_training_region(region: TrainingRegion | str | Path) -> IsaacHeightfieldInput:
    """Provide the local grid, intersecting fence boxes and identity.

    IsaacLab's Newton XPBD backend is currently the contact-tested consumer of
    this 1 cm grid. This function only transfers data; it does not choose a
    physics backend or claim PhysX/MJWarp contact equivalence.
    """

    selected = TrainingRegion.open(region.root if isinstance(region, TrainingRegion) else region)
    samples = selected.heightfield()
    manifest = selected.manifest
    route = dict(manifest["route"])
    spawn = {
        "route_id": route["route_id"],
        "position_xyz_m": list(route["spawn"]["xyz_m"]),
        "heading_yaw_rad": float(route["spawn"]["heading_yaw_rad"]),
        "position_reference": "terrain_surface",
        "topology_verified": False,
    }
    descriptor = {
        "scenario_id": manifest["scenario_id"],
        "profile": manifest["profile"],
        "profile_hash": manifest["profile_hash"],
        "region_manifest_sha256": selected.manifest_sha256,
        "source_manifest_sha256": manifest["source"]["source_manifest_sha256"],
        "source_profile_hash": manifest["source"]["source_profile_hash"],
        "source_collision_samples_sha256": manifest["source"]["source_collision_samples_sha256"],
        "local_samples_sha256": manifest["files"]["collision/heightfield.npz"]["sha256"],
        "source_grid_slice_yx": manifest["source"]["source_grid_slice_yx"],
        "route": route,
        "validation_status": manifest["validation_status"],
        "heightfield_contact_status": "source_mjcf_recorded",
    }
    return IsaacHeightfieldInput(
        height_m=samples.height_m.astype(np.float32),
        x_m=samples.x_m.astype(np.float32),
        y_m=samples.y_m.astype(np.float32),
        bounds_xy_m=samples.bounds_xy_m,
        scenario=descriptor,
        spawn_points=(spawn,),
        static_boxes=_cropped_static_boxes(selected),
        heightfield_contact=_cropped_heightfield_contact(selected),
        source="verified_cropped_training_region",
    )


def export_isaac_training_region(region: TrainingRegion | str | Path, output: str | Path) -> dict:
    """Export exact region samples and source contact metadata for an offline consumer.

    Unlike the in-memory Isaac adapter's float32 arrays, this copies the
    source region's exact float64 NPZ. The files are backend-neutral and do
    not create or validate a PhysX collider.
    """

    selected = TrainingRegion.open(region.root if isinstance(region, TrainingRegion) else region)
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise ValueError(f"output already exists: {target}")
    source = selected.manifest["source"]
    target.mkdir(parents=True)
    shutil.copyfile(selected.root / "collision/heightfield.npz", target / _OFFLINE_GRID_FILE)
    descriptor = {
        "artifact_type": _OFFLINE_REGION_TYPE,
        "schema_version": _OFFLINE_REGION_SCHEMA,
        "scope": "offline_heightfield_and_fence_boxes_no_physx_contact_validation",
        "scenario_id": selected.manifest["scenario_id"],
        "validation_status": selected.manifest["validation_status"],
        "grid_file": _OFFLINE_GRID_FILE,
        "grid_file_sha256": sha256_file(target / _OFFLINE_GRID_FILE),
        "grid": {
            **selected.manifest["grid"],
            "height_array_order": "height_m[y_index, x_index]",
            "world_frame": "xyz_m_z_up",
            "height_values": "absolute_world_z_m_no_extra_scale_or_offset",
            "sample_dtype": "float64",
        },
        "identity": {
            "region_manifest_sha256": selected.manifest_sha256,
            "region_profile_hash": selected.manifest["profile_hash"],
            "source_manifest_sha256": source["source_manifest_sha256"],
            "source_profile_hash": source["source_profile_hash"],
            "source_collision_samples_sha256": source["source_collision_samples_sha256"],
            "source_grid_slice_yx": source["source_grid_slice_yx"],
            "source_field_xml_sha256": selected.manifest["files"]["field.xml"]["sha256"],
        },
        "collision": {
            "heightfield": _cropped_heightfield_contact(selected),
            "static_boxes": list(_cropped_static_boxes(selected)),
        },
        "route": selected.manifest["route"],
    }
    (target / "descriptor.json").write_text(
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return descriptor


def load_isaac_training_region_export(
    output: str | Path, *, source_region: TrainingRegion | str | Path | None = None
) -> IsaacHeightfieldInput:
    """Check an offline export before constructing an external terrain collider.

    Pass ``source_region`` to verify the data, contact and identity against the
    source; without it, provenance in the descriptor is self-reported. Legacy
    schemas 1 and 2 do not provide a heightfield contact contract.
    """

    root = Path(output).expanduser().resolve()
    descriptor_path = root / "descriptor.json"
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Isaac region descriptor is invalid: {exc}") from exc
    if (
        descriptor.get("artifact_type") != _OFFLINE_REGION_TYPE
        or descriptor.get("schema_version") not in _SUPPORTED_OFFLINE_REGION_SCHEMAS
        or descriptor.get("grid_file") != _OFFLINE_GRID_FILE
    ):
        raise ManifestError("unsupported Isaac region descriptor")
    schema = descriptor["schema_version"]
    grid_path = root / _OFFLINE_GRID_FILE
    if grid_path.is_symlink() or not grid_path.is_file():
        raise AssetIntegrityError("Isaac region grid file is missing")
    if sha256_file(grid_path) != descriptor.get("grid_file_sha256"):
        raise AssetIntegrityError("Isaac region grid hash mismatch")
    with np.load(grid_path, allow_pickle=False) as payload:
        if set(payload.files) != {"x_m", "y_m", "height_m"}:
            raise ManifestError("Isaac region grid arrays are incomplete")
        x = np.asarray(payload["x_m"])
        y = np.asarray(payload["y_m"])
        height = np.asarray(payload["height_m"])
    record = descriptor["grid"]
    bounds = np.asarray(record["bounds_xy_m"], dtype=np.float64)
    resolution = np.asarray(record["resolution_xy_m"], dtype=np.float64)
    if (
        x.dtype != np.float64
        or y.dtype != np.float64
        or height.dtype != np.float64
        or height.shape != (record["rows_y"], record["columns_x"])
        or x.shape != (height.shape[1],)
        or y.shape != (height.shape[0],)
        or bounds.shape != (2, 2)
        or resolution.shape != (2,)
        or not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(height).all())
        or not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0))
        or not np.array_equal(bounds, [[x[0], y[0]], [x[-1], y[-1]]])
        or not np.allclose(np.diff(x), resolution[0], atol=1e-8, rtol=0)
        or not np.allclose(np.diff(y), resolution[1], atol=1e-8, rtol=0)
        or not np.allclose(
            [height.min(), height.max()],
            [record["minimum_height_m"], record["maximum_height_m"]],
            atol=1e-10,
            rtol=0,
        )
        or record.get("height_array_order") != "height_m[y_index, x_index]"
        or record.get("world_frame") != "xyz_m_z_up"
        or record.get("height_values") != "absolute_world_z_m_no_extra_scale_or_offset"
        or record.get("sample_dtype") != "float64"
        or record.get("orientation") != "world_y_ascending_world_x_ascending"
        or record.get("interpolation") != "mujoco_hfield_triangle"
    ):
        raise ManifestError("Isaac region grid geometry disagrees with descriptor")
    boxes: tuple[dict[str, Any], ...] = ()
    heightfield_contact: dict[str, Any] | None = None
    if schema >= 2:
        collision = descriptor.get("collision")
        expected_keys = {"static_boxes"} if schema == 2 else {"heightfield", "static_boxes"}
        if not isinstance(collision, dict) or set(collision) != expected_keys:
            raise ManifestError("Isaac region collision descriptor is invalid")
        boxes = _validate_static_boxes(collision["static_boxes"], bounds)
        if schema == 3:
            heightfield_contact = _validate_heightfield_contact(collision["heightfield"])
    if source_region is not None:
        selected = TrainingRegion.open(
            source_region.root if isinstance(source_region, TrainingRegion) else source_region
        )
        source = selected.manifest["source"]
        expected_identity = {
            "region_manifest_sha256": selected.manifest_sha256,
            "region_profile_hash": selected.manifest["profile_hash"],
            "source_manifest_sha256": source["source_manifest_sha256"],
            "source_profile_hash": source["source_profile_hash"],
            "source_collision_samples_sha256": source["source_collision_samples_sha256"],
            "source_grid_slice_yx": source["source_grid_slice_yx"],
        }
        if schema >= 2:
            expected_identity["source_field_xml_sha256"] = selected.manifest["files"]["field.xml"][
                "sha256"
            ]
        original = selected.heightfield()
        if (
            descriptor.get("identity") != expected_identity
            or not np.array_equal(x, original.x_m)
            or not np.array_equal(y, original.y_m)
            or not np.array_equal(height, original.height_m)
            or (schema >= 2 and boxes != _cropped_static_boxes(selected))
            or (schema == 3 and heightfield_contact != _cropped_heightfield_contact(selected))
        ):
            raise ManifestError("Isaac region export differs from its verified source region")
    return IsaacHeightfieldInput(
        height_m=height,
        x_m=x,
        y_m=y,
        bounds_xy_m=((float(x[0]), float(y[0])), (float(x[-1]), float(y[-1]))),
        scenario={
            "scenario_id": descriptor["scenario_id"],
            "profile_hash": descriptor["identity"]["region_profile_hash"],
            "source_manifest_sha256": descriptor["identity"]["source_manifest_sha256"],
            "source_profile_hash": descriptor["identity"]["source_profile_hash"],
            "region_manifest_sha256": descriptor["identity"]["region_manifest_sha256"],
            "route": descriptor["route"],
            "validation_status": descriptor["validation_status"],
            "heightfield_contact_status": (
                "source_mjcf_recorded" if schema == 3 else "not_recorded_legacy_schema"
            ),
        },
        static_boxes=boxes,
        heightfield_contact=heightfield_contact,
        source="verified_offline_training_region",
    )


__all__ = [
    "IsaacHeightfieldInput",
    "export_isaac_training_region",
    "load_isaac_heightfield",
    "load_isaac_training_region",
    "load_isaac_training_region_export",
]
