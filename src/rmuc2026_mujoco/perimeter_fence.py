"""A bounded, rulebook-sized physical perimeter for local robot experiments.

The rulebook specifies the 28 by 15 m battlefield and a 2.4 m fence top. The
CAD shell does not locate a continuous steel fence. This proxy places its
centerline at the inferred core edge. Its 50 mm thickness, buried base, and
solid treatment of the dart aperture are simulation choices, not official
dimensions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any
import xml.etree.ElementTree as ET


FENCE_SCHEMA = "rmuc2026_rulebook_perimeter_proxy_v1"
FENCE_NAMES = (
    "rmuc2026_perimeter_left",
    "rmuc2026_perimeter_right",
    "rmuc2026_perimeter_bottom",
    "rmuc2026_perimeter_top",
)
CORE_LENGTH_M = 28.0
CORE_WIDTH_M = 15.0
TOP_ABOVE_FIELD_FLOOR_M = 2.4
OUTWARD_OFFSET_M = 0.0
THICKNESS_M = 0.05
BURIAL_M = 0.30
_APPROX_CAD_EXTENTS_M = (29.752008274927, 16.003130912781)


def perimeter_fence_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    """Derive four touching collision boxes from the pinned CAD core frame."""

    if manifest.get("schema_version") != 2:
        raise ValueError("the physical fence currently requires a schema-2 field pack")
    dimensions = manifest["dimensions"]
    if dimensions.get("official_core_battlefield_m") != [CORE_LENGTH_M, CORE_WIDTH_M]:
        raise ValueError("the field's official core dimensions do not match the fence")
    bounds = dimensions["cad_assembly_outer_bounds_after_translation_m"]
    if not isinstance(bounds, list) or len(bounds) != 2 or any(len(row) != 3 for row in bounds):
        raise ValueError("the CAD assembly bounds must have two XYZ corners")
    low, high = ([float(value) for value in row] for row in bounds)
    extents = (high[0] - low[0], high[1] - low[1])
    if any(
        not math.isfinite(value) or abs(value - expected) > 0.02
        for value, expected in zip(extents, _APPROX_CAD_EXTENTS_M)
    ):
        raise ValueError("the CAD assembly does not match the calibrated fence frame")
    cx = (low[0] + high[0]) / 2
    cy = (low[1] + high[1]) / 2
    terrain_centre = manifest["collision"]["geom_center_after_translation_m"]
    if (
        len(terrain_centre) != 3
        or abs(float(terrain_centre[0]) - cx) > 0.05
        or abs(float(terrain_centre[1]) - cy) > 0.05
    ):
        raise ValueError("the inferred fence is not centered on the collision field")
    floor_z = -float(manifest["coordinate_frame"]["recommended_spawn"]["terrain_height_m"])
    if not all(math.isfinite(value) for value in (cx, cy, floor_z)):
        raise ValueError("the fence frame must be finite")
    x_left = cx - CORE_LENGTH_M / 2 - OUTWARD_OFFSET_M
    x_right = cx + CORE_LENGTH_M / 2 + OUTWARD_OFFSET_M
    y_bottom = cy - CORE_WIDTH_M / 2 - OUTWARD_OFFSET_M
    y_top = cy + CORE_WIDTH_M / 2 + OUTWARD_OFFSET_M
    half_thickness = THICKNESS_M / 2
    half_height = (TOP_ABOVE_FIELD_FLOOR_M + BURIAL_M) / 2
    z = floor_z + (TOP_ABOVE_FIELD_FLOOR_M - BURIAL_M) / 2
    panels = [
        {
            "name": FENCE_NAMES[0],
            "pos": [x_left, cy, z],
            "size": [half_thickness, (y_top - y_bottom - THICKNESS_M) / 2, half_height],
        },
        {
            "name": FENCE_NAMES[1],
            "pos": [x_right, cy, z],
            "size": [half_thickness, (y_top - y_bottom - THICKNESS_M) / 2, half_height],
        },
        {
            "name": FENCE_NAMES[2],
            "pos": [cx, y_bottom, z],
            "size": [(x_right - x_left + THICKNESS_M) / 2, half_thickness, half_height],
        },
        {
            "name": FENCE_NAMES[3],
            "pos": [cx, y_top, z],
            "size": [(x_right - x_left + THICKNESS_M) / 2, half_thickness, half_height],
        },
    ]
    return {
        "schema": FENCE_SCHEMA,
        "status": "EXPERIMENTAL_PHYSICAL_PROXY",
        "official_core_battlefield_m": [CORE_LENGTH_M, CORE_WIDTH_M],
        "official_fence_top_above_field_floor_m": TOP_ABOVE_FIELD_FLOOR_M,
        "placement_basis": "pinned_CAD_outer_bounds_center_plus_rulebook_core_rectangle",
        "outward_offset_m": OUTWARD_OFFSET_M,
        "thickness_m": THICKNESS_M,
        "burial_below_field_floor_m": BURIAL_M,
        "dart_aperture_collision": "solid_for_robot_containment_not_an_official_window_model",
        "panels": panels,
        "contact": {
            "contype": 2,
            "conaffinity": 1,
            "friction": [1.0, 0.005, 0.0001],
            "solref": [0.02, 1.0],
        },
        "validation_status": "DRAFT_BLOCKED",
    }


def _format(values: list[float]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def append_perimeter_fence(root: ET.Element, contract: dict[str, Any]) -> None:
    """Add the same physical boundary to both field runtime profiles."""

    worldbody = root.find("./worldbody")
    if worldbody is None:
        raise ValueError("field MJCF lacks worldbody")
    if any(geom.get("name") in FENCE_NAMES for geom in worldbody.findall("./geom")):
        raise ValueError("field MJCF already contains perimeter geoms")
    contact = contract["contact"]
    for panel in contract["panels"]:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": panel["name"],
                "type": "box",
                "pos": _format(panel["pos"]),
                "size": _format(panel["size"]),
                "contype": str(contact["contype"]),
                "conaffinity": str(contact["conaffinity"]),
                "friction": _format(contact["friction"]),
                "solref": _format(contact["solref"]),
                "rgba": "0.08 0.09 0.10 0.55",
            },
        )


def export_fenced_pack(source: Path, output: Path) -> dict[str, Any]:
    """Copy a verified pack, bind fence geoms in both profiles, and reverify."""

    from .manifest import FieldAsset, sha256_file

    asset = FieldAsset.open(source, verify=True)
    if asset.manifest.get("perimeter_fence") is not None:
        raise ValueError("input pack already has a perimeter fence")
    contract = perimeter_fence_contract(dict(asset.manifest))
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise ValueError(f"output already exists: {destination}")
    if destination == asset.root or asset.root in destination.parents:
        raise ValueError("output cannot be nested inside the input pack")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        for path in asset.verified_files:
            relative = path.relative_to(asset.root)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        manifest = json.loads(json.dumps(asset.manifest))
        for profile in asset.available_runtime_profiles:
            relative = str(asset.runtime_profiles[profile]["entrypoint"])
            xml_path = staging / relative
            tree = ET.parse(xml_path)
            append_perimeter_fence(tree.getroot(), contract)
            ET.indent(tree, space="  ")
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            with xml_path.open("ab") as stream:
                stream.write(b"\n")
            record = next(row for row in manifest["contents"]["files"] if row["file"] == relative)
            record["sha256"] = sha256_file(xml_path)
            record["size_bytes"] = xml_path.stat().st_size
        manifest["perimeter_fence"] = contract
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        FieldAsset.open(staging, verify=True)
        staging.rename(destination)
        return contract
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
