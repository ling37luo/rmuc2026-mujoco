"""A bounded physical perimeter for local robot experiments.

The rulebook specifies the 28 by 15 m battlefield and a 2.4 m fence top. The
CAD shell does not locate a continuous steel fence. The current containment
proxy follows the inferred 28 by 15 m raised deck edge. Its inner face overlaps
the deck by 25 mm, blocking a robot before it can drop into the lower skirt
between the playable deck and the outer CAD bounds. The resulting 5.5 mm
overlap at the two fixed fly-ramp outer corners is below the 10 mm heightfield
sample spacing. The wall thickness, buried base, and solid dart-aperture
treatment are simulation choices, not official dimensions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any
import xml.etree.ElementTree as ET


LEGACY_FENCE_SCHEMA = "rmuc2026_rulebook_perimeter_proxy_v1"
RAMP_CLEARANCE_FENCE_SCHEMA = "rmuc2026_rulebook_perimeter_proxy_v2"
SOFT_CONTACT_FENCE_SCHEMA = "rmuc2026_rulebook_perimeter_proxy_v3"
HEIGHTFIELD_EDGE_FENCE_SCHEMA = "rmuc2026_rulebook_perimeter_proxy_v4"
FENCE_SCHEMA = "rmuc2026_rulebook_perimeter_proxy_v5"
FENCE_NAMES = (
    "rmuc2026_perimeter_left",
    "rmuc2026_perimeter_right",
    "rmuc2026_perimeter_bottom",
    "rmuc2026_perimeter_top",
)
CORE_LENGTH_M = 28.0
CORE_WIDTH_M = 15.0
TOP_ABOVE_FIELD_FLOOR_M = 2.4
OUTWARD_OFFSET_X_M = 0.0
OUTWARD_OFFSET_Y_M = 0.40
DECK_EDGE_OUTWARD_OFFSET_Y_M = 0.0
THICKNESS_M = 0.05
BURIAL_M = 0.30
_APPROX_CAD_EXTENTS_M = (29.752008274927, 16.003130912781)


def perimeter_fence_contract(
    manifest: dict[str, Any], *, schema: str = FENCE_SCHEMA
) -> dict[str, Any]:
    """Derive four touching collision boxes from a pinned runtime pack."""

    supported_schemas = {
        LEGACY_FENCE_SCHEMA,
        RAMP_CLEARANCE_FENCE_SCHEMA,
        SOFT_CONTACT_FENCE_SCHEMA,
        HEIGHTFIELD_EDGE_FENCE_SCHEMA,
        FENCE_SCHEMA,
    }
    if schema not in supported_schemas:
        raise ValueError(f"unsupported perimeter fence schema: {schema}")
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
    collision = manifest["collision"]
    terrain_centre = collision["geom_center_after_translation_m"]
    if (
        len(terrain_centre) != 3
        or abs(float(terrain_centre[0]) - cx) > 0.05
        or abs(float(terrain_centre[1]) - cy) > 0.05
    ):
        raise ValueError("the inferred fence is not centered on the collision field")
    floor_z = -float(manifest["coordinate_frame"]["recommended_spawn"]["terrain_height_m"])
    if not all(math.isfinite(value) for value in (cx, cy, floor_z)):
        raise ValueError("the fence frame must be finite")
    half_thickness = THICKNESS_M / 2
    if schema == HEIGHTFIELD_EDGE_FENCE_SCHEMA:
        half_size_xy = collision.get("half_size_xy_m")
        if (
            not isinstance(half_size_xy, list)
            or len(half_size_xy) != 2
            or any(
                not math.isfinite(float(value)) or float(value) <= half_thickness
                for value in half_size_xy
            )
        ):
            raise ValueError("the collision heightfield half-size is invalid")
        terrain_cx, terrain_cy = (float(terrain_centre[index]) for index in range(2))
        half_x, half_y = (float(value) for value in half_size_xy)
        x_left = terrain_cx - half_x + half_thickness
        x_right = terrain_cx + half_x - half_thickness
        y_bottom = terrain_cy - half_y + half_thickness
        y_top = terrain_cy + half_y - half_thickness
        panel_cx = terrain_cx
        panel_cy = terrain_cy
        x_offset = half_x - half_thickness - CORE_LENGTH_M / 2
        y_offset = half_y - half_thickness - CORE_WIDTH_M / 2
        if x_offset < 0.0 or y_offset < 0.0:
            raise ValueError("the collision heightfield is too small to contain the official core")
        heightfield_bounds = [
            [terrain_cx - half_x, terrain_cy - half_y],
            [terrain_cx + half_x, terrain_cy + half_y],
        ]
    else:
        x_offset = 0.0 if schema == LEGACY_FENCE_SCHEMA else OUTWARD_OFFSET_X_M
        y_offset = (
            0.0
            if schema == LEGACY_FENCE_SCHEMA
            else DECK_EDGE_OUTWARD_OFFSET_Y_M
            if schema == FENCE_SCHEMA
            else OUTWARD_OFFSET_Y_M
        )
        x_left = cx - CORE_LENGTH_M / 2 - x_offset
        x_right = cx + CORE_LENGTH_M / 2 + x_offset
        y_bottom = cy - CORE_WIDTH_M / 2 - y_offset
        y_top = cy + CORE_WIDTH_M / 2 + y_offset
        panel_cx = cx
        panel_cy = cy
    half_height = (TOP_ABOVE_FIELD_FLOOR_M + BURIAL_M) / 2
    z = floor_z + (TOP_ABOVE_FIELD_FLOOR_M - BURIAL_M) / 2
    panels = [
        {
            "name": FENCE_NAMES[0],
            "pos": [x_left, panel_cy, z],
            "size": [half_thickness, (y_top - y_bottom - THICKNESS_M) / 2, half_height],
        },
        {
            "name": FENCE_NAMES[1],
            "pos": [x_right, panel_cy, z],
            "size": [half_thickness, (y_top - y_bottom - THICKNESS_M) / 2, half_height],
        },
        {
            "name": FENCE_NAMES[2],
            "pos": [panel_cx, y_bottom, z],
            "size": [(x_right - x_left + THICKNESS_M) / 2, half_thickness, half_height],
        },
        {
            "name": FENCE_NAMES[3],
            "pos": [panel_cx, y_top, z],
            "size": [(x_right - x_left + THICKNESS_M) / 2, half_thickness, half_height],
        },
    ]
    contract = {
        "schema": schema,
        "status": "EXPERIMENTAL_PHYSICAL_PROXY",
        "official_core_battlefield_m": [CORE_LENGTH_M, CORE_WIDTH_M],
        "official_fence_top_above_field_floor_m": TOP_ABOVE_FIELD_FLOOR_M,
        "placement_basis": (
            "pinned_CAD_outer_bounds_center_plus_rulebook_core_rectangle"
            if schema == LEGACY_FENCE_SCHEMA
            else "pinned_CAD_center_plus_core_x_and_source_supported_outer_y_apron"
            if schema in {RAMP_CLEARANCE_FENCE_SCHEMA, SOFT_CONTACT_FENCE_SCHEMA}
            else "collision_heightfield_outer_edge_with_wall_footprint_inside_terrain"
            if schema == HEIGHTFIELD_EDGE_FENCE_SCHEMA
            else "inferred_raised_core_deck_edge"
        ),
        "thickness_m": THICKNESS_M,
        "burial_below_field_floor_m": BURIAL_M,
        "dart_aperture_collision": "solid_for_robot_containment_not_an_official_window_model",
        "panels": panels,
        "contact": {
            "contype": 2,
            "conaffinity": 1,
            "friction": [1.0, 0.005, 0.0001],
            "solref": [
                0.04
                if schema
                in {SOFT_CONTACT_FENCE_SCHEMA, HEIGHTFIELD_EDGE_FENCE_SCHEMA, FENCE_SCHEMA}
                else 0.02,
                1.0,
            ],
        },
        "validation_status": "DRAFT_BLOCKED",
    }
    if schema == LEGACY_FENCE_SCHEMA:
        contract["outward_offset_m"] = 0.0
    else:
        contract["outward_offset_xy_m"] = [x_offset, y_offset]
        contract["visual_rgba"] = [0.08, 0.09, 0.10, 0.22]
    if schema == HEIGHTFIELD_EDGE_FENCE_SCHEMA:
        contract["heightfield_edge_alignment"] = {
            "heightfield_bounds_xy_m": heightfield_bounds,
            "wall_outer_faces_match_heightfield_edges": True,
            "traversable_strip_outside_wall_m": [0.0, 0.0, 0.0, 0.0],
            "wall_footprint_inside_heightfield": True,
        }
    if schema == FENCE_SCHEMA:
        contract["playable_deck_edge_alignment"] = {
            "inferred_core_bounds_xy_m": [
                [cx - CORE_LENGTH_M / 2, cy - CORE_WIDTH_M / 2],
                [cx + CORE_LENGTH_M / 2, cy + CORE_WIDTH_M / 2],
            ],
            "wall_inner_face_overlap_into_core_xy_m": [
                half_thickness - x_offset,
                half_thickness - y_offset,
            ],
            "traversable_lower_skirt_inside_wall_m": [0.0, 0.0, 0.0, 0.0],
            "claim_boundary": (
                "containment proxy follows the inferred raised 28x15m deck edge; "
                "not a surveyed official fence centerline"
            ),
        }
    return contract


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
                "rgba": (
                    "0.08 0.09 0.10 0.55"
                    if contract["schema"] == LEGACY_FENCE_SCHEMA
                    else "0.08 0.09 0.10 0.22"
                ),
            },
        )


def export_fenced_pack(source: Path, output: Path, *, schema: str = FENCE_SCHEMA) -> dict[str, Any]:
    """Copy a verified pack, bind fence geoms in both profiles, and reverify."""

    from .manifest import FieldAsset, sha256_file

    asset = FieldAsset.open(source, verify=True)
    if asset.manifest.get("perimeter_fence") is not None:
        raise ValueError("input pack already has a perimeter fence")
    contract = perimeter_fence_contract(dict(asset.manifest), schema=schema)
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
