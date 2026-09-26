"""Build an opt-in, source-exact contact replacement for official wall parts 402/403.

This is deliberately not a runtime-pack writer.  It creates a locally hashed
collision experiment so the exact convex wall can replace, rather than overlap,
the single-heightfield roof.  The candidate stays blocked until robot and wall
tip contact gates are satisfied by the runtime-pack integration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from .collision_candidate import CONVEX_WALL_EVIDENCE, SOURCE_GLB_SHA256
from .download import OFFICIAL_STEP_SHA256
from .manifest import FieldAsset, sha256_file
from .query import HeightFieldData, load_heightfield
from .wall_tip_repair import WALL_PARTS, _projected_top


EXPECTED_ROOF_NODES_PER_WALL = 7900
ROOF_ERROR_LIMIT_M = 1.0e-5
BOTTOM_ERROR_LIMIT_M = 1.0e-5
WALL_CONTACT_SOLREF = "0.005 1"
CONVEX_PLANE_TOLERANCE_M = 2.0e-6


def _exact_source_convex_mesh(source_mesh: Any, *, source_part_index: int) -> Any:
    """Use the audited source facets as a convex mesh without a SciPy hull.

    The source GLB duplicates vertices at material/normal seams.  Welding those
    coincident vertices restores the closed 20-triangle polyhedron.  Checking
    every source vertex against every outward facet rejects a non-convex edit;
    asking trimesh for ``convex_hull`` would silently require optional SciPy.
    """

    evidence = CONVEX_WALL_EVIDENCE[source_part_index]
    mesh = source_mesh.copy()
    mesh.merge_vertices()
    if (
        len(mesh.vertices) != evidence["unique_vertices"]
        or len(mesh.faces) != evidence["hull_facets"]
        or not mesh.is_watertight
        or not mesh.is_winding_consistent
        or mesh.euler_number != 2
        or mesh.volume <= 0.0
        or not np.isclose(mesh.volume, evidence["hull_volume_m3"], atol=1.0e-5)
    ):
        raise ValueError(f"source part {source_part_index} closed convex topology changed")
    signed_distance = np.einsum(
        "fvc,fc->fv",
        mesh.vertices[None, :, :] - mesh.triangles_center[:, None, :],
        mesh.face_normals,
    )
    if float(np.max(signed_distance)) > CONVEX_PLANE_TOLERANCE_M:
        raise ValueError(f"source part {source_part_index} is no longer convex")
    return mesh


def _replace_wall_roof(
    wall_mesh: Any,
    field: HeightFieldData,
    output_height_m: np.ndarray,
    *,
    expected_nodes: int,
) -> dict[str, Any]:
    """Transfer one exact wall footprint from hfield roof to source mesh.

    Every replaced hfield sample must be the source roof, and the replacement
    must be the source wall's lower face and the existing lower flank.  These
    gates stop a same-looking but geometrically different wall from carving
    floor or other official structure.
    """

    bounds = np.asarray(wall_mesh.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("source wall bounds are invalid")
    x, y = field.x_m, field.y_m
    if (
        x.ndim != 1
        or y.ndim != 1
        or output_height_m.shape != (len(y), len(x))
        or field.height_m.shape != output_height_m.shape
        or not np.allclose(np.diff(x), 0.01, rtol=0.0, atol=1.0e-6)
        or not np.allclose(np.diff(y), 0.01, rtol=0.0, atol=1.0e-6)
    ):
        raise ValueError("wall replacement requires the verified 1 cm world grid")

    ix = np.flatnonzero((x >= bounds[0, 0]) & (x <= bounds[1, 0]))
    iy = np.flatnonzero((y >= bounds[0, 1]) & (y <= bounds[1, 1]))
    if len(ix) < 3 or len(iy) < 3 or ix[0] < 2 or ix[-1] > len(x) - 3:
        raise ValueError("wall replacement footprint or flank is incomplete")
    top = _projected_top(wall_mesh, x[ix], y[iy])
    covered = np.isfinite(top)
    if int(covered.sum()) != expected_nodes or not np.all(covered):
        raise ValueError("source wall roof footprint changed")
    original = field.height_m[np.ix_(iy, ix)]
    top_error = float(np.max(np.abs(original - top)))
    if top_error > ROOF_ERROR_LIMIT_M:
        raise ValueError(f"heightfield roof is not the exact source wall: {top_error}")

    inverted = wall_mesh.copy()
    inverted.vertices[:, 2] *= -1.0
    bottom = -_projected_top(inverted, x[ix], y[iy])
    lower_flank = np.minimum(field.height_m[iy, ix[0] - 2], field.height_m[iy, ix[-1] + 2])
    bottom_gap = float(np.max(bottom - lower_flank[:, None]))
    bottom_burial = float(np.max(lower_flank[:, None] - bottom))
    if not np.isfinite(bottom).all() or bottom_gap > BOTTOM_ERROR_LIMIT_M:
        raise ValueError(f"lower flank would leave a gap beneath the source wall: {bottom_gap}")

    # Assign only after all geometry/ownership checks pass.  The exact wall
    # mesh owns the roof and sides, and the hfield still owns the source floor.
    output_height_m[np.ix_(iy, ix)] = lower_flank[:, None]
    return {
        "roof_nodes_transferred": expected_nodes,
        "roof_source_abs_error_max_m": top_error,
        "maximum_gap_beneath_source_wall_m": max(0.0, bottom_gap),
        "maximum_source_wall_burial_beneath_floor_m": bottom_burial,
        "source_bounds_world_m": bounds.tolist(),
        "lower_flank_height_range_m": [float(lower_flank.min()), float(lower_flank.max())],
        "maximum_removed_roof_height_m": float(np.max(original - lower_flank[:, None])),
        "changed_grid_rectangle_yx": [int(iy[0]), int(iy[-1]), int(ix[0]), int(ix[-1])],
    }


def _verified_source_scene(
    source_root: Path, asset: FieldAsset
) -> tuple[Any, np.ndarray, dict[str, str]]:
    """Load the original GLB in the runtime frame after source/hash checks."""

    import trimesh

    source_manifest_path = source_root / "manifest.json"
    source_sha = sha256_file(source_manifest_path)
    if source_sha != asset.manifest["source_identity"]["field_manifest_sha256"]:
        raise ValueError("source build does not match this runtime pack")
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source["source"]["sha256"] != OFFICIAL_STEP_SHA256:
        raise ValueError("source build is not the audited official STEP")
    if source["collision"]["samples_sha256"] != asset.collision["samples_sha256"]:
        raise ValueError("source and runtime collision samples disagree")
    if source["collision"]["verified_wall_tip_repair"]["status"] != "PASS":
        raise ValueError("candidate requires the verified seven-node wall-tip repair")
    glb_record = source["conversion"]["colored_intermediate_glb"]
    if glb_record["sha256"] != SOURCE_GLB_SHA256:
        raise ValueError("source GLB identity changed")
    glb_path = source_root / glb_record["file"]
    if sha256_file(glb_path) != SOURCE_GLB_SHA256:
        raise ValueError("source GLB file hash changed")

    scene = trimesh.load(glb_path, force="scene", process=False)
    spawn = asset.recommended_spawn
    translation = np.asarray(
        [
            -float(spawn["x_before_translation_m"]),
            -float(spawn["y_before_translation_m"]),
            -float(source["conversion"]["main_floor_height_before_shift_m"])
            - float(spawn["terrain_height_m"]),
        ],
        dtype=np.float64,
    )
    return (
        scene,
        translation,
        {
            "source_manifest_sha256": source_sha,
            "source_glb_sha256": SOURCE_GLB_SHA256,
        },
    )


def _source_walls(source_root: Path, asset: FieldAsset) -> tuple[dict[int, Any], dict[str, str]]:
    """Load two exact GLB parts only after the source/runtime hashes agree."""

    scene, translation, source_identity = _verified_source_scene(source_root, asset)
    nodes = list(scene.graph.nodes_geometry)
    if len(nodes) <= max(WALL_PARTS):
        raise ValueError("source GLB part order is incomplete")
    walls: dict[int, Any] = {}
    for part in WALL_PARTS:
        transform, geometry_name = scene.graph[nodes[part]]
        original = scene.geometry[geometry_name].copy()
        original.apply_transform(transform)
        if len(original.faces) != 20 or len(np.unique(original.vertices, axis=0)) != 12:
            raise ValueError(f"source part {part} topology changed")
        hull = _exact_source_convex_mesh(original, source_part_index=part)
        hull.apply_translation(translation)
        walls[part] = hull
    return walls, source_identity


def build_verified_wall_replacement(
    source_build: Path, runtime_pack: Path
) -> tuple[HeightFieldData, dict[int, Any], dict[str, Any]]:
    """Return a local candidate without modifying either verified input."""

    asset = FieldAsset.open(runtime_pack)
    source_root = Path(source_build).expanduser().resolve()
    field = load_heightfield(asset)
    walls, source_identity = _source_walls(source_root, asset)
    updated = field.height_m.copy()
    reports = []
    regions = []
    for part in WALL_PARTS:
        record = _replace_wall_roof(
            walls[part], field, updated, expected_nodes=EXPECTED_ROOF_NODES_PER_WALL
        )
        record["source_part_index"] = part
        reports.append(record)
        regions.append(record["changed_grid_rectangle_yx"])
    if not (regions[0][1] < regions[1][0] or regions[1][1] < regions[0][0]):
        # The present walls are far apart.  Reject a changed source geometry
        # before one wall's floor can be overwritten by the other.
        if not (regions[0][3] < regions[1][2] or regions[1][3] < regions[0][2]):
            raise ValueError("exact wall ownership rectangles overlap")
    report = {
        "status": "EXPERIMENTAL_BLOCKED",
        "source_identity": source_identity,
        "base_runtime_manifest_sha256": asset.manifest_sha256,
        "base_collision_samples_sha256": asset.collision["samples_sha256"],
        "source_part_indices": list(WALL_PARTS),
        "contact_ownership": "source_convex_mesh_roof_and_sides; existing_hfield_source_bottom",
        "wall_contact_solref": WALL_CONTACT_SOLREF,
        "wall_friction": [1.0, 0.005, 0.0001],
        "walls": reports,
        "outside_wall_rectangles_bitwise_unchanged": True,
        "validation_boundary": (
            "local exact-source physics candidate only; correctly spawned 120 mm sphere "
            "tip trials reached at most 4.67 mm wall penetration without warnings; "
            "matched frozen Fudan 0.5 m/s near-wall trials had no solver warning, but "
            "downhill hfield pairs can reach MuJoCo's 50-contact cap and broader robot "
            "routes are unverified"
        ),
    }
    return HeightFieldData(field.x_m, field.y_m, updated), walls, report


def write_local_wall_contact_candidate(
    source_build: Path, runtime_pack: Path, output_directory: Path
) -> Path:
    """Write only experimental local assets, never a runtime-schema manifest."""

    from PIL import Image

    asset = FieldAsset.open(runtime_pack)
    field, walls, report = build_verified_wall_replacement(source_build, runtime_pack)
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate directory: {output}")
    collision_dir = output / "collision"
    collision_dir.mkdir(parents=True)

    spawn = asset.recommended_spawn
    source_height = field.height_m + float(spawn["terrain_height_m"])
    maximum = float(asset.collision["maximum_height_m"])
    if float(np.min(source_height)) < 0.0 or float(np.max(source_height)) > maximum:
        raise ValueError("candidate source height exceeds the unchanged hfield range")
    npz_path = collision_dir / "rmuc2026_heightfield.npz"
    np.savez_compressed(
        npz_path,
        x_m=field.x_m + float(spawn["x_before_translation_m"]),
        y_m=field.y_m + float(spawn["y_before_translation_m"]),
        height_m=source_height,
    )
    pixels = np.rint(np.clip(source_height / maximum, 0.0, 1.0) * 65535.0).astype(np.uint16)
    png_path = collision_dir / "rmuc2026_heightfield.png"
    Image.fromarray(pixels[::-1]).save(png_path)

    xml_root = ET.parse(asset.entrypoint_for("collision_only")).getroot()
    mesh_assets = xml_root.find("./asset")
    worldbody = xml_root.find("./worldbody")
    assert mesh_assets is not None and worldbody is not None
    for part in WALL_PARTS:
        relative = f"collision/official_wall_{part}.obj"
        walls[part].export(output / relative, file_type="obj")
        name = f"rmuc2026_official_wall_{part}"
        ET.SubElement(mesh_assets, "mesh", name=name, file=relative)
        ET.SubElement(
            worldbody,
            "geom",
            name=name,
            type="mesh",
            mesh=name,
            contype="2",
            conaffinity="1",
            friction="1 0.005 0.0001",
            solref=WALL_CONTACT_SOLREF,
            group="0",
        )
    xml_path = output / "rmuc2026_wall_contact_candidate.xml"
    ET.ElementTree(xml_root).write(xml_path, encoding="utf-8", xml_declaration=True)

    files = []
    for path in (
        npz_path,
        png_path,
        xml_path,
        *(output / f"collision/official_wall_{i}.obj" for i in WALL_PARTS),
    ):
        files.append(
            {
                "file": str(path.relative_to(output)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    report["artifact_type"] = "rmuc2026_wall_contact_local_candidate"
    report["candidate_collision_samples_sha256"] = sha256_file(npz_path)
    report["files"] = files
    report["runtime_schema_pack"] = False
    manifest_path = output / "candidate_manifest.json"
    manifest_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_build", type=Path)
    parser.add_argument("runtime_pack", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args(argv)
    print(
        write_local_wall_contact_candidate(
            args.source_build, args.runtime_pack, args.output_directory
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
