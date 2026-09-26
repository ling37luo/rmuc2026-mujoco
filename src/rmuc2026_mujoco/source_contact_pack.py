"""Export a local runtime candidate with exact-source wall contact ownership.

The source wall replaces the heightfield roof at its footprint.  This exporter
does not claim that whole-field topology or robot traversal is accepted.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from .manifest import FieldAsset, sha256_file
from .training_region import _heightfield_png
from .wall_active_contact import WALL_CONTACT_SOLREF, build_verified_wall_replacement


CONTACT_LAYER_KIND = "official_source_convex_walls_402_403_v1"
CONTACT_FRICTION = (1.0, 0.005, 0.0001)


def _update_file_record(
    manifest: dict, root: Path, relative: str, *, role: str | None = None
) -> None:
    path = root / relative
    records = manifest["contents"]["files"]
    record = next((item for item in records if item["file"] == relative), None)
    if record is None:
        if role is None:
            raise ValueError(f"undeclared file has no role: {relative}")
        record = {"file": relative, "role": role}
        records.append(record)
    record["sha256"] = sha256_file(path)
    record["size_bytes"] = path.stat().st_size


def _append_wall_geoms(xml_path: Path, mesh_records: list[dict]) -> None:
    root = ET.parse(xml_path).getroot()
    asset = root.find("./asset")
    worldbody = root.find("./worldbody")
    if asset is None or worldbody is None:
        raise ValueError("runtime MJCF lacks asset or worldbody")
    for record in mesh_records:
        name = record["name"]
        if root.find(f"./asset/mesh[@name='{name}']") is not None:
            raise ValueError(f"wall mesh already exists in {xml_path}: {name}")
        ET.SubElement(asset, "mesh", {"name": name, "file": record["file"]})
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": name,
                "type": "mesh",
                "mesh": name,
                "contype": str(record["contype"]),
                "conaffinity": str(record["conaffinity"]),
                "friction": " ".join(f"{x:g}" for x in record["friction"]),
                "solref": " ".join(f"{x:g}" for x in record["solref"]),
                "group": "3",
                "rgba": "0.4 0.4 0.4 0",
            },
        )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)


def export_source_wall_pack(
    source_build: str | Path,
    runtime_pack: str | Path,
    output: str | Path,
) -> dict:
    """Derive a hash-bound, opt-in wall-contact pack from a verified schema-4 pack.

    The input pack may include ``interactive_lite``.  Every runtime profile gets
    the same updated heightfield and two collision meshes.  No input is edited.
    """

    asset = FieldAsset.open(runtime_pack, verify=True)
    if asset.manifest["schema_version"] != 4:
        raise ValueError("source-wall export requires a schema-4 runtime pack")
    if asset.collision.get("source_contact_layer") is not None:
        raise ValueError("runtime pack already has a source contact layer")
    field, walls, report = build_verified_wall_replacement(
        Path(source_build).expanduser().resolve(), asset.root
    )
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for original in asset.verified_files:
            relative = original.relative_to(asset.root)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
        manifest = copy.deepcopy(dict(asset.manifest))
        collision = manifest["collision"]
        prior_samples_sha = collision["samples_sha256"]
        spawn = asset.recommended_spawn
        source_height = field.height_m + float(spawn["terrain_height_m"])
        maximum_height = float(collision["maximum_height_m"])
        minimum_height = float(collision["minimum_height_m"])
        if (
            not np.isfinite(source_height).all()
            or float(source_height.min()) < minimum_height - 1e-8
            or float(source_height.max()) > maximum_height + 1e-8
        ):
            raise ValueError("wall replacement exceeds the declared heightfield range")
        samples_relative = str(collision["samples_file"])
        image_relative = str(collision["image_file"])
        np.savez_compressed(
            staging / samples_relative,
            x_m=field.x_m + float(spawn["x_before_translation_m"]),
            y_m=field.y_m + float(spawn["y_before_translation_m"]),
            height_m=source_height,
        )
        normalized = (source_height - minimum_height) / (maximum_height - minimum_height)
        (staging / image_relative).write_bytes(_heightfield_png(normalized))
        _update_file_record(manifest, staging, samples_relative)
        _update_file_record(manifest, staging, image_relative)
        collision["samples_sha256"] = sha256_file(staging / samples_relative)
        collision["image_sha256"] = sha256_file(staging / image_relative)
        # The previous seven hfield roof-tip nodes have been transferred to the
        # source wall meshes, so the old hfield-only repair claim is superseded.
        collision.pop("verified_wall_tip_repair", None)

        mesh_records: list[dict] = []
        for part, mesh in sorted(walls.items()):
            relative = f"collision/official_wall_{part}.obj"
            path = staging / relative
            mesh.export(path, file_type="obj")
            _update_file_record(manifest, staging, relative, role="source_contact_mesh")
            mesh_records.append(
                {
                    "name": f"rmuc2026_official_wall_{part}",
                    "file": relative,
                    "sha256": sha256_file(path),
                    "source_part_index": part,
                    "contype": 2,
                    "conaffinity": 1,
                    "friction": list(CONTACT_FRICTION),
                    "solref": [float(value) for value in WALL_CONTACT_SOLREF.split()],
                }
            )
        for profile in asset.available_runtime_profiles:
            relative = str(asset.runtime_profiles[profile]["entrypoint"])
            _append_wall_geoms(staging / relative, mesh_records)
            _update_file_record(manifest, staging, relative)

        collision["source_contact_layer"] = {
            "kind": CONTACT_LAYER_KIND,
            "status": "EXPERIMENTAL_BLOCKED",
            "source_manifest_sha256": report["source_identity"]["source_manifest_sha256"],
            "source_glb_sha256": report["source_identity"]["source_glb_sha256"],
            "base_collision_samples_sha256": prior_samples_sha,
            "updated_collision_samples_sha256": collision["samples_sha256"],
            "roof_nodes_transferred": sum(
                int(item["roof_nodes_transferred"]) for item in report["walls"]
            ),
            "ownership_regions": report["walls"],
            "mesh_geoms": mesh_records,
            "claim_boundary": report["validation_boundary"],
        }
        manifest["contents"]["file_count_excluding_manifest"] = len(manifest["contents"]["files"])
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        # The source-contact validator is installed by the pack's schema-4
        # reader; this check also compiles every declared file and hash.
        FieldAsset.open(staging, verify=True)
        staging.rename(destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = ["CONTACT_LAYER_KIND", "export_source_wall_pack"]
