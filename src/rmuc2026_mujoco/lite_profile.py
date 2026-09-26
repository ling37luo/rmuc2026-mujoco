"""Build a visual-only light profile from an already verified runtime pack.

The full and light entrypoints use the same collision assets and world geoms.
Only the 38 CAD mesh file references change; the locally generated files stay
inside the new pack and remain bound by its manifest hashes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from .manifest import FieldAsset, sha256_file
from .pack import ExportBlocked


LITE_XML = "rmuc2026_field_interactive_lite.xml"
LITE_METHOD = "quadric_decimation_per_cad_visual_group_v1"


def _mesh_file_record(root: Path, relative: str, role: str) -> dict[str, object]:
    path = root / relative
    return {
        "file": relative,
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _face_targets(face_counts: list[int], roles: list[str], budget: int) -> list[int]:
    """Distribute one ceiling across material groups; never decimate the base shell."""

    minimums = [
        count if role == "base_surface_shell" else min(count, max(4, math.ceil(count * 0.10)))
        for count, role in zip(face_counts, roles)
    ]
    if sum(minimums) > budget:
        raise ExportBlocked(
            f"visual face budget {budget} cannot preserve the grouped source geometry; "
            f"minimum is {sum(minimums)}"
        )
    remaining = budget - sum(minimums)
    capacities = [count - minimum for count, minimum in zip(face_counts, minimums)]
    total_capacity = sum(capacities)
    if not total_capacity:
        return minimums
    ideal = [remaining * capacity / total_capacity for capacity in capacities]
    extras = [min(capacity, int(value)) for capacity, value in zip(capacities, ideal)]
    spare = remaining - sum(extras)
    for index in sorted(
        range(len(face_counts)),
        key=lambda item: (ideal[item] - extras[item], capacities[item]),
        reverse=True,
    ):
        if spare == 0:
            break
        if extras[index] < capacities[index]:
            extras[index] += 1
            spare -= 1
    return [minimum + extra for minimum, extra in zip(minimums, extras)]


def export_interactive_lite_pack(
    source_pack: Path,
    output_dir: Path,
    *,
    target_visual_faces: int,
) -> dict[str, object]:
    """Create a new schema-4 pack with unchanged physics and smaller CAD visuals.

    The source may be a schema-2, -3, or schema-4 runtime pack, including a
    perimeter-fenced pack.  It is hash-verified before any derivative is made.
    The ``build`` extra supplies trimesh and fast-simplification for this one
    export operation; loading the resulting pack needs neither dependency.
    """

    if isinstance(target_visual_faces, bool) or target_visual_faces <= 0:
        raise ValueError("--interactive-lite-faces must be a positive integer")
    source = FieldAsset.open(source_pack, verify=True)
    if int(source.manifest["schema_version"]) < 2:
        raise ExportBlocked("lite-pack requires a runtime pack with named lighting (schema 2+)")
    if "interactive_lite" in source.runtime_profiles:
        raise ExportBlocked(
            "source pack already contains interactive_lite; use its full source pack"
        )
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise ExportBlocked(f"output already exists: {output}")
    if output == source.root or source.root in output.parents:
        raise ExportBlocked("output cannot be inside the source pack")

    try:
        import trimesh
    except ImportError as exc:
        raise ExportBlocked(
            "lite-pack needs the build extra: install rmuc2026-mujoco[build]"
        ) from exc

    source_visuals = list(source.visual_meshes)
    meshes = [
        trimesh.load(source.root / str(record["file"]), force="mesh", process=False)
        for record in source_visuals
    ]
    face_counts = [int(len(mesh.faces)) for mesh in meshes]
    source_faces = sum(face_counts)
    if target_visual_faces >= source_faces:
        raise ExportBlocked(
            f"visual face budget {target_visual_faces} cannot improve this pack: "
            f"full contains only {source_faces} faces"
        )
    if any(count == 0 for count in face_counts):
        raise ExportBlocked("source contains an empty visual mesh")
    roles = [str(record.get("visual_role", "cad_structure")) for record in source_visuals]
    targets = _face_targets(face_counts, roles, target_visual_faces)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for record in source.manifest["contents"]["files"]:
            relative = str(record["file"])
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source.root / relative, destination)
        manifest = json.loads((source.root / "manifest.json").read_text(encoding="utf-8"))
        lite_records: list[dict[str, object]] = []
        file_records: list[dict[str, object]] = []
        replacements: dict[str, str] = {}
        for source_record, mesh, original_faces, wanted in zip(
            source_visuals, meshes, face_counts, targets
        ):
            source_relative = str(source_record["file"])
            lite_relative = (Path("visual_lite") / Path(source_relative).name).as_posix()
            destination = staging / lite_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if wanted == original_faces:
                shutil.copyfile(source.root / source_relative, destination)
                actual_faces = original_faces
            else:
                try:
                    reduced = mesh.simplify_quadric_decimation(face_count=wanted)
                except (ImportError, ValueError, RuntimeError) as exc:
                    raise ExportBlocked(
                        f"CAD visual simplification failed for {source_relative}: {exc}"
                    ) from exc
                actual_faces = int(len(reduced.faces))
                if (
                    actual_faces == 0
                    or actual_faces > original_faces
                    or not np.isfinite(reduced.vertices).all()
                ):
                    raise ExportBlocked(
                        f"CAD visual simplification produced invalid mesh: {source_relative}"
                    )
                payload = reduced.export(file_type="obj")
                destination.write_bytes(
                    payload.encode("utf-8") if isinstance(payload, str) else payload
                )
            replacements[source_relative] = lite_relative
            file_record = _mesh_file_record(staging, lite_relative, "cad_visual_mesh_lite")
            file_records.append(file_record)
            lite_records.append(
                {
                    "source_file": source_relative,
                    "file": lite_relative,
                    "sha256": file_record["sha256"],
                    "source_faces": original_faces,
                    "faces": actual_faces,
                }
            )

        actual_total = sum(int(record["faces"]) for record in lite_records)
        # A disconnected group can be irreducible at its allocated target.
        # Recover the small excess from larger groups that can still decimate
        # without dropping a material group or touching the base shell.
        if actual_total > target_visual_faces:
            for index in sorted(
                range(len(lite_records)),
                key=lambda item: int(lite_records[item]["faces"]),
                reverse=True,
            ):
                if actual_total <= target_visual_faces:
                    break
                if roles[index] == "base_surface_shell":
                    continue
                previous = int(lite_records[index]["faces"])
                if previous <= 4:
                    continue
                wanted = max(4, previous - (actual_total - target_visual_faces) - 64)
                try:
                    reduced = meshes[index].simplify_quadric_decimation(face_count=wanted)
                except (ImportError, ValueError, RuntimeError) as exc:
                    raise ExportBlocked(
                        f"CAD visual simplification failed for {lite_records[index]['source_file']}: {exc}"
                    ) from exc
                achieved = int(len(reduced.faces))
                if achieved < 1 or achieved > previous or not np.isfinite(reduced.vertices).all():
                    raise ExportBlocked(
                        "CAD visual simplification produced an invalid recovery mesh"
                    )
                if achieved == previous:
                    continue
                destination = staging / str(lite_records[index]["file"])
                payload = reduced.export(file_type="obj")
                destination.write_bytes(
                    payload.encode("utf-8") if isinstance(payload, str) else payload
                )
                refreshed = _mesh_file_record(
                    staging, str(lite_records[index]["file"]), "cad_visual_mesh_lite"
                )
                file_records[index] = refreshed
                lite_records[index]["sha256"] = refreshed["sha256"]
                lite_records[index]["faces"] = achieved
                actual_total -= previous - achieved
        if actual_total > target_visual_faces:
            raise ExportBlocked(
                f"group-preserving simplification reached {actual_total} faces, "
                f"above requested budget {target_visual_faces}"
            )

        full_xml = ET.parse(source.entrypoint_for("full"))
        root = full_xml.getroot()
        changed = 0
        for mesh_asset in root.findall("./asset/mesh"):
            old_file = mesh_asset.get("file")
            if old_file in replacements:
                mesh_asset.set("file", replacements[old_file])
                changed += 1
        if changed != len(source_visuals):
            raise ExportBlocked("full MJCF visual meshes disagree with the source manifest")
        ET.indent(root, space="  ")
        (staging / LITE_XML).write_bytes(
            ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"
        )
        manifest["schema_version"] = 4
        manifest["visual_lite"] = {
            "source_profile": "full",
            "method": LITE_METHOD,
            "target_visual_faces": target_visual_faces,
            "source_visual_faces": source_faces,
            "output_visual_faces": actual_total,
            "physics_changed": False,
            "meshes": lite_records,
        }
        manifest["runtime_profiles"]["profiles"]["interactive_lite"] = {
            "entrypoint": LITE_XML,
            "includes_visual_meshes": True,
            "visual_mesh_count": len(lite_records),
            "intended_use": "interactive_visualization_lite",
        }
        file_records.insert(0, _mesh_file_record(staging, LITE_XML, "field_mjcf_interactive_lite"))
        manifest["contents"]["files"].extend(file_records)
        manifest["contents"]["file_count_excluding_manifest"] = len(manifest["contents"]["files"])
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        FieldAsset.open(staging, verify=True)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = ["export_interactive_lite_pack"]
