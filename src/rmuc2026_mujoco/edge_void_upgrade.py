"""Derive a schema-3 source build from audited outer-edge ray evidence.

This is the fast path for an existing, decorated 1 cm source build.  A fresh
official STEP conversion uses the same edge rule in ``_ray_heightfield``.
Neither path edits the input build or an exported runtime pack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from ._conversion import _heightfield_structural_audit
from .collision_candidate import SOURCE_GLB_SHA256
from .download import OFFICIAL_STEP_SHA256
from .edge_void_candidate import EDGE_BAND_M, VOID_DEPTH_M, outer_edge_sample_mask
from .manifest import sha256_file


def derive_edge_void_source_build(base_build: Path, evidence_dir: Path, output_dir: Path) -> Path:
    """Copy a source build and replace only audited, source-missing edge nodes."""

    from PIL import Image

    base = Path(base_build).expanduser().resolve()
    evidence = Path(evidence_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    base_manifest_path = base / "manifest.json"
    candidate_path = evidence / "candidate.json"
    manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    collision = manifest["collision"]
    if (
        manifest.get("artifact_type") != "rmuc2026_official_field_mujoco_asset"
        or manifest.get("status") != "PASS"
        or manifest["source"]["sha256"] != OFFICIAL_STEP_SHA256
        or manifest["conversion"]["colored_intermediate_glb"]["sha256"] != SOURCE_GLB_SHA256
        or candidate.get("status") != "CANDIDATE_ONLY"
        or candidate.get("official_step_sha256") != OFFICIAL_STEP_SHA256
        or candidate.get("source_glb_sha256") != SOURCE_GLB_SHA256
        or candidate.get("base_collision_samples_sha256") != collision["samples_sha256"]
        or collision.get("resolution_m") != 0.01
        or collision.get("minimum_height_m") < 0.0
        or candidate.get("edge_band_m") != EDGE_BAND_M
        or candidate.get("void_depth_m") != VOID_DEPTH_M
    ):
        raise ValueError("source build or edge evidence does not match the audited 1 cm field")
    glb = base / manifest["conversion"]["colored_intermediate_glb"]["file"]
    base_samples = base / collision["samples_file"]
    evidence_samples = evidence / candidate["candidate_samples_file"]
    evidence_mask = evidence / candidate["source_ray_mask_file"]
    for path, expected in (
        (glb, SOURCE_GLB_SHA256),
        (base_samples, collision["samples_sha256"]),
        (evidence_samples, candidate["candidate_samples_sha256"]),
        (evidence_mask, candidate["source_ray_mask_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"source or evidence file hash changed: {path}")
    with (
        np.load(base_samples, allow_pickle=False) as old,
        np.load(evidence_samples, allow_pickle=False) as new,
        np.load(evidence_mask, allow_pickle=False) as ray_mask,
    ):
        x, y = np.asarray(old["x_m"]), np.asarray(old["y_m"])
        original = np.asarray(old["height_m"])
        updated = np.asarray(new["height_m"])
        changed = np.asarray(ray_mask["changed_mask"])
        source_hits = np.asarray(ray_mask["source_hit_mask"])
        exact_axes = (
            np.array_equal(x, new["x_m"])
            and np.array_equal(y, new["y_m"])
            and np.array_equal(x, ray_mask["x_m"])
            and np.array_equal(y, ray_mask["y_m"])
        )
    if (
        not exact_axes
        or original.shape != updated.shape
        or changed.shape != original.shape
        or source_hits.shape != original.shape
        or changed.dtype != np.bool_
        or source_hits.dtype != np.bool_
        or not np.isfinite(original).all()
        or not np.isfinite(updated).all()
        or np.any(changed & source_hits)
        or np.any(changed & ~outer_edge_sample_mask(x, y))
        or not np.all(source_hits[~outer_edge_sample_mask(x, y)])
        or not np.array_equal(original[~changed], updated[~changed])
        or not np.all(updated[changed] == -VOID_DEPTH_M)
        or int(changed.sum()) != candidate["source_ray_misses_changed"]
        or int(changed.sum()) > collision["ray_misses_filled_with_ground"]
    ):
        raise ValueError("edge evidence changes source-backed or interior field samples")
    # Copy rather than hard-link: changing NPZ/PNG and the manifest must never
    # mutate the source build, even on filesystems where hard links are cheap.
    shutil.copytree(base, output, copy_function=shutil.copy2)
    destination_samples = output / collision["samples_file"]
    shutil.copyfile(evidence_samples, destination_samples)
    mask_relative = "collision/edge_void_mask.npz"
    mask_path = output / mask_relative
    np.savez_compressed(mask_path, changed_mask=changed)
    maximum = float(np.max(updated))
    minimum = float(np.min(updated))
    normalized = (updated - minimum) / (maximum - minimum)
    pixels = np.rint(normalized * 65535.0).astype(np.uint16)
    image_path = output / collision["image_file"]
    Image.fromarray(pixels[::-1], mode="I;16").save(image_path)

    collision["samples_sha256"] = sha256_file(destination_samples)
    collision["image_sha256"] = sha256_file(image_path)
    collision["minimum_height_m"] = minimum
    collision["maximum_height_m"] = maximum
    collision["ray_misses_filled_with_ground"] -= int(changed.sum())
    collision["structural_audit"] = _heightfield_structural_audit(updated)
    collision["verified_wall_tip_repair"]["collision_samples_sha256"] = collision["samples_sha256"]
    collision["edge_void_provenance"] = {
        "status": "PASS",
        "source_glb_sha256": SOURCE_GLB_SHA256,
        "mask_file": mask_relative,
        "mask_sha256": sha256_file(mask_path),
        "changed_count": int(changed.sum()),
        "edge_band_m": EDGE_BAND_M,
        "sentinel_height_m": -VOID_DEPTH_M,
    }
    collision["claim_boundary"] = (
        "official STEP-derived single-valued 2.5D top-surface contact proxy; audited "
        "outer-edge source misses use a finite -5 m surrogate void, while interior "
        "ray misses remain filled with ground; underpasses and stacked surfaces remain sealed"
    )
    manifest["build_variant"] = "source_ray_edge_void_derivative_v1"
    manifest["edge_void_derivation"] = {
        "base_manifest_sha256": sha256_file(base_manifest_path),
        "candidate_json_sha256": sha256_file(candidate_path),
        "candidate_samples_sha256": candidate["candidate_samples_sha256"],
        "source_ray_mask_sha256": candidate["source_ray_mask_sha256"],
        "source_hit_nodes_changed": 0,
        "interior_nodes_changed": 0,
    }
    result = output / "manifest.json"
    result.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_build", type=Path)
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    print(derive_edge_void_source_build(args.base_build, args.evidence_dir, args.output_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
