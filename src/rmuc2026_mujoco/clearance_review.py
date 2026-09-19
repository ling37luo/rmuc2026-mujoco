"""Read-only review of legacy clearance evidence against one verified runtime pack.

This checks whether the current heightfield still seals the CAD clearance
candidates and replays the recorded horizontal blocker masks. It never creates
collision geometry or promotes a candidate to a traversable route.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .errors import Rmuc2026Error
from .manifest import FieldAsset, sha256_file
from .query import HeightFieldData, load_heightfield


TARGET_PARTS = (292, 312)
ENTRY_NAMES = (
    "x_negative_entry",
    "x_positive_entry",
    "y_negative_entry",
    "y_positive_entry",
)
SEAL_TOLERANCE_M = 0.02
LEGACY_MULTIHIT_MANIFEST_SHA256 = "1443a7a14f5252ff321e951cd59810bda87673edf67ce39779e54799fc24aff2"
LEGACY_HORIZONTAL_MANIFEST_SHA256 = (
    "7fec1cfecc56f65afe803cc12af34280418dcf106959e935c780b7f3e2b0a55c"
)


class ClearanceReviewError(ValueError):
    """The inputs do not support a hash-bound, read-only review."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ClearanceReviewError(reason)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClearanceReviewError(f"cannot read {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _bound_file(path: Path, expected_sha256: str, label: str) -> str:
    _require(path.is_file(), f"{label} is missing: {path}")
    actual = sha256_file(path)
    _require(actual == expected_sha256, f"{label} SHA-256 mismatch")
    return actual


def _coordinate_contract(source: dict[str, Any]) -> dict[str, Any]:
    """Keep the CAD-to-heightfield transform separate from grid resolution."""

    conversion = source["conversion"]
    collision = source["collision"]
    return {
        "axis_and_units": conversion["axis_and_units"],
        "main_floor_height_before_shift_m": conversion["main_floor_height_before_shift_m"],
        "recommended_spawn": source["recommended_spawn"],
        "geom_center_after_translation_m": collision["geom_center_after_translation_m"],
        "xy_bounds_m": [
            collision["x_min_m"],
            collision["x_max_m"],
            collision["y_min_m"],
            collision["y_max_m"],
        ],
    }


def _nearest(axis: np.ndarray, query: np.ndarray) -> np.ndarray:
    upper = np.clip(np.searchsorted(axis, query, side="left"), 0, len(axis) - 1)
    lower = np.clip(upper - 1, 0, len(axis) - 1)
    return np.where(np.abs(query - axis[lower]) <= np.abs(axis[upper] - query), lower, upper)


def review_samples(
    heightfield: HeightFieldData,
    x_m: np.ndarray,
    y_m: np.ndarray,
    target_lower_z_m: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    seal_tolerance_m: float = SEAL_TOLERANCE_M,
) -> dict[str, float | int]:
    """Replay the legacy nearest-cell seal rule on *current* pack samples."""

    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    lower = np.asarray(target_lower_z_m, dtype=np.float64)
    candidate = np.asarray(candidate_mask, dtype=bool)
    _require(x.ndim == y.ndim == 1 and x.size > 0 and y.size > 0, "invalid audit axes")
    _require(lower.shape == candidate.shape == (y.size, x.size), "audit grid shape mismatch")
    _require(np.isfinite(x).all() and np.isfinite(y).all(), "audit axes are nonfinite")
    _require(np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0), "audit axes are unordered")
    _require(
        x[0] >= heightfield.x_m[0]
        and x[-1] <= heightfield.x_m[-1]
        and y[0] >= heightfield.y_m[0]
        and y[-1] <= heightfield.y_m[-1],
        "audit grid lies outside current heightfield",
    )
    _require(np.isfinite(lower[candidate]).all(), "candidate lower surface is nonfinite")
    _require(math.isfinite(seal_tolerance_m) and seal_tolerance_m >= 0, "invalid seal tolerance")
    count = int(np.count_nonzero(candidate))
    _require(count > 0, "no CAD clearance candidate samples")
    current = heightfield.height_m[
        np.ix_(_nearest(heightfield.y_m, y), _nearest(heightfield.x_m, x))
    ]
    sealed = candidate & (current >= lower - seal_tolerance_m)
    sealed_count = int(np.count_nonzero(sealed))
    return {
        "candidate_count": count,
        "current_heightfield_sealed_count": sealed_count,
        "current_heightfield_sealed_fraction": sealed_count / count,
        "current_heightfield_height_min_m": float(np.min(current[candidate])),
        "current_heightfield_height_max_m": float(np.max(current[candidate])),
    }


def review_entry_mask(
    mask: np.ndarray, spacing_m: float, required_width_m: float
) -> dict[str, Any]:
    """Count contiguous lanes clear at every sampled robot-envelope height."""

    blocked = np.asarray(mask)
    _require(blocked.ndim == 2 and min(blocked.shape) > 0, "invalid horizontal ray mask")
    _require(np.isin(blocked, (0, 1)).all(), "horizontal ray mask is not binary")
    _require(math.isfinite(spacing_m) and spacing_m > 0, "invalid lane spacing")
    _require(math.isfinite(required_width_m) and required_width_m > 0, "invalid width")
    clear = ~np.asarray(blocked, dtype=bool).any(axis=1)
    widest = current = 0
    for value in clear:
        current = current + 1 if value else 0
        widest = max(widest, current)
    width = widest * spacing_m
    return {
        "lane_count": int(blocked.shape[0]),
        "fully_clear_lane_count": int(np.count_nonzero(clear)),
        "widest_connected_clear_width_m": float(width),
        "required_connected_clear_width_m": float(required_width_m),
        "envelope_channel_observed": bool(width >= required_width_m),
    }


def review_clearance(
    pack: Path,
    source_manifest_path: Path,
    multihit_dir: Path,
    horizontal_dir: Path,
) -> dict[str, Any]:
    """Bind the old CAD rays to the current 1 cm pack and replay safe checks."""

    pack = Path(pack).resolve()
    source_manifest_path = Path(source_manifest_path).resolve()
    multihit_dir = Path(multihit_dir).resolve()
    horizontal_dir = Path(horizontal_dir).resolve()
    asset = FieldAsset.open(pack)
    _require(asset.hashes_verified, "pack file hashes were not verified")
    _require(abs(float(asset.collision["resolution_m"]) - 0.01) < 1e-8, "pack is not 1 cm")
    source_sha = _bound_file(
        source_manifest_path,
        str(asset.manifest["source_identity"]["field_manifest_sha256"]),
        "source manifest",
    )
    source = _json(source_manifest_path)
    current_samples_sha = str(asset.collision["samples_sha256"])
    _require(
        source["collision"]["samples_sha256"] == current_samples_sha,
        "source manifest does not bind the current pack samples",
    )
    cad = source["conversion"]["colored_intermediate_glb"]
    cad_sha = str(cad["sha256"])
    _bound_file(source_manifest_path.parent / str(cad["file"]), cad_sha, "source CAD GLB")

    multihit_path = multihit_dir / "manifest.json"
    horizontal_path = horizontal_dir / "manifest.json"
    _bound_file(multihit_path, LEGACY_MULTIHIT_MANIFEST_SHA256, "supported multi-hit manifest")
    _bound_file(horizontal_path, LEGACY_HORIZONTAL_MANIFEST_SHA256, "supported horizontal manifest")
    multihit = _json(multihit_path)
    horizontal = _json(horizontal_path)
    _require(multihit.get("status") == "PASS", "legacy multi-hit audit did not pass")
    _require(
        multihit["inputs"]["fixed_glb_sha256"] == cad_sha
        and horizontal["inputs"]["fixed_glb_sha256"] == cad_sha,
        "legacy audits use a different CAD GLB",
    )
    _bound_file(
        multihit_path,
        str(horizontal["inputs"]["multihit_manifest_sha256"]),
        "multi-hit manifest",
    )
    multi_samples_path = multihit_dir / "clearance_multihit_samples.npz"
    _bound_file(
        multi_samples_path,
        str(multihit["artifacts"][multi_samples_path.name]["sha256"]),
        "multi-hit samples",
    )
    _require(
        horizontal["inputs"]["multihit_manifest_sha256"] == sha256_file(multihit_path),
        "horizontal audit is not tied to multi-hit audit",
    )
    rays_path = horizontal_dir / "clearance_horizontal_rays.npz"
    _bound_file(
        rays_path,
        str(horizontal["artifacts"][rays_path.name]["sha256"]),
        "horizontal blocker rays",
    )
    old_samples_sha = str(multihit["inputs"]["heightfield_samples_sha256"])
    _require(
        multihit["inputs"]["field_manifest_sha256"]
        == horizontal["inputs"]["field_manifest_sha256"],
        "legacy audits use different field manifests",
    )
    legacy_source_path = Path(multihit["inputs"]["field_build"]) / "manifest.json"
    legacy_source_sha = _bound_file(
        legacy_source_path,
        str(multihit["inputs"]["field_manifest_sha256"]),
        "legacy source manifest",
    )
    legacy_source = _json(legacy_source_path)
    _require(
        legacy_source["collision"]["samples_sha256"] == old_samples_sha,
        "legacy source manifest does not bind the audited samples",
    )
    _require(
        legacy_source["conversion"]["colored_intermediate_glb"]["sha256"] == cad_sha,
        "legacy source manifest does not bind the audited CAD",
    )
    _require(
        _coordinate_contract(legacy_source) == _coordinate_contract(source),
        "legacy and current CAD-to-heightfield coordinate contracts differ",
    )

    heightfield = load_heightfield(asset)
    spacing = float(horizontal["ray_contract"]["lane_spacing_m"])
    required_width = float(horizontal["robot_envelope"]["required_connected_clear_width_m"])
    targets = []
    with (
        np.load(multi_samples_path, allow_pickle=False) as samples,
        np.load(rays_path, allow_pickle=False) as rays,
    ):
        for part in TARGET_PARTS:
            prefix = f"part_{part}_"
            result = review_samples(
                heightfield,
                samples[prefix + "x_m"],
                samples[prefix + "y_m"],
                samples[prefix + "target_lower_z_m"],
                samples[prefix + "potential_candidate_mask"],
            )
            expected = next(row for row in multihit["targets"] if row["source_part_index"] == part)[
                "summary"
            ]["potential_candidate_ray_count"]
            _require(result["candidate_count"] == expected, f"part {part} candidate count changed")
            entries = {
                entry: review_entry_mask(
                    rays[prefix + entry + "_blocked_ray_mask"], spacing, required_width
                )
                for entry in ENTRY_NAMES
            }
            targets.append({"source_part_index": part, **result, "entries": entries})

    any_channel = any(
        entry["envelope_channel_observed"]
        for target in targets
        for entry in target["entries"].values()
    )
    reasons = [
        "legacy horizontal rays show no axis-aligned robot-envelope entry"
        if not any_channel
        else "a sampled entry alone does not prove a swept or curved route",
        "facility semantics and physical traversability remain unverified",
        "hybrid collision contact ownership has not been established",
    ]
    return {
        "artifact_type": "rmuc2026_read_only_clearance_review",
        "status": "BLOCKED",
        "integrity_status": "PASS",
        "collision_changed": False,
        "validation_status": asset.manifest["validation_status"],
        "pack_manifest_sha256": asset.manifest_sha256,
        "current_1cm_samples_sha256": current_samples_sha,
        "source_manifest_sha256": source_sha,
        "legacy_source_manifest_sha256": legacy_source_sha,
        "coordinate_contract_match": True,
        "cad_glb_sha256": cad_sha,
        "legacy_2cm_samples_sha256": old_samples_sha,
        "cross_resolution_binding": (
            "legacy 2 cm CAD ray and blocker masks share the exact CAD GLB hash; "
            "only their heightfield seal check is replayed against this verified 1 cm pack"
        ),
        "multihit_manifest_sha256": sha256_file(multihit_path),
        "horizontal_manifest_sha256": sha256_file(horizontal_path),
        "targets": targets,
        "axis_aligned_envelope_channel_observed": any_channel,
        "blocking_reasons": reasons,
        "claim_boundary": (
            "This read-only result neither verifies diagonal or curved access nor authorizes "
            "underpass collision, route traversal, or a change to DRAFT_BLOCKED."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--multihit-audit", type=Path, required=True)
    parser.add_argument("--horizontal-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = review_clearance(
            args.pack, args.source_manifest, args.multihit_audit, args.horizontal_audit
        )
    except (ClearanceReviewError, KeyError, OSError, ValueError, Rmuc2026Error) as exc:
        result = {
            "artifact_type": "rmuc2026_read_only_clearance_review",
            "status": "BLOCKED",
            "integrity_status": "FAIL",
            "blocking_reasons": [str(exc)],
        }
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        try:
            _require(
                not args.output.exists(), "output already exists; review will not overwrite it"
            )
            args.output.write_text(payload, encoding="utf-8")
        except (ClearanceReviewError, OSError) as exc:
            print(
                json.dumps(
                    {
                        "artifact_type": "rmuc2026_read_only_clearance_review",
                        "status": "BLOCKED",
                        "integrity_status": "FAIL",
                        "blocking_reasons": [str(exc)],
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
    print(payload, end="")
    return 0 if result["integrity_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
