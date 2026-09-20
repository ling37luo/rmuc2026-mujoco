"""Contract for local, non-activated CAD collision candidate evidence.

The registry records places that need contact work.  It is deliberately not a
source of MuJoCo geoms: a source-bound bounding box is not proof that the
heightfield can relinquish contact ownership at that location.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from typing import Any


REGISTRY_ARTIFACT_TYPE = "rmuc2026_collision_candidate_registry"
REGISTRY_COORDINATE_FRAME = "translated_m_z_up"
REGISTRY_STATUS = "AUDIT_ONLY"
REGISTRY_ACTIVATION = "disabled"
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,79}\Z")


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _vector(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must contain three coordinates")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{label} must contain numeric coordinates")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite coordinates")
    return result  # type: ignore[return-value]


def validate_candidate_registry(
    payload: object,
    *,
    source_manifest_sha256: str,
    official_step_sha256: str,
    source_glb_sha256: str,
    collision_samples_sha256: str,
) -> dict[str, Any]:
    """Validate a registry without implying any candidate is safe to activate."""

    if not isinstance(payload, dict):
        raise ValueError("collision candidate registry must be a JSON object")
    required = {
        "schema_version",
        "artifact_type",
        "source_manifest_sha256",
        "official_step_sha256",
        "source_glb_sha256",
        "collision_samples_sha256",
        "coordinate_frame",
        "status",
        "activation",
        "candidates",
    }
    if set(payload) != required:
        raise ValueError("collision candidate registry has missing or unsupported fields")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported collision candidate registry schema")
    if payload["artifact_type"] != REGISTRY_ARTIFACT_TYPE:
        raise ValueError("unsupported collision candidate registry artifact_type")
    for key, expected in (
        ("source_manifest_sha256", source_manifest_sha256),
        ("official_step_sha256", official_step_sha256),
        ("source_glb_sha256", source_glb_sha256),
        ("collision_samples_sha256", collision_samples_sha256),
    ):
        if _sha(payload[key], key) != _sha(expected, f"expected {key}"):
            raise ValueError(f"collision candidate registry {key} is stale or mismatched")
    if payload["coordinate_frame"] != REGISTRY_COORDINATE_FRAME:
        raise ValueError("unsupported collision candidate coordinate frame")
    if payload["status"] != REGISTRY_STATUS or payload["activation"] != REGISTRY_ACTIVATION:
        raise ValueError("collision candidates must remain audit-only and disabled")
    candidates = payload["candidates"]
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 256:
        raise ValueError("collision candidate registry must contain 1 to 256 candidates")
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        label = f"candidates[{index}]"
        if not isinstance(candidate, dict):
            raise ValueError(f"{label} must be an object")
        allowed = {
            "id",
            "category",
            "source_part_indices",
            "bounds_world_m",
            "blocking_reasons",
            "evidence",
        }
        required_candidate = allowed - {"evidence"}
        if not required_candidate <= set(candidate) or not set(candidate) <= allowed:
            raise ValueError(f"{label} has missing or unsupported fields")
        identifier = candidate["id"]
        if not isinstance(identifier, str) or _IDENTIFIER.fullmatch(identifier) is None:
            raise ValueError(f"{label}.id is invalid")
        if identifier in seen:
            raise ValueError(f"duplicate collision candidate id: {identifier}")
        seen.add(identifier)
        if not isinstance(candidate["category"], str) or candidate["category"] not in {
            "wall",
            "boundary",
        }:
            raise ValueError(f"{label}.category must be wall or boundary")
        parts = candidate["source_part_indices"]
        if (
            not isinstance(parts, list)
            or not parts
            or len(parts) > 256
            or any(
                isinstance(part, bool) or not isinstance(part, int) or part < 0 for part in parts
            )
            or len(parts) != len(set(parts))
        ):
            raise ValueError(f"{label}.source_part_indices must be distinct nonnegative integers")
        bounds = candidate["bounds_world_m"]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"{label}.bounds_world_m must have two corners")
        low = _vector(bounds[0], f"{label}.bounds_world_m[0]")
        high = _vector(bounds[1], f"{label}.bounds_world_m[1]")
        if any(upper <= lower for lower, upper in zip(low, high)):
            raise ValueError(f"{label}.bounds_world_m must have positive extent")
        reasons = candidate["blocking_reasons"]
        if (
            not isinstance(reasons, list)
            or not reasons
            or any(
                not isinstance(reason, str) or _IDENTIFIER.fullmatch(reason) is None
                for reason in reasons
            )
            or len(reasons) != len(set(reasons))
        ):
            raise ValueError(f"{label}.blocking_reasons must be distinct identifiers")
        evidence = candidate.get("evidence", {})
        if not isinstance(evidence, dict):
            raise ValueError(f"{label}.evidence must be an object")
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("collision candidate registry must be finite JSON") from exc
    return payload


def combine_candidate_registries(
    registries: Sequence[object],
    *,
    source_manifest_sha256: str,
    official_step_sha256: str,
    source_glb_sha256: str,
    collision_samples_sha256: str,
) -> dict[str, Any]:
    """Join independently audited regions without enabling any collision.

    Each input is validated against the *same* source identities first; the
    combined result is validated again to catch duplicate candidate IDs.
    """

    if not registries:
        raise ValueError("at least one collision candidate registry is required")
    candidates: list[dict[str, Any]] = []
    for registry in registries:
        verified = validate_candidate_registry(
            registry,
            source_manifest_sha256=source_manifest_sha256,
            official_step_sha256=official_step_sha256,
            source_glb_sha256=source_glb_sha256,
            collision_samples_sha256=collision_samples_sha256,
        )
        candidates.extend(verified["candidates"])
    combined: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": REGISTRY_ARTIFACT_TYPE,
        "source_manifest_sha256": source_manifest_sha256,
        "official_step_sha256": official_step_sha256,
        "source_glb_sha256": source_glb_sha256,
        "collision_samples_sha256": collision_samples_sha256,
        "coordinate_frame": REGISTRY_COORDINATE_FRAME,
        "status": REGISTRY_STATUS,
        "activation": REGISTRY_ACTIVATION,
        "candidates": candidates,
    }
    return validate_candidate_registry(
        combined,
        source_manifest_sha256=source_manifest_sha256,
        official_step_sha256=official_step_sha256,
        source_glb_sha256=source_glb_sha256,
        collision_samples_sha256=collision_samples_sha256,
    )
