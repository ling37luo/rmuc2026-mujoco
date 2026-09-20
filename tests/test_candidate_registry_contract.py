"""The candidate registry is source-bound evidence, never active collision."""

from __future__ import annotations

from copy import deepcopy

import pytest

from rmuc2026_mujoco.candidate_registry_contract import (
    REGISTRY_ACTIVATION,
    REGISTRY_ARTIFACT_TYPE,
    REGISTRY_COORDINATE_FRAME,
    REGISTRY_STATUS,
    combine_candidate_registries,
    validate_candidate_registry,
)


MANIFEST_SHA = "a" * 64
STEP_SHA = "b" * 64
GLB_SHA = "c" * 64
SAMPLES_SHA = "d" * 64


def _registry() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": REGISTRY_ARTIFACT_TYPE,
        "source_manifest_sha256": MANIFEST_SHA,
        "official_step_sha256": STEP_SHA,
        "source_glb_sha256": GLB_SHA,
        "collision_samples_sha256": SAMPLES_SHA,
        "coordinate_frame": REGISTRY_COORDINATE_FRAME,
        "status": REGISTRY_STATUS,
        "activation": REGISTRY_ACTIVATION,
        "candidates": [
            {
                "id": "synthetic_wall_1",
                "category": "wall",
                "source_part_indices": [1, 2],
                "bounds_world_m": [[0.0, 0.0, 0.0], [1.0, 0.1, 0.5]],
                "blocking_reasons": ["HFIELD_OWNERSHIP_UNVERIFIED"],
                "evidence": {"note": "invented geometry for contract test"},
            }
        ],
    }


def _validate(payload: object) -> dict[str, object]:
    return validate_candidate_registry(
        payload,
        source_manifest_sha256=MANIFEST_SHA,
        official_step_sha256=STEP_SHA,
        source_glb_sha256=GLB_SHA,
        collision_samples_sha256=SAMPLES_SHA,
    )


def test_audit_only_source_bound_registry_is_accepted() -> None:
    candidate = _registry()
    assert _validate(candidate) == candidate


def test_independent_wall_and_boundary_evidence_can_be_combined() -> None:
    wall = _registry()
    boundary = deepcopy(wall)
    boundary["candidates"][0]["id"] = "synthetic_boundary_1"
    boundary["candidates"][0]["category"] = "boundary"
    combined = combine_candidate_registries(
        [wall, boundary],
        source_manifest_sha256=MANIFEST_SHA,
        official_step_sha256=STEP_SHA,
        source_glb_sha256=GLB_SHA,
        collision_samples_sha256=SAMPLES_SHA,
    )
    assert combined["activation"] == "disabled"
    assert [item["id"] for item in combined["candidates"]] == [
        "synthetic_wall_1",
        "synthetic_boundary_1",
    ]


def test_combination_rejects_duplicate_candidates() -> None:
    with pytest.raises(ValueError, match="duplicate collision candidate id"):
        combine_candidate_registries(
            [_registry(), _registry()],
            source_manifest_sha256=MANIFEST_SHA,
            official_step_sha256=STEP_SHA,
            source_glb_sha256=GLB_SHA,
            collision_samples_sha256=SAMPLES_SHA,
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("activation",), "enabled", "audit-only"),
        (("schema_version",), 1.0, "schema"),
        (("status",), "PASS", "audit-only"),
        (("source_manifest_sha256",), "d" * 64, "stale"),
        (("official_step_sha256",), "d" * 64, "stale"),
        (("source_glb_sha256",), "d" * 64, "stale"),
        (("collision_samples_sha256",), "e" * 64, "stale"),
        (("coordinate_frame",), "raw_step", "coordinate frame"),
        (("candidates", 0, "source_part_indices"), [], "source_part_indices"),
        (("candidates", 0, "category"), [], "category"),
        (("candidates", 0, "source_part_indices"), [1, 1], "source_part_indices"),
        (("candidates", 0, "bounds_world_m"), [[0, 0, 0], [0, 1, 1]], "positive extent"),
        (("candidates", 0, "blocking_reasons"), [], "blocking_reasons"),
        (("candidates", 0, "evidence"), {"gap_m": float("nan")}, "finite JSON"),
    ],
)
def test_registry_rejects_activation_stale_sources_or_unbounded_claims(
    path: tuple[str | int, ...], value: object, message: str
) -> None:
    payload = deepcopy(_registry())
    node = payload
    for key in path[:-1]:
        node = node[key]  # type: ignore[index]
    node[path[-1]] = value  # type: ignore[index]
    with pytest.raises(ValueError, match=message):
        _validate(payload)
