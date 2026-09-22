"""Framework-neutral input for an optional Isaac/Isaac Lab adapter.

This module deliberately imports no Isaac package.  It exposes the verified
heightfield and identity metadata an external Isaac consumer needs to build a
high-parallel environment without changing the field geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import FieldAsset
from .query import field_bounds, load_heightfield
from .scenarios import scenario_descriptor
from .turning import screen_turn_spawns, turn_phase_registry


@dataclass(frozen=True)
class IsaacHeightfieldInput:
    height_m: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    bounds_xy_m: tuple[tuple[float, float], tuple[float, float]]
    scenario: dict[str, Any]
    spawn_points: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.height_m.shape),
            "bounds_xy_m": [list(row) for row in self.bounds_xy_m],
            "scenario": self.scenario,
            "spawn_points": [dict(point) for point in self.spawn_points],
            "source": "verified_runtime_pack_heightfield",
        }


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
    return IsaacHeightfieldInput(
        height_m=np.asarray(samples.height_m, dtype=np.float32),
        x_m=np.asarray(samples.x_m, dtype=np.float32),
        y_m=np.asarray(samples.y_m, dtype=np.float32),
        bounds_xy_m=field_bounds(field),
        scenario=descriptor,
        spawn_points=spawn_points,
    )


__all__ = ["IsaacHeightfieldInput", "load_isaac_heightfield"]
