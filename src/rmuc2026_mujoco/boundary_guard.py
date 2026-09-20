"""Bounded stop signal for robots approaching unsupported field edges.

This monitor reads a verified schema-3 source-miss mask. It never edits the
field, a robot, contacts, or control. Consumers must stop stepping when it
returns a record; a stop does not make the edge physically traversable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from .errors import ManifestError
from .manifest import FieldAsset


@dataclass
class FieldBoundaryGuard:
    """Watch source misses and saturated heightfield contacts near an outer edge.

    `observe` must be called once before each physics step. The caller supplies
    only measured state and contact counts, so this class does not depend on a
    particular robot model or import MuJoCo into the package's core API.
    """

    x_m: np.ndarray
    y_m: np.ndarray
    source_miss_mask: np.ndarray
    edge_band_m: float
    contact_cap_edge_margin_m: float = 0.4
    source_miss_steps: int = 20
    contact_cap_steps: int = 4
    contact_cap: int = 50
    minimum_outward_speed_m_s: float = 0.05
    _last_xy: tuple[float, float] | None = field(default=None, init=False, repr=False)
    _last_time_s: float | None = field(default=None, init=False, repr=False)
    _miss_count: int = field(default=0, init=False, repr=False)
    _cap_count: int = field(default=0, init=False, repr=False)
    stop_record: dict[str, Any] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.x_m = np.array(self.x_m, dtype=np.float64, copy=True)
        self.y_m = np.array(self.y_m, dtype=np.float64, copy=True)
        self.source_miss_mask = np.array(self.source_miss_mask, copy=True)
        if (
            self.x_m.ndim != 1
            or self.y_m.ndim != 1
            or len(self.x_m) < 2
            or len(self.y_m) < 2
            or not np.isfinite(self.x_m).all()
            or not np.isfinite(self.y_m).all()
            or not np.all(np.diff(self.x_m) > 0)
            or not np.all(np.diff(self.y_m) > 0)
            or self.source_miss_mask.shape != (len(self.y_m), len(self.x_m))
            or self.source_miss_mask.dtype != np.bool_
            or not math.isfinite(self.edge_band_m)
            or self.edge_band_m <= 0
            or not math.isfinite(self.contact_cap_edge_margin_m)
            or not 0 < self.contact_cap_edge_margin_m <= self.edge_band_m
            or self.source_miss_steps < 1
            or self.contact_cap_steps < 1
            or self.contact_cap < 1
            or not math.isfinite(self.minimum_outward_speed_m_s)
            or self.minimum_outward_speed_m_s < 0
        ):
            raise ValueError("invalid field boundary guard contract")
        self.x_m.flags.writeable = False
        self.y_m.flags.writeable = False
        self.source_miss_mask.flags.writeable = False

    @classmethod
    def from_asset(cls, asset: FieldAsset) -> FieldBoundaryGuard:
        """Bind the guard to a fully hash-verified schema-3 runtime pack."""

        if not asset.hashes_verified or asset.manifest.get("schema_version") != 3:
            raise ManifestError("boundary guard requires a hash-verified schema-3 pack")
        collision = asset.collision
        provenance = collision["edge_void_provenance"]
        spawn = asset.recommended_spawn
        with np.load(asset.file(str(collision["samples_file"])), allow_pickle=False) as samples:
            x = np.asarray(samples["x_m"], dtype=np.float64) - float(
                spawn["x_before_translation_m"]
            )
            y = np.asarray(samples["y_m"], dtype=np.float64) - float(
                spawn["y_before_translation_m"]
            )
        with np.load(asset.file(str(provenance["mask_file"])), allow_pickle=False) as masks:
            miss = np.asarray(masks["changed_mask"], dtype=np.bool_)
        return cls(x, y, miss, edge_band_m=float(provenance["edge_band_m"]))

    def reset(self) -> None:
        """Clear persistence after a robot reset without changing its physical state."""

        self._last_xy = None
        self._last_time_s = None
        self._miss_count = 0
        self._cap_count = 0
        self.stop_record = None

    def _over_source_miss(self, x: float, y: float) -> bool:
        if x < self.x_m[0] or x > self.x_m[-1] or y < self.y_m[0] or y > self.y_m[-1]:
            return True
        column = int(np.clip(np.searchsorted(self.x_m, x), 1, len(self.x_m) - 1))
        row = int(np.clip(np.searchsorted(self.y_m, y), 1, len(self.y_m) - 1))
        if x - self.x_m[column - 1] < self.x_m[column] - x:
            column -= 1
        if y - self.y_m[row - 1] < self.y_m[row] - y:
            row -= 1
        return bool(self.source_miss_mask[row, column])

    def _outward_edge(self, x: float, y: float, vx: float, vy: float) -> tuple[str, float] | None:
        edges = (
            ("left", x - float(self.x_m[0]), -vx),
            ("right", float(self.x_m[-1]) - x, vx),
            ("bottom", y - float(self.y_m[0]), -vy),
            ("top", float(self.y_m[-1]) - y, vy),
        )
        eligible = [
            (name, speed, distance)
            for name, distance, speed in edges
            if distance <= self.contact_cap_edge_margin_m
            and speed >= self.minimum_outward_speed_m_s
        ]
        if not eligible:
            return None
        name, speed, _distance = min(eligible, key=lambda item: item[2])
        return name, speed

    def observe(
        self,
        *,
        sim_time_s: float,
        base_xy_m: tuple[float, float],
        field_contact_count: int,
        max_field_pair_contacts: int,
    ) -> dict[str, Any] | None:
        """Return a stop record when either bounded edge rule fires."""

        if self.stop_record is not None:
            return self.stop_record
        x, y = map(float, base_xy_m)
        t = float(sim_time_s)
        if (
            not all(map(math.isfinite, (x, y, t)))
            or field_contact_count < 0
            or max_field_pair_contacts < 0
            or max_field_pair_contacts > field_contact_count
        ):
            raise ValueError("boundary observation is invalid")
        if self._last_time_s is not None and t <= self._last_time_s:
            if t == self._last_time_s:
                return None
            raise ValueError("boundary observation time moved backwards; call reset")
        outward = None
        if self._last_xy is not None and self._last_time_s is not None:
            dt = t - self._last_time_s
            outward = self._outward_edge(
                x, y, (x - self._last_xy[0]) / dt, (y - self._last_xy[1]) / dt
            )
        self._last_xy = (x, y)
        self._last_time_s = t

        self._miss_count = (
            self._miss_count + 1 if self._over_source_miss(x, y) and field_contact_count == 0 else 0
        )
        self._cap_count = (
            self._cap_count + 1
            if outward is not None and max_field_pair_contacts >= self.contact_cap
            else 0
        )
        reason = None
        if self._miss_count >= self.source_miss_steps:
            reason = "source_miss_without_field_contact"
        elif self._cap_count >= self.contact_cap_steps:
            reason = "outward_edge_heightfield_contact_saturation"
        if reason is not None:
            self.stop_record = {
                "reason": reason,
                "sim_time_s": t,
                "base_xy_m": [x, y],
                "field_contact_count": field_contact_count,
                "max_field_pair_contacts": max_field_pair_contacts,
                "consecutive_source_miss_steps": self._miss_count,
                "consecutive_contact_cap_steps": self._cap_count,
                "outward_edge": outward[0] if outward is not None else None,
                "outward_speed_m_s": outward[1] if outward is not None else None,
                "physical_state_modified": False,
            }
        return self.stop_record
