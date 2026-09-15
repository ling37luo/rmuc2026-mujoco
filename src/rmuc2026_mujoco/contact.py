"""Local, opt-in contact refinements; no CAD assets or material claims included."""

from __future__ import annotations

import numpy as np


def refine_planar_ramps(
    x_m: np.ndarray,
    y_m: np.ndarray,
    height_m: np.ndarray,
    geometries: list[dict],
    *,
    inset_m: float = 0.15,
    blend_m: float = 0.06,
    maximum_correction_m: float = 0.03,
) -> tuple[np.ndarray, dict]:
    """Fit only inset ramp interiors to supplied, independently audited planes.

    Keep the original heightfield as the sole collision owner. A transition
    band fades corrections to zero before the footprint boundary; no mesh or
    second surface is added. Source arrays are never modified. The caller must
    bind plane records to the exact field and independently test the candidate.
    """
    x, y, source = (np.asarray(a, dtype=np.float64) for a in (x_m, y_m, height_m))
    if (
        x.ndim != 1
        or y.ndim != 1
        or len(x) < 2
        or len(y) < 2
        or source.shape != (len(y), len(x))
        or not all(np.isfinite(a).all() for a in (x, y, source))
        or not np.all(np.diff(x) > 0)
        or not np.all(np.diff(y) > 0)
    ):
        raise ValueError("finite increasing axes and matching height samples are required")
    if not all(np.isfinite(v) and v > 0 for v in (inset_m, blend_m, maximum_correction_m)):
        raise ValueError("refinement limits must be finite and positive")
    output = source.copy()
    occupied = np.zeros(source.shape, dtype=bool)
    records = []
    for geometry in geometries:
        low = np.asarray(geometry["low_edge_center_xyz_m"], dtype=float)
        normal = np.asarray(geometry["normal_xyz"], dtype=float)
        uphill = np.asarray(geometry["uphill_unit_xy"], dtype=float)
        length = float(geometry["horizontal_run_m"])
        width = float(geometry["surface_width_m"])
        if (
            low.shape != (3,)
            or normal.shape != (3,)
            or uphill.shape != (2,)
            or not all(np.isfinite(a).all() for a in (low, normal, uphill))
            or not np.isfinite([length, width]).all()
            or not np.isclose(np.linalg.norm(uphill), 1.0, atol=1e-6)
            or not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-6)
            or normal[2] <= 0
            or min(length, width) <= 2 * (inset_m + blend_m)
        ):
            raise ValueError("invalid audited ramp geometry")
        dx, dy = x[None, :] - low[0], y[:, None] - low[1]
        along = dx * uphill[0] + dy * uphill[1]
        lateral = -dx * uphill[1] + dy * uphill[0]
        clearance = np.minimum(np.minimum(along, length - along), width / 2 - abs(lateral))
        weight = np.clip((clearance - inset_m) / blend_m, 0.0, 1.0)
        mask = weight > 0
        if not np.any(mask) or np.any(mask & occupied):
            raise ValueError("empty or overlapping ramp refinement footprints")
        plane = low[2] - (normal[0] * dx + normal[1] * dy) / normal[2]
        correction = plane - source
        if np.max(abs(correction[mask])) > maximum_correction_m:
            raise ValueError("ramp disagreement exceeds the permitted correction")
        output[mask] = source[mask] + weight[mask] * correction[mask]
        occupied |= mask
        core = weight == 1
        if not np.any(core):
            raise ValueError("ramp has no fully resolved interior")
        records.append(
            {
                "changed_footprint_samples": int(mask.sum()),
                "core_samples": int(core.sum()),
                "before_core_error_max_m": float(np.max(abs(correction[core]))),
                "after_core_error_max_m": float(np.max(abs(output[core] - plane[core]))),
                "maximum_applied_correction_m": float(np.max(abs(output[mask] - source[mask]))),
            }
        )
    return output, {
        "status": "CANDIDATE",
        "profile": "audited_planes_v1",
        "records": records,
        "inset_m": inset_m,
        "blend_m": blend_m,
        "outside_footprints_bitwise_equal": np.array_equal(output[~occupied], source[~occupied]),
        "collision_owner": "existing heightfield only",
        "claim_boundary": "inset static ramp interiors only; edge, flight, and robot traversal require evaluation",
    }
