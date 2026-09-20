"""Optional dynamic collision proxy for the rulebook's movable energy unit.

The official drawing fixes the outer envelope and approximate mass, but not
the complete collision mesh or friction.  This primitive approximation is
therefore opt-in and is never silently installed into a runtime field pack.
"""

from __future__ import annotations

import math
from typing import Any

from .errors import MujocoModelError
from .manifest import FieldAsset
from .mjcf import HFIELD_NAME, inject_exact_heightfield
from .query import surface_at


RULEBOOK_VERSION = "V2.0.0"
RULEBOOK_SHA256 = "59d65aac5bac75fd6bbab157e224eb936b49cd2b9d60381fbe196fad635b473a"
RULEBOOK_PDF_PAGE = 64
OVERALL_HEIGHT_M = 0.150
SOLID_END_DIAMETER_M = 0.095
OPEN_END_DIAMETER_M = 0.080
NOMINAL_MASS_KG = 0.400
MASS_RANGE_KG = (0.350, 0.450)
DEFAULT_FRICTION = (1.0, 0.005, 0.0001)


def _finite_triplet(values: tuple[float, float, float], *, label: str) -> tuple[float, ...]:
    if len(values) != 3 or any(not math.isfinite(float(value)) for value in values):
        raise ValueError(f"{label} must contain three finite numbers")
    return tuple(float(value) for value in values)


def add_energy_unit(
    spec: Any,
    *,
    name: str,
    center_xyz_m: tuple[float, float, float],
    mass_kg: float = NOMINAL_MASS_KG,
    friction: tuple[float, float, float] = DEFAULT_FRICTION,
) -> dict[str, object]:
    """Add a free, asymmetric 400 g energy-unit proxy to a caller-owned MjSpec.

    The 95/80 mm ends and 150 mm overall height follow rulebook Figure 4-39.
    The twelve lower rim sectors and six ribs are a collision approximation;
    they preserve a central opening but are not a detailed gripper model.
    """

    if not isinstance(name, str) or not name or "/" in name or any(c.isspace() for c in name):
        raise ValueError("energy-unit name must be nonempty and contain no whitespace or slash")
    center = _finite_triplet(center_xyz_m, label="center_xyz_m")
    contact_friction = _finite_triplet(friction, label="friction")
    if not MASS_RANGE_KG[0] <= mass_kg <= MASS_RANGE_KG[1] or not math.isfinite(mass_kg):
        raise ValueError("energy-unit mass is outside the rulebook 400±50 g range")
    if any(value <= 0.0 for value in contact_friction):
        raise ValueError("energy-unit friction values must be positive")
    if not hasattr(spec, "worldbody"):
        raise ValueError("spec must be a MuJoCo MjSpec")

    import mujoco

    body = spec.worldbody.add_body(name=name, pos=list(center))
    body.add_freejoint(name=f"{name}_free")
    body.add_geom(
        name=f"{name}_solid_end",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0, 0.0, 0.060],
        size=[SOLID_END_DIAMETER_M / 2.0, 0.015],
        mass=mass_kg * 0.45,
        friction=contact_friction,
        rgba=[0.38, 0.40, 0.42, 1.0],
    )
    for index in range(12):
        angle = math.tau * index / 12.0
        body.add_geom(
            name=f"{name}_open_end_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.034 * math.cos(angle), 0.034 * math.sin(angle), -0.0625],
            quat=[math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)],
            size=[0.005, 0.009, 0.0125],
            mass=mass_kg * 0.25 / 12.0,
            friction=contact_friction,
            rgba=[0.38, 0.40, 0.42, 1.0],
        )
    for index in range(6):
        angle = math.tau * index / 6.0
        body.add_geom(
            name=f"{name}_rib_{index}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[0.031 * math.cos(angle), 0.031 * math.sin(angle), -0.0025],
            size=[0.0045, 0.0475],
            mass=mass_kg * 0.30 / 6.0,
            friction=contact_friction,
            rgba=[0.83, 0.78, 0.63, 1.0],
        )
    return {
        "status": "OPTIONAL_APPROXIMATE_DYNAMIC_PROXY",
        "body_name": name,
        "rulebook_version": RULEBOOK_VERSION,
        "rulebook_sha256": RULEBOOK_SHA256,
        "rulebook_pdf_page": RULEBOOK_PDF_PAGE,
        "rulebook_figure": "4-39",
        "mass_kg": mass_kg,
        "outer_height_m": OVERALL_HEIGHT_M,
        "solid_end_diameter_m": SOLID_END_DIAMETER_M,
        "open_end_diameter_m": OPEN_END_DIAMETER_M,
        "friction": list(contact_friction),
        "friction_is_official": False,
        "physical_geoms": 19,
        "gripper_clearance_validated": False,
        "claim_boundary": "outer envelope and nominal mass only; rim sectors/ribs and friction are proxies",
    }


def load_field_with_energy_unit(
    asset: FieldAsset,
    *,
    x_m: float,
    y_m: float,
    name: str = "rmuc2026_energy_unit",
    drop_clearance_m: float = 0.05,
    profile: str = "collision_only",
) -> tuple[Any, Any, dict[str, object]]:
    """Make a separate local field scene with one unit above screened flat terrain.

    The returned scene is not a runtime pack and does not rewrite its assets.
    This heightfield-only placement check does not establish free grasp space.
    """

    if not asset.hashes_verified:
        raise MujocoModelError("energy-unit placement requires a hash-verified field pack")
    if not math.isfinite(drop_clearance_m) or not 0.0 <= drop_clearance_m <= 0.5:
        raise ValueError("drop_clearance_m must be between 0 and 0.5 m")
    surface = surface_at(asset, x_m, y_m, window_radius_m=0.06)
    if surface.maximum_window_slope_deg > 5.0 or surface.local_relief_upper_bound_m > 0.01:
        raise MujocoModelError("energy-unit test pose needs locally flat 12 cm terrain")

    import mujoco

    spec = mujoco.MjSpec.from_file(str(asset.entrypoint_for(profile)))
    center = (float(x_m), float(y_m), surface.height_m + 0.075 + drop_clearance_m)
    record = add_energy_unit(spec, name=name, center_xyz_m=center)
    try:
        model = spec.compile()
    except (RuntimeError, ValueError) as exc:
        raise MujocoModelError(f"energy-unit field scene could not compile: {exc}") from exc
    inject_exact_heightfield(model, asset, hfield_name=HFIELD_NAME)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    record["field_manifest_sha256"] = asset.manifest_sha256
    record["initial_center_xyz_m"] = list(center)
    record["placement_source"] = "screened_single_heightfield_not_multilevel_verified"
    return model, data, record
