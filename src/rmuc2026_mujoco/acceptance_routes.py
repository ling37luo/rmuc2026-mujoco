"""Small, hash-bound route presets for field physics acceptance.

The JSON is generated locally from one verified runtime pack.  Routes are
wheel-probe approaches, not official robot trajectories.  Missing topology
evidence is recorded as omitted rather than represented by guessed points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .fly_routes import fly_route_descriptor
from .manifest import FieldAsset
from .slope_catalog import slope_catalog


def _fence_routes(contract: dict[str, Any]) -> list[dict[str, Any]]:
    panels = {record["name"].rsplit("_", 1)[-1]: record for record in contract["panels"]}
    if set(panels) != {"left", "right", "bottom", "top"}:
        raise ValueError("fence must declare the four expected panels")
    left_x = float(panels["left"]["pos"][0])
    right_x = float(panels["right"]["pos"][0])
    bottom_y = float(panels["bottom"]["pos"][1])
    top_y = float(panels["top"]["pos"][1])
    center_x = (left_x + right_x) / 2
    center_y = (bottom_y + top_y) / 2
    if not (left_x < center_x < right_x and bottom_y < center_y < top_y):
        raise ValueError("fence rectangle is invalid")

    def blocked(
        name: str,
        start: tuple[float, float],
        end: tuple[float, float],
        panel: str,
    ) -> dict:
        return {
            "id": name,
            "start_xy_m": list(start),
            "end_xy_m": list(end),
            "expect": "block",
            "blocking_geom": panels[panel]["name"],
            "directions": ["forward"],
            "basis": "manifest_perimeter_fence_panel_center",
        }

    return [
        blocked("fence_left", (left_x + 0.5, center_y), (left_x - 0.2, center_y), "left"),
        blocked("fence_right", (right_x - 0.5, center_y), (right_x + 0.2, center_y), "right"),
        blocked("fence_bottom", (center_x, bottom_y + 0.5), (center_x, bottom_y - 0.2), "bottom"),
        blocked("fence_top", (center_x, top_y - 0.5), (center_x, top_y + 0.2), "top"),
    ]


def _source_wall_routes(layer: dict[str, Any]) -> list[dict[str, Any]]:
    meshes = {record["source_part_index"]: record for record in layer["mesh_geoms"]}
    regions = {record["source_part_index"]: record for record in layer["ownership_regions"]}
    if set(meshes) != {402, 403} or set(regions) != {402, 403}:
        raise ValueError("source-contact layer must contain exactly official walls 402/403")
    routes = []
    for part in (402, 403):
        bounds = regions[part]["source_bounds_world_m"]
        lower, upper = ([float(value) for value in corner] for corner in bounds)
        if len(lower) != 3 or len(upper) != 3 or any(a >= b for a, b in zip(lower, upper)):
            raise ValueError(f"official wall {part} bounds are invalid")
        axis = 0 if upper[0] - lower[0] < upper[1] - lower[1] else 1
        center = [(lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2]
        start = center.copy()
        end = center.copy()
        start[axis] = lower[axis] - 0.3
        end[axis] = upper[axis] + 0.3
        routes.append(
            {
                "id": f"official_wall_{part}",
                "start_xy_m": start,
                "end_xy_m": end,
                "expect": "block",
                "blocking_geom": meshes[part]["name"],
                "directions": ["forward", "reverse"],
                "basis": "source_contact_layer_ownership_bounds",
            }
        )
    return routes


def build_preset_routes(pack: FieldAsset | str | Path) -> dict[str, Any]:
    """Build bounded route probes from declared field metadata and route audits."""

    asset = pack if isinstance(pack, FieldAsset) else FieldAsset.open(pack, verify=True)
    routes: list[dict[str, Any]] = []
    omitted: list[dict[str, str]] = []
    fence = asset.manifest.get("perimeter_fence")
    if fence is None:
        omitted.append({"id": "perimeter", "status": "UNVERIFIED", "reason": "no perimeter fence"})
    else:
        routes.extend(_fence_routes(fence))
        for x_side in ("left", "right"):
            for y_side in ("bottom", "top"):
                omitted.append(
                    {
                        "id": f"fence_corner_{x_side}_{y_side}",
                        "status": "UNVERIFIED_SOURCE_OBSTRUCTED",
                        "reason": (
                            "diagonal corner approach intersects official CAD obstacles, "
                            "including source parts 178/185 and 279-294/298-314; "
                            "it cannot isolate fence contact"
                        ),
                    }
                )

    source_layer = asset.collision.get("source_contact_layer")
    if source_layer is None:
        omitted.append(
            {
                "id": "official_wall_402_403",
                "status": "UNVERIFIED",
                "reason": "no activated source-contact layer in this pack",
            }
        )
    else:
        routes.extend(_source_wall_routes(source_layer))

    catalog = slope_catalog(asset)
    for band in ("slope_3_6", "slope_6_10", "slope_10_15"):
        candidate = next(
            (
                item
                for item in catalog["patches"]
                if item["band_id"] == band and item.get("route") is not None
            ),
            None,
        )
        if candidate is None:
            omitted.append(
                {"id": band, "status": "UNVERIFIED", "reason": "no screened continuous route"}
            )
            continue
        route = candidate["route"]
        routes.append(
            {
                "id": f"ordinary_{route['route_id']}",
                "start_xy_m": route["low_xyz_m"][:2],
                "end_xy_m": route["high_xyz_m"][:2],
                "expect": "traverse",
                "directions": ["forward", "reverse"],
                "basis": "heightfield_screened_ordinary_slope_route",
            }
        )

    for scenario in ("fly_ramp_north", "fly_ramp_south"):
        descriptor = fly_route_descriptor(asset, scenario)
        routes.append(
            {
                "id": f"{scenario}_approach_to_high_seam",
                "start_xy_m": descriptor["approach_xyz_m"][:2],
                "end_xy_m": descriptor["takeoff_xyz_m"][:2],
                "expect": "traverse",
                "directions": ["forward", "reverse"],
                "basis": "audited_fly_route_approach_to_takeoff",
            }
        )

    omitted.append(
        {
            "id": "stairs_basic",
            "status": "UNVERIFIED",
            "reason": (
                "no accepted stair-part semantic or hash-bound low/high approach route; "
                "the source stepped-support groups remain heuristic and can mix levels/underpasses"
            ),
        }
    )
    return {
        "schema_version": 1,
        "source_manifest_sha256": asset.manifest_sha256,
        "collision_samples_sha256": asset.collision["samples_sha256"],
        "route_count": len(routes),
        "routes": routes,
        "omitted": omitted,
        "claim_boundary": (
            "Wheel-centreline route candidates from verified metadata; dynamic outcomes remain "
            "unverified until the acceptance runner executes them. Fence placement is a proxy."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_preset_routes(args.pack)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"routes": result["route_count"], "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
