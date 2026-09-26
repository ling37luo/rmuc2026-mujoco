from __future__ import annotations

from pathlib import Path

from rmuc2026_mujoco import FieldAsset
from rmuc2026_mujoco.acceptance_routes import (
    _fence_routes,
    _source_wall_routes,
    build_preset_routes,
)


def test_fence_presets_cover_four_unobstructed_panel_centers() -> None:
    names = ("left", "right", "bottom", "top")
    positions = ((-14, 0, 1), (14, 0, 1), (0, -7.5, 1), (0, 7.5, 1))
    contract = {
        "panels": [
            {"name": f"rmuc2026_perimeter_{name}", "pos": list(pos)}
            for name, pos in zip(names, positions)
        ]
    }
    routes = _fence_routes(contract)

    assert len(routes) == 4
    assert {route["id"] for route in routes} == {
        "fence_left",
        "fence_right",
        "fence_bottom",
        "fence_top",
    }
    assert all(route["expect"] == "block" for route in routes)
    assert all(route["directions"] == ["forward"] for route in routes)
    assert routes[0]["blocking_geom"] == "rmuc2026_perimeter_left"


def test_source_wall_presets_require_declared_mesh_and_bounds() -> None:
    layer = {
        "mesh_geoms": [
            {"source_part_index": part, "name": f"rmuc2026_official_wall_{part}"}
            for part in (402, 403)
        ],
        "ownership_regions": [
            {
                "source_part_index": part,
                "source_bounds_world_m": [[part / 100, 0, 0], [part / 100 + 0.1, 1, 1]],
            }
            for part in (402, 403)
        ],
    }
    routes = _source_wall_routes(layer)

    assert [route["id"] for route in routes] == ["official_wall_402", "official_wall_403"]
    assert routes[0]["blocking_geom"] == "rmuc2026_official_wall_402"
    assert routes[0]["directions"] == ["forward", "reverse"]
    assert routes[0]["start_xy_m"][0] < 4.02
    assert routes[0]["end_xy_m"][0] > 4.12


def test_complete_presets_bind_pack_and_omit_unlocated_stairs(
    field_asset_dir: Path, monkeypatch
) -> None:
    asset = FieldAsset.open(field_asset_dir, verify=True)
    monkeypatch.setattr(
        "rmuc2026_mujoco.acceptance_routes.slope_catalog",
        lambda _asset: {
            "patches": [
                {
                    "band_id": "slope_6_10",
                    "route": {
                        "route_id": "slope_6_10_00",
                        "low_xyz_m": [0.0, 0.0, 0.0],
                        "high_xyz_m": [0.5, 0.0, 0.1],
                    },
                }
            ]
        },
    )
    monkeypatch.setattr(
        "rmuc2026_mujoco.acceptance_routes.fly_route_descriptor",
        lambda _asset, scenario: {
            "approach_xyz_m": [0.0, 0.0, 0.0],
            "takeoff_xyz_m": [0.8, 0.0, 0.2],
        },
    )

    result = build_preset_routes(asset)

    assert result["source_manifest_sha256"] == asset.manifest_sha256
    assert result["route_count"] == 3
    assert all(route["expect"] == "traverse" for route in result["routes"])
    assert {item["id"] for item in result["omitted"]} >= {
        "perimeter",
        "official_wall_402_403",
        "stairs_basic",
    }
