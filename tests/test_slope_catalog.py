from __future__ import annotations

import numpy as np
import pytest

from rmuc2026_mujoco import HeightFieldData, catalog_slope_patches, get_scenario


def test_ordinary_slope_catalog_fits_a_planar_traversal_patch() -> None:
    axis = np.linspace(-3.0, 3.0, 61)
    xx, _ = np.meshgrid(axis, axis)
    height = np.tan(np.deg2rad(9.0)) * xx
    data = HeightFieldData(x_m=axis, y_m=axis, height_m=height)

    patches = catalog_slope_patches(data, max_per_band=1, sampling_m=0.20)

    assert len(patches) == 1
    patch = patches[0]
    assert patch.band_id == "slope_6_10"
    assert 8.5 <= patch.slope_angle_deg < 9.5
    assert patch.height_gain_m > 0.1
    assert patch.fit_residual_p95_m < 1.0e-8
    assert patch.topology_verified is False


def test_slope_catalog_rejects_bad_limits() -> None:
    axis = np.linspace(-1.0, 1.0, 11)
    data = HeightFieldData(x_m=axis, y_m=axis, height_m=np.zeros((11, 11)))

    with pytest.raises(ValueError, match="max_per_band"):
        catalog_slope_patches(data, max_per_band=0)


def test_slope_basic_excludes_dedicated_fly_ramps() -> None:
    spec = get_scenario("slope_basic")

    assert spec.region["fly_ramps_excluded"] == ["fly_ramp_north", "fly_ramp_south"]
    assert spec.region["slope_bands_deg"][-1] == [10.0, 15.0]
