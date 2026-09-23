from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from rmuc2026_mujoco import FieldAsset, MujocoModelError
from rmuc2026_mujoco.jump_probe import (
    FREE_FLIGHT_SPEEDS_M_S,
    JumpProbeConfig,
    LandingProfile,
    detect_landing_profile,
    run_fixed_fly_ramp_jump_probe,
)
from rmuc2026_mujoco.ramp_audit import FlyRampGeometry
from rmuc2026_mujoco.wheel_probe import (
    PreparedWheelProbe,
    WheelProbeConfig,
    WheelSupportProfile,
)


def test_detect_landing_profile_measures_gap_face_and_top() -> None:
    along = np.arange(1.30, 2.02, 0.001)
    height = np.full_like(along, -0.01)
    height[along >= 1.83] = 0.17

    profile = detect_landing_profile(
        along,
        height,
        takeoff_seam_along_m=1.18,
    )

    assert profile.landing_edge_along_m == pytest.approx(1.83, abs=0.0011)
    assert profile.measured_gap_m == pytest.approx(0.65, abs=0.0011)
    assert profile.gap_floor_height_m == pytest.approx(-0.01)
    assert profile.landing_top_height_m == pytest.approx(0.17)


def test_detect_landing_profile_rejects_missing_raised_landing() -> None:
    along = np.arange(1.30, 2.02, 0.001)
    height = np.full_like(along, -0.01)

    with pytest.raises(MujocoModelError, match="could not find the landing face"):
        detect_landing_profile(
            along,
            height,
            takeoff_seam_along_m=1.18,
        )


def test_jump_probe_matrix_separates_low_speed_outcomes_from_required_gate(
    field_asset_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = FieldAsset.open(field_asset_dir)
    model = mujoco.MjModel.from_xml_string(
        """<mujoco>
  <option timestep=".001"/>
  <worldbody><body><freejoint/><geom type="sphere" size=".05" mass="1"/></body></worldbody>
</mujoco>"""
    )
    support = WheelSupportProfile(
        along_m=np.asarray([-0.3, 1.4]),
        center_height_m=np.asarray([0.06, 0.06]),
    )

    def fake_measure(
        _asset: FieldAsset,
        ramp: FlyRampGeometry,
        *,
        config: JumpProbeConfig,
    ) -> LandingProfile:
        assert config.official_gap_m == 0.65
        return LandingProfile(
            takeoff_seam_along_m=ramp.cad_high_seam_along_m,
            landing_edge_along_m=ramp.cad_high_seam_along_m + 0.65,
            measured_gap_m=0.65,
            gap_floor_height_m=0.0,
            landing_top_height_m=0.2,
        )

    def fake_build(
        _asset: FieldAsset,
        ramp: FlyRampGeometry,
        *,
        runtime_profile: str,
        config: WheelProbeConfig,
    ) -> PreparedWheelProbe:
        assert runtime_profile == "collision_only"
        config.validated()
        return PreparedWheelProbe(
            model=model,
            data=mujoco.MjData(model),
            ramp=ramp,
            support_profile=support,
            manifest_sha256=asset.manifest_sha256,
            runtime_profile=runtime_profile,
        )

    calls: list[tuple[str, float]] = []

    def fake_trial(
        _model: mujoco.MjModel,
        _data: mujoco.MjData,
        *,
        ramp: FlyRampGeometry,
        speed_m_s: float,
        jump_config: JumpProbeConfig,
        **_kwargs: object,
    ) -> dict[str, object]:
        calls.append((ramp.route_id, speed_m_s))
        landed = speed_m_s >= jump_config.required_success_speed_m_s
        return {
            "trial_id": f"{ramp.route_id}:free_flight:{speed_m_s:.1f}",
            "outcome": "LANDED_ON_TOP" if landed else "SHORT_OR_LIP_IMPACT",
            "required_to_land": landed,
            "status": "PASS",
            "solver_warnings": {},
            "checks": {
                "finite_state": True,
                "finite_contact_forces": True,
                "no_solver_warning": True,
                "became_airborne": True,
                "first_recontact_detected": True,
                "stable_post_landing_observation": landed,
            },
        }

    monkeypatch.setattr("rmuc2026_mujoco.jump_probe.measure_runtime_landing_profile", fake_measure)
    monkeypatch.setattr("rmuc2026_mujoco.jump_probe.build_wheel_probe_model", fake_build)
    monkeypatch.setattr("rmuc2026_mujoco.jump_probe.run_jump_probe_trial", fake_trial)
    output = tmp_path / "jump-probe.json"

    result = run_fixed_fly_ramp_jump_probe(asset, output_path=output)

    assert result["status"] == "PASS"
    assert result["summary"] == {
        "dimension_records_passed": 2,
        "dimension_record_count": 2,
        "landed_on_top_trials": 4,
        "short_or_lip_impact_trials": 6,
        "required_speed_trials_passed": 4,
        "required_speed_trial_count": 4,
        "solver_warning_count": 0,
    }
    assert len(calls) == 2 * len(FREE_FLIGHT_SPEEDS_M_S)
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_jump_probe_configuration_rejects_unordered_intervals() -> None:
    with pytest.raises(ValueError, match="landing-top sample interval"):
        JumpProbeConfig(
            landing_top_inset_start_m=0.2,
            landing_top_inset_end_m=0.1,
        ).validated()
