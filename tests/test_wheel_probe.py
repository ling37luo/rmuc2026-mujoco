from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from rmuc2026_mujoco import FieldAsset, MujocoModelError
from rmuc2026_mujoco.ramp_audit import FIXED_FLY_RAMPS, FlyRampGeometry
from rmuc2026_mujoco.wheel_probe import (
    PROBE_BODY_NAME,
    PROBE_PATH_JOINT_NAME,
    PROBE_SPIN_JOINT_NAME,
    PROBE_VERTICAL_JOINT_NAME,
    PROBE_WHEEL_GEOM_NAME,
    PreparedWheelProbe,
    WheelProbeConfig,
    WheelSupportProfile,
    run_fixed_fly_ramp_wheel_probe,
    run_wheel_probe_trial,
)


def _flat_ramp() -> FlyRampGeometry:
    return FlyRampGeometry(
        route_id="synthetic_flat_ramp",
        source_part_index=1,
        low_edge_center_xyz_m=(0.0, 0.0, 0.0),
        high_edge_center_xyz_m=(0.4, 0.0, 0.0),
        normal_xyz=(0.0, 0.0, 1.0),
        uphill_unit_xy=(1.0, 0.0),
        horizontal_run_m=0.4,
        surface_width_m=1.0,
        slope_angle_degrees=0.0,
        cad_low_seam_along_m=0.0,
        cad_high_seam_along_m=0.4,
    )


def _flat_probe_model(*, extra_heightfield: bool = False) -> tuple[mujoco.MjModel, mujoco.MjData]:
    extra_asset = (
        '<hfield name="unexpected_second_heightfield" nrow="2" ncol="2" size="1 1 .1 .05"/>'
        if extra_heightfield
        else ""
    )
    model = mujoco.MjModel.from_xml_string(
        f"""<mujoco>
  <option timestep=".001" solver="Newton"/>
  <asset>
    <hfield name="rmuc2026_collision" nrow="3" ncol="3" size="2 2 .1 .05"/>
    {extra_asset}
  </asset>
  <worldbody>
    <geom name="rmuc2026_field_collision" type="hfield" hfield="rmuc2026_collision"
          contype="2" conaffinity="1" friction="1 .005 .0001" solref=".02 1"/>
    <body name="{PROBE_BODY_NAME}">
      <joint name="{PROBE_PATH_JOINT_NAME}" type="slide" axis="1 0 0" damping=".02"/>
      <joint name="{PROBE_VERTICAL_JOINT_NAME}" type="slide" axis="0 0 1" damping=".02"/>
      <joint name="{PROBE_SPIN_JOINT_NAME}" type="hinge" axis="0 1 0" damping=".001"/>
      <geom name="{PROBE_WHEEL_GEOM_NAME}" type="cylinder"
            fromto="0 -.04 0 0 .04 0" size=".06" mass="2"
            friction="1 .005 .0001" condim="6"/>
    </body>
  </worldbody>
</mujoco>"""
    )
    return model, mujoco.MjData(model)


def test_model_probe_runs_finite_contact_dynamics_on_one_heightfield() -> None:
    model, data = _flat_probe_model()
    ramp = _flat_ramp()
    along = np.arange(-0.3, 0.6, 0.0025)
    support = WheelSupportProfile(along_m=along, center_height_m=np.full_like(along, 0.06))

    result = run_wheel_probe_trial(
        model,
        data,
        ramp=ramp,
        direction="uphill",
        speed_m_s=1.0,
        start_center_height_m=0.062,
        support_profile=support,
    )

    assert result["dynamic_status"] == "PASS"
    assert result["checks"] == {
        "reached_finish": True,
        "finite_state": True,
        "finite_contact_forces": True,
        "no_solver_warning": True,
        "no_pose_teleport": True,
        "no_stall": True,
        "no_edge_snag": True,
        "no_abnormal_bounce": True,
        "commanded_speed_tracked": True,
    }
    assert result["maximum_progress_m"] >= result["required_progress_m"]
    assert result["contact_step_count"] > 0
    assert result["solver_warnings"] == {
        "INERTIA": 0,
        "CONTACTFULL": 0,
        "CNSTRFULL": 0,
        "BADQPOS": 0,
        "BADQVEL": 0,
        "BADQACC": 0,
        "BADCTRL": 0,
    }
    json.dumps(result, allow_nan=False)


def test_model_probe_rejects_more_than_one_heightfield() -> None:
    model, data = _flat_probe_model(extra_heightfield=True)

    with pytest.raises(MujocoModelError, match="exactly one heightfield"):
        run_wheel_probe_trial(
            model,
            data,
            ramp=_flat_ramp(),
            direction="uphill",
            speed_m_s=0.5,
            start_center_height_m=0.062,
        )


def test_full_matrix_covers_twelve_trials_and_writes_json(
    field_asset_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = FieldAsset.open(field_asset_dir)
    model, data = _flat_probe_model()
    support = WheelSupportProfile(
        along_m=np.asarray([-0.3, 1.4]),
        center_height_m=np.asarray([0.06, 0.06]),
    )

    def fake_build(
        _asset: FieldAsset,
        ramp: FlyRampGeometry,
        *,
        runtime_profile: str,
        config: WheelProbeConfig,
    ) -> PreparedWheelProbe:
        assert runtime_profile == "collision_only"
        return PreparedWheelProbe(
            model=model,
            data=data,
            ramp=ramp,
            support_profile=support,
            manifest_sha256=asset.manifest_sha256,
            runtime_profile=runtime_profile,
        )

    calls: list[tuple[str, str, float]] = []

    def fake_trial(
        _model: mujoco.MjModel,
        _data: mujoco.MjData,
        *,
        ramp: FlyRampGeometry,
        direction: str,
        speed_m_s: float,
        **_kwargs: object,
    ) -> dict[str, object]:
        calls.append((ramp.route_id, direction, speed_m_s))
        return {
            "trial_id": f"{ramp.route_id}:{direction}:{speed_m_s:.1f}",
            "dynamic_status": "PASS",
            "solver_warnings": {},
            "checks": {"finite_state": True, "finite_contact_forces": True},
        }

    monkeypatch.setattr("rmuc2026_mujoco.wheel_probe.build_wheel_probe_model", fake_build)
    monkeypatch.setattr("rmuc2026_mujoco.wheel_probe.run_wheel_probe_trial", fake_trial)
    output = tmp_path / "wheel-probe.json"

    result = run_fixed_fly_ramp_wheel_probe(asset, output_path=output)

    assert result["status"] == "PASS"
    assert result["probe"]["trial_count"] == 12
    assert result["summary"]["passed_trials"] == 12
    assert len(calls) == 12
    assert {route for route, _direction, _speed in calls} == {
        ramp.route_id for ramp in FIXED_FLY_RAMPS
    }
    assert {direction for _route, direction, _speed in calls} == {"uphill", "downhill"}
    assert {speed for _route, _direction, speed in calls} == {0.3, 0.5, 1.0}
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_probe_configuration_rejects_nonpositive_thresholds() -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        WheelProbeConfig(maximum_airborne_duration_s=0.0).validated()
