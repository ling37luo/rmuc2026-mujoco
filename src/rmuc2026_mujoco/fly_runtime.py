"""One continuous robot episode over an audited fly ramp."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import random
import tempfile
import time

import numpy as np

from .control import load_controller
from .fly_progress import FlyRampProgress
from .fly_routes import APPROACH_BEFORE_LOW_EDGE_M, fly_route_descriptor
from .manifest import FieldAsset
from .mjcf import FIELD_ATTACH_PREFIX, compose_with_robot
from .query import load_heightfield
from .scenarios import scenario_descriptor
from .slope_runtime import _root_joint, reset_slope_spawn


FLY_SCENARIOS = ("fly_ramp_north", "fly_ramp_south")


class FlyRampSession:
    """Reuse one composed robot/field model across independent jump attempts."""

    def __init__(
        self,
        asset,
        *,
        robot=None,
        controller=None,
        scenario_id="fly_ramp_north",
        speed_mps=2.2,
        approach_distance_m=APPROACH_BEFORE_LOW_EDGE_M,
        profile="collision_only",
        mode="policy",
        friction_preset=None,
        record_trajectory=False,
    ):
        if scenario_id not in FLY_SCENARIOS:
            raise ValueError(f"fly-ramp scenario must be one of {FLY_SCENARIOS}")
        if not math.isfinite(speed_mps) or speed_mps <= 0:
            raise ValueError("fly-ramp speed must be positive and finite")
        if (robot is None) != (controller is None):
            raise ValueError(
                "provide both --robot and --controller, or neither for the example rover"
            )
        self.asset = asset if isinstance(asset, FieldAsset) else FieldAsset.open(asset, verify=True)
        self.bounds = load_heightfield(self.asset).bounds_xy_m
        self.routes = {}
        self._approach_routes = {}
        self.scenario_id = scenario_id
        self.approach_distance_m = float(approach_distance_m)
        self.route = self._route_for(scenario_id, self.approach_distance_m)
        self.profile = profile
        self.profile_hash = scenario_descriptor(self.asset, scenario_id, profile=profile)[
            "profile_hash"
        ]
        self.speed_mps = speed_mps
        self.mode = mode
        self.friction_preset = friction_preset
        self.record_trajectory = record_trajectory
        self.robot_kind = "external_robot" if robot is not None else "example_four_wheel_rover"
        if robot is None:
            from .fly_demo import FlyRoverController, fly_rover_xml

            xml = fly_rover_xml()
            with tempfile.TemporaryDirectory(prefix="rmuc-fly-rover-") as directory:
                path = Path(directory) / "rover.xml"
                path.write_text(xml, encoding="utf-8")
                self.model, self.data = compose_with_robot(
                    self.asset, path, profile=profile, friction_preset=friction_preset
                )
            self.robot_hash = hashlib.sha256(xml.encode()).hexdigest()
            self.controller = FlyRoverController(self.route, speed_mps)
        else:
            robot_path = Path(robot).expanduser().resolve()
            self.robot_hash = hashlib.sha256(robot_path.read_bytes()).hexdigest()
            self.model, self.data = compose_with_robot(
                self.asset, robot_path, profile=profile, friction_preset=friction_preset
            )
            self.controller = load_controller(controller, self.model, self.data, mode=mode)
        joint = _root_joint(self.model)
        self.qadr = int(self.model.jnt_qposadr[joint])
        self.dofadr = int(self.model.jnt_dofadr[joint])
        self.body = int(self.model.jnt_bodyid[joint])
        self.field_geom = self.model.geom(f"{FIELD_ATTACH_PREFIX}rmuc2026_field_collision").id
        self.field_geoms = {
            i
            for i in range(self.model.ngeom)
            if (self.model.geom(i).name or "").startswith(FIELD_ATTACH_PREFIX)
            and (self.model.geom_contype[i] or self.model.geom_conaffinity[i])
        }
        self.reset()

    def _route_for(self, scenario_id, approach_distance_m):
        if approach_distance_m == APPROACH_BEFORE_LOW_EDGE_M:
            if scenario_id not in self.routes:
                self.routes[scenario_id] = fly_route_descriptor(self.asset, scenario_id)
            return self.routes[scenario_id]
        key = (scenario_id, approach_distance_m)
        if key not in self._approach_routes:
            self._approach_routes[key] = fly_route_descriptor(
                self.asset, scenario_id, approach_distance_m
            )
        return self._approach_routes[key]

    def reset(
        self,
        *,
        scenario_id=None,
        speed_mps=None,
        approach_distance_m=None,
        lateral_offset_m=0.0,
        heading_offset_deg=0.0,
        seed=None,
    ):
        if scenario_id is not None:
            if scenario_id not in FLY_SCENARIOS:
                raise ValueError(f"fly-ramp scenario must be one of {FLY_SCENARIOS}")
            self.scenario_id = scenario_id
            self.profile_hash = scenario_descriptor(self.asset, scenario_id, profile=self.profile)[
                "profile_hash"
            ]
        if approach_distance_m is not None:
            self.approach_distance_m = float(approach_distance_m)
        self.route = self._route_for(self.scenario_id, self.approach_distance_m)
        if speed_mps is not None:
            if not math.isfinite(speed_mps) or speed_mps <= 0:
                raise ValueError("fly-ramp speed must be positive and finite")
            self.speed_mps = speed_mps
        if not math.isfinite(lateral_offset_m) or not math.isfinite(heading_offset_deg):
            raise ValueError("fly-ramp pose offsets must be finite")
        if abs(lateral_offset_m) >= self.route["surface_width_m"] / 2:
            raise ValueError("fly-ramp lateral offset is outside the slope face")
        self.seed = None if seed is None else int(seed)
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed % (2**32))
            if callable(getattr(self.controller, "set_seed", None)):
                self.controller.set_seed(self.seed)
        if callable(getattr(self.controller, "configure_route", None)):
            self.controller.configure_route(self.route, "uphill", self.speed_mps)
        self.lateral_offset_m = float(lateral_offset_m)
        self.heading_offset_deg = float(heading_offset_deg)
        self.initial_pose = reset_slope_spawn(
            self.model,
            self.data,
            {
                "low_xyz_m": self.route["approach_xyz_m"],
                "heading_yaw_rad": self.route["heading_yaw_rad"],
                "uphill_unit_xy": self.route["uphill_unit_xy"],
            },
            controller=self.controller,
            lateral_offset_m=self.lateral_offset_m,
            heading_offset_rad=math.radians(self.heading_offset_deg),
        )
        self.progress = FlyRampProgress(self.route)
        self.steps = 0
        self.rows = []
        self.phase_states = {}
        self.failure = None
        self.warnings = {}
        self.finite = True
        self.field_contact_steps = 0
        self.obstacle_contact_steps = 0
        self.max_penetration_m = 0.0
        self.field_normal_force_peak_by_robot_geom_n = {}
        self.max_tilt_deg = 0.0
        self.max_abs_effort = 0.0
        self.started = time.perf_counter()

    @property
    def completed(self):
        return self.progress.status == "COMPLETE"

    def step(self):
        import mujoco

        if self.model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER:
            mujoco.mj_step1(self.model, self.data)
            self.controller(self.model, self.data, step=self.steps, mode=self.mode)
            mujoco.mj_step2(self.model, self.data)
        else:
            mujoco.mj_forward(self.model, self.data)
            self.controller(self.model, self.data, step=self.steps, mode=self.mode)
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        mujoco.mj_forward(self.model, self.data)
        d = self.data
        self.finite = bool(all(np.isfinite(x).all() for x in (d.qpos, d.qvel, d.qacc)))
        self.warnings.update(
            {mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(d.warning) if w.number}
        )
        position = np.asarray(d.xpos[self.body], dtype=float)
        velocity = np.asarray(d.qvel[self.dofadr : self.dofadr + 2], dtype=float)
        u = np.asarray(self.route["uphill_unit_xy"], dtype=float)
        tilt = math.degrees(math.acos(np.clip(d.xmat[self.body].reshape(3, 3)[2, 2], -1, 1)))
        self.max_tilt_deg = max(self.max_tilt_deg, tilt)
        if d.actuator_force.size:
            self.max_abs_effort = max(self.max_abs_effort, float(np.max(np.abs(d.actuator_force))))
        contact_points = []
        obstacle_contacts = 0
        for index, contact in enumerate(d.contact):
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            field_geom = geom1 if geom1 in self.field_geoms else geom2
            if field_geom not in self.field_geoms:
                continue
            self.max_penetration_m = max(self.max_penetration_m, max(0.0, -float(contact.dist)))
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, d, index, force)
            if force[0] > 0.1:
                robot_geom = geom2 if field_geom == geom1 else geom1
                robot_name = self.model.geom(robot_geom).name or f"geom_{robot_geom}"
                self.field_normal_force_peak_by_robot_geom_n[robot_name] = max(
                    self.field_normal_force_peak_by_robot_geom_n.get(robot_name, 0.0),
                    float(force[0]),
                )
                if field_geom == self.field_geom:
                    contact_points.append(np.asarray(contact.pos, dtype=float).tolist())
                else:
                    obstacle_contacts += 1
        self.field_contact_steps += bool(contact_points or obstacle_contacts)
        self.obstacle_contact_steps += bool(obstacle_contacts)
        delta = position[:2] - np.asarray(self.route["low_seam_xyz_m"][:2])
        lateral = float(delta @ np.array([-u[1], u[0]]))
        reason = None
        if not self.finite or self.warnings:
            reason = "numerical_instability"
        elif tilt >= 60.0 and self.progress.phase != "flight":
            reason = "robot_tipped"
        elif np.any(position[:2] < self.bounds[0]) or np.any(position[:2] > self.bounds[1]):
            reason = "outside_field"
        elif abs(lateral) > self.route["surface_width_m"] / 2 + 0.10:
            reason = "outside_ramp_corridor"
        if reason is not None and self.failure is None:
            self.failure = {"reason": reason, "time_s": float(d.time)}
        previous_events = set(self.progress.events)
        self.progress.update(
            time_s=float(d.time),
            position_xyz_m=position,
            forward_speed_mps=float(velocity @ u),
            contact_points_xyz_m=contact_points,
            tilt_deg=tilt,
            obstacle_contacts=obstacle_contacts,
            failure=reason,
        )
        for event_name, event_time_s in self.progress.events.items():
            if event_name not in previous_events:
                self.phase_states[event_name] = {
                    "event_time_s": float(event_time_s),
                    "sample_time_s": float(d.time),
                    "position_xyz_m": position.tolist(),
                    "velocity_xyz_mps": d.qvel[self.dofadr : self.dofadr + 3].tolist(),
                    "forward_speed_mps": float(velocity @ u),
                    "lateral_speed_mps": float(velocity @ np.array([-u[1], u[0]])),
                    "vertical_speed_mps": float(d.qvel[self.dofadr + 2]),
                    "lateral_offset_m": lateral,
                    "body_up_xyz": d.xmat[self.body].reshape(3, 3)[:, 2].tolist(),
                    "tilt_deg": tilt,
                    "field_contacts": len(contact_points),
                    "obstacle_contacts": obstacle_contacts,
                }
        if self.progress.failure_reason and self.failure is None:
            self.failure = {
                "reason": self.progress.failure_reason,
                "time_s": float(d.time),
            }
        if (
            self.record_trajectory
            and self.steps % max(1, round(0.02 / self.model.opt.timestep)) == 0
        ):
            self.rows.append(
                {
                    "time_s": float(d.time),
                    "position_xyz_m": position.tolist(),
                    "forward_speed_mps": float(velocity @ u),
                    "along_m": float(delta @ u),
                    "lateral_m": lateral,
                    "tilt_deg": tilt,
                    "field_contacts": len(contact_points),
                    "obstacle_contacts": obstacle_contacts,
                    "phase": self.progress.phase,
                    "qpos": d.qpos.tolist(),
                    "qvel": d.qvel.tolist(),
                    "warnings": dict(self.warnings),
                }
            )
        return self.finite and not self.warnings

    def report(self):
        return {
            "scenario_id": self.scenario_id,
            "route": self.route,
            "source_manifest_sha256": self.asset.manifest_sha256,
            "profile": self.profile,
            "profile_hash": self.profile_hash,
            "heightfield_samples_sha256": self.asset.collision["samples_sha256"],
            "robot_kind": self.robot_kind,
            "robot_mjcf_sha256": self.robot_hash,
            "controller_identity": getattr(self.controller, "identity", None),
            "control": self.mode,
            "friction_preset": self.friction_preset,
            "commanded_speed_mps": self.speed_mps,
            "approach_distance_m": self.approach_distance_m,
            "seed": self.seed,
            "lateral_offset_m": self.lateral_offset_m,
            "heading_offset_deg": self.heading_offset_deg,
            "timestep_s": float(self.model.opt.timestep),
            "steps": self.steps,
            "sim_time_s": float(self.data.time),
            "initial_pose": self.initial_pose.tolist(),
            "steps_per_second": self.steps / max(1e-9, time.perf_counter() - self.started),
            "physics_status": "PASS" if self.finite and not self.warnings else "FAIL",
            **self.progress.report(),
            "phase_states": dict(self.phase_states),
            "first_failure": self.failure,
            "field_contact_steps": self.field_contact_steps,
            "obstacle_contact_steps": self.obstacle_contact_steps,
            "max_penetration_m": self.max_penetration_m,
            "field_normal_force_peak_by_robot_geom_n": dict(
                self.field_normal_force_peak_by_robot_geom_n
            ),
            "max_tilt_deg": self.max_tilt_deg,
            "max_abs_actuator_force": self.max_abs_effort,
            "finite": self.finite,
            "warnings": dict(self.warnings),
            "validation_status": self.asset.manifest.get("validation_status"),
        }


__all__ = ["FLY_SCENARIOS", "FlyRampSession"]
