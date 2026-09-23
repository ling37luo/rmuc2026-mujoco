"""Executable ordinary-slope interaction with user controllers or an example rover."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from queue import SimpleQueue
import tempfile
import threading
import time

import numpy as np

from .control import load_controller
from .display import FieldDisplayController
from .manifest import FieldAsset
from .mjcf import FIELD_ATTACH_PREFIX, compose_with_robot
from .query import load_heightfield
from .scenarios import scenario_descriptor
from .slope_catalog import slope_catalog
from .slope_demo import RoverController, rover_xml
from .slope_progress import SlopeProgress
from .slope_routes import select_slope_route
from .viewer import SafePassiveViewerSession


def _root_joint(model):
    import mujoco

    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("slope reset requires one freejoint robot")
    return int(free[0])


def reset_slope_spawn(model, data, route, *, direction="uphill", controller=None):
    """Reset all state, place the robot at a route end, find terrain support.

    Only initial pose is adjusted. No forces, contact parameters or ongoing
    motion are modified. Robot collision masks must already see the field.
    """
    import mujoco

    mujoco.mj_resetData(model, data)
    if callable(getattr(controller, "reset", None)):
        controller.reset(model, data)
    joint = _root_joint(model)
    qadr = int(model.jnt_qposadr[joint])
    point = route["high_xyz_m" if direction == "downhill" else "low_xyz_m"]
    heading = route["heading_yaw_rad"] + (math.pi if direction == "downhill" else 0)
    data.qpos[qadr : qadr + 2] = point[:2]
    data.qpos[qadr + 3 : qadr + 7] = [math.cos(heading / 2), 0, 0, math.sin(heading / 2)]
    field_geom = model.geom(f"{FIELD_ATTACH_PREFIX}rmuc2026_field_collision").id

    def penetrates(z):
        data.qpos[qadr + 2] = z
        mujoco.mj_forward(model, data)
        return any(
            (int(c.geom1) == field_geom or int(c.geom2) == field_geom) and c.dist < 0
            for c in data.contact
        )

    high = float(point[2]) + max(1.0, float(data.qpos[qadr + 2]))
    if penetrates(high):
        raise ValueError("robot intersects terrain above the route spawn")
    low = high
    for _ in range(math.ceil((high - point[2] + 0.5) / 0.02)):
        low -= 0.02
        if penetrates(low):
            break
        high = low
    else:
        raise ValueError("no robot/terrain contact at spawn; check robot collision masks")
    for _ in range(12):
        middle = (high + low) / 2
        if penetrates(middle):
            low = middle
        else:
            high = middle
    data.qpos[qadr + 2] = high + 0.002
    data.qvel[:] = 0
    data.qacc_warmstart[:] = 0
    mujoco.mj_forward(model, data)
    return data.qpos[qadr : qadr + 7].copy()


class SlopeSession:
    """Identical reset, control and stepping for the viewer and headless runs."""

    def __init__(
        self,
        asset,
        *,
        robot=None,
        controller=None,
        patch=None,
        direction="uphill",
        speed=0.3,
        mode="human",
        profile="collision_only",
        friction_preset=None,
    ):
        if direction not in {"uphill", "downhill", "roundtrip"}:
            raise ValueError("direction must be uphill, downhill or roundtrip")
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be positive and finite")
        if robot is not None and controller is None:
            raise ValueError("slope interaction with --robot requires --controller module:factory")
        if robot is None and controller is not None:
            raise ValueError("--controller requires --robot for slope_basic")
        self.asset = asset if isinstance(asset, FieldAsset) else FieldAsset.open(asset, verify=True)
        self.catalog = slope_catalog(self.asset)
        self.route = select_slope_route(self.catalog, patch)
        self.bounds = load_heightfield(self.asset).bounds_xy_m
        self.direction, self.mode, self.profile = direction, mode, profile
        self.speed = speed
        self.profile_hash = scenario_descriptor(self.asset, "slope_basic", profile=profile)[
            "profile_hash"
        ]
        self.robot_kind = "external_robot" if robot is not None else "example_four_wheel_rover"
        if robot is None:
            xml = rover_xml()
            with tempfile.TemporaryDirectory(prefix="rmuc-slope-rover-") as directory:
                robot_path = Path(directory) / "rover.xml"
                robot_path.write_text(xml, encoding="utf-8")
                self.model, self.data = compose_with_robot(
                    self.asset, robot_path, profile=profile, friction_preset=friction_preset
                )
            self.robot_hash = hashlib.sha256(xml.encode()).hexdigest()
            self.controller = RoverController(self.route, direction, speed)
        else:
            self.robot_hash = hashlib.sha256(Path(robot).read_bytes()).hexdigest()
            self.model, self.data = compose_with_robot(
                self.asset, robot, profile=profile, friction_preset=friction_preset
            )
            self.controller = load_controller(controller, self.model, self.data, mode=mode)
            if callable(getattr(self.controller, "configure_route", None)):
                self.controller.configure_route(self.route, direction, speed)
        self.friction_preset = friction_preset
        joint = _root_joint(self.model)
        self.qadr = int(self.model.jnt_qposadr[joint])
        self.body = int(self.model.jnt_bodyid[joint])
        self.field_geoms = {
            i
            for i in range(self.model.ngeom)
            if (self.model.geom(i).name or "").startswith(FIELD_ATTACH_PREFIX)
            and (self.model.geom_contype[i] or self.model.geom_conaffinity[i])
        }
        self.episodes = []
        self.reset()

    def reset(self):
        self.initial_pose = reset_slope_spawn(
            self.model, self.data, self.route, direction=self.direction, controller=self.controller
        )
        self.steps = 0
        self.rows = []
        self.progress = SlopeProgress(self.route, self.direction)
        self.failure = None
        self.warnings = {}
        self.max_penetration = self.max_tilt = 0.0
        self.field_contact_steps = 0
        self.finite = True
        self.start_time = time.perf_counter()

    @property
    def phase(self):
        return self.progress.phase

    @property
    def completed(self):
        return self.progress.status == "COMPLETE"

    def step(self):
        import mujoco

        # Refresh sensors before the external low-level controller. The split
        # API preserves Euler; other integrators use a full forward + step.
        if self.model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER:
            mujoco.mj_step1(self.model, self.data)
            self.controller(self.model, self.data, step=self.steps, mode=self.mode)
            mujoco.mj_step2(self.model, self.data)
        else:
            mujoco.mj_forward(self.model, self.data)
            self.controller(self.model, self.data, step=self.steps, mode=self.mode)
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        # Kinematics after integration are used for telemetry and route gates.
        mujoco.mj_forward(self.model, self.data)
        d = self.data
        self.finite = bool(all(np.isfinite(x).all() for x in (d.qpos, d.qvel, d.qacc)))
        self.warnings.update(
            {mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(d.warning) if w.number}
        )
        position = np.asarray(d.xpos[self.body])
        u = np.asarray(self.route["uphill_unit_xy"])
        delta = position[:2] - np.asarray(self.route["low_xyz_m"][:2])
        along = float(delta @ u)
        across = float(delta @ np.array([-u[1], u[0]]))
        tilt = math.degrees(math.acos(np.clip(d.xmat[self.body].reshape(3, 3)[2, 2], -1, 1)))
        self.max_tilt = max(self.max_tilt, tilt)
        contacts = [
            c
            for c in d.contact
            if int(c.geom1) in self.field_geoms or int(c.geom2) in self.field_geoms
        ]
        self.field_contact_steps += bool(contacts)
        penetration = max((max(0.0, -float(c.dist)) for c in contacts), default=0.0)
        self.max_penetration = max(self.max_penetration, penetration)
        reason = None
        if not self.finite or self.warnings:
            reason = "numerical_instability"
        elif tilt > 60:
            reason = "robot_tipped"
        elif np.any(position[:2] < self.bounds[0]) or np.any(position[:2] > self.bounds[1]):
            reason = "outside_field"
        elif abs(across) > self.route["width_m"] / 2:
            reason = "outside_route"
        if reason and self.failure is None:
            self.failure = {"reason": reason, "time_s": float(d.time)}
        self.progress.update(
            time_s=float(d.time),
            along_m=along,
            contacts=len(contacts),
            tilt_deg=tilt,
            penetration_m=penetration,
            failure=self.failure,
        )
        if self.steps % max(1, round(0.02 / self.model.opt.timestep)) == 0:
            self.rows.append(
                {
                    "time_s": float(d.time),
                    "position_xyz_m": position.tolist(),
                    "along_m": along,
                    "lateral_m": across,
                    "tilt_deg": tilt,
                    "field_contacts": len(contacts),
                    "phase": self.phase,
                    "uphill_status": self.progress.legs["uphill"]["status"],
                    "downhill_status": self.progress.legs["downhill"]["status"],
                    "qpos": d.qpos.tolist(),
                    "qvel": d.qvel.tolist(),
                    "finite": self.finite,
                    "warnings": dict(self.warnings),
                }
            )
        return self.finite and not self.warnings

    def report(self):
        return {
            "scenario_id": "slope_basic",
            "route": self.route,
            "source_manifest_sha256": self.asset.manifest_sha256,
            "profile": self.profile,
            "profile_hash": self.profile_hash,
            "heightfield_samples_sha256": self.asset.collision["samples_sha256"],
            "robot_kind": self.robot_kind,
            "robot_mjcf_sha256": self.robot_hash,
            "controller_identity": getattr(self.controller, "identity", None),
            "direction": self.direction,
            "control": self.mode,
            "speed_mps": self.speed,
            "friction_preset": self.friction_preset,
            "timestep_s": float(self.model.opt.timestep),
            "steps": self.steps,
            "sim_time_s": float(self.data.time),
            "initial_pose": self.initial_pose.tolist(),
            "steps_per_second": self.steps / max(1e-9, time.perf_counter() - self.start_time),
            "physics_status": "PASS" if self.finite and not self.warnings else "FAIL",
            **self.progress.report(),
            "first_failure": self.failure,
            "field_contact_steps": self.field_contact_steps,
            "max_penetration_m": self.max_penetration,
            "max_tilt_deg": self.max_tilt,
            "finite": self.finite,
            "warnings": dict(self.warnings),
            "validation_status": self.asset.manifest.get("validation_status"),
        }


def draw_slope_route(scene, route):
    """Viewer-only route markers; they cannot generate a contact."""
    import mujoco

    points = route["waypoints_xyz_m"]
    scene.ngeom = 0
    for index, point in enumerate(points[: scene.maxgeom]):
        color = [0.15, 0.8, 0.45, 1] if index == 0 else [1, 0.65, 0.05, 1]
        mujoco.mjv_initGeom(
            scene.geoms[index],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            [0.025, 0, 0],
            np.asarray(point) + [0, 0, 0.035],
            np.eye(3).ravel(),
            color,
        )
        scene.ngeom += 1


def view_slope(args, asset):
    """CLI integration; same session runs headless or with a native window."""
    import mujoco

    session = SlopeSession(
        asset,
        robot=args.robot,
        controller=args.controller,
        patch=args.patch,
        direction=args.direction,
        speed=args.speed,
        mode=args.control,
        profile=args.profile,
        friction_preset=args.friction_preset,
    )
    if args.headless:
        for _ in range(args.steps):
            if not session.step():
                break
    else:
        import mujoco.viewer
        from .cli import _start_display_key_listener

        display = FieldDisplayController(asset, lighting=args.lighting, livery=args.livery)
        close = threading.Event()
        keys = SimpleQueue()
        context = mujoco.viewer.launch_passive(session.model, session.data)
        print(
            f"SLOPE_VIEW patch={session.route['route_id']} robot={session.robot_kind}", flush=True
        )
        print(session.progress.status_line(session.data.time), flush=True)
        print(
            "W/S forward/reverse; A/D steer; E straighten; X stop; 1-4 speed; "
            "R reset; L/G lighting/livery; Esc close. Commands persist until changed.",
            flush=True,
        )
        lifecycle = SafePassiveViewerSession(context)
        with lifecycle as viewer:
            lifecycle.listener = _start_display_key_listener(
                display,
                close,
                on_key=keys.put,
                reserved_keys=("l", "g", "w", "s", "a", "d", "e", "x", "r", "1", "2", "3", "4"),
            )
            if lifecycle.listener is None and args.control == "human":
                raise ValueError(
                    "human slope driving needs the optional viewer dependencies and an X11 "
                    "window; install the viewer extra or use --control policy"
                )
            with viewer.lock():
                viewer.cam.lookat[:] = (
                    np.asarray(session.route["low_xyz_m"]) + session.route["high_xyz_m"]
                ) / 2
                viewer.cam.distance = max(3.0, session.route["length_m"] * 2)
                viewer.cam.elevation = -35
                viewer.cam.azimuth = math.degrees(session.route["heading_yaw_rad"]) + 90
                draw_slope_route(viewer.user_scn, session.route)
                display.apply(session.model, viewer.opt.geomgroup)
            started = time.monotonic()
            last_status = tuple(leg["status"] for leg in session.progress.legs.values())
            while viewer.is_running() and not close.is_set():
                frame_start = time.monotonic()
                if args.duration and frame_start - started >= args.duration:
                    break
                with viewer.lock():
                    while not keys.empty():
                        key = keys.get()
                        if key == "R":
                            session.episodes.append(
                                {"summary": session.report(), "rows": session.rows}
                            )
                            session.reset()
                            last_status = None
                        elif callable(getattr(session.controller, "press_name", None)):
                            session.controller.press_name(key)
                    # Advance a render frame of physics, not just one 2 ms step.
                    for _ in range(max(1, round(1 / (60 * session.model.opt.timestep)))):
                        if not session.step():
                            close.set()
                            break
                    display.apply(session.model, viewer.opt.geomgroup)
                current_status = tuple(leg["status"] for leg in session.progress.legs.values())
                if current_status != last_status:
                    print(session.progress.status_line(session.data.time), flush=True)
                    last_status = current_status
                viewer.sync()
                time.sleep(max(0, 1 / 60 - (time.monotonic() - frame_start)))
    report = session.report()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.telemetry or Path("runs") / f"slope_interaction_{stamp}.json"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"summary": report, "rows": session.rows, "previous_episodes": session.episodes},
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({**report, "telemetry": str(output)}, indent=2))
    return (
        0
        if report["physics_status"] == "PASS"
        and (args.control == "human" or report["traversal_status"] == "COMPLETE")
        else 2
    )
