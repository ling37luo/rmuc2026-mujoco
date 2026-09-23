"""Public four-wheel fly-ramp example; no team robot or learned policy."""

from __future__ import annotations

import math

import numpy as np


WHEEL_RADIUS_M = 0.08
HALF_TRACK_M = 0.21
MAX_WHEEL_SPEED_RAD_S = 40.0
MAX_FORWARD_SPEED_M_S = 3.0


def fly_rover_xml() -> str:
    """A small skid-steer rover whose actuators cover the fly-ramp speed sweep."""
    wheels, actuators = [], []
    for name, x, y in (
        ("fl", 0.16, HALF_TRACK_M),
        ("fr", 0.16, -HALF_TRACK_M),
        ("rl", -0.16, HALF_TRACK_M),
        ("rr", -0.16, -HALF_TRACK_M),
    ):
        wheels.append(f'''
        <body name="{name}" pos="{x} {y} -.03">
          <joint name="{name}" axis="0 1 0" damping=".02" armature=".01"/>
          <geom name="wheel_{name}" type="cylinder" size="{WHEEL_RADIUS_M} .035"
                quat=".7071067812 .7071067812 0 0" mass=".4"
                friction="1 .005 .0001" rgba=".12 .15 .18 1"/>
        </body>''')
        actuators.append(
            f'<velocity name="{name}" joint="{name}" kv="1" '
            f'ctrlrange="-{MAX_WHEEL_SPEED_RAD_S} {MAX_WHEEL_SPEED_RAD_S}" '
            'forcerange="-6 6"/>'
        )
    return f"""<mujoco model="RMUC fly-ramp example rover">
      <option timestep=".002" integrator="implicitfast"/>
      <default><geom contype="1" conaffinity="2" solref=".02 1"/></default>
      <worldbody><body name="rover" pos="0 0 .2">
        <freejoint/>
        <geom name="chassis" type="box" size=".21 .15 .045" mass="4"
              rgba=".1 .55 .8 1"/>
        <geom name="nose" type="box" pos=".18 0 .055" size=".035 .1 .01"
              mass=".02" contype="0" conaffinity="0" rgba="1 .65 .1 1"/>
        {"".join(wheels)}
      </body></worldbody><actuator>{"".join(actuators)}</actuator>
    </mujoco>"""


class FlyRoverController:
    """Follow one fly-ramp centerline, then brake beyond its landing edge."""

    identity = "example_fly_rover_v1"

    def __init__(self, route, speed):
        self.configure_route(route, "forward", speed)
        self.reset(None, None)

    def configure_route(self, route, direction, speed):
        speed = float(speed)
        if not math.isfinite(speed) or not 0 < speed <= MAX_FORWARD_SPEED_M_S:
            raise ValueError(f"example fly rover speed must be in (0, {MAX_FORWARD_SPEED_M_S}] m/s")
        self.route, self.direction, self.speed = route, direction, speed
        self.origin = np.asarray(route["approach_xyz_m"][:2], dtype=float)
        self.uphill = np.asarray(route["uphill_unit_xy"], dtype=float)
        self.uphill /= np.linalg.norm(self.uphill)
        self.lateral = np.array([-self.uphill[1], self.uphill[0]])
        self.target_along = float(
            (np.asarray(route["landing_target_xyz_m"][:2]) - self.origin) @ self.uphill
        )
        self.heading = float(route["heading_yaw_rad"])

    def reset(self, model, data):
        self.vx = 0.0
        self.yaw = 0.0
        self.phase = "approach"

    def set_seed(self, seed):
        # The example controller is deterministic; retain the run seed for telemetry.
        self.seed = int(seed)

    def press_name(self, name):
        if name in {"1", "2", "3", "4"}:
            self.speed = (1.5, 2.0, 2.2, 2.5)[int(name) - 1]
            if self.vx:
                self.vx = math.copysign(self.speed, self.vx)
        elif name == "W":
            self.vx = self.speed
        elif name == "S":
            self.vx = -self.speed
        elif name == "A":
            self.yaw = 0.6
        elif name == "D":
            self.yaw = -0.6
        elif name == "E":
            self.yaw = 0.0
        elif name == "X":
            self.vx = self.yaw = 0.0

    def __call__(self, model, data, *, step, mode):
        if mode == "policy":
            delta = np.asarray(data.qpos[:2]) - self.origin
            along = float(delta @ self.uphill)
            if along >= self.target_along:
                self.phase = "settle"
            if self.phase == "settle":
                self.vx = self.yaw = 0.0
            else:
                # Keep driving across the lip and through the landing edge.
                # Slowing at takeoff would turn this into a jump-short test.
                self.vx = self.speed
                lateral_error = float(delta @ self.lateral)
                q = data.qpos[3:7]
                heading = math.atan2(
                    2 * (q[0] * q[3] + q[1] * q[2]),
                    1 - 2 * (q[2] ** 2 + q[3] ** 2),
                )
                heading_error = math.atan2(
                    math.sin(self.heading - heading), math.cos(self.heading - heading)
                )
                self.yaw = float(np.clip(2.5 * heading_error - 1.3 * lateral_error, -0.6, 0.6))
        if data.time < 0.3:
            data.ctrl[:] = 0.0
            return
        left = (self.vx - self.yaw * HALF_TRACK_M) / WHEEL_RADIUS_M
        right = (self.vx + self.yaw * HALF_TRACK_M) / WHEEL_RADIUS_M
        data.ctrl[:] = [left, right, left, right]
