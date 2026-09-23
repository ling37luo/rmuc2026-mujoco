"""Original four-wheel example robot; not a team's RM robot or learned policy."""

from __future__ import annotations

import math
import numpy as np


def rover_xml() -> str:
    wheels, actuators = [], []
    for name, x, y in (
        ("fl", 0.16, 0.21),
        ("fr", 0.16, -0.21),
        ("rl", -0.16, 0.21),
        ("rr", -0.16, -0.21),
    ):
        wheels.append(f'''
        <body name="{name}" pos="{x} {y} -.03">
          <joint name="{name}" axis="0 1 0" damping=".02" armature=".01"/>
          <geom name="wheel_{name}" type="cylinder" size=".08 .035"
                quat=".7071067812 .7071067812 0 0" mass=".4"
                friction="1 .005 .0001" rgba=".12 .15 .18 1"/>
        </body>''')
        actuators.append(
            f'<velocity name="{name}" joint="{name}" kv="1" ctrlrange="-16 16" forcerange="-3 3"/>'
        )
    return f"""<mujoco model="RMUC slope example rover">
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


class RoverController:
    """Wheel-speed example, with route following or simple latched keyboard input."""

    def __init__(self, route, direction, speed):
        self.configure_route(route, direction, speed)
        self.reset(None, None)

    def configure_route(self, route, direction, speed):
        self.route, self.direction, self.speed = route, direction, speed

    def reset(self, model, data):
        self.vx = self.yaw = 0.0
        self.phase = "downhill" if self.direction == "downhill" else "uphill"
        self.reached_at = None

    def press_name(self, name):
        if name in {"1", "2", "3", "4"}:
            self.speed = (0.15, 0.3, 0.5, 0.8)[int(name) - 1]
            self.vx = math.copysign(self.speed, self.vx) if self.vx else 0.0
        elif name == "W":
            self.vx = self.speed
        elif name == "S":
            self.vx = -self.speed
        elif name == "A":
            self.yaw = 0.5
        elif name == "D":
            self.yaw = -0.5
        elif name == "E":
            self.yaw = 0.0
        elif name == "X":
            self.vx = self.yaw = 0.0

    def __call__(self, model, data, *, step, mode):
        if mode == "policy":
            low = np.asarray(self.route["low_xyz_m"][:2])
            u = np.asarray(self.route["uphill_unit_xy"])
            pos = data.qpos[:2]
            along = float((pos - low) @ u)
            lateral = float((pos - low) @ np.array([-u[1], u[0]]))
            target = self.route["length_m"] if self.phase == "uphill" else 0.0
            error = target - along
            q = data.qpos[3:7]
            heading = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
            desired = self.route["heading_yaw_rad"]
            if self.direction == "downhill":
                desired += math.pi
            heading_error = math.atan2(math.sin(desired - heading), math.cos(desired - heading))
            forward_sign = -1 if self.direction == "downhill" else 1
            self.vx = float(np.clip(2 * error * forward_sign, -self.speed, self.speed))
            self.yaw = float(np.clip(3 * heading_error - 2 * lateral * np.sign(error), -0.6, 0.6))
            if abs(error) < 0.05:
                self.vx = 0.0
                if self.reached_at is None:
                    self.reached_at = float(data.time)
                if self.direction == "roundtrip" and self.phase == "uphill":
                    if data.time - self.reached_at > 0.5:
                        self.phase = "downhill"
                        self.reached_at = None
            else:
                self.reached_at = None
        if data.time < 0.3:
            data.ctrl[:] = 0.0
        else:
            data.ctrl[:] = [
                (self.vx - self.yaw * 0.21) / 0.08,
                (self.vx + self.yaw * 0.21) / 0.08,
            ] * 2
