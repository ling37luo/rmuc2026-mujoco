from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from rmuc2026_mujoco.corner_check import TraceDigest, drive_corner, plan_corner_route
from rmuc2026_mujoco.query import HeightFieldData
from rmuc2026_mujoco.slope_demo import rover_xml


def test_corner_planner_routes_around_a_barrier_deterministically():
    axis = np.linspace(0, 8, 161)
    x, y = np.meshgrid(axis, axis)
    height = np.zeros_like(x)
    height[(x > 1) & (x < 1.4) & (y < 2.5)] = 1
    grid = HeightFieldData(axis, axis, height)
    corner = {"id": "left_bottom", "xy_m": [0, 0]}
    first, path = plan_corner_route(grid, corner, [[4.45, 0.45]])
    second, again = plan_corner_route(grid, corner, [[4.45, 0.45]])
    assert first == second
    np.testing.assert_array_equal(path, again)
    assert np.max(path[:, 1]) > 2.8
    assert np.linalg.norm(path[-1]) < 0.7
    assert all(not (0.65 < x < 1.75 and y < 2.8) for x, y in path)
    with pytest.raises(ValueError, match="no supported perimeter entry"):
        plan_corner_route(grid, corner, [[1.15, 0.45]])


def test_corner_probe_drives_and_retreats_without_pose_interventions():
    import xml.etree.ElementTree as ET

    root = ET.fromstring(rover_xml())
    world = root.find("worldbody")
    world.append(
        ET.fromstring(
            '<geom name="rmuc2026_field/floor" type="plane" size="5 5 .1" contype="2" conaffinity="1"/>'
        )
    )
    world.append(
        ET.fromstring(
            '<geom name="rmuc2026_field/wall" type="box" pos="-.05 0 .5" size=".05 5 .5" contype="2" conaffinity="1"/>'
        )
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)

    def reset(*, qpos):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

    def step(action):
        data.ctrl[:] = action
        mujoco.mj_step(model, data)

    env = SimpleNamespace(model=model, data=data, reset=reset, step=step)
    route = dict(
        id="synthetic",
        waypoints_xy_m=[[1.0, 0.45], [0.4, 0.45]],
        corner_xy_m=[0, 0],
        initial_height_m=0,
    )
    report = drive_corner(env, route, 0.5)
    assert report["status"] == "PASS"
    assert report["blocking_contacts"]
    assert report["phases"][-1]["phase"] == "hold"
    assert np.linalg.norm(data.qpos[:2] - [1, 0.45]) < 0.08
    assert not report["warnings"]


def test_trace_digest_includes_contact_identity():
    data = SimpleNamespace(qpos=np.zeros(3), qvel=np.zeros(3))
    first, second = TraceDigest(), TraceDigest()
    first.record(data, {"wheel | ground": 1})
    second.record(data, {"wheel | wall": 1})
    assert first.report() != second.report()
