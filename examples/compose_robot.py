"""Compose a user-owned robot MJCF with a verified local field asset pack."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer

from rmuc2026_mujoco import FieldAsset, compose_with_robot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", help="local directory containing manifest.json")
    parser.add_argument("robot", type=Path, help="path to a separately licensed robot MJCF")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="viewer duration in seconds; 0 waits until the window is closed",
    )
    args = parser.parse_args()
    if args.duration < 0.0:
        parser.error("--duration must be non-negative")

    asset = FieldAsset.open(args.asset, verify=True)
    model, data = compose_with_robot(asset, args.robot)

    started = time.monotonic()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and (
            args.duration == 0.0 or time.monotonic() - started < args.duration
        ):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(float(model.opt.timestep))


if __name__ == "__main__":
    main()
