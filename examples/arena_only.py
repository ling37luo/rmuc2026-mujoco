"""Open an integrity-checked local RMUC 2026 field pack in MuJoCo."""

from __future__ import annotations

import argparse
import time

import mujoco.viewer

from rmuc2026_mujoco import FieldAsset, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", help="local directory containing manifest.json")
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
    model, data = load_model(asset)
    report = asset.report()
    print(
        f"verified {report.visual_mesh_count} meshes and "
        f"{report.verified_bytes} bytes; validation_scope="
        f"{asset.manifest.get('validation_scope')}"
    )

    started = time.monotonic()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and (
            args.duration == 0.0 or time.monotonic() - started < args.duration
        ):
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
