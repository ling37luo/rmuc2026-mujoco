"""Interactive replay of the same fly-ramp session used by automatic runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from queue import SimpleQueue
import threading
import time

from .display import FieldDisplayController
from .fly_runtime import FlyRampSession
from .viewer import SafePassiveViewerSession


def view_fly(args, asset):
    """Show or headlessly step one robot attempt, then save phase telemetry."""

    session = FlyRampSession(
        asset,
        robot=args.robot,
        controller=args.controller,
        scenario_id=args.scenario,
        speed_mps=2.5 if args.speed is None else args.speed,
        approach_distance_m=args.approach_distance,
        profile=args.profile,
        mode=args.control,
        friction_preset=args.friction_preset,
        record_trajectory=True,
    )
    previous = []
    if args.headless:
        for _ in range(args.steps):
            if not session.step() or session.progress.status != "INCOMPLETE":
                break
    else:
        import mujoco.viewer

        from .cli import _configure_camera, _configure_fly_camera, _start_display_key_listener

        display = FieldDisplayController(asset, lighting=args.lighting, livery=args.livery)
        close = threading.Event()
        keys = SimpleQueue()
        context = mujoco.viewer.launch_passive(session.model, session.data)
        print(f"FLY_VIEW route={session.route['route_id']} robot={session.robot_kind}", flush=True)
        print(
            "W/S forward/reverse; A/D steer; E straighten; X stop; 1-4 speed; "
            "R reset; L/G lighting/livery; Esc close.",
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
                    "human fly-ramp driving needs optional viewer dependencies and an X11 window"
                )
            with viewer.lock():
                if args.camera == "spawn":
                    _configure_fly_camera(viewer, session.route)
                else:
                    _configure_camera(viewer, asset, args.camera)
                display.apply(session.model, viewer.opt.geomgroup)
            started = time.monotonic()
            last_phase = session.progress.phase
            while viewer.is_running() and not close.is_set():
                frame_start = time.monotonic()
                if args.duration and frame_start - started >= args.duration:
                    break
                with viewer.lock():
                    while not keys.empty():
                        key = keys.get()
                        if key == "R":
                            previous.append({"summary": session.report(), "rows": session.rows})
                            session.reset()
                        elif callable(getattr(session.controller, "press_name", None)):
                            session.controller.press_name(key)
                    if session.progress.status == "INCOMPLETE":
                        for _ in range(max(1, round(1 / (60 * session.model.opt.timestep)))):
                            if not session.step():
                                close.set()
                                break
                            if session.progress.status != "INCOMPLETE":
                                break
                    display.apply(session.model, viewer.opt.geomgroup)
                if session.progress.phase != last_phase:
                    print(
                        f"FLY_PROGRESS t={session.data.time:.2f}s phase={session.progress.phase} "
                        f"speed={session.progress.takeoff_speed_mps}",
                        flush=True,
                    )
                    last_phase = session.progress.phase
                viewer.sync()
                time.sleep(max(0, 1 / 60 - (time.monotonic() - frame_start)))
    report = session.report()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.telemetry or Path("runs") / f"fly_interaction_{stamp}.json"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"summary": report, "rows": session.rows, "previous_episodes": previous}, indent=2
        )
        + "\n"
    )
    print(json.dumps({**report, "telemetry": str(output)}, indent=2))
    return 0 if report["physics_status"] == "PASS" else 2


__all__ = ["view_fly"]
