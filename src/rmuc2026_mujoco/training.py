"""Small MuJoCo training/debug adapter with no RL framework dependency."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

try:
    import resource
except ImportError:  # pragma: no cover - Windows has no resource module
    resource = None

from .control import ControlCallback
from .manifest import FieldAsset
from .mjcf import compose_with_robot, load_model
from .scenarios import get_scenario, scenario_descriptor


@dataclass(frozen=True)
class StepResult:
    observation: np.ndarray
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class TelemetryRecorder:
    """Record compact, robot-agnostic physics rows."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, model: Any, data: Any, *, step: int, elapsed_s: float) -> dict[str, Any]:
        import mujoco

        warnings = {
            mujoco.mjtWarning(index).name: int(item.number)
            for index, item in enumerate(data.warning)
            if int(item.number)
        }
        row = {
            "step": int(step),
            "time_s": float(data.time),
            "qpos": np.asarray(data.qpos, dtype=float).tolist(),
            "qvel": np.asarray(data.qvel, dtype=float).tolist(),
            "contacts": int(data.ncon),
            "warnings": warnings,
            "finite": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
            "steps_per_second": float((step + 1) / elapsed_s) if elapsed_s > 0.0 else None,
        }
        self.rows.append(row)
        return row


class MuJoCoScenario:
    """Load one field scenario and expose reset/step/telemetry primitives."""

    def __init__(
        self,
        asset: FieldAsset | str | Path,
        robot_xml: str | Path | None = None,
        *,
        scenario: str = "full_eval",
        profile: str | None = None,
        seed: int | None = None,
        env_count: int = 1,
        patch: str | None = None,
        direction: str = "uphill",
    ) -> None:
        self.asset = asset if isinstance(asset, FieldAsset) else FieldAsset.open(asset, verify=True)
        self.scenario = get_scenario(scenario)
        self.profile = profile or self.scenario.profile
        self.robot_xml = None if robot_xml is None else Path(robot_xml).expanduser().resolve()
        if self.robot_xml is None:
            self.model, self.data = load_model(self.asset, profile=self.profile)
        else:
            self.model, self.data = compose_with_robot(
                self.asset, self.robot_xml, profile=self.profile
            )
        self.seed = seed
        self.env_count = int(env_count)
        self.route = None
        if scenario == "slope_basic":
            from .slope_catalog import slope_catalog
            from .slope_routes import select_slope_route
            from .slope_runtime import reset_slope_spawn

            self.route = select_slope_route(slope_catalog(self.asset), patch)
            if self.robot_xml is not None:
                reset_slope_spawn(self.model, self.data, self.route, direction=direction)
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.recorder = TelemetryRecorder()
        self._last_steps_per_second: float | None = None
        self._qpos0 = np.array(self.data.qpos, dtype=float, copy=True)
        self._qvel0 = np.array(self.data.qvel, dtype=float, copy=True)

    @property
    def observation(self) -> np.ndarray:
        return np.concatenate((np.asarray(self.data.qpos), np.asarray(self.data.qvel))).copy()

    def reset(
        self,
        *,
        qpos: np.ndarray | None = None,
        qvel: np.ndarray | None = None,
    ) -> np.ndarray:
        import mujoco

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._qpos0 if qpos is None else np.asarray(qpos, dtype=float)
        self.data.qvel[:] = self._qvel0 if qvel is None else np.asarray(qvel, dtype=float)
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.recorder = TelemetryRecorder()
        self._last_steps_per_second = None
        return self.observation

    def step(self, action: np.ndarray | list[float] | None = None) -> StepResult:
        import mujoco

        if action is not None:
            values = np.asarray(action, dtype=float).reshape(-1)
            if values.size != int(self.model.nu):
                raise ValueError(f"action has {values.size} values; model expects {self.model.nu}")
            self.data.ctrl[:] = values
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        warnings = {
            mujoco.mjtWarning(index).name: int(item.number)
            for index, item in enumerate(self.data.warning)
            if int(item.number)
        }
        finite = bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        info = {
            "time_s": float(self.data.time),
            "contacts": int(self.data.ncon),
            "warnings": warnings,
            "finite": finite,
        }
        return StepResult(self.observation, not finite, False, info)

    def run(
        self,
        steps: int,
        *,
        controller: ControlCallback | None = None,
        telemetry: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        started = time.perf_counter()
        result: StepResult | None = None
        for step in range(steps):
            if controller is not None:
                controller(self.model, self.data, step=step, mode="policy")
            result = self.step()
            row = self.recorder.record(
                self.model, self.data, step=step, elapsed_s=time.perf_counter() - started
            )
            if telemetry is not None:
                telemetry(row)
            if result.terminated:
                break
        elapsed = time.perf_counter() - started
        self._last_steps_per_second = len(self.recorder.rows) / elapsed if elapsed else None
        return {
            "scenario_id": self.scenario.scenario_id,
            "steps": len(self.recorder.rows),
            "sim_time_s": float(self.data.time),
            "steps_per_second": self._last_steps_per_second,
            "terminated": bool(result.terminated) if result else False,
            "warnings": self.recorder.rows[-1]["warnings"] if self.recorder.rows else {},
            "cpu_rss_bytes": _cpu_rss_bytes(),
            "gpu_memory_bytes": None,
        }

    def metadata(self) -> dict[str, Any]:
        import mujoco

        descriptor = scenario_descriptor(self.asset, self.scenario.scenario_id, profile=self.profile)
        robot_hash = None
        if self.robot_xml is not None:
            robot_hash = hashlib.sha256(self.robot_xml.read_bytes()).hexdigest()
        return {
            "scenario_id": self.scenario.scenario_id,
            "source_manifest_sha256": self.asset.manifest_sha256,
            "slope_route": self.route,
            "profile": self.profile,
            "profile_hash": descriptor["profile_hash"],
            "heightfield_samples_sha256": self.asset.collision["samples_sha256"],
            "robot_mjcf_sha256": robot_hash,
            "timestep_s": float(self.model.opt.timestep),
            "solver": mujoco.mjtSolver(int(self.model.opt.solver)).name,
            "seed": self.seed,
            "env_count": self.env_count,
            "steps_per_second": self._last_steps_per_second,
            "memory": {
                "cpu_rss_bytes": _cpu_rss_bytes(),
                "gpu_memory_bytes": None,
            },
        }


def _cpu_rss_bytes() -> int | None:
    if resource is None:
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


__all__ = ["MuJoCoScenario", "StepResult", "TelemetryRecorder"]
