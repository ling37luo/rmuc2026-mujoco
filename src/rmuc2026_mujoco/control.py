"""Minimal controller protocol used by the generic viewer and runner."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Protocol


class ControlCallback(Protocol):
    """Write the next actuator command into ``data``.

    The callback owns robot-specific actuator semantics.  It may write
    ``data.ctrl`` or apply forces directly; the field package does not inspect
    the robot's joint names.
    """

    def __call__(self, model: Any, data: Any, *, step: int, mode: str) -> None: ...


class NoOpController:
    """Controller for a passive viewer or a robot with its own external loop."""

    def __call__(self, model: Any, data: Any, *, step: int, mode: str) -> None:
        del model, data, step, mode


def load_controller(spec: str, model: Any, data: Any, *, mode: str) -> ControlCallback:
    """Load ``module:factory`` and accept either a callback or a factory."""

    if ":" not in spec:
        raise ValueError("controller must use module:object syntax")
    module_name, object_name = spec.split(":", 1)
    target = getattr(_load_module(module_name), object_name)
    if not callable(target):
        raise TypeError(f"controller target {spec!r} is not callable")
    try:
        candidate = target(model, data, mode=mode)
    except TypeError:
        candidate = target
    if not callable(candidate):
        raise TypeError(f"controller target {spec!r} did not return a callable")
    return _adapt_callback(candidate)


def _load_module(name: str) -> Any:
    """Load an installed module or a local ``examples.foo`` module.

    Console scripts launched by ``uv run`` do not always put the repository
    root on ``sys.path``.  Resolving a matching local ``.py`` file keeps the
    documented example command working without packaging private controllers.
    """

    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as import_error:
        relative = Path(name)
        if name.endswith(".py"):
            candidates = (relative,)
        else:
            relative = Path(*name.split("."))
            candidates = (relative.with_suffix(".py"), relative / "__init__.py")
        path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise import_error
        module_name = f"rmuc2026_user_controller_{abs(hash(path))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise import_error
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _adapt_callback(callback: Any) -> ControlCallback:
    signature = inspect.signature(callback)
    names = tuple(signature.parameters)

    def call(model: Any, data: Any, *, step: int, mode: str) -> None:
        if "step" in names or "mode" in names:
            callback(model, data, step=step, mode=mode)
        elif len(names) >= 3:
            callback(model, data, step)
        else:
            callback(model, data)

    for name in ("reset", "press_name", "configure_route"):
        method = getattr(callback, name, None)
        if callable(method):
            setattr(call, name, method)
    if hasattr(callback, "identity"):
        call.identity = callback.identity
    return call


__all__ = ["ControlCallback", "NoOpController", "load_controller"]
