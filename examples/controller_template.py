"""Tiny controller module for ``rmuc2026-field view --controller``.

Replace the body with a user's keyboard or policy logic.  The field package
does not assume actuator names, wheel layout, or an action space.
"""

from __future__ import annotations


def make(model, data, *, mode: str = "human"):
    del mode

    def control(model, data, *, step: int, mode: str) -> None:
        del model, step, mode
        if data.ctrl.size:
            data.ctrl[:] = 0.0

    return control
