"""Minimal external controller template for ``rmuc2026-field run``.

Copy this module into a user's controller project and replace ``step`` with
the robot-specific observation/action mapping.  The field package never
assumes a joint order.
"""


class ExampleTurnController:
    def reset(self, model, data, spawn, seed):
        del model, data, spawn, seed

    def step(self, model, data, command, step_index):
        del command, step_index
        if getattr(model, "nu", 0):
            data.ctrl[:] = 0.0

    def observe(self, model, data):
        del model, data
        return None


def make(model=None, data=None):
    del model, data
    return ExampleTurnController()
