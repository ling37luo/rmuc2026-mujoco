from types import SimpleNamespace

from rmuc2026_mujoco import control


def test_controller_factory_receives_selected_field_asset(monkeypatch):
    selected_asset = object()
    calls = []

    def make(model, data, *, mode, field_asset=None):
        assert field_asset is selected_asset
        assert mode == "policy"

        def step(model, data, *, step, mode):
            calls.append((model, data, step, mode))

        return step

    monkeypatch.setattr(control, "_load_module", lambda _: SimpleNamespace(make=make))
    model, data = object(), object()
    callback = control.load_controller(
        "user:make", model, data, mode="policy", field_asset=selected_asset
    )
    callback(model, data, step=7, mode="policy")
    assert calls == [(model, data, 7, "policy")]


def test_controller_adapter_keeps_keyboard_contract(monkeypatch):
    class UserController:
        viewer_keys = ("Up", "Down", "Left", "Right", "space")

        def __init__(self):
            self.presses = []

        def __call__(self, _model, _data, *, step, mode):
            assert (step, mode) == (1, "human")

        def press_name(self, name):
            self.presses.append(name)

        def release_name(self, name):
            self.presses.append(f"release:{name}")

    user = UserController()
    monkeypatch.setattr(
        control, "_load_module", lambda _: SimpleNamespace(make=lambda *_args, **_kwargs: user)
    )
    model, data = object(), object()
    callback = control.load_controller("user:make", model, data, mode="human")
    assert callback.viewer_keys == user.viewer_keys
    callback.press_name("W")
    callback.release_name("UP")
    callback(model, data, step=1, mode="human")
    assert user.presses == ["W", "release:UP"]
