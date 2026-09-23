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
