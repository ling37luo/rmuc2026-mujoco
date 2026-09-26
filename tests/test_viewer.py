from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import rmuc2026_mujoco.viewer as viewer_module
from rmuc2026_mujoco.viewer import (
    ForegroundWindow,
    SafePassiveViewerSession,
    ViewerFocusGuard,
    X11ViewerKeyInterceptor,
    stop_keyboard_listener,
    update_viewer_status_overlay,
    viewer_viewport_size,
    wait_for_passive_viewer_shutdown,
)


class _StopDisplay:
    def __init__(self, listener: "_KeyboardListener", events: list[str]) -> None:
        self.listener = listener
        self.events = events

    def record_disable_context(self, context: object) -> None:
        assert context is self.listener._context
        self.events.append("listener.disable_record")
        self.listener.alive = False

    def flush(self) -> None:
        self.events.append("listener.flush")


class _KeyboardListener:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.alive = True
        self._context = object()
        self._display_stop = _StopDisplay(self, events)

    def stop(self) -> None:
        self.events.append("listener.stop")

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float) -> None:
        assert timeout > 0.0
        self.events.append("listener.join")


class _PassiveViewer:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.simulator = object()
        self.polls_after_close = 0

    def close(self) -> None:
        self.events.append("viewer.close")

    def _sim(self) -> object | None:
        self.events.append("viewer.poll")
        self.polls_after_close += 1
        if self.polls_after_close >= 3:
            self.simulator = None
        return self.simulator


class _CloseFailureViewer:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def close(self) -> None:
        self.events.append("viewer.close")
        raise RuntimeError("native close failed")

    def _sim(self) -> None:
        self.events.append("viewer.poll")
        return None


def test_keyboard_listener_uses_xorg_stop_connection_and_joins() -> None:
    events: list[str] = []
    listener = _KeyboardListener(events)

    assert stop_keyboard_listener(listener) is True
    assert events == [
        "listener.stop",
        "listener.disable_record",
        "listener.flush",
        "listener.join",
    ]

    assert stop_keyboard_listener(listener) is True
    assert events[-1] == "listener.stop"


def test_passive_viewer_waits_for_native_simulator_release() -> None:
    events: list[str] = []
    viewer = _PassiveViewer(events)
    viewer.close()

    assert wait_for_passive_viewer_shutdown(viewer) is True
    assert events == ["viewer.close", "viewer.poll", "viewer.poll", "viewer.poll"]


def test_session_preserves_badqacc_error_after_ordered_native_cleanup() -> None:
    events: list[str] = []
    listener = _KeyboardListener(events)
    viewer = _PassiveViewer(events)

    with pytest.raises(FloatingPointError, match="deliberate BADQACC"):
        with SafePassiveViewerSession(viewer, listener) as active_viewer:
            assert active_viewer is viewer
            raise FloatingPointError("deliberate BADQACC")

    assert events == [
        "listener.stop",
        "listener.disable_record",
        "listener.flush",
        "listener.join",
        "viewer.close",
        "viewer.poll",
        "viewer.poll",
        "viewer.poll",
    ]


def test_session_does_not_mask_badqacc_when_native_close_also_fails(capsys) -> None:
    events: list[str] = []
    viewer = _CloseFailureViewer(events)

    with pytest.raises(FloatingPointError, match="deliberate BADQACC"):
        with SafePassiveViewerSession(viewer):
            raise FloatingPointError("deliberate BADQACC")

    assert events == ["viewer.close", "viewer.poll"]
    assert "RuntimeError: native close failed" in capsys.readouterr().err


def test_session_reports_native_close_failure_without_body_error() -> None:
    viewer = _CloseFailureViewer([])

    with pytest.raises(RuntimeError, match="native close failed"):
        with SafePassiveViewerSession(viewer):
            pass


def test_focus_guard_binds_one_owned_mujoco_window_and_fails_closed() -> None:
    foreground: list[ForegroundWindow | None] = [None]
    guard = ViewerFocusGuard(lambda: foreground[0], process_id=41)

    assert guard() is False
    foreground[0] = ForegroundWindow(7, 99, "MuJoCo : other process")
    assert guard() is False
    foreground[0] = ForegroundWindow(7, 41, "Terminal")
    assert guard() is False

    foreground[0] = ForegroundWindow(7, 41, "MuJoCo : rmuc2026_field")
    assert guard() is True
    assert guard.target_window_id == 7

    foreground[0] = ForegroundWindow(8, 41, "MuJoCo : another viewer")
    assert guard() is False
    foreground[0] = ForegroundWindow(7, 41, "MuJoCo : rmuc2026_field")
    assert guard() is True


def test_focus_guard_treats_probe_failure_as_not_focused() -> None:
    def failed_probe() -> None:
        raise RuntimeError("window manager unavailable")

    assert ViewerFocusGuard(failed_probe, process_id=41)() is False


def test_live_viewport_ignores_panels_and_rejects_invalid_dimensions() -> None:
    class Viewport:
        width = 913
        height = 720

    class Viewer:
        viewport = Viewport()

    viewer = Viewer()
    assert viewer_viewport_size(viewer) == (913, 720)
    viewer.viewport.width = 0
    assert viewer_viewport_size(viewer) is None


def test_status_overlay_uses_native_bottom_right_text_when_available() -> None:
    class Viewer:
        texts: list[tuple[int, int, str, str]] = []

        def set_texts(self, value) -> None:
            self.texts.append(value)

    viewer = Viewer()
    assert (
        update_viewer_status_overlay(
            viewer,
            lighting="flat",
            livery="off",
            camera="overview",
            shortcuts=True,
        )
        is True
    )
    font, position, title, content = viewer.texts[-1]
    assert (font, position, title) == (100, 3, "RMUC 2026 FIELD")
    assert "lighting  flat" in content
    assert "livery   off" in content
    assert "camera   overview" in content
    assert "L lighting | G livery | Esc close" in content


def test_status_overlay_distinguishes_passive_robot_and_unavailable_shortcuts() -> None:
    class Viewer:
        texts: list[tuple[int, int, str, str]] = []

        def set_texts(self, value) -> None:
            self.texts.append(value)

    viewer = Viewer()
    assert update_viewer_status_overlay(
        viewer,
        lighting="flat",
        livery="off",
        camera="overview",
        shortcuts=False,
        robot_control="view only (no controller)",
    )
    content = viewer.texts[-1][3]
    assert "L/G unavailable; use --lighting/--livery" in content
    assert "robot    view only (no controller)" in content


def test_status_overlay_falls_back_without_native_text_api() -> None:
    assert (
        update_viewer_status_overlay(
            object(),
            lighting="flat",
            livery="off",
            camera="native",
            shortcuts=False,
        )
        is False
    )


def test_x11_key_interceptor_releases_every_passive_grab() -> None:
    class Connection:
        events: list[str] = []

        def pending_events(self) -> int:
            return 0

        def next_event(self) -> None:
            raise AssertionError("no event should be pending")

        def sync(self) -> None:
            self.events.append("sync")

        def close(self) -> None:
            self.events.append("close")

    class Window:
        id = 17
        released: list[tuple[int, int]] = []

        def ungrab_key(self, keycode: int, modifiers: int, *, onerror=None) -> None:
            assert callable(onerror)
            self.released.append((keycode, modifiers))

    connection = Connection()
    window = Window()
    interceptor = X11ViewerKeyInterceptor(connection, window, ((46, 32768), (42, 32768)))

    assert interceptor.window_id == 17
    interceptor.close()
    interceptor.close()

    assert window.released == [(46, 32768), (42, 32768)]
    assert connection.events == ["sync", "close"]


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="the grabbed-key dispatch path is Linux/X11 only",
)
def test_x11_key_interceptor_dispatches_grabbed_press_and_release() -> None:
    from Xlib import X

    class Connection:
        def __init__(self) -> None:
            self.events = [
                SimpleNamespace(type=X.KeyPress, detail=46),
                SimpleNamespace(type=X.KeyRelease, detail=46),
                SimpleNamespace(type=X.KeyPress, detail=42),
                SimpleNamespace(type=X.KeyRelease, detail=42),
                SimpleNamespace(type=X.KeyPress, detail=99),
            ]

        def pending_events(self) -> int:
            return len(self.events)

        def next_event(self):
            return self.events.pop(0)

    class StopAfterOneDrain:
        def __init__(self) -> None:
            self.calls = 0

        def wait(self, _timeout: float) -> bool:
            self.calls += 1
            return self.calls > 1

    interceptor = object.__new__(X11ViewerKeyInterceptor)
    interceptor._connection = Connection()
    interceptor._key_names = {46: "L", 42: "G"}
    interceptor._stop_event = StopAfterOneDrain()
    received: list[tuple[str, bool]] = []
    interceptor.set_key_callback(lambda name, is_press: received.append((name, is_press)))

    interceptor._drain_events()

    assert received == [("L", True), ("L", False), ("G", True), ("G", False)]


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="the live keyboard grab path is Linux/X11 only",
)
def test_x11_async_grab_error_rolls_back_and_fails_closed(monkeypatch) -> None:
    from Xlib import XK, display

    class Connection:
        sync_count = 0
        closed = False

        def keysym_to_keycode(self, keysym: int) -> int:
            mapping = {
                XK.string_to_keysym("l"): 46,
                XK.string_to_keysym("g"): 42,
                XK.string_to_keysym("Num_Lock"): 77,
                XK.string_to_keysym("Scroll_Lock"): 78,
            }
            return mapping[keysym]

        def get_modifier_mapping(self):
            return [[], [], [], [], [77], [], [], []]

        def sync(self) -> None:
            self.sync_count += 1

        def close(self) -> None:
            self.closed = True

    class Window:
        id = 17
        requested: list[tuple[int, int]] = []
        released: list[tuple[int, int]] = []

        def grab_key(
            self,
            keycode: int,
            modifiers: int,
            _owner_events: bool,
            _pointer_mode: int,
            _keyboard_mode: int,
            *,
            onerror,
        ) -> None:
            self.requested.append((keycode, modifiers))
            if len(self.requested) == 2:
                assert onerror(RuntimeError("already grabbed"), object()) is True

        def ungrab_key(self, keycode: int, modifiers: int, *, onerror) -> None:
            assert callable(onerror)
            assert onerror(RuntimeError("already released"), object()) is True
            self.released.append((keycode, modifiers))

    connection = Connection()
    window = Window()
    # create_viewer_key_interceptor only checks that a display is *named* before
    # it builds grabs; the connection and window below are fakes, so the grab
    # path must be reachable on a headless runner that has no X server at all.
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(display, "Display", lambda: connection)
    monkeypatch.setattr(viewer_module, "_find_x11_viewer_window", lambda *_args: window)

    assert viewer_module.create_viewer_key_interceptor() is None
    assert window.released == window.requested
    assert connection.sync_count == 2
    assert connection.closed is True
