"""Safe native-viewer lifecycle helpers shared by field integrations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
import sys
import threading
import time
from typing import Any


KEYBOARD_LISTENER_SHUTDOWN_TIMEOUT_S = 2.0
PASSIVE_VIEWER_SHUTDOWN_TIMEOUT_S = 5.0
PASSIVE_VIEWER_SHUTDOWN_POLL_S = 0.01
VIEWER_WINDOW_DISCOVERY_TIMEOUT_S = 2.0
VIEWER_WINDOW_DISCOVERY_POLL_S = 0.02


@dataclass(frozen=True)
class ForegroundWindow:
    """Small, backend-independent description of the focused desktop window."""

    window_id: int
    process_id: int
    title: str


class ViewerFocusGuard:
    """Allow global-key callbacks only for one focused MuJoCo viewer window.

    The first focused MuJoCo window owned by this process becomes the target.
    Thereafter, another window from the same process cannot accidentally take
    over the shortcuts.  Missing or ambiguous window-manager data always
    returns ``False`` so the launch-time display switches remain the fallback.
    """

    def __init__(
        self,
        probe: Callable[[], ForegroundWindow | None],
        *,
        process_id: int | None = None,
        target_window_id: int | None = None,
    ) -> None:
        self._probe = probe
        self._process_id = os.getpid() if process_id is None else int(process_id)
        self._target_window_id = target_window_id

    @property
    def target_window_id(self) -> int | None:
        return self._target_window_id

    def __call__(self) -> bool:
        try:
            window = self._probe()
        except Exception:
            return False
        if window is None:
            return False
        if window.process_id != self._process_id:
            return False
        if "mujoco" not in window.title.casefold():
            return False
        if self._target_window_id is None:
            self._target_window_id = window.window_id
        return window.window_id == self._target_window_id


def create_viewer_focus_guard() -> ViewerFocusGuard | None:
    """Create a fail-closed foreground-window check for the current platform."""

    probe = _platform_foreground_window_probe()
    return None if probe is None else ViewerFocusGuard(probe)


class X11ViewerKeyInterceptor:
    """Own passive grabs for viewer-local keys so MuJoCo cannot also handle them."""

    def __init__(
        self,
        connection: Any,
        window: Any,
        grabs: tuple[tuple[int, int], ...],
    ) -> None:
        self._connection = connection
        self._window = window
        self._grabs = grabs
        self._closed = False
        self._stop_event = threading.Event()
        self._drain_thread = threading.Thread(
            target=self._drain_events,
            name="rmuc2026-x11-key-grab",
            daemon=True,
        )
        self._drain_thread.start()

    @property
    def window_id(self) -> int:
        return int(self._window.id)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        self._drain_thread.join(KEYBOARD_LISTENER_SHUTDOWN_TIMEOUT_S)
        _release_x11_grabs(self._connection, self._window, self._grabs)
        try:
            self._connection.close()
        except Exception:
            pass

    def _drain_events(self) -> None:
        while not self._stop_event.wait(0.02):
            try:
                while self._connection.pending_events():
                    self._connection.next_event()
            except Exception:
                return


class FocusScopedKeyboardListener:
    """Release viewer-local key grabs together with the pynput listener."""

    def __init__(self, listener: Any, interceptor: X11ViewerKeyInterceptor) -> None:
        self._listener = listener
        self._interceptor = interceptor

    def stop(self) -> None:
        try:
            self._listener.stop()
        finally:
            self._interceptor.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._listener, name)


def create_viewer_key_interceptor(
    keys: tuple[str, ...] = ("l", "g"),
) -> tuple[X11ViewerKeyInterceptor, ViewerFocusGuard] | None:
    """Grab only L/G on this process's X11 MuJoCo window, or fail closed.

    X11 passive grabs activate only while the target window owns keyboard
    focus.  Other applications keep receiving the same keys.  Platforms that
    cannot make this selective guarantee keep launch-time display options but
    do not install interactive shortcuts.
    """

    if not sys.platform.startswith("linux") or not os.environ.get("DISPLAY"):
        return None
    try:
        from Xlib import X, XK, display

        connection = display.Display()
    except Exception:
        return None
    try:
        window = _find_x11_viewer_window(connection, os.getpid())
        if window is None:
            connection.close()
            return None
        keycodes = tuple(
            int(connection.keysym_to_keycode(XK.string_to_keysym(key))) for key in keys
        )
        if any(keycode <= 0 for keycode in keycodes):
            connection.close()
            return None
        # Reserve plain L/G with active lock-key combinations.
        # Modified native shortcuts such as Alt+L and Alt+G remain available.
        lock_masks = {X.LockMask}
        modifier_mapping = connection.get_modifier_mapping()
        for keysym_name in ("Num_Lock", "Scroll_Lock"):
            lock_keycode = int(connection.keysym_to_keycode(XK.string_to_keysym(keysym_name)))
            for index, keycodes_for_modifier in enumerate(modifier_mapping):
                if lock_keycode in keycodes_for_modifier:
                    lock_masks.add(1 << index)
        passive_modifiers = (0,)
        for mask in sorted(lock_masks):
            passive_modifiers += tuple(modifiers | mask for modifiers in passive_modifiers)
        grabs = tuple(
            (keycode, modifiers) for keycode in keycodes for modifiers in passive_modifiers
        )
        grab_errors: list[Any] = []

        def record_grab_error(error: Any, _request: Any) -> bool:
            grab_errors.append(error)
            return True

        for keycode, modifiers in grabs:
            window.grab_key(
                keycode,
                modifiers,
                False,
                X.GrabModeAsync,
                X.GrabModeAsync,
                onerror=record_grab_error,
            )
        connection.sync()
        if grab_errors:
            _release_x11_grabs(connection, window, grabs)
            connection.close()
            return None
    except Exception:
        try:
            connection.close()
        except Exception:
            pass
        return None
    interceptor = X11ViewerKeyInterceptor(connection, window, grabs)
    guard = ViewerFocusGuard(
        _x11_foreground_window,
        process_id=os.getpid(),
        target_window_id=interceptor.window_id,
    )
    return interceptor, guard


def _release_x11_grabs(
    connection: Any,
    window: Any,
    grabs: tuple[tuple[int, int], ...],
) -> None:
    """Best-effort rollback for a complete or partially installed grab set."""

    def ignore_error(_error: Any, _request: Any) -> bool:
        return True

    for keycode, modifiers in grabs:
        try:
            window.ungrab_key(keycode, modifiers, onerror=ignore_error)
        except Exception:
            pass
    try:
        connection.sync()
    except Exception:
        pass


def viewer_viewport_size(viewer: Any) -> tuple[int, int] | None:
    """Read the live 3D framebuffer rectangle, excluding MuJoCo side panels."""

    try:
        viewport = viewer.viewport
        width = int(viewport.width)
        height = int(viewport.height)
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def update_viewer_status_overlay(
    viewer: Any,
    *,
    lighting: str,
    livery: str,
    camera: str,
    shortcuts: bool,
    robot_control: str | None = None,
) -> bool:
    """Put persistent display state in the native viewer when supported.

    ``Handle.set_texts`` is available in current MuJoCo releases.  Older
    handles remain usable: callers can keep the terminal status as a fallback.
    Numeric enum values intentionally avoid importing MuJoCo during unit tests
    (100 is the normal font scale and 3 is the bottom-right grid position).
    """

    set_texts = getattr(viewer, "set_texts", None)
    if not callable(set_texts):
        return False
    shortcut_text = (
        "L lighting | G livery | Esc close"
        if shortcuts
        else "L/G unavailable; use --lighting/--livery"
    )
    content = f"lighting  {lighting}\nlivery   {livery}\ncamera   {camera}\n{shortcut_text}"
    if robot_control is not None:
        content += f"\nrobot    {robot_control}"
    try:
        set_texts((100, 3, "RMUC 2026 FIELD", content))
    except Exception:
        return False
    return True


def _platform_foreground_window_probe() -> Callable[[], ForegroundWindow | None] | None:
    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY"):
            return None
        try:
            import Xlib  # noqa: F401
        except ImportError:
            return None
        return _x11_foreground_window
    if sys.platform == "win32":
        return _windows_foreground_window
    if sys.platform == "darwin":
        try:
            import AppKit  # noqa: F401
            import Quartz  # noqa: F401
        except ImportError:
            return None
        return _macos_foreground_window
    return None


def _x11_foreground_window() -> ForegroundWindow | None:
    """Read EWMH foreground-window metadata through a short-lived connection."""

    try:
        from Xlib import X, display

        connection = display.Display()
    except Exception:
        return None
    try:
        root = connection.screen().root
        active = root.get_full_property(
            connection.intern_atom("_NET_ACTIVE_WINDOW"),
            X.AnyPropertyType,
        )
        if active is None or len(active.value) != 1:
            return None
        window_id = int(active.value[0])
        if window_id == 0:
            return None
        window = connection.create_resource_object("window", window_id)
        pid_property = window.get_full_property(
            connection.intern_atom("_NET_WM_PID"),
            X.AnyPropertyType,
        )
        if pid_property is None or len(pid_property.value) != 1:
            return None
        title_property = window.get_full_property(
            connection.intern_atom("_NET_WM_NAME"),
            X.AnyPropertyType,
        )
        raw_title = title_property.value if title_property is not None else window.get_wm_name()
        title = _window_title(raw_title)
        if not title:
            return None
        return ForegroundWindow(window_id, int(pid_property.value[0]), title)
    except Exception:
        return None
    finally:
        try:
            connection.close()
        except Exception:
            pass


def _find_x11_viewer_window(connection: Any, process_id: int) -> Any | None:
    """Wait briefly for the one EWMH client window created by launch_passive."""

    from Xlib import X

    root = connection.screen().root
    client_list_atom = connection.intern_atom("_NET_CLIENT_LIST")
    deadline = time.monotonic() + VIEWER_WINDOW_DISCOVERY_TIMEOUT_S
    while True:
        matches = []
        clients = root.get_full_property(client_list_atom, X.AnyPropertyType)
        for window_id in () if clients is None else clients.value:
            try:
                window = connection.create_resource_object("window", int(window_id))
                pid_property = window.get_full_property(
                    connection.intern_atom("_NET_WM_PID"),
                    X.AnyPropertyType,
                )
                if pid_property is None or int(pid_property.value[0]) != process_id:
                    continue
                title_property = window.get_full_property(
                    connection.intern_atom("_NET_WM_NAME"),
                    X.AnyPropertyType,
                )
                raw_title = (
                    title_property.value if title_property is not None else window.get_wm_name()
                )
                if "mujoco" in _window_title(raw_title).casefold():
                    matches.append(window)
            except Exception:
                continue
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1 or time.monotonic() >= deadline:
            return None
        time.sleep(VIEWER_WINDOW_DISCOVERY_POLL_S)


def _windows_foreground_window() -> ForegroundWindow | None:
    try:
        import ctypes

        user32 = ctypes.windll.user32
        window_id = int(user32.GetForegroundWindow())
        if window_id == 0:
            return None
        process_id = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(window_id, ctypes.byref(process_id))
        title_length = int(user32.GetWindowTextLengthW(window_id))
        title_buffer = ctypes.create_unicode_buffer(title_length + 1)
        user32.GetWindowTextW(window_id, title_buffer, title_length + 1)
        title = title_buffer.value
        if not title:
            return None
        return ForegroundWindow(window_id, int(process_id.value), title)
    except Exception:
        return None


def _macos_foreground_window() -> ForegroundWindow | None:
    try:
        import AppKit
        import Quartz

        process_id = int(
            AppKit.NSWorkspace.sharedWorkspace().frontmostApplication().processIdentifier()
        )
        options = (
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
        )
        windows = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)
        for window in windows:
            if int(window.get(Quartz.kCGWindowOwnerPID, -1)) != process_id:
                continue
            title = str(window.get(Quartz.kCGWindowName, ""))
            if title:
                return ForegroundWindow(int(window[Quartz.kCGWindowNumber]), process_id, title)
    except Exception:
        return None
    return None


def _window_title(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    try:
        return bytes(value).decode("utf-8", errors="replace")
    except (TypeError, ValueError):
        return ""


def stop_keyboard_listener(listener: Any) -> bool:
    """Stop and join a pynput listener before releasing viewer resources.

    pynput 1.8.2's Xorg backend can leave its RECORD thread blocked when it
    disables the context through the recording connection itself. The backend
    already owns a second connection specifically for stopping that context;
    use it as a bounded fallback so no keyboard callback can outlive the model,
    data, or viewer it controls.
    """

    stop_succeeded = True
    try:
        listener.stop()
    except Exception:
        stop_succeeded = False
    try:
        alive = bool(listener.is_alive())
    except Exception:
        alive = False
    if not alive:
        return stop_succeeded
    stop_display = getattr(listener, "_display_stop", None)
    record_context = getattr(listener, "_context", None)
    if stop_display is not None and record_context is not None:
        try:
            stop_display.record_disable_context(record_context)
            stop_display.flush()
        except Exception:
            # The listener may have completed between is_alive() and this
            # fallback. Its public stop request is still authoritative.
            pass
    try:
        listener.join(KEYBOARD_LISTENER_SHUTDOWN_TIMEOUT_S)
    except Exception:
        # Callback failures must not prevent the native viewer from receiving
        # its close request and completing GL teardown.
        pass
    try:
        return stop_succeeded and not bool(listener.is_alive())
    except Exception:
        return True


def wait_for_passive_viewer_shutdown(viewer: Any) -> bool:
    """Wait until MuJoCo's daemon render thread has destroyed its GL context."""

    simulator_ref = getattr(viewer, "_sim", None)
    if not callable(simulator_ref):
        # Current MuJoCo exposes the render-loop lifetime through Handle._sim.
        # Keep a short grace period for a future Handle without that hook.
        time.sleep(PASSIVE_VIEWER_SHUTDOWN_POLL_S)
        return True
    deadline = time.monotonic() + PASSIVE_VIEWER_SHUTDOWN_TIMEOUT_S
    while simulator_ref() is not None:
        if time.monotonic() >= deadline:
            return False
        time.sleep(PASSIVE_VIEWER_SHUTDOWN_POLL_S)
    return True


@dataclass
class SafePassiveViewerSession:
    """Stop input first, then close and fully drain a passive MuJoCo viewer."""

    viewer: Any
    listener: Any | None = None

    def __enter__(self) -> Any:
        return self.viewer

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        listener_stopped = True if self.listener is None else stop_keyboard_listener(self.listener)
        cleanup_error: Exception | None = None
        try:
            self.viewer.close()
        except Exception as error:
            cleanup_error = error
        try:
            viewer_stopped = wait_for_passive_viewer_shutdown(self.viewer)
        except Exception as error:
            viewer_stopped = False
            if cleanup_error is None:
                cleanup_error = error
        if not listener_stopped:
            print(
                "RMUC2026_VIEWER_CLEANUP=WARNING: keyboard listener did not join before timeout",
                file=sys.stderr,
                flush=True,
            )
        if not viewer_stopped:
            print(
                "RMUC2026_VIEWER_CLEANUP=WARNING: MuJoCo render thread did not stop before timeout",
                file=sys.stderr,
                flush=True,
            )
        if cleanup_error is not None:
            print(
                f"RMUC2026_VIEWER_CLEANUP=WARNING: {type(cleanup_error).__name__}: {cleanup_error}",
                file=sys.stderr,
                flush=True,
            )
            if exc_type is None:
                raise cleanup_error
        return False
