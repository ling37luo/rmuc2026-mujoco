from __future__ import annotations

import json
import math
from pathlib import Path
from queue import SimpleQueue
import threading

import pytest

from rmuc2026_mujoco import DownloadedStep, FieldAsset
from rmuc2026_mujoco.cli import (
    _configure_camera,
    _controller_reserved_keys,
    _drain_controller_keys,
    _start_display_key_listener,
    _viewer_physics_substeps,
    build_parser,
    main,
)
from rmuc2026_mujoco.download import DownloadedRulebook


def test_verify_cli_emits_machine_readable_pass(field_asset_dir: Path, capsys) -> None:
    assert main(["verify", str(field_asset_dir)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "PASS"
    assert output["collision_shape"] == [3, 4]
    assert output["validation_status"] == "DRAFT_BLOCKED"


def test_source_cli_is_read_only_and_reports_unverified_redistribution(capsys) -> None:
    assert main(["source"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["download_started"] is False
    assert output["included_in_package"] is False
    assert output["redistribution_permission"] == "UNVERIFIED"
    assert output["redistribution_authorized"] is False
    assert output["license_review"] == "NO_EXPLICIT_GRANT_FOUND"
    review = output["rulebook_review"]
    assert review["official_rule_centre"] == (
        "https://bbs.robomaster.com/wiki/20204847/809871?source=7"
    )
    assert review["latest_reviewed_version"] == "V2.2.0"
    assert review["latest_reviewed_publication_date"] == "2026-08-07"
    assert review["latest_reviewed_size_bytes"] == 21_018_604
    assert review["latest_reviewed_sha256"] == (
        "88ae3c7d0bbd7312c8095eff2603910437e292a0b53f46c9378f26144c09a6f5"
    )
    assert review["latest_reviewed_url"].endswith("%E5%86%8CV2.2.0%EF%BC%8820260807%EF%BC%89.pdf")
    assert review["comparison_scope"] == "rulebook_chapter_4_only"
    assert review["static_field_chapter_vs_v2_0_0"] == "NO_CHANGE_DETECTED"
    assert review["comparison_method"] == "normalised_text_and_embedded_image_hashes"
    assert review["future_updates_require_new_review"] is True


def test_setup_cli_has_a_safe_user_cache_default() -> None:
    args = build_parser().parse_args(["setup", "local-field-pack"])

    assert args.step_cache.name == "RMUC2026_V2.0.0.stp"
    assert args.step_cache.parent.name == "rmuc2026-mujoco"
    assert args.rulebook_cache.name == "RMUC2026_rulebook_V2.0.0.pdf"
    assert args.output == Path("local-field-pack")


def test_view_cli_defaults_to_full_without_friction_override() -> None:
    args = build_parser().parse_args(["view", "local-field-pack"])

    assert args.profile == "full"
    assert args.friction_preset is None
    assert args.camera == "overview"


def test_view_cli_accepts_headless_profile_and_unofficial_friction() -> None:
    args = build_parser().parse_args(
        ["view", "local-field-pack", "--profile", "collision_only", "--friction", "low"]
    )

    assert args.profile == "collision_only"
    assert args.friction_preset == "low"


def test_view_cli_accepts_generic_robot_control_options() -> None:
    args = build_parser().parse_args(
        [
            "view",
            "field",
            "--robot",
            "robot.xml",
            "--control",
            "policy",
            "--controller",
            "controller:make",
            "--headless",
            "--steps",
            "12",
        ]
    )
    assert args.robot == Path("robot.xml")
    assert args.control == "policy"
    assert args.controller == "controller:make"
    assert args.headless is True
    assert args.steps == 12


def test_run_cli_accepts_slope_descriptor_for_isaac() -> None:
    args = build_parser().parse_args(
        ["run", "field", "--robot", "robot.xml", "--scenario", "slope_basic", "--backend", "isaac"]
    )
    assert args.scenario == "slope_basic"
    assert args.backend == "isaac"


def test_scenarios_cli_lists_public_registry(capsys) -> None:
    assert main(["scenarios"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert [item["scenario_id"] for item in output["scenarios"]] == [
        "full_eval",
        "turn_basic",
        "stairs_basic",
        "slope_basic",
        "fly_ramp_north",
        "fly_ramp_south",
        "boundary_contact",
    ]


def test_overview_camera_uses_verified_field_bounds(field_asset_dir: Path) -> None:
    class Camera:
        lookat = [99.0, 99.0, 99.0]
        distance = 0.0
        azimuth = 0.0
        elevation = 0.0

    class Viewport:
        width = 1600
        height = 900

    class Global:
        fovy = 45.0

    class Visual:
        global_ = Global()

    class Model:
        vis = Visual()

    class Viewer:
        cam = Camera()
        viewport = Viewport()
        m = Model()

    viewer = Viewer()
    viewport = _configure_camera(viewer, FieldAsset.open(field_asset_dir), "overview")

    assert viewport == (1600, 900)
    assert viewer.cam.lookat == [0.5, 0.0, 0.75]
    assert math.isclose(viewer.cam.distance, 6.675981399053374)
    assert viewer.cam.azimuth == 135.0
    assert viewer.cam.elevation == -50.0

    viewer.viewport.width = 600
    viewer.viewport.height = 900
    _configure_camera(viewer, FieldAsset.open(field_asset_dir), "overview")
    assert math.isclose(viewer.cam.distance, 8.880680507502843)


def test_display_keys_are_focus_scoped_and_toggle_once_per_press(monkeypatch) -> None:
    class Key:
        esc = object()
        alt = object()

    class Listener:
        def __init__(self, **callbacks) -> None:
            self.callbacks = callbacks
            self.started = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.started = False

    class Interceptor:
        closed = False

        def close(self) -> None:
            self.closed = True

    class Keyboard:
        pass

    Keyboard.Key = Key
    Keyboard.Listener = Listener

    class Character:
        def __init__(self, char: str) -> None:
            self.char = char

    class Display:
        presses: list[str] = []

        def press_name(self, name: str) -> bool:
            self.presses.append(name)
            return name in {"L", "G"}

    focused = [True]
    now = [10.0]
    monkeypatch.setattr("rmuc2026_mujoco.cli.time.monotonic", lambda: now[0])
    changed: list[str] = []
    driving: list[str] = []
    close_requested = threading.Event()
    interceptor = Interceptor()
    listener = _start_display_key_listener(
        Display(),
        close_requested,
        on_change=lambda: changed.append("changed"),
        on_key=driving.append,
        focus_check=lambda: focused[0],
        keyboard_module=Keyboard,
        key_interceptor=interceptor,
    )
    assert listener is not None
    assert listener.started is True
    assert listener.callbacks["suppress"] is False
    press = listener.callbacks["on_press"]
    release = listener.callbacks["on_release"]

    l_key = Character("l")
    press(l_key)
    press(l_key)
    assert Display.presses == ["L"]
    assert changed == ["changed"]
    release(l_key)
    now[0] += 0.1
    press(l_key)
    assert Display.presses == ["L", "L"]

    release(l_key)
    focused[0] = False
    g_key = Character("g")
    press(g_key)
    focused[0] = True
    press(g_key)
    assert Display.presses == ["L", "L"]
    release(g_key)
    now[0] += 0.1
    press(g_key)
    assert Display.presses == ["L", "L", "G"]
    release(g_key)

    press(Key.alt)
    press(l_key)
    release(l_key)
    release(Key.alt)
    assert Display.presses == ["L", "L", "G"]

    w_key = Character("w")
    focused[0] = False
    press(w_key)
    release(w_key)
    assert driving == []
    focused[0] = True
    press(w_key)
    press(w_key)
    assert driving == ["W"]
    release(w_key)

    focused[0] = False
    assert press(Key.esc) is None
    assert close_requested.is_set() is False
    release(Key.esc)
    focused[0] = True
    assert press(Key.esc) is False
    assert close_requested.is_set() is True
    listener.stop()
    assert interceptor.closed is True


def test_display_keys_fail_closed_without_selective_native_interception(monkeypatch) -> None:
    class Keyboard:
        class Key:
            esc = object()

        class Listener:
            def __init__(self, **_callbacks) -> None:
                raise AssertionError("listener must not start without an interceptor")

    monkeypatch.setattr("rmuc2026_mujoco.cli.create_viewer_key_interceptor", lambda: None)

    assert (
        _start_display_key_listener(
            object(),
            threading.Event(),
            keyboard_module=Keyboard,
        )
        is None
    )

    with pytest.raises(ValueError, match="must be supplied together"):
        _start_display_key_listener(
            object(),
            threading.Event(),
            focus_check=lambda: True,
            keyboard_module=Keyboard,
        )


def test_generic_viewer_forwards_controller_keys_on_simulation_thread() -> None:
    class Controller:
        viewer_keys = ("W", "a", "s", "d", "Up", "Down", "Left", "Right", "space", "g", "W")

        def __init__(self) -> None:
            self.presses: list[str] = []

        def press_name(self, name: str) -> None:
            self.presses.append(name)

        def release_name(self, name: str) -> None:
            self.presses.append(f"release:{name}")

    controller = Controller()
    assert _controller_reserved_keys(controller) == (
        "w",
        "a",
        "s",
        "d",
        "Up",
        "Down",
        "Left",
        "Right",
        "space",
    )
    assert _controller_reserved_keys(object()) == ()

    keys: SimpleQueue[tuple[str, str]] = SimpleQueue()
    keys.put(("press", "UP"))
    keys.put(("press", "SPACE"))
    keys.put(("release", "UP"))
    _drain_controller_keys(keys, controller.press_name, controller.release_name)
    assert controller.presses == ["UP", "SPACE", "release:UP"]


def test_controller_viewer_key_declaration_rejects_non_character_keys() -> None:
    class Controller:
        viewer_keys = ("escape",)

        def press_name(self, _name: str) -> None:
            pass

    with pytest.raises(ValueError, match="character or named"):
        _controller_reserved_keys(Controller())


def test_generic_listener_forwards_arrow_and_space_press_release_in_order() -> None:
    class Key:
        esc = object()
        up = object()
        down = object()
        left = object()
        right = object()
        space = object()

    class Listener:
        def __init__(self, **callbacks) -> None:
            self.callbacks = callbacks

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    class Interceptor:
        def close(self) -> None:
            pass

    class Keyboard:
        pass

    Keyboard.Key = Key
    Keyboard.Listener = Listener

    class Display:
        def press_name(self, _name: str) -> bool:
            return False

    received: list[tuple[str, str]] = []
    listener = _start_display_key_listener(
        Display(),
        threading.Event(),
        on_key=lambda name: received.append(("press", name)),
        on_key_release=lambda name: received.append(("release", name)),
        focus_check=lambda: True,
        keyboard_module=Keyboard,
        key_interceptor=Interceptor(),
    )
    assert listener is not None
    press = listener.callbacks["on_press"]
    release = listener.callbacks["on_release"]
    press(Key.up)
    press(Key.up)
    release(Key.up)
    press(Key.space)
    release(Key.space)
    assert received == [
        ("press", "UP"),
        ("release", "UP"),
        ("press", "SPACE"),
        ("release", "SPACE"),
    ]
    listener.stop()


def test_grabbed_and_global_key_events_do_not_toggle_display_twice() -> None:
    class Key:
        esc = object()

    class Listener:
        def __init__(self, **callbacks) -> None:
            self.callbacks = callbacks

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    class Interceptor:
        callback = None

        def set_key_callback(self, callback) -> None:
            self.callback = callback

        def close(self) -> None:
            pass

    class Keyboard:
        pass

    Keyboard.Key = Key
    Keyboard.Listener = Listener

    class Character:
        def __init__(self, char: str) -> None:
            self.char = char

    class Display:
        presses: list[str] = []

        def press_name(self, name: str) -> bool:
            self.presses.append(name)
            return name in {"L", "G"}

    changed: list[str] = []
    interceptor = Interceptor()
    listener = _start_display_key_listener(
        Display(),
        threading.Event(),
        on_change=lambda: changed.append("changed"),
        focus_check=lambda: True,
        keyboard_module=Keyboard,
        key_interceptor=interceptor,
    )
    assert listener is not None
    assert interceptor.callback is not None
    press = listener.callbacks["on_press"]
    release = listener.callbacks["on_release"]
    for name in ("L", "G"):
        interceptor.callback(name, True)
        press(Character(name.lower()))
        interceptor.callback(name, False)
        release(Character(name.lower()))

    assert Display.presses == ["L", "G"]
    assert changed == ["changed", "changed"]
    listener.stop()


def test_generic_viewer_steps_about_one_display_frame() -> None:
    assert _viewer_physics_substeps(0.002) == 8
    assert _viewer_physics_substeps(0.001) == 17
    assert _viewer_physics_substeps(1.0 / 60.0) == 1
    assert _viewer_physics_substeps(0.000001) == 256


def test_setup_cli_downloads_then_builds_locally(monkeypatch, tmp_path: Path, capsys) -> None:
    step = tmp_path / "cache" / "RMUC2026_V2.0.0.stp"
    output = tmp_path / "runtime-pack"
    calls: list[tuple[str, Path]] = []

    def fake_download(destination, *, acknowledge_reference_only, progress):
        assert acknowledge_reference_only is True
        assert progress is not None
        calls.append(("download", Path(destination)))
        return DownloadedStep(
            path=step,
            size_bytes=1_254_821_405,
            sha256="8dfe9ebd761e44d91361b3e593bc05416329112217b58cb35800b3cde2ffae33",
            reused=False,
        )

    def fake_build(step_path, output_dir, **options):
        assert Path(step_path) == step
        assert options == {
            "target_visual_faces": 2_300_000,
            "heightfield_resolution_m": 0.02,
            "include_edge_void": False,
            "include_surface_guide": False,
            "rulebook_pdf": None,
        }
        calls.append(("build", Path(output_dir)))
        return {
            "artifact_type": "rmuc2026_mujoco_runtime_asset_pack",
            "status": "PASS",
            "validation_status": "DRAFT_BLOCKED",
        }

    monkeypatch.setattr("rmuc2026_mujoco.cli.download_official_step", fake_download)
    monkeypatch.setattr("rmuc2026_mujoco.cli.build_runtime_asset_pack", fake_build)

    assert (
        main(
            [
                "setup",
                str(output),
                "--step-cache",
                str(step),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert calls == [("download", step), ("build", output)]
    assert result["status"] == "PASS"
    assert result["official_step_reused"] is False


def test_setup_cli_rejects_unsafe_build_resolution(tmp_path: Path, capsys) -> None:
    assert (
        main(
            [
                "setup",
                str(tmp_path / "output"),
                "--heightfield-resolution",
                "0.5",
            ]
        )
        == 2
    )
    assert "within [0.01, 0.10]" in capsys.readouterr().err


def test_setup_with_surface_guide_downloads_verified_rulebook(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    step = tmp_path / "cache" / "RMUC2026_V2.0.0.stp"
    rulebook = tmp_path / "cache" / "rulebook.pdf"
    output = tmp_path / "runtime-pack"
    calls: list[str] = []

    monkeypatch.setattr(
        "rmuc2026_mujoco.cli.download_official_step",
        lambda *_args, **_kwargs: DownloadedStep(
            step,
            1_254_821_405,
            "8dfe9ebd761e44d91361b3e593bc05416329112217b58cb35800b3cde2ffae33",
            True,
        ),
    )

    def fake_rulebook(destination, **_kwargs):
        calls.append("rulebook")
        assert Path(destination) == rulebook
        return DownloadedRulebook(rulebook, 21_012_597, "5" * 64, False)

    def fake_build(step_path, output_dir, **options):
        calls.append("build")
        assert Path(step_path) == step
        assert Path(output_dir) == output
        assert options["include_surface_guide"] is True
        assert options["rulebook_pdf"] == rulebook
        assert options["heightfield_resolution_m"] == 0.01
        assert options["include_edge_void"] is True
        return {
            "artifact_type": "rmuc2026_mujoco_runtime_asset_pack",
            "status": "PASS",
            "validation_status": "DRAFT_BLOCKED",
        }

    monkeypatch.setattr(
        "rmuc2026_mujoco.cli.download_official_rulebook_v2_0_0",
        fake_rulebook,
    )
    monkeypatch.setattr("rmuc2026_mujoco.cli.build_runtime_asset_pack", fake_build)

    assert (
        main(
            [
                "setup",
                str(output),
                "--step-cache",
                str(step),
                "--rulebook-cache",
                str(rulebook),
                "--heightfield-resolution",
                "0.01",
                "--experimental-edge-void",
                "--include-surface-guide",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert calls == ["rulebook", "build"]
    assert result["official_rulebook_reused"] is False


def test_build_surface_guide_requires_local_rulebook(tmp_path: Path, capsys) -> None:
    assert (
        main(
            [
                "build",
                "--step",
                str(tmp_path / "field.step"),
                "--output",
                str(tmp_path / "pack"),
                "--include-surface-guide",
            ]
        )
        == 2
    )
    assert "--rulebook is required" in capsys.readouterr().err


def test_surface_cli_emits_geometry_and_claim_boundary(
    field_asset_dir: Path,
    capsys,
) -> None:
    assert (
        main(
            [
                "surface",
                str(field_asset_dir),
                "-0.5",
                "-0.5",
                "--window-radius",
                "0.51",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["height_m"] == 0.25
    assert output["query_schema_version"] == 1
    assert output["query_algorithm"] == "mujoco_hfield_triangle_v1"
    assert output["manifest_sha256"]
    assert output["heightfield_samples_sha256"]
    assert output["topology_verified"] is False
    assert "underpasses" in output["claim_boundary"]


def test_spawns_cli_reports_screening_contract_without_overclaiming(
    field_asset_dir: Path,
    capsys,
) -> None:
    assert (
        main(
            [
                "spawns",
                str(field_asset_dir),
                "--count",
                "2",
                "--footprint-radius",
                "0",
                "--boundary-margin",
                "0",
                "--minimum-separation",
                "1",
                "--max-slope-deg",
                "90",
                "--max-relief",
                "2",
                "--ground-height-min",
                "-1",
                "--ground-height-max",
                "2",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "RUNTIME_PROXY_SCREENED_NOT_TOPOLOGY_VERIFIED"
    assert output["query_schema_version"] == 1
    assert output["query_algorithm"] == "mujoco_hfield_triangle_spawn_screen_v1"
    assert output["manifest_sha256"]
    assert output["heightfield_samples_sha256"]
    assert output["candidate_count"] == 2
    assert output["topology_verified"] is False
    assert output["selection_contract"]["slope_interpolation"] == "mujoco_hfield_triangle"
    assert (
        output["selection_contract"]["ground_height_range_applies_to"] == "entire_screening_window"
    )
    assert all(candidate["topology_verified"] is False for candidate in output["candidates"])


def test_spawns_cli_rejects_inverted_height_range(field_asset_dir: Path, capsys) -> None:
    assert (
        main(
            [
                "spawns",
                str(field_asset_dir),
                "--ground-height-min",
                "1",
                "--ground-height-max",
                "-1",
            ]
        )
        == 2
    )
    assert "ordered" in capsys.readouterr().err
