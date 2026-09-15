from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tarfile
from types import ModuleType
import zipfile

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_release_contents.py"


def _load_audit_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("release_content_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit_module() -> ModuleType:
    return _load_audit_module()


def _wheel(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, mode="w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def _sdist(path: Path, source: Path, arcname: str) -> Path:
    source.write_text("from pathlib import Path\n", encoding="utf-8")
    with tarfile.open(path, mode="w:gz") as archive:
        archive.add(source, arcname=arcname)
    return path


def test_accepts_code_only_wheel_and_sdist(tmp_path: Path, audit_module: ModuleType) -> None:
    wheel = _wheel(
        tmp_path / "safe.whl",
        {
            "rmuc2026_mujoco/__init__.py": b"from .manifest import FieldAsset\n",
            "rmuc2026_mujoco-0.1.dist-info/METADATA": b"Summary: standalone from RL-Lab\n",
        },
    )
    sdist = _sdist(tmp_path / "safe.tar.gz", tmp_path / "module.py", "safe/module.py")

    assert audit_module.audit_archive(wheel)["status"] == "PASS"
    assert audit_module.audit_archive(sdist)["status"] == "PASS"


@pytest.mark.parametrize("asset", ["field.obj", "heightfield.npz", "official.step", "render.png"])
def test_rejects_packaged_field_assets(
    tmp_path: Path, audit_module: ModuleType, asset: str
) -> None:
    wheel = _wheel(tmp_path / "unsafe.whl", {f"package/data/{asset}": b"payload"})

    with pytest.raises(audit_module.ReleaseAuditError, match="forbidden asset member"):
        audit_module.audit_archive(wheel)


@pytest.mark.parametrize(
    "asset",
    [
        "robot.urdf",
        "robot.xacro",
        "world.sdf",
        "robot.mjcf",
        "policy.ckpt",
        "policy.safetensors",
        "policy.engine",
    ],
)
def test_rejects_mechanical_and_model_files(
    tmp_path: Path, audit_module: ModuleType, asset: str
) -> None:
    wheel = _wheel(tmp_path / "unsafe.whl", {f"package/data/{asset}": b"payload"})

    with pytest.raises(audit_module.ReleaseAuditError, match="forbidden asset member"):
        audit_module.audit_archive(wheel)


@pytest.mark.parametrize(
    "directory",
    ["runs", "artifacts", "checkpoints", "logs", "runtime_output", "runtime-output"],
)
def test_rejects_generated_runtime_directories(
    tmp_path: Path, audit_module: ModuleType, directory: str
) -> None:
    wheel = _wheel(
        tmp_path / "unsafe.whl",
        {f"package/{directory}/evidence.json": b'{"status":"PASS"}\n'},
    )

    with pytest.raises(audit_module.ReleaseAuditError, match="generated runtime directory"):
        audit_module.audit_archive(wheel)


@pytest.mark.parametrize(
    "payload",
    [
        b'<mujoco model="field"></mujoco>\n',
        b'<?xml version="1.0"?><robot name="wheel-leg"/>\n',
        b'<SDF version="1.9"><world name="arena"/></SDF>\n',
        b'<world name="arena"></world>\n',
    ],
)
def test_rejects_robot_or_scene_payload_hidden_as_xml(
    tmp_path: Path, audit_module: ModuleType, payload: bytes
) -> None:
    wheel = _wheel(tmp_path / "unsafe.whl", {"package/config.xml": payload})

    with pytest.raises(audit_module.ReleaseAuditError, match="robot/scene XML payload"):
        audit_module.audit_archive(wheel)


def test_allows_only_hash_pinned_original_xml_example(
    tmp_path: Path, audit_module: ModuleType
) -> None:
    example = Path(__file__).parents[1] / "examples" / "simple_robot.xml"
    member = "rmuc2026_mujoco-0.1/examples/simple_robot.xml"
    safe = _wheel(tmp_path / "safe.whl", {member: example.read_bytes()})
    tampered = _wheel(
        tmp_path / "tampered.whl",
        {member: example.read_bytes().replace(b"demo_robot", b"other_robot", 1)},
    )

    assert audit_module.audit_archive(safe)["status"] == "PASS"
    with pytest.raises(audit_module.ReleaseAuditError, match="robot/scene XML payload"):
        audit_module.audit_archive(tampered)


@pytest.mark.parametrize(
    "source",
    [
        b"CACHE = '" + b"/" + b"home/alice/Documents/RL-Lab/runs'\n",
        b"CACHE = r'C:" + b"\\" + b"Users\\alice\\RL-Lab'\n",
    ],
)
def test_rejects_host_absolute_paths(
    tmp_path: Path, audit_module: ModuleType, source: bytes
) -> None:
    wheel = _wheel(tmp_path / "unsafe.whl", {"package/config.py": source})

    with pytest.raises(audit_module.ReleaseAuditError, match="host absolute path"):
        audit_module.audit_archive(wheel)


@pytest.mark.parametrize(
    "source",
    [
        b"from rmuc_field.pipeline import convert\n",
        b"import wheel_legged_gym.envs\n",
        b"from scripts.private_builder import build\n",
    ],
)
def test_rejects_rl_lab_internal_imports(
    tmp_path: Path, audit_module: ModuleType, source: bytes
) -> None:
    wheel = _wheel(tmp_path / "unsafe.whl", {"package/module.py": source})

    with pytest.raises(audit_module.ReleaseAuditError, match="RL-Lab/internal import"):
        audit_module.audit_archive(wheel)


def test_rejects_archive_path_traversal(tmp_path: Path, audit_module: ModuleType) -> None:
    wheel = _wheel(tmp_path / "unsafe.whl", {"../outside.py": b"pass\n"})

    with pytest.raises(audit_module.ReleaseAuditError, match="unsafe archive member path"):
        audit_module.audit_archive(wheel)


def test_rejects_asset_payload_renamed_as_data(tmp_path: Path, audit_module: ModuleType) -> None:
    wheel = _wheel(
        tmp_path / "unsafe.whl",
        {"package/data.bin": b"ISO-10303-21;\nHEADER;\n"},
    )

    with pytest.raises(audit_module.ReleaseAuditError, match="forbidden STEP payload"):
        audit_module.audit_archive(wheel)


def test_rejects_unknown_binary_member_even_without_known_signature(
    tmp_path: Path, audit_module: ModuleType
) -> None:
    wheel = _wheel(tmp_path / "unsafe.whl", {"package/policy.bin": b"opaque-model-data"})

    with pytest.raises(audit_module.ReleaseAuditError, match="unsupported non-text"):
        audit_module.audit_archive(wheel)


def test_rejects_oversized_member_before_reading(tmp_path: Path, audit_module: ModuleType) -> None:
    wheel = _wheel(
        tmp_path / "unsafe.whl",
        {"package/blob.bin": b"x" * (audit_module.MAX_MEMBER_BYTES + 1)},
    )

    with pytest.raises(audit_module.ReleaseAuditError, match="archive member exceeds"):
        audit_module.audit_archive(wheel)
