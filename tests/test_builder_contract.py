from __future__ import annotations

import ast
from pathlib import Path

import pytest

from rmuc2026_mujoco import builder


def test_runtime_import_does_not_eagerly_import_cad_stack() -> None:
    source = Path(builder.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports = {
        alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names
    }
    assert not {"OCP", "trimesh", "PIL", "rtree", "cadquery"} & top_level_imports


def test_runtime_pack_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "already-there"
    output.mkdir()

    with pytest.raises(FileExistsError, match="output already exists"):
        builder.build_runtime_asset_pack(tmp_path / "unused.step", output)


def test_builder_source_has_no_rl_lab_imports() -> None:
    package_root = Path(builder.__file__).parent
    for name in ("builder.py", "_conversion.py", "pack.py"):
        text = (package_root / name).read_text(encoding="utf-8")
        assert "src.rmuc_field" not in text
        assert "scripts.rmuc2026" not in text
        assert "fudan_" not in text
