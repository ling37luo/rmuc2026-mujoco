#!/usr/bin/env python3
"""Fail closed when a release archive contains non-redistributable assets.

This check inspects archives without extracting them.  It is intentionally
independent of the package runtime so it can run before anything is uploaded.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
from typing import Iterable, Iterator
import zipfile


# The repository distributes conversion code, not official or derived field
# geometry, renders, policies, or training checkpoints.
FORBIDDEN_ASSET_SUFFIXES = frozenset(
    {
        ".3ds",
        ".7z",
        ".blend",
        ".bmp",
        ".dae",
        ".exr",
        ".fbx",
        ".glb",
        ".gltf",
        ".hdr",
        ".jpeg",
        ".jpg",
        ".msh",
        ".npz",
        ".obj",
        ".onnx",
        ".pdf",
        ".ply",
        ".png",
        ".pt",
        ".pth",
        ".rar",
        ".step",
        ".stl",
        ".stp",
        ".tif",
        ".tiff",
        ".usd",
        ".usda",
        ".usdc",
        ".usdz",
        ".webm",
        ".webp",
        ".zip",
    }
)

TEXT_SUFFIXES = frozenset(
    {
        "",
        ".cfg",
        ".css",
        ".ini",
        ".json",
        ".md",
        ".py",
        ".pyi",
        ".rst",
        ".sh",
        ".toml",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)

# These roots belong to RL-Lab or the unpublished upstream training stack and
# must not leak into the standalone package's import graph.
FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "rmuc_field",
        "scripts",
        "src",
        "wheel_legged_gym",
    }
)

POSIX_HOST_PATH = re.compile(
    rb"(?<![A-Za-z0-9+.-]://)(?<![A-Za-z0-9_])"
    rb"/(?:home|Users|private/var|mnt|media)/[^\x00\r\n\t\"'<> ]+"
)
WINDOWS_HOST_PATH = re.compile(
    rb"(?i)\b[A-Z]:\\(?:Users|Documents and Settings)\\[^\x00\r\n\t\"'<>]+"
)
MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_CONTENT_BYTES = 32 * 1024 * 1024
FORBIDDEN_FILE_SIGNATURES = (
    (b"ISO-10303-21;", "STEP"),
    (b"glTF", "GLB"),
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"%PDF-", "PDF"),
    (b"\xff\xd8\xff", "JPEG"),
    (b"\x93NUMPY", "NumPy array"),
)


@dataclass(frozen=True)
class ArchiveMember:
    name: str
    data: bytes


class ReleaseAuditError(RuntimeError):
    """Raised when an archive violates the public-release boundary."""


def _safe_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or normalized.startswith("/") or path.is_absolute():
        raise ReleaseAuditError(f"unsafe absolute archive member: {name!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ReleaseAuditError(f"unsafe archive member path: {name!r}")
    if re.match(r"^[A-Za-z]:", normalized):
        raise ReleaseAuditError(f"unsafe drive-qualified archive member: {name!r}")
    return normalized


def _zip_members(path: Path) -> Iterator[ArchiveMember]:
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = _safe_member_name(info.filename)
            if info.is_dir():
                continue
            unix_kind = (info.external_attr >> 16) & 0o170000
            if unix_kind == stat.S_IFLNK:
                raise ReleaseAuditError(f"symbolic-link archive member is not allowed: {name!r}")
            if info.file_size > MAX_MEMBER_BYTES:
                raise ReleaseAuditError(
                    f"archive member exceeds {MAX_MEMBER_BYTES} bytes: {name!r} ({info.file_size})"
                )
            yield ArchiveMember(name=name, data=archive.read(info))


def _tar_members(path: Path) -> Iterator[ArchiveMember]:
    with tarfile.open(path, mode="r:*") as archive:
        for info in archive.getmembers():
            name = _safe_member_name(info.name)
            if info.isdir():
                continue
            if not info.isfile():
                raise ReleaseAuditError(
                    f"non-regular archive member is not allowed: {name!r} ({info.type!r})"
                )
            if info.size > MAX_MEMBER_BYTES:
                raise ReleaseAuditError(
                    f"archive member exceeds {MAX_MEMBER_BYTES} bytes: {name!r} ({info.size})"
                )
            handle = archive.extractfile(info)
            if handle is None:
                raise ReleaseAuditError(f"could not read archive member: {name!r}")
            yield ArchiveMember(name=name, data=handle.read())


def _members(path: Path) -> Iterable[ArchiveMember]:
    lower_name = path.name.lower()
    if lower_name.endswith(".whl") or lower_name.endswith(".zip"):
        return _zip_members(path)
    if lower_name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar")):
        return _tar_members(path)
    raise ReleaseAuditError(f"unsupported release archive type: {path}")


def _forbidden_imports(source: bytes, member_name: str) -> list[str]:
    try:
        tree = ast.parse(source, filename=member_name)
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise ReleaseAuditError(
            f"cannot parse packaged Python source {member_name!r}: {exc}"
        ) from exc

    violations: list[str] = []
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
        for module in modules:
            root = module.split(".", 1)[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                violations.append(module)
    return sorted(set(violations))


def audit_archive(path: Path) -> dict[str, object]:
    archive = path.expanduser().resolve()
    if not archive.is_file():
        raise ReleaseAuditError(f"release archive is not a regular file: {archive}")

    violations: list[str] = []
    member_count = 0
    content_bytes = 0
    member_names: set[str] = set()
    for member in _members(archive):
        member_count += 1
        content_bytes += len(member.data)
        if member.name in member_names:
            violations.append(f"duplicate archive member: {member.name}")
        member_names.add(member.name)
        if content_bytes > MAX_ARCHIVE_CONTENT_BYTES:
            raise ReleaseAuditError(
                f"archive exceeds {MAX_ARCHIVE_CONTENT_BYTES} uncompressed bytes"
            )
        suffix = PurePosixPath(member.name).suffix.lower()
        if suffix in FORBIDDEN_ASSET_SUFFIXES:
            violations.append(f"forbidden asset member: {member.name}")
            continue

        for signature, format_name in FORBIDDEN_FILE_SIGNATURES:
            if member.data.startswith(signature):
                violations.append(
                    f"forbidden {format_name} payload under non-asset name: {member.name}"
                )
                break

        if suffix not in TEXT_SUFFIXES:
            continue
        for match in POSIX_HOST_PATH.finditer(member.data):
            value = match.group(0).decode("utf-8", errors="replace")
            violations.append(f"host absolute path in {member.name}: {value}")
        for match in WINDOWS_HOST_PATH.finditer(member.data):
            value = match.group(0).decode("utf-8", errors="replace")
            violations.append(f"host absolute path in {member.name}: {value}")
        if suffix in {".py", ".pyi"}:
            for module in _forbidden_imports(member.data, member.name):
                violations.append(f"RL-Lab/internal import in {member.name}: {module}")

    if member_count == 0:
        violations.append("release archive contains no regular files")
    if violations:
        detail = "\n  - ".join(violations)
        raise ReleaseAuditError(f"release content audit failed for {archive.name}:\n  - {detail}")

    return {
        "archive": str(archive),
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "members": member_count,
        "uncompressed_bytes": content_bytes,
        "status": "PASS",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit wheel/sdist contents before publishing rmuc2026-mujoco."
    )
    parser.add_argument("archives", nargs="+", type=Path, help="wheel and/or source archive")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    failed = False
    for archive in args.archives:
        try:
            result = audit_archive(archive)
        except (OSError, ReleaseAuditError, tarfile.TarError, zipfile.BadZipFile) as exc:
            failed = True
            print(f"RELEASE_CONTENT_AUDIT=FAIL: {exc}", file=sys.stderr)
        else:
            print(
                "RELEASE_CONTENT_AUDIT=PASS; "
                f"archive={result['archive']}; sha256={result['sha256']}; "
                f"members={result['members']}; uncompressed_bytes={result['uncompressed_bytes']}"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
