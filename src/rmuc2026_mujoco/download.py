"""Verified access to the official RMUC 2026 reference STEP.

The STEP is downloaded from RoboMaster's own server and is never mirrored by
this project.  Keeping this logic explicit avoids accidentally treating a
public download as permission to redistribute it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import BinaryIO, Callable
from urllib.request import Request, urlopen


OFFICIAL_STEP_URL = (
    "https://hz-rm-bbs-web-prod.oss-cn-hangzhou.aliyuncs.com/"
    "f637e13eadfc4602a76c7a05c54f71271782803452461/RMUC2026_V2.0.0.stp"
)
OFFICIAL_STEP_SIZE = 1_254_821_405
OFFICIAL_STEP_SHA256 = "8dfe9ebd761e44d91361b3e593bc05416329112217b58cb35800b3cde2ffae33"
OFFICIAL_STEP_PRODUCT = "00_RMUC2026_FINALS_ASM"
OFFICIAL_SOURCE_PAGE = "https://bbs.robomaster.com/article/814728?source=8"
OFFICIAL_RULE_CENTRE = "https://bbs.robomaster.com/wiki/20204847/809871?source=7"
OFFICIAL_RULEBOOK_V2_0_0_URL = (
    "https://bbs-web-static.robomaster.com/"
    "5fbd46b110e542faaea519a6dd3065761782460789174/"
    "RoboMaster%202026%20%E6%9C%BA%E7%94%B2%E5%A4%A7%E5%B8%88%E8%B6%85%E7%BA%A7%"
    "E5%AF%B9%E6%8A%97%E8%B5%9B%E6%AF%94%E8%B5%9B%E8%A7%84%E5%88%99%E6%89%8B%"
    "E5%86%8CV2.0.0%EF%BC%8820260626%EF%BC%89.pdf"
)
OFFICIAL_RULEBOOK_V2_0_0_SIZE = 21_012_597
OFFICIAL_RULEBOOK_V2_0_0_SHA256 = "59d65aac5bac75fd6bbab157e224eb936b49cd2b9d60381fbe196fad635b473a"
LAST_REVIEWED_RULEBOOK_VERSION = "V2.2.0"
LAST_REVIEWED_RULEBOOK_PUBLICATION_DATE = "2026-08-07"
LAST_REVIEWED_RULEBOOK_URL = (
    "https://hz-rm-bbs-web-prod.oss-cn-hangzhou.aliyuncs.com/"
    "e354d2750e17485e8f67e5b5a9d1a2891786094242879/"
    "RoboMaster%202026%20%E6%9C%BA%E7%94%B2%E5%A4%A7%E5%B8%88%E8%B6%85%E7%BA%A7"
    "%E5%AF%B9%E6%8A%97%E8%B5%9B%E6%AF%94%E8%B5%9B%E8%A7%84%E5%88%99%E6%89%8B"
    "%E5%86%8CV2.2.0%EF%BC%8820260807%EF%BC%89.pdf"
)
LAST_REVIEWED_RULEBOOK_SIZE = 21_018_604
LAST_REVIEWED_RULEBOOK_SHA256 = "88ae3c7d0bbd7312c8095eff2603910437e292a0b53f46c9378f26144c09a6f5"
LAST_RULE_REVIEW_DATE = "2026-09-15"


class DownloadError(RuntimeError):
    """Raised when the official source cannot be fetched or verified."""


@dataclass(frozen=True)
class DownloadedStep:
    path: Path
    size_bytes: int
    sha256: str
    reused: bool


@dataclass(frozen=True)
class DownloadedRulebook:
    """Identity of the pinned local V2.0.0 rulebook reference."""

    path: Path
    size_bytes: int
    sha256: str
    reused: bool


def _sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_official_step(path: Path) -> DownloadedStep:
    """Verify the immutable size, header identity, and SHA-256 of the STEP."""

    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise DownloadError(f"STEP path must not be a symlink: {requested}")
    candidate = requested.resolve()
    if not candidate.is_file():
        raise DownloadError(f"not a regular STEP file: {candidate}")
    size = candidate.stat().st_size
    if size != OFFICIAL_STEP_SIZE:
        raise DownloadError(f"STEP size mismatch: expected={OFFICIAL_STEP_SIZE}, actual={size}")
    with candidate.open("rb") as handle:
        header = handle.read(64 * 1024).decode("latin-1", errors="replace")
    if "ISO-10303-21" not in header or OFFICIAL_STEP_PRODUCT not in header:
        raise DownloadError("STEP header does not identify the pinned RMUC 2026 Finals assembly")
    digest = _sha256(candidate)
    if digest != OFFICIAL_STEP_SHA256:
        raise DownloadError(
            f"STEP SHA-256 mismatch: expected={OFFICIAL_STEP_SHA256}, actual={digest}"
        )
    return DownloadedStep(candidate, size, digest, reused=True)


def _copy_and_hash(
    source: BinaryIO,
    destination: BinaryIO,
    *,
    expected_size: int,
    progress: Callable[[int, int], None] | None,
    chunk_size: int = 8 * 1024 * 1024,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = source.read(chunk_size)
        if not chunk:
            break
        destination.write(chunk)
        digest.update(chunk)
        total += len(chunk)
        if progress is not None:
            progress(total, expected_size)
    destination.flush()
    os.fsync(destination.fileno())
    return total, digest.hexdigest()


def download_official_step(
    destination: Path,
    *,
    acknowledge_reference_only: bool,
    progress: Callable[[int, int], None] | None = None,
    timeout_seconds: float = 60.0,
) -> DownloadedStep:
    """Download the pinned STEP directly from RoboMaster and verify it.

    ``acknowledge_reference_only`` must be explicit because the publisher calls
    the model a reference and this project's review found no explicit standard
    redistribution grant. This is a conservative distribution policy, not a
    legal determination. This function downloads from the official URL; it
    never uses a project mirror or silently accepts a different file.
    """

    if not acknowledge_reference_only:
        raise DownloadError(
            "explicit acknowledgement required: the official STEP is reference-only "
            f"material; review {OFFICIAL_SOURCE_PAGE}"
        )
    requested = Path(destination).expanduser()
    if requested.is_symlink():
        raise DownloadError(f"destination must not be a symlink: {requested}")
    target = requested.resolve()
    if target.exists():
        verified = verify_official_step(target)
        return DownloadedStep(
            verified.path,
            verified.size_bytes,
            verified.sha256,
            reused=True,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink():
        raise DownloadError(f"destination parent must not be a symlink: {target.parent}")

    request = Request(OFFICIAL_STEP_URL, headers={"User-Agent": "rmuc2026-mujoco/0.2"})
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.name}.",
            suffix=".part",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                size, digest = _copy_and_hash(
                    response,
                    temporary,
                    expected_size=OFFICIAL_STEP_SIZE,
                    progress=progress,
                )
        if size != OFFICIAL_STEP_SIZE:
            raise DownloadError(
                f"downloaded STEP size mismatch: expected={OFFICIAL_STEP_SIZE}, actual={size}"
            )
        if digest != OFFICIAL_STEP_SHA256:
            raise DownloadError(
                f"downloaded STEP SHA-256 mismatch: expected={OFFICIAL_STEP_SHA256}, actual={digest}"
            )
        assert temporary_path is not None
        if target.exists():
            raise DownloadError(
                f"destination appeared during download; refusing overwrite: {target}"
            )
        os.replace(temporary_path, target)
        temporary_path = None
        verified = verify_official_step(target)
        return DownloadedStep(
            verified.path,
            verified.size_bytes,
            verified.sha256,
            reused=False,
        )
    except DownloadError:
        raise
    except Exception as exc:
        raise DownloadError(f"official STEP download failed: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def verify_official_rulebook_v2_0_0(path: Path) -> DownloadedRulebook:
    """Verify the exact V2.0.0 PDF used by the optional overhead guide."""

    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise DownloadError(f"rulebook path must not be a symlink: {requested}")
    candidate = requested.resolve()
    if not candidate.is_file():
        raise DownloadError(f"not a regular rulebook PDF: {candidate}")
    size = candidate.stat().st_size
    if size != OFFICIAL_RULEBOOK_V2_0_0_SIZE:
        raise DownloadError(
            f"rulebook size mismatch: expected={OFFICIAL_RULEBOOK_V2_0_0_SIZE}, actual={size}"
        )
    with candidate.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise DownloadError("rulebook file does not have a PDF header")
    digest = _sha256(candidate)
    if digest != OFFICIAL_RULEBOOK_V2_0_0_SHA256:
        raise DownloadError(
            "rulebook SHA-256 mismatch: "
            f"expected={OFFICIAL_RULEBOOK_V2_0_0_SHA256}, actual={digest}"
        )
    return DownloadedRulebook(candidate, size, digest, reused=True)


def download_official_rulebook_v2_0_0(
    destination: Path,
    *,
    acknowledge_reference_only: bool,
    progress: Callable[[int, int], None] | None = None,
    timeout_seconds: float = 60.0,
) -> DownloadedRulebook:
    """Download and verify the exact official PDF used by the local surface guide."""

    if not acknowledge_reference_only:
        raise DownloadError(
            "explicit acknowledgement required: the official rulebook is reference-only "
            f"material; review {OFFICIAL_RULE_CENTRE}"
        )
    requested = Path(destination).expanduser()
    if requested.is_symlink():
        raise DownloadError(f"destination must not be a symlink: {requested}")
    target = requested.resolve()
    if target.exists():
        verified = verify_official_rulebook_v2_0_0(target)
        return DownloadedRulebook(
            verified.path,
            verified.size_bytes,
            verified.sha256,
            reused=True,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink():
        raise DownloadError(f"destination parent must not be a symlink: {target.parent}")

    request = Request(
        OFFICIAL_RULEBOOK_V2_0_0_URL,
        headers={"User-Agent": "rmuc2026-mujoco/0.2"},
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.name}.",
            suffix=".part",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                size, digest = _copy_and_hash(
                    response,
                    temporary,
                    expected_size=OFFICIAL_RULEBOOK_V2_0_0_SIZE,
                    progress=progress,
                )
        if size != OFFICIAL_RULEBOOK_V2_0_0_SIZE:
            raise DownloadError(
                "downloaded rulebook size mismatch: "
                f"expected={OFFICIAL_RULEBOOK_V2_0_0_SIZE}, actual={size}"
            )
        if digest != OFFICIAL_RULEBOOK_V2_0_0_SHA256:
            raise DownloadError(
                "downloaded rulebook SHA-256 mismatch: "
                f"expected={OFFICIAL_RULEBOOK_V2_0_0_SHA256}, actual={digest}"
            )
        assert temporary_path is not None
        if target.exists():
            raise DownloadError(
                f"destination appeared during download; refusing overwrite: {target}"
            )
        os.replace(temporary_path, target)
        temporary_path = None
        verified = verify_official_rulebook_v2_0_0(target)
        return DownloadedRulebook(
            verified.path,
            verified.size_bytes,
            verified.sha256,
            reused=False,
        )
    except DownloadError:
        raise
    except Exception as exc:
        raise DownloadError(f"official rulebook download failed: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


__all__ = [
    "DownloadedRulebook",
    "DownloadedStep",
    "DownloadError",
    "LAST_REVIEWED_RULEBOOK_PUBLICATION_DATE",
    "LAST_REVIEWED_RULEBOOK_SHA256",
    "LAST_REVIEWED_RULEBOOK_SIZE",
    "LAST_REVIEWED_RULEBOOK_URL",
    "LAST_REVIEWED_RULEBOOK_VERSION",
    "LAST_RULE_REVIEW_DATE",
    "OFFICIAL_RULE_CENTRE",
    "OFFICIAL_RULEBOOK_V2_0_0_SHA256",
    "OFFICIAL_RULEBOOK_V2_0_0_SIZE",
    "OFFICIAL_RULEBOOK_V2_0_0_URL",
    "OFFICIAL_SOURCE_PAGE",
    "OFFICIAL_STEP_PRODUCT",
    "OFFICIAL_STEP_SHA256",
    "OFFICIAL_STEP_SIZE",
    "OFFICIAL_STEP_URL",
    "download_official_rulebook_v2_0_0",
    "download_official_step",
    "verify_official_rulebook_v2_0_0",
    "verify_official_step",
]
