from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

import rmuc2026_mujoco.download as download


def _fixture_bytes() -> bytes:
    return b"ISO-10303-21;\nHEADER;\n00_RMUC2026_FINALS_ASM\nENDSEC;\n"


def _patch_identity(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    monkeypatch.setattr(download, "OFFICIAL_STEP_SIZE", len(payload))
    monkeypatch.setattr(download, "OFFICIAL_STEP_SHA256", hashlib.sha256(payload).hexdigest())


def _patch_rulebook_identity(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    monkeypatch.setattr(download, "OFFICIAL_RULEBOOK_V2_0_0_SIZE", len(payload))
    monkeypatch.setattr(
        download,
        "OFFICIAL_RULEBOOK_V2_0_0_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )


def test_download_requires_explicit_reference_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(download.DownloadError, match="explicit acknowledgement"):
        download.download_official_step(
            tmp_path / "field.step",
            acknowledge_reference_only=False,
        )


def test_download_is_atomic_verified_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _fixture_bytes()
    _patch_identity(monkeypatch, payload)
    calls = 0

    def fake_urlopen(_request: object, *, timeout: float) -> io.BytesIO:
        nonlocal calls
        assert timeout == 9.0
        calls += 1
        return io.BytesIO(payload)

    monkeypatch.setattr(download, "urlopen", fake_urlopen)
    destination = tmp_path / "nested" / "field.step"
    progress: list[tuple[int, int]] = []
    first = download.download_official_step(
        destination,
        acknowledge_reference_only=True,
        timeout_seconds=9.0,
        progress=lambda current, total: progress.append((current, total)),
    )
    second = download.download_official_step(
        destination,
        acknowledge_reference_only=True,
    )

    assert first.path == destination.resolve()
    assert first.reused is False
    assert second.reused is True
    assert destination.read_bytes() == payload
    assert calls == 1
    assert progress[-1] == (len(payload), len(payload))
    assert not list(destination.parent.glob("*.part"))


def test_bad_download_never_replaces_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _fixture_bytes()
    _patch_identity(monkeypatch, payload)
    monkeypatch.setattr(download, "urlopen", lambda *_args, **_kwargs: io.BytesIO(b"bad"))
    destination = tmp_path / "field.step"

    with pytest.raises(download.DownloadError, match="size mismatch"):
        download.download_official_step(destination, acknowledge_reference_only=True)

    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_verify_rejects_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _fixture_bytes()
    _patch_identity(monkeypatch, payload)
    real = tmp_path / "real.step"
    real.write_bytes(payload)
    link = tmp_path / "link.step"
    link.symlink_to(real)

    with pytest.raises(download.DownloadError, match="symlink"):
        download.verify_official_step(link)


def test_rulebook_download_is_atomic_verified_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"%PDF-1.7\nverified official fixture\n%%EOF\n"
    _patch_rulebook_identity(monkeypatch, payload)
    calls = 0

    def fake_urlopen(_request: object, *, timeout: float) -> io.BytesIO:
        nonlocal calls
        assert timeout == 7.0
        calls += 1
        return io.BytesIO(payload)

    monkeypatch.setattr(download, "urlopen", fake_urlopen)
    destination = tmp_path / "cache" / "rulebook.pdf"
    first = download.download_official_rulebook_v2_0_0(
        destination,
        acknowledge_reference_only=True,
        timeout_seconds=7.0,
    )
    second = download.download_official_rulebook_v2_0_0(
        destination,
        acknowledge_reference_only=True,
    )

    assert first.path == destination.resolve()
    assert first.reused is False
    assert second.reused is True
    assert destination.read_bytes() == payload
    assert calls == 1
    assert not list(destination.parent.glob("*.part"))


def test_rulebook_download_requires_reference_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(download.DownloadError, match="explicit acknowledgement"):
        download.download_official_rulebook_v2_0_0(
            tmp_path / "rulebook.pdf",
            acknowledge_reference_only=False,
        )
