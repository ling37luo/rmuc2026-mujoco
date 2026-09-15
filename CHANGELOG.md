# Changelog

All notable changes to this project will be documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project intends to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-09-15

### Added

- Standalone, fail-closed loading of a user-supplied RMUC 2026 field asset pack.
- Manifest, path-containment, and SHA-256 integrity validation.
- Exact floating-point heightfield injection after MuJoCo model compilation.
- Standalone MJCF generation, field height queries, CLI inspection, and viewer.
- Explicit official-source download plus one-command local `setup` workflow.
- Robot-only MJCF composition through MuJoCo `MjSpec`.
- Synthetic-only tests and CI across supported Python versions.

### Known limitations

- No official or official-source-derived field asset is distributed.
- The current recognized collision representation is an unofficial,
  conservative 2.5-D heightfield and cannot model underpasses, overhangs,
  stacked surfaces, vertical walls, or dynamic mechanisms.
- `DRAFT_BLOCKED` validation remains unsuitable as evidence of whole-field
  physical fidelity or an official competition simulation.
