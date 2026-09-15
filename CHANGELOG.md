# Changelog

All notable changes to this project will be documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project intends to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Local surface queries for height, normal, slope, grid resolution, and relief.
- Deterministic runtime-proxy heightfield screening for candidate spawn points.
- A bounds-aware `overview` camera, plus spawn-centred and unchanged `native`
  viewer camera modes.
- A design-reference ledger that separates reusable engineering patterns from
  third-party code and asset licensing.

### Changed

- Development builds now identify as `0.2.0.dev0`; the published `v0.1.0`
  tag and release artifacts remain unchanged.
- Quick-start commands now use an isolated `python3` environment and work on
  Ubuntu systems without a global `python` alias.
- Runtime verification now validates the complete heightfield PNG, rejects
  redirected/included MJCF assets and transformed field geometry, and matches
  every visual mesh reference to the manifest.
- Source and archive audits now reject generated evidence directories,
  mechanical descriptions, checkpoints, and unknown binary payloads.

### Accuracy boundary

- Surface slopes follow MuJoCo's actual heightfield triangles, but every surface
  and spawn result remains non-topology-verified. Ray misses have no runtime
  validity mask, so these additions do not promote generated packs beyond
  `DRAFT_BLOCKED`.
- New local builds call the collision representation a top-surface proxy;
  legacy `v0.1.0` manifests using the old `conservative_*` schema token remain
  readable for compatibility.

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
  single-valued 2.5-D heightfield proxy and cannot model underpasses, overhangs,
  stacked surfaces, vertical walls, or dynamic mechanisms.
- `DRAFT_BLOCKED` validation remains unsuitable as evidence of whole-field
  physical fidelity or an official competition simulation.
