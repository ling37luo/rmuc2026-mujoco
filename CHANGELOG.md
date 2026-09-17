# Changelog

All notable changes to this project will be documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project intends to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Runtime packs now carry the collision sample-provenance record:
  `collision.ray_misses_filled_with_ground`, `collision.isolated_spikes_replaced`,
  and the source build's `collision.structural_audit`. These say how much of the
  runtime surface is synthetic, which the pack previously reduced to a prose
  claim boundary. A source build that never measured a counter records `null`
  instead of a fabricated zero.

### Fixed

- Pack verification now compares the collision bootstrap PNG against the
  verified float samples it is supposed to quantize. Previously only the two
  hashes were checked, so a pack whose image and samples disagreed still
  verified; a consumer that loads the entrypoint XML directly with MuJoCo reads
  the image instead of the samples and would have received a different field.
  The comparison allows one LSB of 16-bit rounding.

## [0.2.0a1] - 2026-09-15

### Added

- Runtime-pack schema 2 with named field lights, a public
  `FieldDisplayController`, and `L` flat/shadow switching.
- Optional local-only complete rulebook overhead livery and `G` visibility
  switching; the RGB image retains its baked robots, obstacles, field modules,
  and shadows, while the group 4 mesh is excluded from collision.
- A 1 cm collision option plus hash-bound static and 12-trial dynamic evidence
  for the two fixed 17-degree fly ramps.
- Optional bounded planar-ramp refinement with unchanged exterior samples and
  a transition band, without adding overlapping collision geometry.
- Actual-contact regression tests for low-friction robot/field pairs.
- Deterministic visual-mesh cleanup for degenerate, repeated, and oppositely
  wound coincident CAD faces before local OBJ export.
- Local surface queries for height, normal, slope, grid resolution, and relief.
- Deterministic runtime-proxy heightfield screening for candidate spawn points.
- A bounds-aware `overview` camera, plus spawn-centred and unchanged `native`
  viewer camera modes.
- A design-reference ledger that separates reusable engineering patterns from
  third-party code and asset licensing.

### Changed

- The runtime documentation now makes the field's 2 ms timestep part of the
  1 cm integration boundary and calls out that a parent robot specification can
  replace global MuJoCo options during composition.
- Interactive viewers now stop and join the keyboard hook before closing the
  MuJoCo window, then wait for the native render thread to release its GL
  context. Physics exceptions therefore exit with their Python status instead
  of racing a daemon thread and producing a later segmentation fault.
- On Linux/X11, plain `L` and `G` are intercepted only for the unique MuJoCo
  window owned by the current process, avoiding MuJoCo's built-in lighting and
  fog shortcuts. Other platforms retain the launch-time display switches.
- Full-profile rendering now records a separate, deterministic display RGBA,
  darkens the broad base shell, caps washed-out whites, and defaults to the
  original even lighting with cast shadows disabled. Display changes do not
  alter field physics or source-colour provenance.
- Fine heightfields retain the established 2 cm envelope and spawn anchor, so
  changing sample spacing does not shift the field coordinate frame.
- Runtime loading remains backward-compatible with existing schema 1 packs;
  schema 2 controls are enabled only when their complete contracts are present,
  including both complete-guide and earlier filtered-marking livery packs.
- Friction presets now take precedence in generated contacts and update explicit
  field pairs. Global contact overrides fail closed; robot geom values remain
  unchanged. Higher field priority also selects field solver parameters.
- Local surface queries calculate triangle slopes only inside the requested
  window, avoiding full-field gradient allocations while preserving results.
- The first 0.2 alpha identifies as `0.2.0a1`; official and derived field assets
  remain local and are excluded from its code-only tag and release archives.
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
