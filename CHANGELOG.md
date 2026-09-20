# Changelog

All notable changes to this project will be documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project intends to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- An optional 400 g movable energy-unit collision probe based on the official
  rulebook's 150 mm height and asymmetric 95/80 mm end diameters. It is a
  primitive contact test, never part of the default field pack; rib/rim detail
  and friction remain explicitly approximate.
- An exact-source 1 cm heightfield repair for seven wall-end roof samples on
  official GLB parts 402/403. The source GLB hash, changed sample values, and
  output collision-file hash are recorded and checked in the manifests. It
  retains one heightfield contact owner and adds no wall collision geom; other
  sources or grid resolutions explicitly report the repair as not applicable. This is a
  bounded physical correction, not whole-wall collision validation.
- Source- and heightfield-hash-bound, audit-only collision candidate tools for
  STEP-derived wall parts and all four finite field edges. They document
  existing heightfield ownership and definite source-face gaps without adding
  unverified wall or perimeter collision to runtime packs.

### Documentation

- Clarified that the STEP base shell lacks a continuous body-height fence,
  while official V2.0.0 rulebook §4.1 specifies a 28 × 15 m field with a
  steel perimeter fence rising 2.4 m above the floor. Its exact placement and
  openings remain unresolved, so no fence proxy is activated by this change.

## [0.2.1] - 2026-09-20

### Fixed

- The optional `G` rulebook livery no longer bridges low terrain while the
  robot contacts the lower collision surface. The visual mesh checks signed
  height error in both directions against the unchanged 1 cm heightfield,
  refines rejected cells locally from 5 to 2.5 and 1.25 cm, and keeps refined
  edges aligned with neighboring coarse triangles. Its nominal visual lift is 5 mm rather
  than 18 mm; retained triangles stay 2–8 mm above the sampled collision
  surface. This changes appearance only, not contact geometry or materials.

## [0.2.0] - 2026-09-19

This code-only release improves local visual-guide generation and runtime-pack
verification. Generated field packs remain `DRAFT_BLOCKED`; no official or
source-derived field assets are included.

### Added

- Runtime packs now carry the collision sample-provenance record:
  `collision.ray_misses_filled_with_ground`, `collision.isolated_spikes_replaced`,
  and the source build's `collision.structural_audit`. These say how much of the
  runtime surface is synthetic, which the pack previously reduced to a prose
  claim boundary. A source build that never measured a counter records `null`
  instead of a fabricated zero.

### Fixed

- The optional full rulebook livery now follows a 5 cm visual grid instead of
  20 cm, reducing stepped gaps along raised-platform edges. Cells that would
  hide an interior 1 cm terrain protrusion are omitted; the collision
  heightfield, 18 mm visual clearance, and group 4 zero-contact contract stay
  unchanged.
- Boundary-connected near-white rulebook page margins now become transparent
  in the locally generated runtime texture. Interior RGB pixels and world UVs
  are preserved; the manifest records the source-image hash and processing
  contract while existing schema 2 packs remain readable. The published JSON
  schema accepts the new fields as a validated pair.
- Pack verification now compares the collision bootstrap PNG against the
  verified float samples it is supposed to quantize. Previously only the two
  hashes were checked, so a pack whose image and samples disagreed still
  verified; a consumer that loads the entrypoint XML directly with MuJoCo reads
  the image instead of the samples and would have received a different field.
  The comparison allows one LSB of 16-bit rounding.
- `ValidationReport` no longer reports `PASS` when hashes were not verified.
  `FieldAsset.open(..., verify=False)` now reports `PASS_SIZE_ONLY` and exposes
  `hashes_verified`, so a consumer cannot read a size-only check as an
  integrity claim.
- The `test` extra now declares `python-xlib`, which `tests/test_viewer.py`
  imports directly. CI installs only `.[test]`, so the synthetic-fixture job
  had failed on every branch since `0bbee0d`, while the same suite passed in a
  developer environment that also installs the `viewer` extra.

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
