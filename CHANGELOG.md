# Changelog

All notable changes to this project will be documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project intends to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.2.0] - 2026-09-23

### Added

- Added executable north/south fly-ramp robot episodes and a parallel speed and
  pose-offset matrix on the existing runtime pack. Reports separate physical
  health from robot jump success and identify takeoff, airborne, recontact and
  stable-landing phases.
- Added a public four-wheel fly-ramp example and interactive viewer reset,
  plus pack-bound approach/landing routes for MuJoCo and the optional Isaac
  descriptor. No field geometry or collision parameters were changed.
- Added source-bound, 1 cm north/south fly-ramp training-region exports with
  recorded grid slices, routes, hashes and perimeter contact. The public
  MuJoCo batch runner can evaluate either region with external robots and
  controllers across independent worker processes.
- Added an optional Isaac data handoff with exact height samples, contact
  metadata and cropped perimeter boxes, plus an Isaac Sim 6.1 PhysX contact
  probe for 1, 4 and 16 parallel environments.

### Changed

- Fly-ramp reports distinguish route completion from numerical health and
  expose peak field-contact force per robot geom. Package metadata and the
  public `__version__` now identify version `1.2.0`.

This is a code and interface release. Bounded MuJoCo and PhysX terrain-contact
checks passed; robot landing quality and friction-response parity are not
certified. The full field and derived regions remain `DRAFT_BLOCKED`, and crop
edges require trainer resets. Generated field packs, official-source assets
and robot/policy files are not distributed.

## [1.1.0] - 2026-09-23

### Added

- Added the robot-agnostic `turn_basic` benchmark with screened flat-ground
  starts, spin/arc/reversal phases, and parallel MuJoCo execution. External
  controllers keep responsibility for robot observations and actions.
- Added ordinary-slope route screening for `slope_basic`. The field's verified
  heightfield provides three continuous approach, slope, and exit routes in
  the current local pack without introducing a second map or changing geometry.
- Added a slope interaction viewer and example four-wheel rover. External
  robots and policies use the same model composition and controller callback.
- Added independent uphill, downhill, and roundtrip results with contact,
  tilt, penetration, failure, and timing records for each leg.
- Added automatic MuJoCo slope evaluation across routes, directions, speeds,
  repeats, and worker processes, with deterministic case identities and
  optional per-episode trajectories.

### Changed

- `view --scenario slope_basic` defaults to uphill-only evaluation. A roundtrip
  succeeds after uphill and downhill complete in the same episode. Reaching
  or crossing an endpoint counts without the former 0.25 s dwell requirement.
- `run --scenario slope_basic` runs a headless evaluation matrix; `run` keeps
  the existing turning behavior for `turn_basic`.
- Package metadata and the public `__version__` identify version `1.1.0`.

This is a code and interface release. It does not distribute generated field
packs, official assets, or private robots and policies. The current full-field
runtime pack remains `DRAFT_BLOCKED`; the bounded slope tests do not establish
whole-field collision accuracy or Isaac training results.

## [1.0.0] - 2026-09-22

### Added

- Added the public scenario registry: `full_eval`, `turn_basic`,
  `stairs_basic`, both audited fly ramps, and `boundary_contact`.
- Added the robot-agnostic controller callback interface and a shared viewer,
  headless, and MuJoCo scenario runner based on `compose_with_robot`.
- Added per-step telemetry and run metadata for contacts, solver warnings,
  finite state, profile hashes, robot MJCF hashes, throughput, and memory.
- Added a dependency-free Isaac heightfield input adapter and documentation for
  optional high-parallel consumers.
- Added the `view --robot`, `--control`, `--scenario`, `--headless`, and
  `--telemetry` entry points, plus local controller-module loading.
- Added a field-owned free-flight wheel probe for the two fixed fly ramps. It
  measures the runtime heightfield gap and separates takeoff-lip contact from
  reaching the landing top.

### Changed

- The public Python package and project metadata are now version `1.0.0`.
- Documentation now describes the complete field baseline and the lightweight
  collision-only training profiles without adding an RL framework dependency.

This is a code and interface release. Generated runtime packs remain local and
`DRAFT_BLOCKED`; official STEP/rulebook assets and private robot or policy
files are not distributed.

## [0.3.1] - 2026-09-21

### Fixed

- Added the missing `rtree` dependency to the `test` extra so a clean CI
  environment can execute the source-ray edge test. Runtime behavior and
  generated field packs are unchanged from 0.3.0.

## [0.3.0] - 2026-09-21

This code-only release adds bounded physical interaction tools for local RMUC
runtime packs. Generated packs remain `DRAFT_BLOCKED` and are not distributed.

### Fixed

- Fresh `fence-pack` exports put the four wall centrelines on the inferred
  28 × 15 m raised-deck edge. This blocks the lower CAD skirt where a robot
  could fall and become trapped against the heightfield-edge fence introduced
  in `0.3.0a4`.
- The wall inner faces overlap the raised deck by 25 mm. The two fixed fly-ramp
  outer corners overlap the adjacent wall by about 5.5 mm, below the 10 mm
  heightfield sample spacing; all 12 fixed-ramp wheel trials still pass.
- The verifier keeps support for the v1–v4 contracts. New local exports use
  the v5 perimeter contract and retain the stable `solref="0.04 1"` contact.

## [0.3.0a4] - 2026-09-21

This code-only prerelease closes the traversable strip between the physical
perimeter proxy and the finite heightfield. Local packs remain
`DRAFT_BLOCKED` and are not distributed.

### Fixed

- Fresh `fence-pack` exports place every 50 mm wall fully inside the runtime
  heightfield and align each wall's outer face with the corresponding
  heightfield edge. The previous candidate left about 0.09 m outside its
  north/south walls and about 0.88 m outside its east/west walls, where a
  robot could straddle the terrain edge and fence.
- The manifest records the exact heightfield bounds, zero traversable exterior
  strip, and edge-alignment contract. The verifier continues to accept v1–v3
  fence contracts; fresh exports use v4 with the v3 soft-contact parameters.
- Moving the walls to the collision edge increases clearance from the fixed
  north/south fly-ramp outer corners to about 0.488 m and 0.491 m without
  changing the heightfield, ramp geometry, friction, or solver parameters.

## [0.3.0a3] - 2026-09-20

This code-only prerelease corrects the physical fence's interference with
the two fly ramps. Local runtime packs remain `DRAFT_BLOCKED` and are not
distributed.

### Fixed

- The previous north/south fence inner faces overlapped the outer corners of
  both fixed fly ramps by about 5.5 mm. Fresh `fence-pack` exports now keep
  the east/west walls on the supported core edge and move only the north/south
  walls 0.40 m onto the source-supported outer apron. This leaves about
  0.395 m between each ramp edge and the fence inner face, without modifying
  the heightfield or ramp friction. The lighter visual alpha makes the solid
  proxy less obstructive.
- A hard-fence Fudan approach from the north apron produced `BADQACC` during
  sustained wall contact. The fence-only `solref` time constant is now 0.04 s
  instead of 0.02 s; matched A/B routes and all four 6 s robot approaches
  complete without solver warnings. The heightfield solver setting remains
  unchanged, and its 50-contact pair limit is still reached in robot trials.
- The verifier recognizes v1, v2 and v3 perimeter contracts, so existing
  verified fenced packs remain readable. New exports use v3.

## [0.3.0a2] - 2026-09-20

This remains a code-only, `DRAFT_BLOCKED` pre-release. No generated pack or
official asset is distributed.

### Added

- A repository-owned `fence-pack` command that derives a new, fully verified
  schema-2 runtime pack with four touching physical perimeter geoms in both
  display and collision-only profiles. They sit on the inferred 28 × 15 m core
  edge and reach the rulebook's 2.4 m top height. Four sphere and four Fudan
  approach routes contacted the fence without crossing it or producing a
  numerical warning; full perimeter physics remains unaccepted.
- A source-hash-bound `ramp-source-audit` command. It proves that the 390
  greater-than-1 mm dominant-plane samples on the two 17-degree fly ramps
  match higher adjacent official CAD parts, rather than an erroneous
  heightfield. It preserves the existing collision samples.
- Manifest and MJCF verification for the optional fence contract. RL-Lab may
  copy the declared field geoms but does not generate them.

### Changed

- The fence centreline was moved from an earlier local 0.15 m outside-core
  experiment to the inferred core edge, so its inner contact face is on
  source-supported ground. The earlier pack is not promoted.

## [0.3.0a1] - 2026-09-20

This is a code-only prerelease of bounded field-contact tools. It does not
promote any generated runtime pack beyond `DRAFT_BLOCKED` or activate the
experimental wall and edge candidates by default. The final 0.3 physical
acceptance gate remains open.

### Added

- A field-owned, read-only `FieldBoundaryGuard` for fully verified schema-3
  packs. It signals a stop after persistent source-miss contact loss or when
  one heightfield/robot pair stays at MuJoCo's 50-contact limit while moving
  outward within 0.4 m of the grid edge. It does not alter geometry, state,
  controls, or the solver; short top and left robot routes stopped before
  falling or a numerical warning. Safe travel on those edges is not claimed.

- Experimental runtime schema 3 for the audited 1 cm official-source field. A hashed ray
  mask removes 277,806 unsupported outer-edge samples using a finite -5 m
  surrogate void, while preserving every source-hit and interior sample. The
  MJCF origin, height range, PNG bootstrap and exact float injection share one
  versioned formula; schema 1/2 packs remain readable. This removes false
  playable-height support, not the single-heightfield topology limit. It is
  opt-in: a frozen robot leaving the edge can later produce `BADQACC` if its
  consumer continues stepping without a stop rule; full edge traversal is
  still unaccepted.
- Local-only exact wall-contact and chassis perimeter candidates with bounded
  wheel, sphere and frozen-robot evidence. They are disabled in exported packs
  until contact-owner and placement contracts are complete.
- A 2.3 million default visual face budget for fresh builds. Mesh cleanup made
  the old 450,000-face budget unattainable on some official parts without
  losing large areas; the builder still rejects a requested budget it cannot
  meet and reports the actual count.
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

- Prioritized robot-driving ground, walls, finite boundaries and contact
  behavior. Movable props and game mechanisms remain optional interaction
  probes rather than field-completion gates.
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
