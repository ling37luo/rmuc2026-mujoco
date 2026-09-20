# Roadmap

This roadmap separates loader engineering from claims about field fidelity.
Completing a code item never upgrades a geometry-validation status by itself.

## 0.1 — asset-free local builder and loader

- pin and verify the official STEP identity;
- perform the CAD conversion only on the user's machine;
- export relocatable, hash-bound runtime packs;
- provide full-visual and collision-only MuJoCo profiles;
- compose a separately licensed robot MJCF with the field;
- expose explicit, unofficial friction presets;
- test and audit wheels/source archives without third-party assets.

## 0.2.0 — bounded static-interaction foundations

This release adds a 1 cm heightfield option, deterministic terrain-surface and
candidate-spawn screening, and evidence tied to the two fixed 17-degree fly
ramps. It also improves the optional local-only visual guide and its display
controls. These are bounded capabilities, not whole-field collision approval:
generated runtime packs retain `DRAFT_BLOCKED`, and spawn screening is not
topology-verified.

## Planned static-interaction work

- resolve the source-backed top-edge part 236/552 contact transition before
  offering it as a robot route. Both the existing and source-negative local
  1 cm trials reach MuJoCo's 50-contact-per-pair heightfield limit, and a
  matched robot can leave the field. Nine local exact-source convex roof
  triangles improve two approaches up to the official edge, but one still
  fails after crossing it, and the triangles' shared side contacts are not
  proven exclusive. Source fidelity and safe route behavior require separate
  gates;
- decompose the source-backed bottom-edge part-244 roof beyond the one tested
  convex triangle and validate its adjacent seams. The local one-triangle
  candidate improved 0.3/0.5 m/s outward routes but raised the 1.0 m/s
  acceleration peak and left 50-contact heightfield pairs;
- define and validate robot out-of-bounds termination before promoting the
  source-ray-masked schema-3 pack. Initial frozen-robot edge routes lost field
  contact and later produced `BADQACC`; keep the finite -5 m floor labeled
  as a surrogate rather than a source-measured depth;
- finish the exact 402/403 wall replacement contract and reverse/high-side,
  repeated-impact robot routes before enabling its convex contacts in a runtime
  pack. Twelve short forward approaches did not penetrate the walls, but
  startup heightfield contact pairs still reached 50 points;
- register the chassis perimeter proxy against the official-source boundary,
  resolve its open stretches and window positions, then repeat robot routes
  before any default activation;
- add a versioned registry of robot-validated spawn poses and static routes;
- add query APIs for semantic, unsupported, or unverified zones;
- add multi-hit vertical and horizontal-blocker audits before any multi-level
  structure can be promoted;
- use the source-bound wall and four-edge audit-only candidate registries to
  select genuinely missing barriers; the current wall footprints already
  belong to the heightfield except for seven repaired 1 cm wall-end roof
  samples on exact-source parts 402/403;
- use the official rulebook's 28 × 15 m, 2.4 m-high perimeter-fence requirement
  and depicted dart window to scope a labeled fence proxy; the STEP base shell
  does not fix its exact centerline, thickness, or openings, so keep it out of
  the default collision pack until placement and contact tests pass;
- replace only independently accepted structure footprints with mutually
  exclusive primitive or convex collision while preserving the remaining
  heightfield bit-for-bit;
- validate contact ownership so one XY location cannot accidentally collide
  with both the heightfield and its replacement geometry;
- add visual LODs with geometry-coverage and silhouette regression gates.

The intended collision profiles are `fast_heightfield` (current heightfield),
`hybrid_static` (connected base heightfield plus accepted regular/convex static
geometry), and eventually `dynamic_facilities` (explicit moving bodies and
joints). These are design targets, distinct from the current `full` visual and
`collision_only` loading profiles, not current validation claims.

Coordinates and meshes for these features must be generated locally unless
the upstream rightsholder grants written redistribution permission.

## 0.3 — robot interaction and remaining topology

- represent accepted underpasses and stacked surfaces with hybrid collision;
- register material zones from measured or clearly labeled non-official data;
- publish paired geometry, contact, and robot-independent traversal tests.

Movable facilities, energy-unit handling, outposts and bases are optional
test fixtures. They are not prerequisites for the field's main purpose:
reproducible robot locomotion and contact interaction.

This stage remains blocked until each promoted facility has sufficient source
evidence and an explicit behavior contract. The project will not replace the
whole field with unqualified triangle-mesh collision merely to remove the
`DRAFT_BLOCKED` label.

## Prebuilt assets

The Python API is designed so a prebuilt asset archive can be added later.
Such an archive will be published only after written permission covers the
official-source-derived OBJ, heightfield, and preview media. Until then,
GitHub, package indexes, CI, and releases remain code-only.
