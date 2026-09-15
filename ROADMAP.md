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

## 0.2 — validated static interaction

- add a versioned registry of safe spawn poses and static routes;
- add query APIs for unsupported or unverified zones;
- replace only independently accepted ramp footprints with mutually exclusive
  static mesh collision while preserving the remaining heightfield bit-for-bit;
- validate contact ownership so one XY location cannot accidentally collide
  with both the heightfield and its replacement mesh;
- add visual LODs with geometry-coverage and silhouette regression gates.

Coordinates and meshes for these features must be generated locally unless
the upstream rightsholder grants written redistribution permission.

## 0.3 — multi-level and dynamic facilities

- represent accepted underpasses and stacked surfaces with hybrid collision;
- model movable facilities as explicit MuJoCo bodies and joints;
- register material zones from measured or clearly labeled non-official data;
- publish paired geometry, contact, and robot-independent traversal tests.

This stage remains blocked until each promoted facility has sufficient source
evidence and an explicit behavior contract. The project will not replace the
whole field with unqualified triangle-mesh collision merely to remove the
`DRAFT_BLOCKED` label.

## Prebuilt assets

The Python API is designed so a prebuilt asset archive can be added later.
Such an archive will be published only after written permission covers the
official-source-derived OBJ, heightfield, and preview media. Until then,
GitHub, package indexes, CI, and releases remain code-only.
