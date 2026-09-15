# Design references and reuse boundary

This document records the external material reviewed while designing the
field builder. It is an engineering bibliography, not a statement that those
projects supplied this repository's code, dimensions, or assets.

## Authoritative RMUC references

- The [RoboMaster 2026 field drawing and movable-prop publication][field-source]
  remains the source of the pinned `RMUC2026_V2.0.0.stp` identity used by the
  local builder.
- The [official rule centre][rule-centre] listed rulebook V2.2.0 when this
  review was performed on 2026-09-15. A local comparison of normalized text and
  embedded-image hashes found no change in rulebook chapter 4 relative to
  V2.0.0. That result is deliberately limited to chapter 4; later publications
  and other chapters must be reviewed rather than assumed compatible.

The field publication calls the material a reference, and this review found no
explicit SPDX, Creative Commons, or OSI redistribution grant. This is not a
legal determination. The code therefore keeps the official STEP, rulebook, and
every derived mesh/heightfield on the user's machine. The repository records
source URLs, version identities, hashes, and other provenance metadata only.

## Collision and model-assembly patterns

- [MuJoCo contact parameter mixing](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters):
  equal-priority geom friction uses an element-wise maximum. Therefore changing
  only the field friction can leave low-friction trials ineffective. Our
  explicit sensitivity mode sets field priority and explicit-pair friction,
  and reports the solver-parameter precedence as part of the experiment.
- [MuJoCo XML reference][mujoco-xml] and [model editing][mujoco-edit]: primitive
  collision geometry, contact filtering, explicit bodies/joints, and procedural
  `MjSpec` composition are preferred over one monolithic field mesh.
- [CoACD][coacd] (MIT): a possible future tool for bounded approximate convex
  decomposition of individually accepted irregular static parts. Each convex
  result would remain a separate MuJoCo geom and pass geometry/contact gates.
- [obj2mjcf][obj2mjcf] (MIT): separation of visual meshes, material groups,
  collision processing, and generated MJCF is a useful pipeline pattern.
- [onshape-to-robot][onshape-to-robot] (MIT): reinforces the design rule that
  high-detail visual models and deliberately simple collision proxies should
  be maintained independently.
- [dm_control maze arenas][dm-control] (Apache-2.0): merging adjacent regular
  walls into longer box geoms is a useful pattern for efficient static fields.
- [MuJoCo Menagerie][menagerie] (model-specific licences): several models use
  non-colliding visual meshes with box, sphere, and cylinder collision proxies.
  No Menagerie asset is included, and each model's own licence would have to be
  checked before reuse.
- [MuJoCo Playground rough terrain][playground] (Apache-2.0): its locomotion
  models keep feet-only and full-collision robot variants separate, while rough
  terrain is represented by a heightfield. This is a useful reminder to treat
  robot collision complexity and terrain representation as independent design
  choices; it does not validate this project's planned profiles.

## RoboMaster simulator architecture references

- [rmuc21 Ignition simulator][rmuc21] (MIT): semantic field components and
  movable mechanisms are separate links/joints rather than one static mesh.
- [rmoss_gz_resources][rmoss] (SDF/XML under Apache-2.0; mesh copyrights remain
  with their authors): geometry, controls, and plugins are separated, and
  regular collision uses box/cylinder/sphere proxies.
- [T-DT 2026 rm_simulator][tdt] (MIT): useful only as a reference for game-zone
  and state-machine organisation; it is a 2-D rules simulator, not a source of
  physical RMUC field geometry.

No source code, mesh, texture, measurement table, or generated binary from the
projects above is copied into this repository. Ideas that later lead to adapted
code must be implemented with licence-compatible attribution and isolated
tests. Ambiguously licensed simulator assets are not accepted as substitutes
for official mechanical evidence.

## Resulting implementation direction

1. Keep the current single-valued heightfield and label its topology limits on
   every query result. The current `full` and `collision_only` profiles both
   use that same heightfield; a future explicitly reduced candidate would be
   named `fast_heightfield`, not `fast`.
2. Audit multiple vertical intersections and horizontal blockers before
   promoting any tunnel, bridge, or stacked surface.
3. Build a `hybrid_static` profile from a connected base heightfield plus
   mutually exclusive box/ramp/convex geoms for individually accepted parts.
4. Give each collision location and layer one owner, then test gaps,
   penetrations, drop/slide/ramp behaviour, contact count, and real-time rate.
5. Model movable facilities as explicit bodies, joints, and state machines only
   after their current mechanical and rule contracts are pinned.

Until those gates pass, successful package loading, surface queries, and spawn
screening do not change the manifest status from `DRAFT_BLOCKED`.

[field-source]: https://bbs.robomaster.com/article/814728?source=8
[rule-centre]: https://bbs.robomaster.com/wiki/20204847/809871?source=7
[mujoco-xml]: https://mujoco.readthedocs.io/en/stable/XMLreference.html
[mujoco-edit]: https://mujoco.readthedocs.io/en/latest/programming/modeledit.html
[coacd]: https://github.com/SarahWeiii/CoACD
[obj2mjcf]: https://github.com/kevinzakka/obj2mjcf
[onshape-to-robot]: https://github.com/Rhoban/onshape-to-robot
[dm-control]: https://github.com/google-deepmind/dm_control/blob/main/dm_control/locomotion/arenas/mazes.py
[menagerie]: https://github.com/google-deepmind/mujoco_menagerie
[playground]: https://github.com/google-deepmind/mujoco_playground
[rmuc21]: https://github.com/robomaster-oss/rmuc21_ignition_simulator
[rmoss]: https://github.com/robomaster-oss/rmoss_gz_resources
[tdt]: https://github.com/T-DT-Algorithm-2026/rm_simulator
