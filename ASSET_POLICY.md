# Asset policy

The source distribution and wheel for `rmuc2026-mujoco` contain Python code
and original documentation only. They must not contain the official STEP/PDF,
meshes or heightfields derived from them, screenshots extracted from the
rulebook, Fudan robot descriptions or meshes, RL checkpoints, or locally
generated run evidence.

## Allowed in the code repository

- official download-page URLs, byte sizes, and cryptographic hashes;
- schema examples with invented names and numbers;
- small geometry and heightfields created entirely by the tests;
- code that validates a user-supplied local asset directory.

## Not allowed without separate written redistribution permission

- `RMUC2026_V2.0.0.stp` or the official rulebook PDF;
- OBJ, GLB, NPZ, PNG, texture, or screenshot files derived from those sources;
- Fudan wheel-legged URDF/MJCF/mesh assets, checkpoints, policies, or ONNX files;
- a GitHub Release archive or Git LFS object containing any of the above.

The `source` command only prints the pinned official source identity. The
separate `download` command is opt-in, requires an explicit reference-only
acknowledgement, contacts only the official URL, and verifies the pinned size
and SHA-256. It does not accept third-party terms on a user's behalf. Downloads
and local derivatives stay outside the Python package and retain the same
license boundary.

CI and package tests use synthetic fixtures only. Release verification must
inspect both source and wheel contents and fail if common binary asset suffixes
or generated run directories are present.

## Runtime accuracy boundary

The default schema-2 local asset manifest uses an unofficial 2.5-D
top-surface heightfield proxy for collision. Downward-ray locations with no
geometry hit are filled with ground height, and that pack has no validity
mask distinguishing measured hits from filled samples. Experimental schema 3
binds a source-miss mask at audited outer edges, but replaces those samples
with a finite surrogate void rather than a true hole or safe robot exit. Neither
model must be described as conservative. A single heightfield cannot represent overhangs, tunnels,
stacked surfaces, vertical walls, moving mechanisms, or the open space below
bridges, and it may seal a route that is visibly open in the CAD-derived mesh.
A manifest whose validation scope is `DRAFT_BLOCKED` remains a draft even when
every file hash passes.

Hash verification means only that a local asset pack is the expected pack. It
does not turn the pack into an official RoboMaster simulator, establish legal
redistribution rights, or prove whole-field physical fidelity.
