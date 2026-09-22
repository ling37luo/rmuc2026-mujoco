# Isaac training adapter

Isaac/Isaac Lab is an optional consumer of this repository. Keep its runtime
dependencies in the consumer project and call
`rmuc2026_mujoco.load_isaac_heightfield(pack, scenario=..., profile="collision_only")`.
The returned arrays and descriptor contain the verified heightfield, bounds,
scenario routes, turning phases and spawn-selection contract, source manifest
hash, profile hash, and collision file hashes. In particular, `turn_basic`
uses the same phase names (`spin`, `arc`, `reversal`) and profile identity as
the MuJoCo runner, so a high-parallel Isaac run can be compared with a small
MuJoCo run without silently changing the field.

The consumer may convert those arrays to an Isaac heightfield or simplified
collision actors. It must keep the descriptor with training output and must
not add RL-Lab-specific geometry patches at runtime.
