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

For `fly_ramp_north` and `fly_ramp_south`, the descriptor also contains the
pack-measured approach, takeoff, gap and landing route plus a terrain-surface
spawn. The consumer places its own robot at a collision-supported root height;
the supplied spawn height is the field surface, not a robot pose. The optional
Isaac backend remains a descriptor export, not an Isaac simulation or training
result.

## Fly-ramp PhysX contact probe (UNVERIFIED_RUNTIME)

`physx_fly_contact_smoke.py` is a standalone **Isaac Sim 6.1** probe for the
locally exported north or south fly-ramp region. Its Isaac runtime path has not
been run in this repository's development environment. It is a small contact
and scale test, not a SCUT/Fudan robot or policy adapter.

The input is the two-file output of `export_isaac_training_region()`. First
verify the copied export against its source region on the producing machine:

```python
from rmuc2026_mujoco import load_isaac_training_region_export

load_isaac_training_region_export(
    "/path/to/fly-north-isaac-export",
    source_region="/path/to/source-fly-north-region",
)
```

Then copy the export to an [Isaac Sim 6.1 supported Linux host](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html)
and run, from this repository's root:

```bash
# This check uses ordinary Python/NumPy and does not start Isaac or PhysX.
uv run --locked --project . python adapters/isaac/physx_fly_contact_smoke.py \
  /path/to/fly-north-isaac-export --envs 16 --check-only

# Run each size as its own fresh process with the installed Isaac Sim runtime.
/path/to/isaac-sim/python.sh adapters/isaac/physx_fly_contact_smoke.py \
  /path/to/fly-north-isaac-export --envs 1 --output /path/to/new-north-1.json
/path/to/isaac-sim/python.sh adapters/isaac/physx_fly_contact_smoke.py \
  /path/to/fly-north-isaac-export --envs 4 --output /path/to/new-north-4.json
/path/to/isaac-sim/python.sh adapters/isaac/physx_fly_contact_smoke.py \
  /path/to/fly-north-isaac-export --envs 16 --output /path/to/new-north-16.json
```

Repeat for the south export. The script authors **one independent static
triangle-mesh collider per environment**; it does not share one contact surface
and call that 16 environments. Mesh vertices are the original height samples,
with the same diagonal used by MuJoCo's heightfield; only the required USD
float32 point conversion is applied. Each collider has four independent 120 mm
diameter sphere probes at approach, low seam, takeoff and landing. Eight
non-node vertical PhysX scene rays per environment compare the cooked collider
height with the source triangle. Each case runs 500 steps at 1 ms by default.

On the real 213×614 north and south v4 source grids, 200 non-node samples per
side were compared against MuJoCo `mj_ray` on the heightfield geom. Direct
barycentric height on these emitted triangle faces differed by at most
`2.15e-8 m`; the opposite diagonal would differ by as much as `0.115 m` at
the selected saddle cells. The USD float32 point conversion is bounded by
`2.29e-7 m` in these regions. This is **offline MuJoCo/grid evidence only**;
PhysX collision cooking, ray hits and contact behavior remain unverified until
the host run succeeds.

The resulting JSON records source/profile hashes, sample order, point
quantization, ray error, first contact on every probe, wall-clock throughput
and process memory. `PHYSX_CONTACT_PROBE_PASS` means all sphere probes touched
and all collider rays matched within 1 mm in that run. Inspect the Isaac
console log for warnings and record GPU memory with `nvidia-smi` separately;
the script does not parse solver logs or query peak VRAM.
It does not establish robot landing quality, MuJoCo/PhysX friction parity or
large-scale training throughput. Retain the full field as the final policy
evaluation scene.

The runtime authoring follows official [Omni Physics static triangle-mesh
collision](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/rigid_bodies_articulations/collision.html),
[scene-query](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/physics_umbrella/physics_umbrella_runtime.html),
[Isaac Sim simulation stepping](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/py/source/extensions/isaacsim.core.simulation_manager/docs/index.html)
and [experimental contact sensor](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/sensors/isaacsim_sensors_physics_contact.html)
interfaces. A supported host must still execute and, if needed, adapt the
unverified script against its installed 6.1 runtime before relying on results.
