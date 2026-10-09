# Isaac adapter

Isaac is an optional consumer. Install its runtime separately; the core field
package has no Isaac or RL-framework dependency. For complete MuJoCo examples,
see [Training and evaluation](../../docs/TRAINING.md).

## Data handoff

`load_isaac_heightfield(pack, scenario=..., profile="collision_only")` returns
height samples, coordinates, boundaries, routes and source/profile hashes.
Keep this identity with training results. A terrain spawn is a surface position;
the consumer must place its own robot at the appropriate supported height.

For fly-ramp regions, first run the two `training-region` commands in the
[training guide](../../docs/TRAINING.md#local-fly-ramp-regions). Then export them
from the repository root using its activated Python environment:

```python
from pathlib import Path
from rmuc2026_mujoco import (
    export_isaac_training_region,
    load_isaac_training_region_export,
)

root = Path("runs/fly-regions")
for side in ("north", "south"):
    source = root / side
    output = root / f"isaac_export_schema3_{side}"
    export_isaac_training_region(source, output)
    load_isaac_training_region_export(output, source_region=source)
```

Use new export directories. Each export contains exact height samples and a
schema-3 descriptor, including any source perimeter boxes intersecting the crop.
The contact probe requires regions from a fenced pack, as in the training guide.
The directory names above are the layout expected by the matrix runner below.
Artificial crop boundaries require environment resets, not extra field walls.

## Optional PhysX contact probe

The probe targets Isaac Sim 6.1. It uses independent static terrain colliders
and 120 mm sphere probes to check terrain, seam and perimeter contact. It does
not load a private robot or train a policy.

Check the export without starting Isaac:

```bash
python adapters/isaac/physx_fly_contact_smoke.py   ./runs/fly-regions/isaac_export_schema3_north --envs 16 --check-only
```

Set `ISAAC_PYTHON` to your installed Isaac launcher (`python.sh`, or the Python
executable in its isolated pip environment). Run this command from the repository
Python environment; it starts each Isaac case in a separate process:

```bash
python adapters/isaac/run_physx_fly_matrix.py   --isaac-python "$ISAAC_PYTHON" --regions-root ./runs/fly-regions   --output-new-dir ./runs/physx-fly-check --device cuda:0 --steps 500
```

Choose a new output directory for each matrix. It runs north and south at 1, 4
and 16 environments and writes per-case reports, console logs and a summary.
After those counts pass, add `--env-counts 16 64` to compare a larger layout.
The standalone probe also accepts `--envs 64`; larger counts remain unmeasured.
The reports include asset identities, contact results, height error, throughput,
process memory and GPU memory sampled across the whole device.

Terrain triangles retain the source heightfield diagonal. PhysX uses float32
points and the source sliding-friction coefficient; MuJoCo torsional/rolling
friction, `solref` and contact bits have no equivalent mapping in this probe.
A successful matrix demonstrates the tested contact cases. It does not establish
robot landing quality, friction-response equivalence or large-scale training
throughput.

Installation: [requirements](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html)
and [Python environment](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_python.html).
