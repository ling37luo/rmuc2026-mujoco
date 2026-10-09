# rmuc2026-mujoco

An open-source RMUC 2026 field builder and simulation interface for MuJoCo.
Build the field from the pinned official CAD source, attach your robot and
controller, and use the same field for interactive evaluation and training
scenarios. The runtime depends on MuJoCo and NumPy, with no RL framework required.

The repository distributes code. Official sources and generated field assets
are downloaded and built locally; see [Asset policy](ASSET_POLICY.md).

## Quick start

Run these commands from the cloned repository. Python 3.10–3.12 are tested.
The official STEP download is about 1.25 GB; allow at least 5 GiB of free disk
space for conversion. Loading a finished pack does not require the CAD tools.

```bash
git clone https://github.com/ling37luo/rmuc2026-mujoco.git
cd rmuc2026-mujoco
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[build,viewer]'

rmuc2026-field setup ./local-rmuc2026-field \
  --heightfield-resolution 0.01 --include-surface-guide
rmuc2026-field verify ./local-rmuc2026-field
rmuc2026-field view ./local-rmuc2026-field
```

`setup` downloads verified sources into a local cache and creates a new pack.
To inspect the pinned source identity first, run `rmuc2026-field source`.
Existing output directories are not overwritten; reuse a completed pack or
choose a new output directory. For source contact checks, add
`--keep-source-build ./runs/field-source` to the build command to retain the CAD
intermediates. If `venv` is unavailable, install your system's
Python venv component or use `uv venv .venv`, activate it, and run `uv pip install '.[build,viewer]'`.

A physical perimeter is an optional addition. For a newly built schema-2 pack:

```bash
rmuc2026-field fence-pack ./local-rmuc2026-field ./runs/field-fenced
rmuc2026-field view ./runs/field-fenced
```

This creates a new pack with a four-sided containment proxy. Its placement and
openings are approximations; the base `setup` output does not include this fence.

## Choose a profile

| Profile | Use | Contents |
| --- | --- | --- |
| `full` | Interactive viewing and final evaluation | CAD visuals and field collision |
| `collision_only` | Headless simulation and batch evaluation | Same physics, without CAD visual meshes |
| `interactive_lite` | Optional visual experiment | Reduced CAD meshes, available only in a `lite-pack` export |

`full` is the viewer default; batch runs default to `collision_only`.
Profiles in the same pack share collision and material parameters. The lite
profile has not demonstrated a consistent resource benefit, so it is not the
recommended performance option. See `rmuc2026-field lite-pack --help` to build one.

## Display and interaction

```bash
rmuc2026-field view ./local-rmuc2026-field \
  --profile full --lighting flat --livery off
```

- **L** toggles cast shadows; **G** toggles the optional rulebook livery.
- These shortcuts require the `viewer` extra and a focused Linux/X11 viewer.
  Other platforms can use the startup flags above.
- Livery is visual only. The source illustration contains baked objects and
  shadows, so it is a guide rather than a clean material texture.

To drive the included example rover on an available ordinary-slope route:

```bash
rmuc2026-field view ./local-rmuc2026-field \
  --scenario slope_basic --profile full --control human --camera spawn
```

**W/S** set forward/reverse motion, **A/D** steer, **E** straightens, **X** stops,
**1–4** select speed, and **R** resets. These example commands stay active until
changed. Autonomous slope and fly-ramp examples are in the [training guide](docs/TRAINING.md).

## Attach your robot

Supply a robot-only MJCF and a controller for its actuators. This public example
checks composition and stepping with a passive sphere; it does not drive a robot:

```bash
rmuc2026-field view ./local-rmuc2026-field \
  --robot examples/simple_robot.xml \
  --controller examples/controller_template.py:make \
  --profile collision_only --headless --steps 1000
```

After creating your own `robot.xml` and `controller.py`, use:

```bash
rmuc2026-field view ./local-rmuc2026-field \
  --robot ./robot.xml --controller ./controller.py:make \
  --control policy --profile full
```

The [controller template](examples/controller_template.py) shows a factory
returning a callback that reads state and writes `data.ctrl`. The template
writes zero actions; replace it with your control logic. Optional `press_name`
and `release_name` methods receive keyboard events. Declare `viewer_keys` to
reserve those keys from MuJoCo's native shortcuts. A viewer without a
controller is passive.

Python integrations use the same composition:

```python
from rmuc2026_mujoco import FieldAsset, compose_with_robot

field = FieldAsset.open("./local-rmuc2026-field", verify=True)
model, data = compose_with_robot(field, "./robot.xml", profile="collision_only")
```

Use the package loaders to inject the verified floating-point heightfield.
Loading the XML directly uses its quantized bootstrap image. Composition
inherits global solver options from the parent robot; check the resulting
`model.opt.timestep` and solver settings. Field entrypoints use a 2 ms timestep.

## Scenarios and current scope

| Scenario | Available support |
| --- | --- |
| `full_eval` | Complete field viewing and external robot evaluation |
| `turn_basic` | Screened flat starts; spin, arc and reversal batches |
| `slope_basic` | Screened routes; separate uphill, downhill and roundtrip results |
| `fly_ramp_north`, `fly_ramp_south` | Jump episodes and same-source local training regions |
| `stairs_basic` | Source-checked isolated steps; paired wheel checks and robot traversal |
| `boundary_contact` | Source-screened perimeter contact and whole-rover corner approach/retreat checks |

The [training guide](docs/TRAINING.md) covers commands, controller interfaces
and parallel execution. The [Isaac adapter](adapters/isaac/README.md) exports
field data and provides a separate PhysX contact probe.

`rmuc2026-field check PACK --source-manifest SOURCE/manifest.json --output runs/check.json`
runs the [portable field regression](docs/TRAINING.md#one-command-field-regression)
with included robots and automatically generated routes. The report separates
field checks from robot task outcomes and identifies unavailable pack features.

The main collision surface is a single-height-per-XY heightfield. Underpasses,
stacked surfaces and overhangs are not represented correctly. An optional
source-wall export adds local convex collision. Selected wall, perimeter and
reachable corner approaches have regression checks; these do not cover every
structure or robot. Dynamic match mechanisms are outside the
current locomotion baseline.

Packs therefore retain `DRAFT_BLOCKED`: file verification and individual route
checks do not establish whole-field physical accuracy. This label does not
prevent using the existing scenario tools; evaluate the contacts and routes
needed by your robot. `dry`, `low` and `high` are optional friction sensitivity
presets, not measured official material values; omit `--friction` to retain the
pack's parameters. The project is an unofficial simulator.

## Documentation and development

- [Training and evaluation](docs/TRAINING.md)
- [Roadmap: six scoped tasks](ROADMAP.md)
- [Changelog](CHANGELOG.md) and [contributing](CONTRIBUTING.md)
- [Technical references](DESIGN_REFERENCES.md)
- [Asset policy](ASSET_POLICY.md) and [third-party notices](THIRD_PARTY_NOTICES.md)

For more options, run `rmuc2026-field --help` or a subcommand's `--help`.
Tests use synthetic fixtures and do not download official assets.

## 中文说明

本仓库提供统一的 RMUC 2026 场地、机器人接入和训练场景接口。官方文件由使用者
在本地下载并生成场地包；仓库中不包含场地资产或特定机器人的策略。
交互查看使用 `full`，批量仿真使用 `collision_only`。普通坡和飞坡已有可运行示例；
台阶、墙面、围栏与可达角落已有局部接触检查，可通过 `check` 统一运行。
复杂台阶组合、多层结构和任意机器人的通行能力仍不属于已验证范围。
