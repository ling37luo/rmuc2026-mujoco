# rmuc2026-mujoco

An **unofficial, asset-free** RMUC 2026 field builder and loader for MuJoCo.
It turns the pinned official STEP file into a relocatable local runtime pack,
checks every generated file by SHA-256, loads the field by itself, and can
compose it with a separately licensed robot MJCF.

The repository contains code and documentation only. It does **not** contain
the official STEP or rulebook, meshes or heightfields derived from them,
Fudan robot files, checkpoints, ONNX policies, or RL-Lab run data.

## What you get

- one-command, direct-from-official-source local setup;
- CAD-derived visual geometry split into material groups;
- a 2 cm collision grid with exact floating-point NPZ injection;
- `full` viewing and mesh-free `collision_only` training profiles;
- explicit, non-official dry/low/high friction sensitivity presets;
- relative paths and a fail-closed manifest/hash contract;
- an arena-only viewer and `MjSpec` robot composition;
- world-space bounds and bilinear terrain-height queries;
- no reinforcement-learning framework in the runtime dependency tree.

## Quick start

Python 3.10+ and MuJoCo 3.3+ are supported. Building from CAD is much heavier
than loading the result: the official STEP is about 1.25 GB and the local
conversion requires at least 5 GiB of free disk space.

```bash
git clone https://github.com/ling37luo/rmuc2026-mujoco.git
cd rmuc2026-mujoco
python -m pip install '.[build]'

# Review the pinned source identity without downloading anything.
rmuc2026-field source

# Explicitly download from RoboMaster, verify size/SHA-256, and build locally.
rmuc2026-field setup ./local-rmuc2026-field \
  --acknowledge-reference-only

rmuc2026-field verify ./local-rmuc2026-field
rmuc2026-field view ./local-rmuc2026-field

# Headless users can skip all CAD visual meshes when composing/loading.
rmuc2026-field view ./local-rmuc2026-field \
  --profile collision_only --friction dry
```

`setup` caches the source at
`~/.cache/rmuc2026-mujoco/RMUC2026_V2.0.0.stp`. Use `--step-cache PATH` to
choose another location. Existing files are reused only after the complete
identity check passes; outputs are never silently overwritten.

If you already have the pinned STEP, the two explicit commands are:

```bash
rmuc2026-field download ./RMUC2026_V2.0.0.stp \
  --acknowledge-reference-only
rmuc2026-field build \
  --step ./RMUC2026_V2.0.0.stp \
  --output ./local-rmuc2026-field
```

The acknowledgement records that the upstream publication is treated as
reference material, not as a redistribution license. It does not accept any
third-party terms on your behalf.

## Use from Python

```python
from rmuc2026_mujoco import FieldAsset, field_bounds, height_at, load_model

field = FieldAsset.open("./local-rmuc2026-field", verify=True)
model, data = load_model(field, profile="full", friction_preset="dry")

print(field_bounds(field))
print(height_at(field, x=0.0, y=0.0))
```

To attach a robot that you are allowed to use:

```python
from pathlib import Path

from rmuc2026_mujoco import FieldAsset, compose_with_robot

field = FieldAsset.open("./local-rmuc2026-field")
model, data = compose_with_robot(
    field,
    Path("my_robot.xml"),
    profile="collision_only",
    friction_preset="dry",
)
```

Supply a **robot-only** MJCF. The package does not guess which floors, lights,
or world bodies in another complete scene should be removed. See
[`examples/arena_only.py`](examples/arena_only.py) and
[`examples/compose_robot.py`](examples/compose_robot.py).

## Runtime-pack contract

A generated pack contains:

```text
local-rmuc2026-field/
├── manifest.json
├── rmuc2026_field.xml
├── rmuc2026_field_collision_only.xml
├── collision/
│   ├── rmuc2026_heightfield.png   # MuJoCo dimensions/bootstrap
│   └── rmuc2026_heightfield.npz   # exact verified float samples
└── visual/
    └── *.obj
```

Every referenced path must remain inside the pack, be a regular file, and
match its declared size and SHA-256. The PNG is not the authoritative contact
surface: the loader replaces its quantized values with the verified NPZ floats
before the first simulation step.

The `dry`, `low`, and `high` friction values are project-provided sensitivity
presets, **not** official RoboMaster material measurements. A preset changes
only the field heightfield geom; it does not overwrite robot-side friction.

## Accuracy boundary

The current collision model is a conservative, single-valued 2.5-D top
surface. It is useful for flat ground, ramps, steps, and bounded static
interaction, but it cannot correctly preserve underpasses, stacked surfaces,
vertical walls, overhangs, or moving field mechanisms. A highest-surface
heightfield can seal an opening that is visibly open in the CAD mesh.

For that reason generated manifests deliberately remain `DRAFT_BLOCKED`:

| Layer | Current status |
| --- | --- |
| Source identity and file integrity | Checked |
| CAD-derived visual layout | Available, simplified |
| Static 2 cm heightfield collision | Available |
| Multi-level / overhanging collision | Not represented |
| Dynamic facilities | Not modeled |
| Official simulator status | No; this project is unofficial |
| Final whole-field policy validation | Not claimed |

Do not use a successful load as proof of official geometry, competition-rule
compliance, robot recovery, policy quality, or source-to-target equivalence.

## Asset and licensing policy

The original code and documentation are MIT-licensed. That license does not
cover RoboMaster/RMUC/DJI material or derivatives. The official publication
does not state a standard asset redistribution license, so this repository
keeps those files on each user's machine and does not mirror them in source,
releases, CI, or package indexes. Read [ASSET_POLICY.md](ASSET_POLICY.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before distributing a locally
generated pack.

If written redistribution permission is obtained, a separately licensed
prebuilt asset release can be added later without changing the Python API.
The staged geometry and interaction work is tracked in [ROADMAP.md](ROADMAP.md).

## Development

All tests use tiny synthetic geometry and never fetch official assets.

```bash
python -m pip install -e '.[test]'
python -m pytest
ruff check .
ruff format --check .
```

## 中文说明

这是一个**非官方、源码仓库不带场地资产**的 RMUC 2026 MuJoCo 模块。
安装 `build` 依赖后，`rmuc2026-field setup` 会直接从 RoboMaster 官方地址
下载指定 STEP，核对固定大小和 SHA-256，并只在你的电脑上生成可搬运场地包。

现在的视觉模型来自官方 CAD；碰撞采用 2 cm 单值高度场，因此平地、坡道和台阶
可以直接用于 MuJoCo，但桥下空间、悬空结构、垂直墙面和动态机关还不是精确碰撞。
所以当前版本适合开发、导航和静态交互验证，不冒充官方比赛仿真器，也不把
`DRAFT_BLOCKED` 写成“全场已精确验收”。

`full` 模式用于看完整 CAD 外观；`collision_only` 不加载 38 组视觉网格，适合
无界面训练。`dry/low/high` 只是本项目用于敏感性测试的非官方摩擦预设，不是赛事
材料实测值。

你可以单独打开地图，也可以把自己的 robot-only MJCF 接进去。以后获得明确的
资产再分发许可后，可以另发预构建地图包，让使用者跳过 1.25 GB STEP 的本地转换；
在此之前，代码可以公开，官方文件和派生资产不能跟着仓库一起上传。
