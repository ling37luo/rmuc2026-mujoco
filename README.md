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
- a configurable 1–10 cm collision grid with exact floating-point NPZ injection;
- `full` viewing and mesh-free `collision_only` training profiles;
- explicit, non-official dry/low/high friction sensitivity presets;
- relative paths and a fail-closed manifest/hash contract;
- an arena-only viewer and `MjSpec` robot composition;
- an automatic whole-field overview camera;
- world-space height, normal, slope, relief, and runtime-proxy spawn screening;
- no reinforcement-learning framework in the runtime dependency tree.

## Quick start

Python 3.10–3.12 and MuJoCo 3.3+ are tested. Building from CAD is much heavier
than loading the result: the official STEP is about 1.25 GB and the local
conversion requires at least 5 GiB of free disk space.

```bash
git clone https://github.com/ling37luo/rmuc2026-mujoco.git
cd rmuc2026-mujoco

# Isolated install; no system `python` alias is required. Minimal Ubuntu images
# may first need: sudo apt install python3-venv
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install '.[build,viewer]'

# Review the pinned source identity without downloading anything.
.venv/bin/rmuc2026-field source

# Explicitly download from RoboMaster, verify size/SHA-256, and build locally.
.venv/bin/rmuc2026-field setup ./local-rmuc2026-field \
  --heightfield-resolution 0.01 \
  --include-surface-guide

.venv/bin/rmuc2026-field verify ./local-rmuc2026-field
.venv/bin/rmuc2026-field view ./local-rmuc2026-field

# Headless users can skip all CAD visual meshes when composing/loading.
.venv/bin/rmuc2026-field view ./local-rmuc2026-field \
  --profile collision_only --friction dry
```

If [`uv`](https://docs.astral.sh/uv/) is already installed, the first three
installation commands can instead be:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python '.[build,viewer]'
```

The executable lives inside `.venv/bin/` unless you activate the environment.
This avoids both Ubuntu's missing `python` alias and externally-managed system
Python errors. If `python3 -m venv` reports that `ensurepip` is unavailable,
install the distribution's `python3-venv` package or use the `uv` route above.

`setup` caches the source at
`~/.cache/rmuc2026-mujoco/RMUC2026_V2.0.0.stp`. Use `--step-cache PATH` to
choose another location. Existing files are reused only after the complete
identity check passes; outputs are never silently overwritten.

If you already have the pinned STEP, the two explicit commands are:

```bash
.venv/bin/rmuc2026-field download ./RMUC2026_V2.0.0.stp \
  --acknowledge-reference-only
.venv/bin/rmuc2026-field build \
  --step ./RMUC2026_V2.0.0.stp \
  --output ./local-rmuc2026-field \
  --heightfield-resolution 0.01 \
  --include-surface-guide \
  --rulebook ./RoboMaster-2026-rulebook-V2.0.0.pdf
```

The acknowledgement records that the upstream publication is treated as
reference material, not as a redistribution license. It does not accept any
third-party terms on your behalf.

`--include-surface-guide` also downloads and verifies the pinned official
V2.0.0 rulebook into the local cache and extracts its overhead illustration
locally. The runtime pack keeps the complete RGB overhead illustration, including
its floor colours, field modules, obstacles, baked robots, and baked shadows.
This matches the earlier `visual5` display instead of reducing the image to a
small set of chromatic markings. The full picture follows the collision
heightfield on a lightweight 20 cm visual grid; cells crossing sharp height
discontinuities are omitted so the texture is not drawn vertically across walls.
The repository never contains the source or generated images. Press `L` in the
interactive viewer to switch between the default flat lighting and cast shadows.
Press `G` to show or hide the optional rulebook livery, which starts hidden and
never participates in contact. The same initial modes can be selected with
`--lighting flat|shadow` and `--livery off|on`. Without the optional `viewer`
dependency, use those startup flags; field loading and physics remain available.
Live `L`/`G` interception is enabled only when Linux/X11 can bind the unique
MuJoCo window owned by the current process. Other platforms and ambiguous
window sessions fail closed and keep the launch-time modes.

## Use from Python

```python
from rmuc2026_mujoco import (
    FieldAsset,
    field_bounds,
    find_spawn_candidates,
    height_at,
    load_model,
    surface_at,
)

field = FieldAsset.open("./local-rmuc2026-field", verify=True)
model, data = load_model(field, profile="full", friction_preset="dry")

print(field_bounds(field))
# This preserves v0.1 compatibility and is bilinear by default.
print(height_at(field, x=0.0, y=0.0))
# Ask explicitly for MuJoCo's triangular collision surface when needed.
print(height_at(field, x=0.0, y=0.0, interpolation="mujoco"))
print(surface_at(field, x=0.0, y=0.0).to_dict())
for candidate in find_spawn_candidates(field, count=4):
    print(candidate.to_dict())
```

The same terrain diagnostics are available without writing Python:

```bash
.venv/bin/rmuc2026-field surface ./local-rmuc2026-field 0 0 \
  --window-radius 0.35
.venv/bin/rmuc2026-field spawns ./local-rmuc2026-field \
  --count 4 --footprint-radius 0.35 --minimum-separation 1.5
```

Spawn results provide the terrain contact height, local normal, centre-face
slope, maximum footprint slope, relief, and grid-boundary clearance. Surface
height and slope follow MuJoCo's two-triangle heightfield cells rather than a
centre-gradient approximation. The filters are deterministic **runtime-proxy
screening**, not certified robot poses: footprint shape, body clearance,
semantic zones, underpasses, and dynamic obstacles still need robot- and
route-specific validation.

The current pack does not include a per-cell ray-hit validity mask. During CAD
conversion, samples with no vertical ray hit can therefore have a filled ground
value that is indistinguishable at runtime from a measured top surface. A
screened result must not be trusted for deployment until the relevant region
has separate topology, clearance, and robot-specific validation.

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
    ├── *.obj
    ├── rmuc2026_surface_guide.obj                 # optional, local only
    └── official_rulebook_v2_overhead_surface.png  # optional, local only
```

Every referenced path must remain inside the pack, be a regular file, and
match its declared size and SHA-256. Verification also compares the collision
bootstrap PNG against the float samples it is supposed to quantize: a consumer
that loads the entrypoint XML directly with MuJoCo reads the image, not the
samples the loader injects. The PNG is not the authoritative contact surface:
the loader replaces its quantized values with the verified NPZ floats before
the first simulation step. `FieldAsset.open(..., verify=False)` skips the hash
and image comparisons and reports `PASS_SIZE_ONLY` instead of `PASS`.

New local builds use runtime-pack schema 2 for the named lighting and optional
livery contracts. The complete livery uses a 20 cm non-contact visual mesh with
MuJoCo's fixed-diagonal heightfield interpolation. The loader continues to accept
schema 1 packs and both schema 2 livery forms: the complete baked guide and the
earlier filtered-marking experiment. Missing schema 2 display controls remain
unavailable instead of being inferred.

Both field entrypoints declare a 2 ms MuJoCo timestep with the Newton solver.
Keep that timestep when evaluating the 1 cm collision candidate. In a composed
model, MuJoCo takes global options from the parent robot specification, so an
integration that replaces the field option must validate its own timestep. A
5 ms parent timestep produced repeatable dense mesh-heightfield contact
instability in the full wheel-legged integration; restoring 2 ms completed a
20 second run under the same 2 m/s command without a numerical warning. That
run does not establish 20 seconds of in-field travel: unconstrained straight
probes reached the finite heightfield boundary after roughly 6.5--8.1 seconds.
Registered in-bounds routes are still required for a whole-route stability
claim. The 2 ms setting is an integration requirement, not a claim that the
current 2.5-D field is competition-ready.

The full profile keeps each original CAD RGBA value in `visual_meshes[*].rgba`
and records the actual renderer colour separately in `display_rgba`. The
`cad_source_contrast_v2` display profile lowers the broad base shell, caps very
bright CAD whites, and preserves colour ordering. Two uniquely named field
lights use the original even illumination and start with cast shadows disabled,
which avoids large-scene shadow-map speckles and dark blocks. `L` changes only
the field key light's `castshadow` flag. `G` changes only display group 4. Both
controls leave state, contacts, friction, solver parameters, and robot-owned
lights unchanged.

The `dry`, `low`, and `high` friction values are project-provided sensitivity
presets, **not** official RoboMaster material measurements. Robot geom values
are preserved. The field receives higher contact priority, so its friction
actually reaches contacts even when a robot has higher authored friction.
This also selects the field's contact solver parameters for automatically
generated pairs. Explicit pairs involving the field receive the selected
friction while retaining their other authored parameters. Global contact
overrides are rejected because they would mask the requested preset.

| Preset | Sliding coefficient | Torsional coefficient (m) | Rolling coefficient (m) |
| --- | ---: | ---: | ---: |
| `low` | 0.35 | 0.002 | 0.00005 |
| `dry` | 1.0 | 0.005 | 0.0001 |
| `high` | 1.5 | 0.01 | 0.0002 |

Torsional and rolling coefficients only affect enabled contact dimensions.
These values support sensitivity tests; they do not label CAD colors as PVC,
rubber, or metal. Actual material calibration needs measured sliding/stopping
data, with robot mass, wheel material, surface condition, and test speed recorded.

### Optional local planar-ramp correction

`rmuc2026_mujoco.contact.refine_planar_ramps` accepts caller-supplied, audited
plane records and world-space height samples. It returns a separate candidate
array and an audit report. It corrects only inset interiors, fades changes
through a transition band, rejects excessive disagreement or overlapping
patches, and leaves other samples unchanged. It adds no collision mesh or
second contact surface. It does not modify or promote a runtime pack; callers
must bind the geometry evidence to the source and test the candidate before use.

## Accuracy boundary

The current collision model is a single-valued 2.5-D top-surface proxy. It is
useful for flat ground, ramps, steps, and bounded static interaction, but it
cannot correctly preserve underpasses, stacked surfaces, vertical walls,
overhangs, or moving field mechanisms. A highest-surface heightfield can seal
an opening that is visibly open in the CAD mesh. Each region therefore needs
its own multi-hit, horizontal-blocker, and clearance checks before being
promoted for physical interaction.

For that reason generated manifests deliberately remain `DRAFT_BLOCKED`:

| Layer | Current status |
| --- | --- |
| Source identity and file integrity | Checked |
| CAD-derived visual layout | Available, simplified |
| Static 1–10 cm heightfield collision | Available; 1 cm is the fine seam candidate |
| Fixed 17° fly-ramp interior and seam audit | Bound to parts 392 and 397 |
| 120 mm wheel dynamics over those two ramps | Separate 12-trial bounded audit |
| Multi-level / overhanging collision | Not represented |
| Dynamic facilities | Not modeled |
| Official simulator status | No; this project is unofficial |
| Final whole-field policy validation | Not claimed |

The repository also provides a read-only recheck for the previously audited
parts 292 and 312. It pins the legacy CAD-ray evidence by hash, verifies the
current 1 cm runtime pack and both source manifests, rejects a changed coordinate
transform, then recomputes whether the
current heightfield seals those candidate gaps. The recorded horizontal ray
masks are replayed for the robot's required entry width. Run it with:

```bash
.venv/bin/python -m rmuc2026_mujoco.clearance_review PACK_DIR \
  --source-manifest SOURCE_MANIFEST.json \
  --multihit-audit MULTIHIT_AUDIT_DIR \
  --horizontal-audit HORIZONTAL_AUDIT_DIR \
  --output NEW_RESULT.json
```

The JSON reports `integrity_status` separately from its top-level `status`.
Matching hashes and a complete recheck can yield integrity `PASS` while the
collision decision remains `BLOCKED`. This audit never edits a runtime pack or
approves an underpass; diagonal access and contact ownership remain separate
checks.

Run the fixed-ramp dynamics gate against an exported pack with:

```bash
.venv/bin/python -m rmuc2026_mujoco.wheel_probe ./local-rmuc2026-field \
  --output ./runs/wheel-probe-result.json
```

The 12 trials use the hash-bound CAD seam endpoints for both directions at
0.3, 0.5, and 1.0 m/s. The nominal part vertex extent is retained as geometry
metadata, but it is not used as a wheel start point beyond the fly-ramp lip.

Do not use a successful load as proof of official geometry, competition-rule
compliance, robot recovery, policy quality, or source-to-target equivalence.

## Asset and licensing policy

The original code and documentation are MIT-licensed. That license does not
cover RoboMaster/RMUC/DJI material or derivatives. Our review found no explicit
standard asset-redistribution grant in the official publication. That is not a
legal determination, so this repository takes the narrower path: it keeps
those files on each user's machine and does not mirror them in source,
releases, CI, or package indexes. Read [ASSET_POLICY.md](ASSET_POLICY.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before distributing a locally
generated pack.

If written redistribution permission is obtained, a separately licensed
prebuilt asset release can be added later without changing the Python API.
The staged geometry and interaction work is tracked in [ROADMAP.md](ROADMAP.md).
The outside projects and collision-design patterns reviewed for future work are
listed in [DESIGN_REFERENCES.md](DESIGN_REFERENCES.md); no code or assets from
those repositories are bundled here.

## Development

All tests use tiny synthetic geometry and never fetch official assets.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

## 中文说明

这是一个**非官方、源码仓库不带场地资产**的 RMUC 2026 MuJoCo 模块。
安装 `build` 依赖后，`rmuc2026-field setup` 会直接从 RoboMaster 官方地址
下载指定 STEP，核对固定大小和 SHA-256，并只在你的电脑上生成可搬运场地包。

打开运行包时，除每个文件的大小与 SHA-256 之外，还会把碰撞引导 PNG 与浮点样本
逐格比对：直接用 MuJoCo 打开入口 XML 的消费者读到的是这张图，而不是加载器注入的
样本。跳过哈希校验的 `FieldAsset.open(..., verify=False)` 只会报告
`PASS_SIZE_ONLY`，不会再被当成完整性通过。

现在的视觉模型来自官方 CAD；碰撞采用可配置 1–10 cm 单值高度场，因此平地、坡道和
台阶可以直接用于 MuJoCo，但桥下空间、悬空结构、垂直墙面和动态机关还不是精确碰撞。
所以当前版本适合开发、导航和静态交互验证，不冒充官方比赛仿真器，也不把
`DRAFT_BLOCKED` 写成“全场已精确验收”。

`full` 模式用于看完整 CAD 外观；`collision_only` 不加载 38 组视觉网格，适合
无界面训练。`dry/low/high` 只是本项目用于敏感性测试的非官方摩擦预设，不是赛事
材料实测值。

带 `--include-surface-guide` 构建时，规则手册俯视图只在本机处理。运行包完整保存俯视图的
RGB 内容，包括地面颜色、场地模块、障碍物、图中烘焙的机器人和阴影；不会再把它过滤成
只有少量红、蓝、橙标线的透明层。视觉曲面使用约 20 cm 网格按 MuJoCo 高度场对角线贴合，
只在突变高度接缝处断开，避免纹理竖跨墙面。默认是无投影平光和隐藏涂装；窗口内按 `L`
切换投影阴影，按 `G` 切换组 4 涂装。两项都只改变显示，不会改变机器人状态、场地接触、
摩擦或求解器参数。读取器仍兼容 schema 1、完整涂装 schema 2 和此前的筛选标线 schema 2。

Ubuntu 默认可能没有 `python` 命令，所以快速开始现在固定使用 `python3` 创建
隔离环境，并从 `.venv/bin/` 调用程序；精简系统还需先安装 `python3-venv`，也可以
直接采用上面的 `uv` 路线。新增的 `surface` 与 `spawns` 命令可查询
坡度、法向、局部起伏并筛选较平坦的候选出生点。坡度按 MuJoCo 高度场单元的
真实三角面计算，而不是中心差分近似；但运行包没有保存逐格射线命中有效掩码，
因此结果只是运行时代理筛选。它们仍只理解单值高度场，不能证明桥下净空、悬挑
结构或机器人本体一定安全，相关区域还需要单独验证。

兼容性说明：`height_at()` 默认仍是 v0.1 的双线性插值；`surface_at()` 以及显式
指定 `height_at(..., interpolation="mujoco")` 时才按 MuJoCo 三角碰撞面计算。

你可以单独打开地图，也可以把自己的 robot-only MJCF 接进去。以后获得明确的
资产再分发许可后，可以另发预构建地图包，让使用者跳过 1.25 GB STEP 的本地转换；
在此之前，代码可以公开，官方文件和派生资产不能跟着仓库一起上传。
