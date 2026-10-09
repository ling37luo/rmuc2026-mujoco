# Training and evaluation

Use one verified field pack for full-field evaluation and local training
scenarios. The repository supplies geometry, reset/stepping interfaces and
measurements; your project supplies the robot, policy, rewards and optimizer.

The commands below assume the [quick start](../README.md#quick-start) is complete,
the environment is activated, and the working directory is the repository root.
`run` is headless. Use `view` when you want a window.

## Deck and perimeter contact

Retain the source build with `setup --keep-source-build ./runs/field-source`,
then check a pack that includes the perimeter proxy:

```bash
rmuc2026-field perimeter-check ./local-rmuc2026-field \
  --source-manifest ./runs/field-source/manifest.json --output ./runs/perimeter.json
```

This selects source-supported strips on all four sides and runs a 120 mm wheel
through angled approach, sliding in both directions, and retreat at 0.3/0.5/1.0 m/s.
Height is unconstrained. The report records penetration, support height, contacts,
acceleration, warnings and phase completion, and compares states and contacts
across all available profiles. Use `--profiles collision_only` for a shorter check.

`PASS` covers the selected strips. Corner panel joins are checked geometrically;
source-obstructed corner ground remains unverified for driving. A wheel pass does
not certify a robot that keeps driving into a wall. Reports remain local, and
the command does not modify the field pack.

## Isolated stairs and ledges

When building a 1 cm pack, add `--keep-source-build ./runs/field-source` to
`setup` or `build`. Source contact checks need that retained CAD build; the
normal build removes it after exporting the runtime pack.

```bash
rmuc2026-field stairs-check ./local-rmuc2026-field \
  --source-manifest ./runs/field-source/manifest.json --output ./runs/stairs.json
rmuc2026-field run ./local-rmuc2026-field --scenario stairs_basic \
  --route-catalog ./runs/stairs.json --directions uphill downhill roundtrip \
  --speeds 0.3 0.5 1.0 --workers 2
rmuc2026-field view ./local-rmuc2026-field --scenario stairs_basic \
  --route-catalog ./runs/stairs.json --direction downhill --control policy
```

`stairs-check` selects isolated risers from reviewed source parts, checks both
treads and the seam, then compares a 120 mm wheel against the field and a local
source-plane reference at 0.3/0.5/1.0 m/s. It records blocked approaches as well
as traversals. A matched blocked approach can pass the contact comparison;
it does not mean the wheel climbed the step. Field geometry is unchanged.

`run` and `view` use the same catalog, robot composition and direction tracking.
Add `--robot` and `--controller` to use your own robot, or `--patch ID` in the
viewer / `--patches ID...` in batches to select a route. `SlopeSession` accepts
`route_catalog` for Python use. Reports separate numerical health from traversal;
small example wheels are not expected to climb every source step.

Keep the report local and use a new filename for another contact check. The
catalog is bound to its field pack. Coverage is limited to selected isolated
steps; compound stairs, adjacent walls and multilevel structures are not certified.

## Ordinary slopes

Inspect the available routes, then run the included four-wheel rover:

```bash
rmuc2026-field slope-catalog ./local-rmuc2026-field
rmuc2026-field run ./local-rmuc2026-field --scenario slope_basic \
  --directions uphill downhill roundtrip --speeds 0.3 0.5 \
  --repeats 2 --workers 2
```

The catalog screens ordinary slopes below 15 degrees and excludes the fly-ramp
corridors. A screened patch is a route candidate, not a dynamic pass. Available
routes depend on the pack. `--patches ID...` selects catalog routes.

| Direction | Start | Completion |
| --- | --- | --- |
| `uphill` | Low endpoint | Reach the high endpoint |
| `downhill` | High endpoint | Reach the low endpoint |
| `roundtrip` | Low endpoint | Complete both legs in one episode |

Watch the automatic example on a selected route:

```bash
rmuc2026-field view ./local-rmuc2026-field \
  --scenario slope_basic --control policy --direction roundtrip --profile full
```

Human viewing defaults to uphill only. Separate uphill and downhill runs do
not count as a roundtrip. Batch reports retain each leg, failure and timeout.

## Fly ramps

Run both ramps or select `fly_ramp_north` / `fly_ramp_south`:

```bash
rmuc2026-field run ./local-rmuc2026-field --scenario fly_ramp \
  --speeds 1.5 2.0 2.5 --repeats 2 --workers 2
rmuc2026-field view ./local-rmuc2026-field \
  --scenario fly_ramp_north --control policy --profile full --camera spawn
```

The included rover is a working interaction example, not a guaranteed jump
policy. Reports follow approach, takeoff, flight, recontact and landing hold.
For fly-ramp batches, `status=PASS` means the simulation completed without
numerical or controller errors; `task_status` and episode outcomes report jump
success. Inspect impact forces as well as route completion.

Use `--approach-distances`, `--lateral-offsets` and `--heading-offsets-deg` for
route variations within the robot's available clearance. These change starts,
not the underlying field geometry.

## External controllers

For slope or fly-ramp runs, supply both `--robot ./robot.xml` and
`--controller ./controller.py:make`. The factory returns the same callback
used by `view`; see the [template](../examples/controller_template.py).
It may also expose:

- `configure_route(route, direction, speed)` to receive the episode route;
- `reset(model, data)` to reset policy state;
- `set_seed(seed)` for private random generators;
- `press_name(key)` / `release_name(key)` for manual input.

The callback owns actuator mapping and policy-update timing. A factory with a
`field_asset` keyword receives the selected field or training region for queries.

Turning has an explicit command interface. Copy the
[turn controller template](../examples/turn_controller_template.py) into your
project and implement `reset(model, data, spawn, seed)`,
`step(model, data, command, step_index)` and optional `observe(model, data)`.
The template writes zero actions and is not a turning controller by itself.
After providing your `robot.xml` and `turn_controller.py`:

```bash
rmuc2026-field run ./local-rmuc2026-field \
  --robot ./robot.xml --controller ./turn_controller.py:make \
  --scenario turn_basic --phase spin --workers 2 --envs-per-worker 2 \
  --duration 8 --seed 42 --telemetry runs/turn-spin.json
```

Phases are `spin`, `arc` and `reversal`. The runner selects screened flat starts,
uses a zero-action first policy interval, then advances physics at 500 Hz and
the controller at 100 Hz. Turning metrics describe the supplied robot and
controller, not a universal pass for every robot.

## Parallel execution and reports

| Runner | Parallel layout |
| --- | --- |
| Turning | `spawn` processes; one `MjModel` and independent `MjData` objects per worker |
| Slopes and fly ramps | One active environment per worker; increase `--workers`, keep `--envs-per-worker=1` |
| External Isaac trainer | Separate consumer of exported field data |

Start with a small worker count and measure throughput and memory before
increasing it. Summaries are the default; use `--trajectory-dir` for detailed
slope/fly trajectories. Those runners create timestamped reports under `runs/`
when `--telemetry` is omitted; give turning runs an explicit telemetry path.
Reports record field/profile/robot identities, seeds, solver settings,
contacts, failures, throughput and worker memory. Numerical health and task
success must both be checked when comparing policies.

Python integrations can use `MuJoCoScenario`, `SlopeSession`, `FlyRampSession`
and their batch helpers for reset, stepping and diagnostics.

## Local fly-ramp regions

Use the 1 cm fenced pack from the [perimeter step](../README.md#quick-start)
to include both terrain and perimeter contact in the optional PhysX probe.
Export the two ramps separately:

```bash
rmuc2026-field training-region ./runs/field-fenced ./runs/fly-regions/north \
  --scenario fly_ramp_north --approach-distance 0.9
rmuc2026-field training-region ./runs/field-fenced ./runs/fly-regions/south \
  --scenario fly_ramp_south --approach-distance 0.9
rmuc2026-field run ./runs/fly-regions/north --scenario fly_ramp_north \
  --speeds 2.0 --workers 2 --footprint-radius 0.35
```

Use new output directories for new exports. The crop copies source grid samples
without resampling, keeps the source perimeter if present, and excludes visual
meshes. Its manifest records source and profile hashes, bounds and the route.
Set `--footprint-radius` for your robot; 0.35 m above is an example.

`TrainingRegion.open()` and `compose_training_region_with_robot()` provide the
same region to Python consumers. Cropped edges are artificial training limits:
reset when `region.contains_footprint(x, y, radius)` returns false. They are not
new physical walls. Use the complete field for final policy evaluation.

## Isaac handoff

`load_isaac_heightfield(pack, scenario=..., profile="collision_only")` supplies
height samples, coordinates, routes and hashes to an external Isaac consumer.
`run --backend isaac` emits a descriptor; it does not start training.

The [Isaac guide](../adapters/isaac/README.md) shows region export and the optional
PhysX probe. Contact probes have been exercised at 1, 4 and 16 environments;
large-scale articulated training throughput and cross-backend friction response
remain unmeasured. Keep each exported profile's identity with training results.
