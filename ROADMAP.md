# Roadmap

The main product is a reproducible full-field interaction baseline. Training
scenarios use the same source geometry and remain independent of any robot or
RL framework. This plan has six tasks; it does not require training a new policy.

## Current baseline

Version 1.2.1 includes local field construction, external robot/controller
composition, turning and slope batches, fly-ramp regions and an Isaac data
handoff. Selected wheel routes and small parallel contact probes have passed.
Whole-field static contact remains incomplete. Development now includes
source-bound isolated stair routes, paired wheel checks and robot traversal.
Complex stair assemblies remain outside that validation.

## Tasks and order

| ID | Task | Priority | Status |
| --- | --- | --- | --- |
| 1 | Concise public documentation and reproducible entry points | First | Complete |
| 2 | Stair approaches, risers and landing transitions | High | Isolated source routes implemented |
| 3 | Perimeter corners and deck-to-fence transitions | High | Selected deck strips and reachable whole-rover corner approaches implemented |
| 4 | Wall contact and collision ownership | High | Source ownership and repeated side/end contact checks implemented |
| 5 | Portable full-field interaction regression | After 2–4 | Versioned `check` suite implemented with public robots and generated routes |
| 6 | Parallel performance and profile selection | After baseline regression | Small-scale measurements available |

### 1. Public documentation

Keep setup, public examples, current limitations and this task list in the
repository. Keep personal sessions, machine-specific reports and intermediate
geometry investigations in local output directories. Example commands must use
included files or clearly identify the files a consumer must provide.

Done when documentation links and command interfaces are checked, and a reader
can distinguish a working example from a controller template or an experiment.

### 2. Stairs

`stairs-check` and `stairs_basic` provide the initial isolated-step suite.
Source geometry and robot traversal outcomes are reported separately.

Identify stair components from the pinned source and export their approach,
riser and landing routes. Check geometry first, then run wheel probes in both
directions at 0.3, 0.5 and 1.0 m/s; follow with the public rover. Replace a local
collision region only if the current heightfield demonstrably fails there.

Done when exported routes reproduce their source geometry and traverse without
unexplained snagging, false support or numerical warnings. Keep robot capability
failures separate from field-contact defects.

### 3. Perimeter and deck edges

`perimeter-check` exercises supported deck strips at each side, including wall
sliding and retreat, with profile parity checks. Obstructed corner ground is
reported separately from the static fence-panel joins. `corner-check` connects
accepted perimeter entries to the nearest supported corner approach for the
public rover. It contacts the next structure and reverses along the same path.
Literal corner access blocked by source structures is reported explicitly.

The selected routes cover all four corner approaches and top/bottom deck strips.
Source obstacles remain in place. Wheel checks exercise angled contact and
sliding; whole-rover checks exercise approach, loaded contact and retreat.

Done when the test footprint cannot drop into an unintended gap or become trapped
between the playable deck and containment proxy on the selected routes.

### 4. Wall contact

`wall-check` compares the source-wall meshes and lowered heightfield regions,
then exercises low/high sides, free ends and fence junctions with repeated
contact and retreat. It records penetration, contact count and acceleration
and compares identical inputs across profiles. Spawn screening excludes the
mesh-owned wall footprints even where the underlying heightfield is flat.

Done when the source-bound wall routes and adjacent ground pass the same public
probe suite, and enabling the replacement preserves accepted slope/fly routes.

### 5. Portable regression

`check` combines public loading, repeatable reset, manual/policy callbacks,
stepping, wheel contact checks and example robot episodes. Routes are generated
from the pack and retained CAD source; all available profiles use the same
inputs. Each section is saved to one local report. Missing features and robot
task failures are explicit, separate from field/health results. Synthetic tests
remain in CI; official-source runs remain local.

Done when another developer can rebuild the field and run the suite without a
private robot, checkpoint, unpublished report or machine-specific path.

### 6. Performance

Measure model loading, simulation throughput and memory with a fixed public
workload. Reuse the MuJoCo 1/4/16 baseline, then increase Isaac environment counts
as hardware permits. Report actual counts and resource use rather than promising
a universal parallel capacity. Check exported terrain, bounds and contact parity.

Compare `full`, `collision_only` and optional `interactive_lite`. Recommend lite
only if it preserves the needed visual structure and improves memory or viewer
frame rate by at least 20%; otherwise keep `collision_only` as the resource option.

## Scope

Focus on drivable static ground, stairs, slopes, fly ramps, walls and boundaries.
Bridge interiors, stacked surfaces and dynamic competition mechanisms are
separate future work. Official material calibration needs measured data; current
friction presets remain sensitivity settings. Geometry changes produce new local
packs, and physical validation is recorded per accepted route or region.
