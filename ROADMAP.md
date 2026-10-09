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
| 3 | Perimeter corners and deck-to-fence transitions | High | Pending |
| 4 | Wall contact and collision ownership | High | Experimental candidate available |
| 5 | Portable full-field interaction regression | After 2–4 | Partial tooling available |
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

Select accessible routes at all four corners and along the top/bottom deck
transitions. Follow the source obstacles when choosing approaches; an obstructed
straight line is not a valid corner test. Exercise angled approach, wall sliding
and retreat to check the continuous support between the deck and fence.

Done when the test footprint cannot drop into an unintended gap or become trapped
between the playable deck and containment proxy on the selected routes.

### 4. Wall contact

Finish the existing source-wall candidate with low-side, high-side, reverse and
repeated-contact cases. Give each replaced region one collision owner and inspect
seams for duplicate support. Record penetration, contact count and acceleration
with identical solver settings across candidate and baseline.

Done when the source-bound wall routes and adjacent ground pass the same public
probe suite, and enabling the replacement preserves accepted slope/fly routes.

### 5. Portable regression

Combine accepted routes into one versioned suite that uses an included robot
and controller. Cover loading, reset, manual/policy input, stepping and result
recording through the public interfaces. Compare all profiles of one pack on
the same inputs. Retain synthetic tests for CI and local-source tests for geometry.

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
