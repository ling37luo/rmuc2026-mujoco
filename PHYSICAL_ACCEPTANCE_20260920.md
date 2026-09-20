# Bounded physical acceptance for the planned 0.3 release

Date: 2026-09-20. Tested code: `93b5acb528fb33eed1c924f4f3204a62f10bb677`.
Decision: **BLOCKED_FOR_V0_3**. This is a code-only project; all generated
field packs remain local and `DRAFT_BLOCKED`. Passing a file hash, unit test,
or short robot route does not certify the whole field.

The tests used verified local official-source-derived runtime packs and the
frozen public Fudan wheel-legged robot at a 2 ms physics timestep. The wall
and edge trials also used a 120 mm, 2 kg spherical probe. Evidence is under
the RL-Lab project's local `runs/` directory, outside this Git repository.
The paths and report SHA-256 values below identify the exact
records; no official or derived asset is redistributed here.

| Gate | Bounded observation | Decision |
| --- | --- | --- |
| Runtime integrity | Schema 2 default, schema 2 seven-sample wall-tip repair, and experimental schema 3 packs passed manifest and file-hash verification. | PASS for integrity only |
| Fixed 17-degree fly ramps | Each of the schema 2 default and schema 3 packs passed 12/12 rolling-wheel routes: both ramps, both directions, 0.3/0.5/1.0 m/s; no solver warning or seam stall. Maximum seam location error was 5 mm against 15 mm. | PASS for those wheel routes and seam positions |
| Strict ramp surface target | Maximum interior plane error was 1.04608 mm on one ramp and 1.04607 mm on the other, exceeding the previously requested 1 mm absolute maximum by about 0.046 mm. The package's current p95 gate passes, but it is a different criterion. | FAIL for the strict maximum |
| Robot baseline | The default pack completed 1,000 stationary physics steps without a numerical warning. The seven-sample wall-tip pack completed 1,000 stationary and 2,000 commanded steps without a warning; late-run forward speed was about 1.67 m/s for a 2 m/s target. | PASS for these short central routes only |
| Exact 402/403 wall candidate | At 0.5 m/s for 5 s, two forward robot approaches had no through-wall event or warning and wall-geom penetration stayed below 1.3 mm. However, both heightfield/robot contact pairs reached MuJoCo's 50-contact limit during startup (105 and 83 steps). The spherical probe also reached 50. Reverse/high-side and repeated-impact routes were not accepted. | BLOCKED for runtime activation |
| Outer edge, schema 2 | The robot remained numerically stable on the tested left route because false playable-height support held it after the source edge. This is not a valid contact pass. | FAIL for source fidelity |
| Outer edge, schema 3 | The spherical probe fell where schema 2 had false support. An unguarded 0.5 m/s robot left the left edge and produced `BADQACC` at about 4.0 s. A test-fixture 50-step no-contact stop caught that one route before the warning, but this is not a pack-level safety contract. | BLOCKED for robot route |
| Source-hit top edge | The matched schema 2 robot route produced `BADQACC` at 1.184 s. The schema 3 route still produced `BADQACC` at 3.376 s under a test-fixture 20-step stop condition: the robot was on source-hit samples with dense field contacts until the failure. A source-miss/no-contact guard cannot cover this event. | BLOCKED |
| Multi-level routes and material zones | No accepted hybrid underpass/stacked-surface contact owner or registered, measured/labeled material zones were available to test. The existing `dry/low/high` friction presets remain sensitivity settings. | NOT READY for 0.3 |

Primary evidence:

- `rmuc2026-field-v030-acceptance/20260920T_wheel_probe_RlU0xP/v0201_threelevel_wheel_probe.json` — SHA-256 `1492710257701f96b3c4072eeaad8506eb7304df1d10f47308529b706e2d9c4e`.
- `rmuc2026-field-v030-acceptance/20260920T_wheel_probe_RlU0xP/schema3_edgevoid_walltip_wheel_probe.json` — SHA-256 `34fba9ed13ace6a266e4464797f0c512b51609b7df45dcb40ed064de61372127`.
- `rmuc2026-field-v030-acceptance/20260920T_wheel_probe_RlU0xP/fudan_2s_smoke/result.json` — SHA-256 `465bc5381301c423b6ba01e5db1fbca51147b20be397aec8d91b92fab6ae66c7`.
- `rmuc2026-physical-acceptance/20260920T_v03_walltip_central1000/result.json` — SHA-256 `113dc8c4731b8d359531681eca0626ef3fc5c77c44b889e1bf8067fd649e3264`.
- `rmuc2026-physical-acceptance/20260920T_v03_walltip_fullspeed4/result.json` — SHA-256 `df5c40028538b802068d570c65cf0af42f5655ec4b7ea97278c6de5e0e08772a`.
- `rmuc2026-v03-acceptance/20260920T_wall_edge_gate_v1/acceptance_summary.json` — SHA-256 `70521e282703eacfef79a7d62d1567ab4334646b382ab90e505405a1ea74b1ff`; this report binds input manifests, lists route criteria, and links the individual wall/edge traces and test-fixture scripts.

Before a 0.3 physical release, establish exclusive contact ownership on the
source-hit top edge and 402/403 wall footprints, rerun reverse and repeated
robot contacts without reaching the heightfield contact cap, and specify a
safe out-of-bounds stop that covers source-hit edges as well as source misses.
Then rerun matched robot routes at the required speeds and close the strict
ramp maximum, hybrid topology, and material-zone gates. Do not promote the
current experimental candidates to a default pack on this evidence.

## Code-prerelease safety follow-up

The `0.3.0a1` code adds a read-only schema-3 boundary stop signal. It was
checked against a new matched 0.5 m/s top-edge start: without the stop, the
robot crossed the source edge and fell to -1.009 m by 1.984 s; with the stop,
four consecutive outward steps at the 50-contact pair limit ended the run at
1.164 s, before the fall and without a numerical warning. A left-edge run
stopped after 20 source-miss/no-contact steps at 3.592 s without a warning.
A separate central 1,000-step run had no guard trigger or numerical warning.
These tests validate bounded termination, not the contact geometry, the
unreproduced earlier top-edge `BADQACC`, or every approach to the perimeter.

Two representative 1.046 mm fly-ramp plane-error nodes were raycast against
the original source GLB. At those nodes the highest source surface is adjacent
part 394 or 398, respectively; the heightfield matches that higher source hit
to floating-point precision. The existing single-ramp-plane maximum is thus
not a valid physical error measurement at those two overlapping nodes. A
full overlap-aware ramp audit is still needed before replacing the strict
1 mm gate.

Additional local evidence under `rmuc2026-v03-acceptance/`:

- `20260920T_sourcehit_guard_v3_nearedge/schema3_top_5_field_guard.json` — SHA-256 `411c6c2b9ba4aa04fb2f193cda28fcdef07f8dc1634be74e119abf17d860eaa2`.
- `20260920T_top_nearedge_control_v1/schema3_top_5.json` — SHA-256 `d2c74a04dc64cc94ed146e7019a961b0557c22d40adcdd6a093c4385f5d2c3dc`.
- `20260920T_sourcehit_guard_v2/schema3_left_5_field_guard.json` — SHA-256 `0479ca4e3eca21ad2a22bfd6ec551bf26ee992dfa6899d3f40f8f46380071fe2`.
- `20260920T_new_guard_central1000/result.json` — SHA-256 `444f7093c8d1c6c645f54108c0047ac6a73976b2834639a2086752a1b4337b18`.
- `20260920T_ramp_overlap_source_v1/result.json` — SHA-256 `db790142cc43fddb72a6f31b210370a05ed42ad4b58b952d0881660ebbb2c6c3`.

## Physical-fence and complete ramp-overlap follow-up

The `0.3.0a2` code can derive a new local schema-2 pack with four touching
box-shaped physical fence geoms at the inferred 28 × 15 m core edge. They
reach 2.4 m above the field floor and use the field's collision bits and
solver settings. No boundary-stop signal is used for this pack. The first
0.15 m outside-core placement sat over source-miss terrain; it was rejected.
The core-edge candidate places its inner contact faces on source-supported
ground in the inspected side stretches. The official fence thickness,
centerline, mesh compliance and dart-window collision are still unknown;
this is an explicitly solid robot-containment proxy.

Four 120 mm sphere approaches (one per side) each completed 1,200 physics
steps, contacted the fence, and remained at least 80 mm inside its centerline
without a numerical warning. Four separate Fudan 0.6 m/s approaches each
completed 3,000 steps with fence contacts, no centerline crossing, no warning
or exception, and a minimum base height above 0.16 m. The field/robot pair
still reached MuJoCo's 50-contact cap in these robot runs. A corner probe
contacted one corner cleanly; the other three were blocked by interior CAD
structures before reaching the fence. The full set of corner poses, speed,
repeated impacts and jumps remains untested. The standard RL-Lab read-only
consumer loaded this exact core-edge candidate and completed a central 2 s
robot run without numerical warnings. It preserved the four declared fence
geoms in the imported field model; its global solver settings still belong
to the robot scene, so this is a compatibility smoke, not perimeter acceptance.

The new full-source ramp audit checked every 1 cm interior sample above the
strict 1 mm dominant-plane target, not just the earlier two representative
nodes. The 197 north and 193 south outliers are the highest surfaces of
adjacent official CAD parts 393/394 and 398/399. Their heightfield/source-top
maximum difference is below `9e-16 m`; overlap-aware interior maximum error
is `1.15e-8 m` north and `9.30e-9 m` south. The ramp samples remain unchanged.
The existing 12/12 two-way wheel routes and 5 mm seam localization evidence
still apply because the heightfield SHA-256 is unchanged.

Local evidence, not distributed with source or release packages:

- Fenced candidate manifest: `rmuc2026-mujoco-package/20260920T_rulebook_perimeter_coreline_candidate_v2/manifest.json` — SHA-256 `72d1094780ab471860cc29ff875a6abf1bbd2ccb621ca9a28a5528ce78c4ee2f`.
- Four robot routes: `rmuc2026-v03-acceptance/20260920T_fence_robot_four_sides_v2/result.json` — SHA-256 `57d9951e541f377adab283b1390b878c6432b75981bbf96a2a075db35fbc1a7f`.
- Standard read-only robot load: `rmuc2026-v03-acceptance/20260920T_fence_robot_readonly_load_coreline_v2/result.json` — SHA-256 `7f2ba025ec20f874794a0e5ca8b2e4d72172f6ccc4b77b1e35e1904888af57e6`.
- Full-source ramp audit: `rmuc2026-v03-acceptance/20260920T_fence_ramp_overlap_v2.json` — SHA-256 `e5c5aa434c188c2060296b5e50e7e44f1d3a70e7d12857e0ec61b9f67a3c3175`.

Decision: **CANDIDATE_ONLY**. The physical fence replaces stop-on-boundary
behavior for this schema-2 pack, but exact official fidelity and full robot
perimeter acceptance remain `DRAFT_BLOCKED`.

## Fly-ramp/perimeter interference correction

The first four-box fence candidate had a specific placement defect: each
fixed fly ramp's outer low corner extended about 5.47 mm beyond the inner
face of its adjacent north/south wall. This could obstruct a robot using the
outer tread even though earlier constrained wheel routes followed the
centerlines and passed. The `0.3.0a3` placement keeps the east/west walls in
place and moves the north/south walls 0.40 m toward the CAD outer apron.
Each ramp now has 0.3945 m of geometric clearance to the wall inner face.
The visual alpha was reduced. The previous v1/v2 fence contracts remain
readable; fresh local exports use v3.

The first moved-wall pack retained a hard fence contact (`solref="0.02 1"`).
A Fudan approach from the north apron reached the wall, then reported
`BADQACC` at 1.400 s; the previous core-edge fence failed at 1.458 s from
the same start. The unfenced control escaped and fell below the field, so it
was not a contact solution. In a matched local A/B, increasing only the
fence contact time constant to 0.04 s completed the 6 s route with no solver
warning and no wall crossing. The candidate pack adopts `solref="0.04 1"`
for the fence alone. Its friction, contact bits, and the entire heightfield
(including the heightfield's `solref="0.02 1"`) are unchanged. This is a
solver-stability sensitivity choice, not measured steel-fence compliance.
[MuJoCo's solver documentation](https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters)
explains the time-constant parameter; the route A/B, rather than that
general guidance, is the evidence for this specific model.

The source-ray-miss mask found 0 misses at 5 cm spacing along all four new
inner wall faces (316 samples on each east/west wall; 560 on each north/south
wall). The final candidate pack passed all 12 two-way, three-speed 120 mm
wheel-probe routes with no solver warning. Four 120 mm sphere approaches
contacted the revised fence, had no centerline crossing or solver warning
over 1,200 steps each. Four bounded 6 s Fudan side approaches contacted the
fence with no warning or centerline crossing; every route still reached the
50-contact heightfield pair cap. The ordinary read-only Fudan viewer
completed a 2 s central robot run; the SCUT `rough_dash` adapter completed a
separate 2 s central closed loop. These central runs do not prove a free
robot can climb, jump, and land across either full fly-ramp route. Robot
contact saturation and official fence details remain open.

Local evidence (not redistributed):

- Final candidate manifest: `rmuc2026-mujoco-package/20260920T_perimeter_ramp_clearance_soft_contact_candidate_v4/manifest.json` — SHA-256 `8e4678b1ac4a5188eba007e9fa14260cfd02b5489b13584ae2627781710e8236`.
- Fence/ramp geometry and source-hit check: `rmuc2026-v03-acceptance/20260920T_perimeter_ramp_clearance_v4_static.json` — SHA-256 `1cb16e0bd2db73c3b5daa4b4b799f61e1fe217f728df1c6821b19a72dc61d85d`.
- Twelve dynamic wheel routes: `rmuc2026-v03-acceptance/20260920T_perimeter_ramp_clearance_v4_wheel.json` — SHA-256 `8fcf9e0be3555fe28a6bf7fd7c34043122c1efc298c4acc74f91d231aa504b24`.
- Four Fudan wall approaches: `rmuc2026-v03-acceptance/20260920T_perimeter_ramp_clearance_v4_robot_four_sides/result.json` — SHA-256 `0eb69dab6fc41bb1e8dffefab683a112281d259b748564a25ef4e5f38364a80a`.
- Hard-fence north-route failure: `rmuc2026-v03-acceptance/20260920T_perimeter_ramp_clearance_v3_robot_four_sides/result.json` — SHA-256 `3bb72a56a904cd251294200372ad364118f2e1f0e9c19db76faa139f2121c14f`.
- Fudan read-only load: `rmuc2026-v03-acceptance/20260920T_perimeter_ramp_clearance_v4_fudan2s/result.json` — SHA-256 `e898b911ae3031f6a3b27cc49ba174fd7de18ae4d91b44b32ebd3fc314a6d5c6`.
- SCUT central closed loop: `scut-rmuc-view/20260920T091215.371060Z/report.json` — SHA-256 `7afefeb6524d1a1b72e07a2e251f78b614960991be8ae22155ba8bb5843352fb`.

Decision: **CANDIDATE_ONLY / DRAFT_BLOCKED**. The demonstrated fence overlap
and one matched hard-contact instability are removed; free robot ramp
traversals and full perimeter physics are not yet accepted.
