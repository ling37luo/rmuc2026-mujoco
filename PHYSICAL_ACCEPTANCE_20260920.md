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
