# Fly-ramp training pack integration status (2026-09-23)

Scope: the two source-bound local fly-ramp regions, neutral contact probes and
parallel simulation interfaces. No robot policy was trained or evaluated here.

The verified source runtime pack has manifest SHA-256
`ade3e578caf1fd217330b8ff0f6717cd0bd6b1f14fec0a081e67d9eefe1e02be`.
The north/south v4 regions have manifest SHA-256
`ea9e4f9b9c64ffbab0a6538be3efb2562b60d950fe9d29203020ca7460b6367e`
and `1ec9bd0511c60bb84187fdeefcd550e65491aec0d3301a39c57c075c1956132b`.
Both use exact `213×614` slices of the source 1 cm collision heightfield;
their manifest, file hashes, grid axes, route and source slice were verified.

| Check | Bounded result |
| --- | --- |
| MuJoCo source/crop contact | A neutral sphere at each approach spawn ran 500 steps at 2 ms in the full and cropped field. Both pairs had 399 contact steps, no numerical warning, and final position differences at most `1.2e-11 m`. |
| MuJoCo fly-zone contact sweep | On each ramp, 301 centerline poses, 91 lateral poses and 193 high gap poses were compared in the full field and crop. Contact pairs and counts matched at every static pose. Eight low-speed rolling segments across both ramps reached their endpoints with identical per-step contact presence and under `4 µm` position divergence. The north uphill segment had one step with two versus one contact point just after the lip, while both scenes retained support. Three vertical drop sites per ramp matched first-contact and sustained-contact steps. No nonfinite state or solver warning occurred. |
| MuJoCo parallel interface | The same 16 public example-rover cases on each ramp were run with 1, 4 and 16 independent workers. All 16 cases per ramp gave the same physics outcomes across worker counts, with zero nonfinite states or solver warnings. At 16 workers the north/south rates were about 8.7k/8.3k physics steps/s and worker RSS was about 75 MiB. The rover did not pass the jump task; its outcome is not a field or policy acceptance result. |
| Isaac data handoff | Float64 X/Y/height exports round-tripped exactly against both verified regions. Schema-3 exports transfer the source-bound perimeter box inside each crop and the heightfield's contact bits, friction and `solref`. North/south exports passed source re-read; their 1/4/16-environment offline preflights passed box-separation checks. |
| Isaac Sim 6.1 PhysX | Both fly-ramp crops passed at 1, 4 and 16 independent environments on the local RTX 5070. Each case ran 500 steps at 1 ms; all 5/20/80 expected sphere contacts and all terrain/fence rays identified their intended colliders. Maximum terrain ray error was `0.0403 mm`, with zero matched PhysX/solver warnings. At 16 environments, north/south reached about 3.89k/3.84k environment-steps/s; process RSS was about 2.74 GB. The six-case local report is `runs/rmuc2026-training-regions/20260923T_fly_local_candidate_v4/physx_six_case_verified_20260923/summary.json` under RL-Lab. |
| Isaac Lab kit-less Newton XPBD | Neutral 120 mm spheres at approach, ramp middle and landing were supported in 1, 4 and 16 independent worlds on both ramps. The largest support-height error was `0.157 mm`; 16-world rates were about 15.9k/17.0k world-steps/s. At 4/16 worlds Newton warned that static heightfield-to-heightfield collision pairs are skipped. Sphere-ground contacts passed. |

At the lateral perimeter, a neutral 120 mm sphere at `z=1.0 m` contacted the
same fence in the full and cropped MuJoCo scenes: north `perimeter_top`, south
`perimeter_bottom`, each with `13.4345 mm` penetration at the probe pose.
This exposed the earlier Isaac handoff's omission of the fence; schema 3 now
records that collision geometry and the source heightfield contact parameters.
The original region manifests and hashes have not changed.

**Decision:** the fly-ramp regions pass the bounded data, MuJoCo/Newton contact,
and Isaac Sim 6.1 PhysX 1/4/16-environment checks above. The PhysX probe checks
the actual collider in raw contact records and binds an explicit
sliding-friction proxy for terrain, fence and test spheres. MuJoCo
torsional/rolling friction, `solref` and contact-bit semantics are not
translated, and friction-response parity was not measured. The runtime probe
and commands are in
[`adapters/isaac/README.md`](adapters/isaac/README.md). Whole-field topology
and the artificial edge of each crop remain `DRAFT_BLOCKED`/trainer reset
boundaries respectively; these results do not certify a robot's flying policy.

The detailed local JSON reports are intentionally outside this code-only
repository under `runs/rmuc2026-training-regions/20260923T_fly_local_candidate_v4/`
and `runs/scut-fly-field-benchmark/20260923T_scut_rough_dash_v1/training_audit/`.
The new schema-3 Isaac exports, `isaac_schema3_{north,south}_{1,4,16}env_offline_preflight.json`,
and `contact_parity_replay_20260923.json` are under the fly-region run
directory. `contact_parity_second_audit_20260923.py` reproduces the MuJoCo
contact comparison from that directory.
