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
| MuJoCo parallel interface | The same 16 public example-rover cases on each ramp were run with 1, 4 and 16 independent workers. All 16 cases per ramp gave the same physics outcomes across worker counts, with zero nonfinite states or solver warnings. At 16 workers the north/south rates were about 8.7k/8.3k physics steps/s and worker RSS was about 75 MiB. The rover did not pass the jump task; its outcome is not a field or policy acceptance result. |
| Isaac data handoff | Float64 X/Y/height exports round-tripped exactly against both verified regions. The offline PhysX mesh preflight accepted 16 spatial offsets and retained source/profile identities, but created no PhysX collider. |
| Isaac Lab kit-less Newton XPBD | Neutral 120 mm spheres at approach, ramp middle and landing were supported in 1, 4 and 16 independent worlds on both ramps. The largest support-height error was `0.157 mm`; 16-world rates were about 15.9k/17.0k world-steps/s. At 4/16 worlds Newton warned that static heightfield-to-heightfield collision pairs are skipped. Sphere-ground contacts passed. |

**Decision:** the fly-ramp regions are usable as verified data inputs and as
bounded MuJoCo/Newton generic-contact training substrates. Their source
identity, contact parity and parallel loading are demonstrated at the scopes
above. Actual Isaac Sim/PhysX collision cooking, scene rays, sphere contacts,
friction and 1/4/16-environment throughput remain **unverified** because that
runtime is unavailable on this host. The runtime probe and commands are in
[`adapters/isaac/README.md`](adapters/isaac/README.md). Whole-field topology
and the artificial edge of each crop remain `DRAFT_BLOCKED`/trainer reset
boundaries respectively; these results do not certify a robot's flying policy.

The detailed local JSON reports are intentionally outside this code-only
repository under `runs/rmuc2026-training-regions/20260923T_fly_local_candidate_v4/`
and `runs/scut-fly-field-benchmark/20260923T_scut_rough_dash_v1/training_audit/`.
