# Local validation log

## 2026-09-23 — executable ordinary-slope interaction

- **USER_PASS (catalog export only):** the user supplied a successful
  `READY_HEIGHTFIELD_SCREENED` catalog output. That output did not demonstrate
  a robot traversing a slope.
- **AGENT_PASS (implementation and validation):** Codex implemented continuous
  route screening, deterministic terrain-contact reset, the shared slope
  session, native viewer controls/markers, and telemetry. The local catalog
  contains nine plane candidates; three have continuous approach/exit routes.
  Six fail the required low or high endpoint footprint check.
- **AGENT_PASS (example rover):** three routes at 0.3 and 0.5 m/s completed
  uphill/crest/reverse-downhill traversals in 16 s each. An additional 10 s
  downhill run completed. All seven runs had finite states and zero numerical
  warnings. These are wheel-speed-controller probes, not learned policies.
- **AGENT_PASS (external robot, one route):** a local-only SCUT `rough_dash`
  consumer completed a 16 s round trip on `slope_6_10_01` at 0.3 m/s. Maximum
  tilt was 3.769 degrees and maximum contact penetration was 1.992 mm, with
  zero numerical warnings. The adapter changed the local robot copy's contact
  affinity to see the existing field bits; field geometry and contact settings
  were unchanged. No terrain assist was enabled. Robot assets, policy and this
  team-specific adapter remain in ignored local storage.
- **AGENT_PASS (GUI):** the native viewer opened and closed normally in a
  bounded run with the external robot. With no driving input, traversal was
  correctly `INCOMPLETE`, while physics remained healthy.
- **AGENT_PASS (software):** 269 tests and Ruff passed. Headless and interactive
  execution share `SlopeSession`. Training reset and Isaac descriptor parity
  were checked against the same source pack.
- **UNVERIFIED:** user-operated keyboard traversal, SCUT performance on all
  three routes/speeds, actual Isaac execution, full-field topology and robot
  clearance beyond the sampled heightfield. `DRAFT_BLOCKED` is unchanged.

Source manifest SHA-256:
`ade3e578caf1fd217330b8ff0f6717cd0bd6b1f14fec0a081e67d9eefe1e02be`.

Small reproduction (after creating a local runtime pack):

```bash
rmuc2026-field view ./local-rmuc2026-field --scenario slope_basic \
  --profile collision_only --control policy --headless --steps 8000
```

Next user check: open `view --scenario slope_basic --control human`, drive up
the marked route and reverse down, then inspect the saved traversal report.
