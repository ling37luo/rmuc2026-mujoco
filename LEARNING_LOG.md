# Local validation log

## 2026-09-23 — separate uphill, downhill and roundtrip results

- **User request:** score uphill and downhill separately; a roundtrip requires
  both records. Codex implemented this in the standalone repository.
- Default slope sessions now request uphill only. Downhill starts at the high
  endpoint; roundtrip records uphill followed by downhill without a reset.
  Each leg has its own timestamps, duration, contact steps, tilt, penetration
  and failure. A failed return leg retains the completed uphill record.
- Reaching or passing the endpoint footprint with contact now counts without
  the old 0.25 s dwell. The 12 cm endpoint tolerance is unchanged. The scenario
  descriptor and run report identify completion rule version 2. Field geometry,
  materials, contact settings and the source pack are unchanged.
- **AGENT_PASS (saved user-run audit):** reclassified the nine episodes in
  `runs/slope_interaction_20260923T021607.342374Z.json` using its 50 Hz trajectory.
  All nine have uphill success; none returned to the low endpoint. The new
  results correctly show downhill and roundtrip incomplete, including the
  five fast passes whose uphill arrival the old dwell rule missed. This is
  an event reclassification, not a new physics run or per-leg penetration audit.
- **AGENT_PASS (new external SCUT simulation, one route):** on
  `slope_6_10_01`, three independent policy runs completed: uphill in 6.652 s,
  downhill in 4.065 s, and roundtrip in 12.272 s (uphill 6.652 s plus return
  5.620 s including the controller's pause at the top). Simulations continued
  for 10/10/16 s, respectively, with finite states and zero numerical warnings.
  Reset reproduced the initial pose, cleared both scores and preserved the
  saved result. The local-only SCUT adapter and robot remain in ignored storage.
- Evidence: `runs/slope_scoring_acceptance_20260923T023326/`, including the
  original-log hash, reclassified events, new telemetry and `summary.json`.
- **AGENT_PASS (software):** 279 tests passed; Ruff and diff checks passed.
  Coverage includes fast passes, independent directions, return-leg failure,
  contact on arrival, immutable recorded results and separation across episodes.
- **UNVERIFIED:** user confirmation of the updated interactive feedback, other
  SCUT slope routes/speeds and Isaac execution. `DRAFT_BLOCKED` is unchanged.

Minimal public reproduction with the example rover (repeat for each direction):

```bash
rmuc2026-field view ./local-rmuc2026-field --scenario slope_basic \
  --profile collision_only --control policy --headless --steps 8000 \
  --direction uphill
```

Use `--direction downhill` for descent only or `--direction roundtrip` for both.
Next user check: drive one uphill pass and inspect `leg_results.uphill`; in a
roundtrip session return down before resetting to earn both records.

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
