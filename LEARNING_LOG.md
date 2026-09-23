# Local validation log

## 2026-09-23 — automatic slope matrix and parallel evaluation

- **User request:** run slope testing automatically for simulation/training
  consumers instead of relying on manual driving. Codex implemented
  `run --scenario slope_basic`, `run_slope_batch` and `run_slope_episode`.
- The matrix covers selected routes, uphill/downhill/roundtrip, speeds and
  repeats. Each case resets independently and ends at success, failure,
  controller error or timeout, then proceeds to the next case. Summary-only
  recording is the default; full trajectories are optional. Spawn workers
  each reuse one composed model/data/controller. Case seeds do not depend on
  worker assignment. No field geometry or contact setting was changed.
- **AGENT_PASS (external SCUT simulation):** first passed a single-worker,
  three-direction smoke at 0.3 m/s on `slope_6_10_01`. Then completed the full
  3 routes x 3 directions x 2 speeds (0.3/0.5 m/s) x 2 repeats matrix with
  two workers: **36/36 PASS**, zero warnings/nonfinite states/timeouts/errors.
  Maximum tilt was 5.342 degrees and maximum penetration 1.992 mm. Approximately
  3,761 physics steps/s including worker setup and trajectory recording;
  worker peak resident memory was about 779/780 MiB. These are local workload
  measurements, not a hardware-independent scaling benchmark.
- **AGENT_PASS (reset/worker parity):** all 18 repeat pairs matched exactly on
  initial pose, completion steps, per-leg results, warnings, tilt and penetration.
  The three matching single-worker and two-worker cases also matched on those
  fields. This deterministic controller does not randomize its behavior by seed;
  the repeats demonstrate reset consistency, not randomized robustness.
- **AGENT_PASS (public example rover):** all three routes x three directions x
  two speeds passed **18/18** with one worker, zero warnings/nonfinite states or
  timeouts. This verifies the standalone entry point without a private robot.
- Evidence in ignored storage: `runs/slope_auto_scut_smoke_20260923.json`,
  `runs/slope_auto_scut_matrix_20260923.json`, 36 trajectories in
  `runs/slope_auto_scut_matrix_20260923_trajectories/`, parity report
  `runs/slope_auto_scut_parity_20260923.json`, and
  `runs/slope_auto_rover_matrix_20260923.json`. SCUT assets/controller remain
  outside distributable code; `runs/local_scut_slope/batch.sh` is a local entry.
- **AGENT_PASS (software):** 291 tests passed, including matrix enumeration,
  failure/timeout/error boundaries, reuse/reset and independent downhill spawn,
  CLI report export and the existing suite. Ruff and diff checks passed.
- **UNVERIFIED:** learned-policy training/convergence, randomized robustness,
  broader speeds/robots and actual Isaac execution. This implements automatic
  evaluation and reusable episode stepping; it does not add a reward or RL
  optimizer. The runtime pack remains `DRAFT_BLOCKED`.

Public reproduction (original example rover, no external controller required):

```bash
rmuc2026-field run ./local-rmuc2026-field --scenario slope_basic \
  --workers 2 --speeds 0.3 0.5 --repeats 2
```

Next consumer task: use these automatic case results as regression feedback
when changing an external policy. Training randomization and rewards belong
in that consumer; the field and episode scoring stay shared.

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
