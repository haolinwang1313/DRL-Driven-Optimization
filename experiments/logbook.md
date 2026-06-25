# Experiment Log

Migration note: This is a pre-migration historical log. Some entries use the old repository layout; see `docs/migration-map.md` for path mappings.

## 2026-04-06T22:32:10.7813131+08:00

### Action
- No new experiments launched in this unattended run.
- Manuscript framing was revised instead toward `surrogate-conditioned benchmark fragility` because the current worktree lacks the artifacts needed for substantive new evidence.
- Manuscript literature positioning was also expanded to acknowledge stronger baseline families already discussed in recent `Applied Energy` work, including `SAC`-style actor-critic baselines and `CMA-ES`.
- The residual old `decision-support` sentence in the methodology section was replaced with neutral `reward scalarization bias` wording.
- `elsarticle/manuscript.tex` was compiled successfully with `latexmk -pdf -interaction=nonstopmode manuscript.tex`.
- A main-text budget-accounting table was added to the NSGA-II comparison section, making query-count symmetry explicit and documenting that hardware-normalized wall-clock / memory traces are still unavailable in the synced artifacts.

### Reason
- The current worktree does not contain `artifacts/publication/` outputs needed for cross-surrogate or compute-accounting analysis.
- No confirmed local GPU / CUDA information is available.
- No confirmed SSH / remote server configuration is available beyond `server.local.example.yaml`.
- No confirmed physical-stack environment is available for EnergyPlus / Radiance / Rhino / Grasshopper validation.

### Deferred Experimental Items
- `cross_surrogate_sensitivity_analysis`
- `cmaes_or_sac_baseline`
- `main_text_budget_accounting`
- `multi_scenario_early_stopping_completion` only if evidence is still intended as a claim

## 2026-04-07T00:00:00+08:00

### Action
- Attempted to continue auto loop into remote-execution setup by locating the server credentials that the user said were already provided.

### Result
- Blocked. The current thread-visible context and terminal snapshot do not contain the actual server address, username, password, or key path.
- Only the template file `server.local.example.yaml` is present in the repo.

### Blocker Summary
- Missing actual remote host
- Missing actual remote username
- Missing actual authentication material (password or key path)
- Missing actual remote project root and results root

### Next Safe Step
- Resume auto loop only after the real remote credentials are present in the accessible thread context or written into a project-local config file.

## 2026-04-07T00:30:00+08:00

### Action
- Read project-local `server.local.yaml` and unlocked remote access.
- Created a project-local `.venv` with `uv run` and installed missing project runtime dependency `paramiko` into that environment only.
- Synced remote publication artifacts into `artifacts/publication/imported` using the existing `sync-publication-results` path and the project-local server config.
- Confirmed that the imported dataset metadata still reports `simulation_mode = fallback_analytic`.
- Ran `tools/analyze_surrogate_checkpoints.py` on two available 2000-sample surrogate checkpoints and wrote `artifacts/publication/diagnostics/checkpoint_sensitivity_analysis.json`.
- Pulled two additional `server_runs` checkpoint bundles into `artifacts/publication/diagnostics/checkpoints`.
- Built `artifacts/publication/diagnostics/benchmark_fragility_summary.json` to summarize benchmark-order changes across three checkpoint contexts.
- Integrated the resulting checkpoint-sensitivity evidence into `elsarticle/appendix.tex` and `elsarticle/manuscript.tex`.
- Recompiled `elsarticle/manuscript.tex` successfully after the appendix update.

### Result
- Remote artifact sync is now operational.
- Publication-mode validation still refuses the synced results for strict publication closure because the imported metadata remains `fallback_analytic`.
- A new diagnostic result is now available locally: two checkpoints with very similar training-fit metrics still show different out-of-bounds behavior on random samples.
- A stronger benchmark-fragility result is now available locally: the imported publication artifact favors DDPG in HV, whereas the later strict-highest-accuracy and surrogate-rebenchmark checkpoints both favor NSGA-II in both HV and IGD.

### Key Diagnostic Numbers
- Imported checkpoint EUIt MAE: `0.2038`
- Local retrained checkpoint EUIt MAE: `0.1885`
- Imported checkpoint EUIt below-min fraction: `0.0008`
- Local retrained checkpoint EUIt below-min fraction: `0.0080`
- Imported checkpoint H above-max fraction: `0.0012`
- Local retrained checkpoint H above-max fraction: `0.0040`
- Triple-better-than-bounds fraction: `0.0004` vs `0.0006`

### Remaining Deferred Experimental Items
- broader multi-checkpoint ranking study beyond the current three-context / four-checkpoint audit
- `cmaes_or_sac_baseline`
- hardware-normalized compute audit
- `multi_scenario_early_stopping_completion` only if evidence is still intended as a claim

## 2026-04-07T01:10:00+08:00

### Action
- Added a project-local `CMA-ES` optimizer path to the codebase and config.
- Executed same-budget `CMA-ES` runs on three checkpoint contexts:
  - `cmaes_local`
  - `cmaes_hp2000`
  - `cmaes_rebench`
- Verified that all three contexts saturate to the same clipped corner solution across all three scalarization scenarios and all 10 seeds per scenario.
- Ran a dedicated `CMA-ES` trajectory probe and wrote `artifacts/publication/diagnostics/cmaes_trajectory_probe.json`.
- Compared `CMA-ES` against same-budget random search on the imported publication artifact.
- Ran deterministic analytic reevaluation of representative `CMA-ES` corner candidates across the three checkpoint contexts.
- Built a representative local CPU-only hardware audit and wrote:
  - `artifacts/publication/diagnostics/hardware_audit_local_cpu.json`
  - `artifacts/publication/diagnostics/hardware_audit_local_cpu_summary.json`
- Integrated the `CMA-ES` and hardware-audit evidence into the manuscript and appendix, then recompiled successfully.

### Result
- `CMA-ES` is no longer a missing baseline.
- The `CMA-ES` corner is not an immediate one-step bug:
  - first perfect reward hit occurs after roughly 2992--3168 evaluations in a representative balanced-scenario run.
- Same-budget random search does not reach that corner on the imported publication artifact:
  - mean reward remains about 0.87 rather than 1.0.
- Deterministic analytic reevaluation of representative `CMA-ES` corner candidates shows heterogeneous drift:
  - imported publication: EG and H fall modestly below the surrogate corner
  - strict-highest-accuracy checkpoint: H falls modestly below the surrogate corner
  - surrogate-rebenchmark checkpoint: the corner degrades strongly to roughly EUIt 68.66, EG 2.56, H 7.45
- Representative local CPU-only hardware audit now exists:
  - DDPG balanced single-seed run: ~99.4 s / 436.6 MB peak RSS
  - NSGA-II fair-budget single run: ~12.5 s / 437.6 MB peak RSS
  - CMA-ES balanced single-seed run: ~0.40 s / 437.9 MB peak RSS
  - random search balanced single-seed run: ~0.34 s / 543.6 MB peak RSS

### Remaining Deferred Experimental Items
- physical-stack or non-`fallback_analytic` publication closure
- broader checkpoint coverage beyond the current accessible bundles
- explicit decision on `SAC`: run minimal check or scope it out
- transferability framework for practitioner-side surrogate self-assessment

## 2026-04-07T01:45:00+08:00

### Action
- Added a practical surrogate self-assessment framework to the discussion and conclusion.
- Scoped current policy-learning conclusions explicitly to the tested deterministic policy-gradient setup, instead of leaving `SAC` implicitly inside the paper's causal claims.
- Recompiled the manuscript successfully after these final text-side updates.

### Result
- The remaining blockers are now mostly evidence-side, not wording-side.
- `SAC` is now an optional strengthening experiment rather than a textual ambiguity.
- The paper's positive contribution is now phrased not only as a warning, but also as a concrete screening procedure for risky surrogates.

## 2026-04-07T12:45:00+08:00

### Action
- Built a 4-candidate mixed probe set spanning `DDPG`, `NSGA-II`, `CMA-ES`, and `RandomSearch`.
- Re-ran `physical-reevaluate-candidates` against that mixed set and wrote `physical_stack_candidate_probe_physprobe_methods_v2.csv`.

### Result
- The project now has a 4-method small-batch physical probe table.
- All 4 representative candidates returned `energyplus_ok = True` and `radiance_ok = True`.
- The current outputs still need metric alignment before publication use:
  - `EUIt` is based on a simple office-program + ideal-air setup;
  - `EG` is still zero because no PV generation model is attached;
  - `H` is a point-in-time Radiance proxy rather than the paper's winter sunlight-hours metric.

## 2026-04-07T11:30:00+08:00

### Action
- Confirmed that the remote project `.venv` already contains the `ladybug`, `honeybee`, and `dragonfly` package stack.
- Installed `honeybee-openstudio` into the remote project `.venv`.
- Replaced the incompatible Ubuntu 24.04 `EnergyPlus` build on the remote server with a compatible Ubuntu 22.04 build and updated the Honeybee Energy config accordingly.
- Successfully ran a minimum remote `Honeybee -> OpenStudio -> EnergyPlus` proof-of-life on a shoe-box `HBJSON`.
- Configured `honeybee-radiance` against the installed Radiance path and successfully ran a minimum remote `Honeybee -> Radiance -> octree -> rtrace` proof-of-life on a shoe-box `HBJSON`.
- Updated `tools/install_physical_stack_remote.py` so it selects a compatible Ubuntu 22.04 vs 24.04 EnergyPlus build instead of hardcoding the Ubuntu 24.04 package.

### Result
- Physical-stack proof-of-life now exists for both `EnergyPlus` and `Radiance` on the remote server.
- The remaining blocker has narrowed to pipeline integration: the repo-side `paper_repro/simulation.py` still only implements the analytic fallback path, so the project cannot yet emit non-`fallback_analytic` publication artifacts through its own code path.

## 2026-04-07T12:10:00+08:00

### Action
- Added a project-level `physical_stack` candidate reevaluation path:
  - new module `paper_repro/physical_stack.py`
  - new CLI command `physical-reevaluate-candidates`
- Implemented nearest-dataset-block projection from optimizer candidates to existing block geometry records in `simulated_blocks.jsonl`.
- Implemented remote `Honeybee -> EnergyPlus` and `Honeybee -> Radiance` probing for the matched block geometries.
- Ran the new path on the first 2 representative candidates from `top_candidate_reevaluation.csv`.

### Result
- The new path ran successfully and wrote:
  - `artifacts/publication/reevaluation/physical_stack_candidate_probe_physprobe.csv`
  - `artifacts/publication/diagnostics/physical_stack_result_a0f9293dbcd0.json`
- Both representative candidates returned `simulation_mode = physical_stack_probe`, `energyplus_ok = True`, and `radiance_ok = True`.
- Representative outputs:
  - candidate 0 -> matched sample 396, projection distance 0.3561, physical EUIt 307.378, physical radiance mean ~63.17
  - candidate 1 -> matched sample 174, projection distance 0.0, physical EUIt 204.023, physical radiance mean ~63.17
- This establishes the first repo-native non-`fallback_analytic` candidate reevaluation path, but the path is still nearest-block based and not yet aligned to final publication metrics.

## 2026-04-07T02:00:00+08:00

### Action
- Probed the remote server for physical-stack executables and existing non-`fallback_analytic` result bundles.
- Read the local physical-stack installer helper and the current simulation implementation to determine whether physical-stack installation would actually unlock a non-fallback pipeline.

### Result
- No remote `EnergyPlus` or `Radiance` executables were detected on the server in the current accessible paths.
- All visible synced `simulated_samples.meta.json` files still report `simulation_mode = fallback_analytic`.
- The current `paper_repro/simulation.py` implementation in this repo still uses the analytic fallback path; it does not yet contain a real physical-stack simulation branch that would automatically convert installation into non-fallback results.
- Therefore, the next highest-value blocker is not \"install tools blindly\" but \"build or restore a real non-fallback simulation path\" if physical-stack publication closure is required.

## 2026-04-08T10:00:00+08:00

### Action
- Continued aligning the `physical_stack` probe to the paper's `EUIt / EG / H` definitions.
- Updated the physical probe model to use paper-style window ratios, loads, ventilation, and thermostat settings.
- Added a rooftop PV generation proxy for `EG`.
- Moved `H` toward a January 20 daylight-hours proxy using sampled winter-hour Radiance runs.
- Verified the new metric path on a single representative candidate.

### Result
- Single-candidate aligned probes work and now return nonzero `EG` plus an hour-count-style `H` proxy.
- Larger aligned probes still tend to stall long enough to hit client-side timeouts before the remote result file is returned.

### Current Interpretation
- The main blocker is now execution robustness, not environment setup or path existence.
- The next highest-value change is to make `physical-reevaluate-candidates` asynchronous / resumable, with local polling of remote result files.

### Latest Async Probe
- Fresh async job `29931e73e7c3` is currently running on the remote server under the direct Python worker:
  - `/home/ac/Dogtor_Project/DDPG/.venv/bin/python /home/ac/Dogtor_Project/DDPG/artifacts/physical_stack_batches/probe_29931e73e7c3.py`
- This is now the primary probe to harvest in the next loop iteration.
- Remote process-tree inspection confirms that this worker has spawned a real `honeybee-energy -> EnergyPlus` child chain rather than idling in Python.

### Async Progress Semantics
- The newest async worker now writes partial-progress fields (`current_case_index`, `total_cases`, `completed_cases`) into `status.json`, so future loops can distinguish startup success from mid-run progress.

## 2026-04-08T11:00:00+08:00

### Action
- Implemented async / resumable execution support for `physical-reevaluate-candidates`.
- Added CLI support for:
  - `--async`
  - `--job-id`
  - `--wait-seconds`
- Added persisted local job metadata files under `artifacts/publication/diagnostics/physical_stack_job_<id>.json`.
- Submitted and polled a first async physical probe job.

### Result
- The probe system can now submit long-running remote jobs without blocking the local foreground shell.
- The remaining issue is no longer whether async control exists, but whether long-running completion semantics are robust enough for larger aligned probes.
- After fixing remote environment inheritance and a generated-script indentation error, a fresh async job now starts the remote Python worker directly rather than leaving only a shell wrapper behind.
- A later async job now also updates its remote `status.json` from `submitted` to `running`, which confirms that status propagation is working.

## 2026-04-08T13:00:00+08:00

### Harvest Check
- Polled the current lead async job `621a1455274b`.
- Result file is still not present, but the remote `status.json` remains `running` and includes per-case progress fields.
- Remote process-tree inspection confirms the worker is still attached to a real `honeybee-energy -> EnergyPlus` child chain.

### Current Decision
- Do not queue additional large physical jobs yet.
- Keep the next loop focused on harvesting the first completed async result.

## 2026-04-08T13:15:00+08:00

### Harvest Check
- Re-polled the lead async job `621a1455274b`.
- The remote `status.json` still reports `running` with `current_case_index = 1`, `total_cases = 1`, and `completed_cases = 0`.
- Remote process-tree inspection still shows the worker attached to a live `honeybee-energy -> EnergyPlus` chain.

### Current Decision
- Continue waiting for the first completed async result.
- Do not submit additional large physical probes until this first aligned result is harvested.

## 2026-04-08T13:30:00+08:00

### Harvest Check
- Re-polled the fresh lead async job `6e80a1552e92`.
- The remote `status.json` still reports `running` with `current_case_index = 1`, `total_cases = 1`, and `completed_cases = 0`.
- Remote process-tree inspection confirms the worker is still attached to a live `honeybee-energy -> EnergyPlus` chain.

### Current Decision
- Continue waiting for the first completed async result from `6e80a1552e92`.
- Do not submit additional large physical probes until this fresh aligned result is harvested.

## 2026-04-08T13:30:00+08:00

### Action
- Added stale-job detection for async physical probes: if a job still reports `running` but no live remote PID exists and no result file is present, it is now marked failed.
- Used the new logic to mark the old lead job `621a1455274b` as stale.
- Submitted a fresh replacement job `6e80a1552e92`.

### Result
- The new replacement job enters a healthy `running` state with `current_case_index = 1`, `total_cases = 1`, and `completed_cases = 0`.
- The async runner is now cleaner operationally: old stale jobs can be retired and replaced without manual ambiguity.

## 2026-04-08T22:41:20+08:00

### Action
- Hardened the async `physical_stack` remote worker in `paper_repro/physical_stack.py`.
- Added per-stage status reporting into remote `status.json`, including:
  - `stage`
  - `matched_sample_id`
  - `timeout_seconds`
  - `timestamp`
- Added remote log emission for each long-running subprocess stage so later loops can inspect whether a probe is stuck in `EnergyPlus`, result parsing, or Radiance.
- Added subprocess timeouts for cleanup, `honeybee-energy simulate`, `honeybee-energy result`, `honeybee-radiance translate`, sky generation, octree build, and point-in-time raytrace.
- Retired the anomalously slow lead job `6e80a1552e92` and confirmed it is now marked failed via the stale-job path.
- Submitted a new hardened async probe `513150aac010` and harvested its first completed result into:
  - `artifacts/publication/diagnostics/physical_stack_result_513150aac010.json`
  - `artifacts/publication/reevaluation/physical_stack_candidate_probe_asynccheck12.csv`

### Result
- The async worker no longer behaves like a black box: a fresh run now exposes exact stage-level progress in both `status.json` and the remote job log.
- The first fully harvested hardened async probe completed successfully instead of hanging indefinitely.
- The harvested candidate is:
  - method: `DDPG`
  - scenario: `Balanced_Performance`
  - matched sample id: `396`
  - projection distance: `0.3561`
- The harvested physical probe output is:
  - `physical_EUIt = 989.42`
  - `physical_EG_total_production = 1.7601`
  - `physical_H_proxy = 9.0`
  - `energyplus_ok = True`
  - `radiance_ok = True`

### Interpretation
- The main async-execution blocker has materially improved: the system can now replace stale jobs, emit stage-aware progress, and harvest a completed async physical result.
- However, the first harvested aligned physical result also shows that metric alignment remains a first-order blocker:
  - surrogate / fallback `EUIt` for this candidate remains near `66`
  - harvested physical `EUIt` is `989.42`
- Therefore the next highest-value loop should **not** expand to a larger physical batch yet.
- The next loop should instead diagnose and reduce the `EUIt / EG / H` alignment gap before treating async physical probes as publication-grade evidence.

## 2026-04-09T00:00:00+08:00

### Action
- Diagnosed the first harvested alignment failure more directly.
- Confirmed that the current `physical_stack` worker was constructing each multi-story residential building as one tall single room instead of a stack of floor-by-floor conditioned rooms.
- Patched the remote worker template in `paper_repro/physical_stack.py` so each assigned building is now generated as:
  - one `Room.from_box(...)` per floor
  - with `room.story` set per level
  - followed by `Model.solve_adjacency(intersect=False)` to recover internal floor/ceiling adjacencies
- Kept the manuscript-matching load assumptions unchanged in this pass:
  - occupant density `0.03 person/m^2`
  - lighting `5 W/m^2`
  - equipment `1.9 W/m^2`
  - ventilation `30 m^3/(h*person)`
  - heating / cooling `18 / 26 C`
- Submitted a fresh async validation probe under the new geometry semantics:
  - job id `ad82ac8175be`
  - output suffix `asynccheck13`

### Result
- The new geometry-aware job launched successfully and advanced into:
  - `stage = energyplus_simulate`
  - `matched_sample_id = 396`
- This means the next loop no longer needs to rediscover the likely root-cause hypothesis for the `EUIt` inflation; it can focus directly on harvesting and comparing the new geometry-corrected result.

### Current Interpretation
- The strongest current hypothesis is that the previous `physical_EUIt = 989.42` mismatch was caused primarily by geometric conditioning error rather than by occupancy/load assumptions alone.
- The next high-value check is whether the geometry-corrected async probe materially reduces `physical_EUIt` toward the surrogate/fallback scale.

## 2026-04-09T00:20:00+08:00

### Action
- Reworked the `physical_stack` worker again to separate the thermal and daylight models:
  - `EUIt` now uses a compressed thermal model with one representative floor-height room per building footprint plus `room.multiplier = floors`.
  - `EG / H` probing still uses a full-height fast Radiance mass model so shading and rooftop area remain tied to the original building height.
- Submitted and harvested a fresh async probe under this compressed geometry path:
  - job id `73c66e9d11e5`
  - output suffix `asynccheck14`

### Result
- The new compressed-model probe completed successfully without the long `EnergyPlus` timeout seen in the explicit per-floor geometry run.
- For the same representative candidate (`matched_sample_id = 396`), the harvested `physical_EUIt` moved from:
  - previous hardened async run: `989.42`
  - compressed thermal model run: `142.115`
- `physical_EG_total_production` remained unchanged at `1.7601`, and `physical_H_proxy` remained `9.0`.

### Interpretation
- This is a strong confirmation that geometric thermal-model semantics were a first-order cause of the earlier `EUIt` inflation.
- However, the compressed-model result is still far above the surrogate/fallback scale near `66`, so geometry alone does not close the alignment gap.
- The next likely causes are now narrower and more model-assumption-specific:
  - hidden office-style thermal assumptions still inherited through `office_program`
  - remaining envelope / schedule mismatch relative to the manuscript's residential assumptions
- Therefore the next loop should stop treating geometry as the main blocker and start diagnosing the residual residential-assumption mismatch in the compressed energy model.

## 2026-04-09T00:35:00+08:00

### Action
- Continued the residual `EUIt` alignment experiments instead of expanding the physical batch.
- Upgraded the compressed thermal model again from:
  - one representative floor per building with full-floor multiplier
  to
  - a three-tier thermal abstraction per building:
    - one `ground` thermal room
    - one `middle` thermal room with `multiplier = floors - 2`
    - one `top` thermal room
- Set thermal boundary conditions explicitly:
  - ground floor bottom remains `Ground`
  - inter-floor top / bottom surfaces become `Adiabatic`
  - top floor roof remains `Outdoors`
- Added window ratio generation back onto the energy model as well as the radiance model, so the thermal run is no longer windowless.
- Submitted a fresh async validation probe under this new thermal-boundary semantics:
  - job id `0005fad4a1bf`
  - output suffix `asynccheck15`

### Current Status
- The new `ground/middle/top` probe launched successfully and is currently in:
  - `stage = energyplus_simulate`
  - `matched_sample_id = 396`
- It has not completed within the short polling window yet, so no new `EUIt` number is available in this round.

### Interpretation
- The experiment stack has now moved beyond coarse geometry repair into boundary-condition correction.
- The next loop should harvest `0005fad4a1bf` before changing additional thermal assumptions, because this run directly tests whether the residual `EUIt` gap is driven by roof / floor exposure semantics.

## 2026-04-09T01:00:00+08:00

### Action
- Continued the `EUIt` alignment ablation on the same representative candidate (`matched_sample_id = 396`) instead of expanding physical probe coverage.
- Tested two more physically meaningful model adjustments:
  1. simplified energy-model windows to one coarse aperture per facade while keeping the radiance model unchanged;
  2. replaced `Always On` internal-load schedules with a residential-like constant-fraction sensitivity schedule and replaced year-round `18/26 C` thermostat control with a season-aware heating/cooling schedule.
- Harvested the resulting async probes:
  - `fcfa1af72c6d` -> simplified windows
  - `b15b7dde7afd` -> residential-like load schedules
  - `7a6aa8069931` -> residential-like load schedules + season-aware thermostat schedule

### Result
- Window simplification does **not** materially change `EUIt`:
  - previous compressed thermal model: `142.115`
  - simplified-window run: `144.397`
- By contrast, load/schedule assumptions are highly influential:
  - residential-like schedules reduce `physical_EUIt` to `89.2`
  - adding season-aware thermostat control reduces it further to `74.379`
- The latest result is now much closer to the surrogate/fallback scale near `66`, while `EG` and the current `H` proxy remain unchanged in this candidate-level test.

### Interpretation
- The main residual `EUIt` mismatch is **not** driven by fenestration discretization.
- The dominant remaining alignment lever is the operational / schedule side of the physical model.
- The physical stack path has now crossed from “grossly misaligned” into “same-order-of-magnitude and narrowing”:
  - `989.42` -> `142.115` -> `89.2` -> `74.379`
- The next loop should focus on final calibration rather than broad architectural rewrites:
  - either tighten the residential load/schedule assumptions further,
  - or hold this as the current best-aligned physical probe and start testing a second representative candidate before overfitting to one case.

## 2026-04-09T10:50:00+08:00

### Action
- Extended the current best-aligned physical probe path from one representative candidate to two:
  - `DDPG / Balanced_Performance` representative candidate (`sample_396`)
  - `NSGA-II` representative candidate (`sample_174`)
- Confirmed that the current best-aligned thermal settings transfer reasonably across both candidates:
  - first candidate physical `EUIt = 74.379`
  - second candidate physical `EUIt = 74.057`
- Continued `H` proxy diagnosis with two additional low-cost adjustments:
  1. changed the daylight aggregation from “hourly mean over all points” to “average sunlight-hours over individual points” across the full `08:00-16:00` January 20 window;
  2. restricted / subsampled the south-facing test points to a small representative ground-floor set rather than using every available aperture center.

### Result
- The revised `H` aggregation is no longer binary in the strictest sense:
  - first candidate moved from `9.0` to `1.0588`
  - second candidate moved from `0.0` to `0.0142`
- However, the newer `H` proxy is still far below the surrogate / fallback values near `7.5`.
- Subsampling the ground-floor test points did **not** materially change the second candidate result:
  - before subsampling: `0.0142405`
  - after subsampling: `0.0142405`

### Interpretation
- `EUIt` is now the strongest-aligned physical quantity among the three paper targets.
- `EG` is at least directionally aligned and stable under the current simplified PV proxy.
- `H` remains the least trustworthy part of the physical probe path.
- The latest diagnostics strongly suggest that the main `H` blocker is no longer thresholding or sensor-count dilution, but the deeper fact that the current Radiance probe is still not a faithful implementation of the paper's original “ground-floor windowsill sunlight-hours” metric.

## 2026-04-09T10:55:00+08:00

### Action
- Continued physical alignment with a two-candidate validation mindset rather than a single-candidate overfit loop.
- Verified the current best-aligned thermal path on the second representative candidate (`NSGA-II`, matched sample `174`).
- Isolated the `H` pipeline as a separate problem and ran targeted diagnostics:
  - expanded the January 20 window to `08:00--16:00`;
  - changed `H` aggregation to average per-point sunlight hours instead of a block-level binary hour counter;
  - fixed a code-path bug where south-facing point subsampling was silently bypassed;
  - retried a ground-floor-only / lowest-row fallback when no valid points were found after filtering.

### Result
- The best-aligned thermal settings now transfer across two representative candidates:
  - `DDPG / Balanced`: surrogate `EUIt = 66.0`, physical `EUIt = 74.379`
  - `NSGA-II`: surrogate `EUIt = 69.44`, physical `EUIt = 74.057`
- This is the first point in the loop where the physical `EUIt` values are both:
  - in the same order of magnitude as the surrogate/fallback values; and
  - stable across more than one representative method candidate.
- The `H` probe is improved but still compressed:
  - `DDPG / Balanced`: surrogate `H = 7.85`, physical `H_proxy = 1.0588`
  - `NSGA-II`: surrogate `H = 7.57`, physical `H_proxy = 0.72`
- The latest `H` fixes did recover non-null values for both representative candidates, but they did **not** lift the proxy into the same numerical regime as the surrogate target.

### Interpretation
- The project has now crossed an important threshold for the physical probe story:
  - `EUIt` can be defended as a partially aligned physical quantity for small-batch candidate checks.
  - `EG` remains a simplified but stable rooftop-PV proxy.
  - `H` still behaves like a structurally compressed auxiliary proxy, not a publication-grade reproduction of the manuscript's sunlight-hours metric.

## 2026-04-09T12:20:00+08:00

### Action
- Continued the `H`-specific reconstruction rather than broadening the physical candidate batch.
- Diagnosed whether the poor `H` proxy was mainly caused by:
  - point aggregation,
  - point-count dilution,
  - or the physical placement of the Radiance sensor points.
- Confirmed that simple threshold sweeps (`1000`, `500`, `300`, `100` lux) did **not** materially change the current `H` values, so the problem is not just the illuminance cutoff.
- Replaced the south-facing sensor construction with a more paper-like geometry:
  - sample points are now generated as windowsill-style points directly on the ground-floor south-facing wall faces,
  - rather than derived from aperture centers across all available apertures.
- Re-ran both representative candidates under this windowsill-style `H` definition:
  - `DDPG / Balanced` -> `artifacts/publication/reevaluation/physical_stack_candidate_probe_asynccheck28.csv`
  - `NSGA-II` -> `artifacts/publication/reevaluation/physical_stack_candidate_probe_asynccheck27.csv`

### Result
- The two representative candidates now produce the following windowsill-style `H` proxies:
  - `DDPG / Balanced`: `physical_H_proxy = 0.3673`
  - `NSGA-II`: `physical_H_proxy = 0.3673`
- This is a notable qualitative change from the earlier inconsistent pair:
  - `DDPG / Balanced`: `1.0588`
  - `NSGA-II`: `0.72`
- `EUIt` and `EG` stay stable under this `H`-only reconstruction:
  - `DDPG / Balanced`: `physical_EUIt = 74.379`
  - `NSGA-II`: `physical_EUIt = 74.057`

### Interpretation
- The current best `H` proxy is now more internally consistent across representative candidates, but it is also clearly **much lower** than the surrogate/fallback `H` values near `7.5`.
- That makes the current physical `H` evidence more conservative and more believable than the earlier aperture-center proxy, but it also confirms that the present Radiance reconstruction is still not a publication-grade replica of the manuscript's original sunlight-hours metric.
- The practical implication for the next loop is:
  - keep `EUIt` as the strongest physical-alignment result,
  - keep `EG` as a simplified but stable proxy,
  - treat `H` explicitly as a still-compressed structural check rather than as a final aligned target.
- Therefore the next loop should split the physical story explicitly:
  - use `EUIt` (and cautiously `EG`) as the current strongest non-`fallback_analytic` evidence path;
  - treat `H` as still-under-reconstruction rather than pretending all three targets are equally aligned.

## 2026-04-09T14:20:00+08:00

### Action
- Pushed the `H` reconstruction one step closer to the original manuscript definition.
- Prototyped a Jan~20 direct-sun-hours workflow using:
  - `honeybee-radiance sunpath epw`
  - manual `oconv` assembly with scene + aperture + hourly suns
  - `honeybee-radiance dc scontrib`
- Verified that this direct-sun-hours route is technically viable on the current windowsill-style ground-floor sensor points.
- Integrated the direct-sun-hours route into `paper_repro/physical_stack.py`, with fallback to the older point-in-time proxy only if the new path fails.
- Re-ran both representative candidates under the new direct-sun-hours implementation:
  - `NSGA-II` representative -> `asynccheck29`
  - `DDPG / Balanced` representative -> `asynccheck30`

### Result
- The direct-sun-hours path materially improves `H` alignment:
  - `DDPG / Balanced`: surrogate `H = 7.850`, physical `H_proxy = 7.525`
  - `NSGA-II`: surrogate `H = 7.569`, physical `H_proxy = 6.075`
- `EUIt` stays stable under the new `H` definition:
  - `DDPG / Balanced`: `physical_EUIt = 74.379`
  - `NSGA-II`: `physical_EUIt = 74.057`
- This is the first `H` reconstruction in the project that is both:
  - based on a direct-sun-hours style Radiance workflow; and
  - numerically close enough to the surrogate target range to support bounded discussion in the manuscript.

### Interpretation
- The current physical evidence hierarchy is now stronger than in previous loops:
  - `EUIt`: strongest-aligned quantity
  - `EG`: simplified but stable rooftop-PV proxy
  - `H`: meaningfully reconstructed and now in-range, though still supported only by a small-batch direct-sun-hours check rather than by a full publication-pipeline replacement
