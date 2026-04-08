# Experiment Log

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
