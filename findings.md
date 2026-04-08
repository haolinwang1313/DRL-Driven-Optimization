# Findings

## Current Verdict

- Current stage: `review loop`
- Reviewer backend used: `claude-review` MCP bridge
- Concrete backend returned by reviewer: `kimi-k2.5`
- Latest score: `8.0/10`
- Latest verdict: `almost`
- Updated reviewer score after checkpoint/CMA-ES/hardware-audit round: `8.0/10`
- Main manuscript framing has now been tightened toward `surrogate-conditioned benchmark fragility`
- Main manuscript now compiles successfully after the framing rewrite and baseline-positioning update
- Remote execution is now unlocked through project-local `server.local.yaml`
- Imported publication artifacts are now available locally under `artifacts/publication/imported`
- A project-level `physical_stack` candidate probe path now exists and has produced first non-`fallback_analytic` results on representative candidates

## Blocking Findings

1. The paper still lacks a diagnostic core.
Current cautionary framing is descriptive; reviewer now requires quantified cross-surrogate sensitivity or mechanistic explanation of why DDPG fails differently from NSGA-II.
Current progress: a two-checkpoint fit/extrapolation audit has now been completed and written to `artifacts/publication/diagnostics/checkpoint_sensitivity_analysis.json`. It shows similar training-fit metrics but different extrapolative bound-violation rates. In addition, a three-context benchmark summary in `artifacts/publication/diagnostics/benchmark_fragility_summary.json` now shows that the imported publication artifact favors DDPG in HV, whereas the later strict-highest-accuracy and surrogate-rebenchmark checkpoints both favor NSGA-II in both HV and IGD. Both pieces of evidence are now reflected in the appendix.

2. External validity is still too weak for `Applied Energy`.
Single-case, single-selected-surrogate, fallback-analytic evidence is still viewed as near publication-blocking unless paired with stronger surrogate diagnostics or sensitivity analysis.
Current progress: stronger surrogate diagnostics are now in place, representative `CMA-ES` corner candidates have also been checked through deterministic analytic reevaluation, and remote proof-of-life now exists for both `EnergyPlus` and `Radiance`. Even so, the synced remote artifacts still report `simulation_mode = fallback_analytic`, so physical-stack closure is still unavailable at the pipeline level.
Additional progress: the project can now map representative candidates to nearest known block geometries and run a remote `physical_stack_probe`, returning physical `EUIt` plus Radiance sensor outputs for small-batch candidate checks.

3. The optimizer benchmark set remains incomplete.
Random search plus NSGA-II is not considered enough. Reviewer explicitly prefers `CMA-ES`; a weaker fallback is literature-based positioning against recent Applied Energy optimizer results.
Current progress: manuscript now includes literature-based positioning for `SAC`, related actor-critic baselines, and `CMA-ES`; a same-budget `CMA-ES` empirical baseline has now also been executed across three checkpoint contexts, and its behavior is discussed in Appendix B. The manuscript now scopes the current policy-learning conclusion explicitly to the tested deterministic policy-gradient setup, so `SAC` is no longer an implicit unresolved claim.

4. Main-text compute accounting is missing.
Reviewer wants wall-clock time, surrogate-training cost, query count, and memory footprint in the main paper, not only fairness language in appendix text.
Current progress: a main-text budget-accounting table has now been added, and a representative local CPU-only hardware audit is now available. Representative local wall-clock / peak RSS values are roughly DDPG 99.4 s / 436.6 MB, NSGA-II 12.5 s / 437.6 MB, CMA-ES 0.40 s / 437.9 MB, and random search 0.34 s / 543.6 MB. This substantially closes the reviewer complaint, although it remains a local reference rather than a heterogeneous production-hardware audit.

5. Early stopping should be demoted unless fully evidenced.
Balanced-only or partial early-stopping evidence should not remain as a real remediation claim.

6. Preference language still needs auditing.
Any wording that implies successful preference optimization beyond "preference-biased but overlapping regions" remains risky.

7. Strict publication validation is still blocked by simulation mode.
The synced remote artifacts are accessible, but `configs/revision.yaml` still rejects them for publication closure because the imported metadata remains `fallback_analytic`.
Current progress: environment-level blockers have been reduced materially. Remote proof-of-life now exists for both `EnergyPlus` and `Radiance`, and the Ubuntu-version mismatch in `EnergyPlus` has been corrected on the server. The blocker is now primarily a code-path gap: `paper_repro/simulation.py` still only implements the analytic fallback path, so the project cannot yet produce non-`fallback_analytic` artifacts through its own pipeline.
Additional progress: an initial project-level physical candidate probe path has now been integrated. The remaining gap is no longer path existence, but metric alignment and promotion from nearest-block probe outputs to publication-grade `EUIt` / `EG` / `H` reevaluation.

8. The new positive contribution now needs actionable framing.
Reviewer now sees the paper as close to ready, but still expects the positive contribution to be phrased as actionable surrogate self-assessment rather than only as a warning.

## Recommended Next Actions

1. Add a quantified cross-surrogate / checkpoint sensitivity study across at least 2-3 checkpoints if existing artifacts can support it.
Status: substantially advanced. A two-checkpoint fit/extrapolation audit and a three-context benchmark-order summary are now integrated into the manuscript. The remaining gap is broader checkpoint coverage beyond the currently accessible bundles.
2. Add a stronger baseline.
Preferred: `CMA-ES`.
Team-priority alternative: `SAC` if the next pass deliberately stays within RL-vs-RL comparison, but this is weaker than `CMA-ES` for the current reviewer objection.
Fallback if compute is unavailable: literature-based positioning against at least 3 recent Applied Energy comparator papers.
Status: `CMA-ES` empirical closure is now in place. The manuscript now scopes current policy-learning conclusions to the tested deterministic policy-gradient setup, so a minimal `SAC` run is optional strengthening rather than a hard textual blocker.
3. Add a main-text budget-accounting table.
Status: completed in partial form. Query-count accounting is now in the main text; hardware-normalized compute audit is still missing from available artifacts.
4. Demote early stopping to future work unless full evidence already exists in artifacts not present in this worktree.
5. Run a final wording audit on abstract, introduction, discussion, and conclusions after the next evidence-bearing revision.
6. Add an explicit transferability / surrogate self-assessment framework to the discussion.
Status: completed.

7. Align the new `physical_stack` probe outputs with the paper's `EUIt`, `EG`, and `H` definitions.
Status: in progress. The project now has a 4-method small-batch physical probe table, but the current outputs are still a residential-agnostic `EUIt`, zero `EG` generation, and a point-in-time `H` proxy.

9. Make the aligned physical probe path asynchronous/resumable.
Status: now high priority. The aligned single-candidate probe works, but larger synchronous runs still time out before the remote result file is returned.

10. Stabilize async physical probe completion semantics.
Status: in progress. Async submission and later polling now work, but long-running completion still needs one more hardening pass before larger aligned probes can rely on it.

Update: remote worker launch and `submitted -> running` status transitions are now both confirmed. The remaining async gap is result harvesting and practical multi-candidate expansion.

Additional update: the latest async worker now emits partial-progress fields in `status.json`, so future loops can tell whether a job is merely alive or actually advancing through cases.
Additional update: the latest async worker has been confirmed to spawn a real `honeybee-energy -> EnergyPlus` child chain, so the current job is compute-bound rather than hung at startup.

11. Confirm remote worker launch semantics.
Status: completed. A fresh async job now launches the remote Python worker directly; the remaining async gap is result/status harvesting for long jobs.

12. Harvest the first completed async physical result before expanding batch size.
Status: active. The current lead job is still `running` and compute-bound; the immediate next step is result collection, not additional submission.

Update: repeated polling still shows the lead job in a healthy `running` state with a live `honeybee-energy -> EnergyPlus` child chain. The next loop should remain focused on harvesting rather than expansion.

Fresh lead-job update: replacement job `6e80a1552e92` is also still in a healthy `running` state with a live `honeybee-energy -> EnergyPlus` child chain. The next loop should continue harvesting this job rather than expand batch size.

Additional update: stale-job detection is now implemented. The old lead job `621a1455274b` was retired as stale, and a fresh replacement job `6e80a1552e92` is now the clean lead job to harvest.
