# Paper Improvement Log

Migration note: This is a pre-migration historical log. Some entries use the old repository layout; see `docs/migration-map.md` for path mappings.

## Score Progression

| Round | Score | Verdict | Key Changes |
|-------|-------|---------|-------------|
| Round 1 | 4.5/10 | No | Synchronized latest revision results; corrected stale HV/IGD, seed-count, episode-count, and re-evaluation statements; tightened claims around surrogate-only evidence |
| Round 2 | 5.0/10 | Almost | Added surrogate-fidelity audit table, budget-accounting table, objective-space distance table, and stronger claim boundaries on preference separation and benchmark fairness |

## Round 1 Review

Review file: `artifacts/publication/reports/reviews/round1_external_review.md`

### Main Issues Raised
1. No physical-stack or empirical validation, so the surrogate-only comparison remains circular for an Applied Energy revision.
2. A 500-sample dataset in 12 dimensions remains sparse for optimization-sensitive surrogate modeling.
3. Benchmark fairness between pooled DDPG archives and the imported NSGA-II archive was not clearly accounted for.
4. Tight clustering of the three DDPG scenarios weakened the preference-articulation narrative.
5. The selected DDPG candidate showed larger re-evaluation error than the selected NSGA-II candidate.

### Fixes Implemented
1. Rewrote the abstract, results, discussion, and conclusion to align all claims with the synced imported evidence and explicitly acknowledge the larger DDPG re-evaluation error.
2. Corrected stale technical details in the main paper and supplement, including DDPG episodes, imported-result interpretation, and outdated benchmark statements.
3. Updated the supplement's convergence and re-evaluation sections to match the current imported results rather than the earlier short-trace/stale-table narrative.

## Round 2 Review

Review file: `artifacts/publication/reports/reviews/round2_external_review.md`

### Main Issues Raised
1. Aggregate archive-budget fairness is still not closed, because each DDPG scenario pools 20 seeds while the imported NSGA-II archive currently reflects one 24,000-evaluation run.
2. The surrogate-fidelity audit is helpful, but still does not replace independent physical validation.
3. No physical-stack or empirical validation is yet available.
4. Objective-space distances below 0.02 mean the tested reward weights do not produce strong preference separation.

### Fixes Implemented
1. Added a supplementary surrogate-fidelity audit table with RMSE, MAE, nRMSE, MAPE, and $R^2$ for EUIt, EG, and H.
2. Added a supplementary budget-accounting and objective-space separation table documenting per-seed budgets, pooled archive asymmetry, and normalized scenario distances.
3. Tightened the manuscript framing again so the method is presented as a surrogate-region identification and morphology-conditioning tool, not as evidence of strong preference-articulated Pareto exploration.
4. Added an explicit interpolation-region caveat tied to the current 500-sample, 12-dimensional surrogate dataset.

## Current Status

- Main manuscript compiles successfully: `elsarticle/manuscript.pdf`
- Supplement compiles successfully: `elsarticle/supplement.pdf`
- Remaining log warnings:
  - `elsarticle/manuscript.log`: one residual `Overfull \\hbox (15.0pt too wide)` near the front matter / abbreviations area
  - `elsarticle/supplement.log`: only minor `Underfull \\hbox` warnings in the budget table note

## Post-Review Follow-Up

After the second external review, the automation loop was extended to attack the biggest remaining blocker: archive-budget fairness.

### Additional execution performed
1. Fixed a checkpoint-compatibility issue in `paper_repro/surrogate.py` so locally retrained surrogate bundles can be loaded reliably.
2. Fixed a `fair_budget` return-shape bug in `paper_repro/pipeline.py` that previously prevented `run-optimizers --nsga2-only` from completing.
3. Retrained a local surrogate from the synced 500-sample dataset using `configs/revision.local.yaml`.
4. Successfully ran 20 `fair_budget` NSGA-II seeds, each with 24,000 evaluations.

### What this follow-up revealed
1. The fairness experiment path is now executable end-to-end.
2. On the locally retrained surrogate, the resulting NSGA-II archive collapsed to a degenerate saturated solution family:
   - 2,000 rows across 20 seeds
   - 2,000 unique action vectors
   - only 1 unique objective tuple after surrogate clipping: EUIt = 66.0, EG = 2.85, H = 7.85
3. This indicates that the current local surrogate / NSGA-II objective formulation is itself unstable or overly saturating at the target bounds.

### Interpretation

This follow-up is useful for research debugging, but it is not yet safe to write back into the paper as a validated headline result. The immediate implication is that the next technical step should focus on diagnosing surrogate saturation and objective clipping before using the new fair-budget NSGA-II archive as manuscript evidence.

### Saturation diagnosis completed

The saturation diagnosis is documented in `artifacts/publication/reports/surrogate_saturation_diagnosis.md`.

Core conclusion:
1. the local fair-budget NSGA-II collapse is real;
2. it persists even after fixing the evaluator-path fairness bug;
3. it is caused by a rare but optimizer-exploitable region where the surrogate predicts all three objectives beyond observed bounds;
4. hard clipping then collapses those distinct raw predictions to a single ideal-point plateau;
5. tightening the current feasible-radius heuristic alone is not enough, because the penalty scale is still too weak.

### Guardrail follow-up implemented

To push the automation beyond diagnosis:

1. `paper_repro/optimizers.py` was updated so both DDPG and NSGA-II can use a shared `surrogate_guardrail` configuration.
2. NSGA-II result export was corrected so archived objective values now come from the same guarded `env.evaluate()` path used during optimization, instead of bypassing the guardrail at write time.
3. `configs/revision.yaml` and `configs/revision.local.yaml` now include explicit guardrail settings.

Follow-up results:

1. The original total-collapse failure mode was reduced, but not fully eliminated.
2. A 2-seed fair-budget NSGA-II diagnostic under the new guardrail no longer collapses to one exact objective tuple, but still hugs the target bounds very closely.
3. A 2-seed guarded DDPG rerun for the balanced scenario completed successfully, although the learning curves indicate unstable late-episode behavior under the stronger guardrail.
4. Static re-evaluation of the imported DDPG and imported NSGA-II archives under the same new guardrail shows that the imported DDPG archive still retains clearly better mean objective values than the imported NSGA-II archive.

This indicates progress, but not manuscript-safe closure. The guardrail redesign is now a live research branch rather than a finished result.

### Server multi-seed DDPG reruns completed

The guarded server-side DDPG reruns requested for stronger training evidence have now completed for all three scenarios:

1. `Balanced_Performance`
2. `Energy_Saving_Focus`
3. `Energy_Generation_Focus`

All 12 shards (`4` shards per scenario, `5` seeds per shard) were synced back from `/home/ac/Dogtor_Project/DDPG/artifacts/publication/optimization` and merged locally into:

- `artifacts/publication/optimization/ddpg_results_guardrail_full.csv`
- `artifacts/publication/optimization/ddpg_logs_guardrail_full.json`
- `artifacts/publication/optimization/ddpg_logs_all_guardrail_full.json`

The merged guarded DDPG summary is:

- Balanced: `EUIt ≈ 66.56`, `EG ≈ 2.803`, `H ≈ 7.811`
- Saving: `EUIt ≈ 66.48`, `EG ≈ 2.788`, `H ≈ 7.720`
- Generation: `EUIt ≈ 66.65`, `EG ≈ 2.818`, `H ≈ 7.796`

The main paper now uses these merged multi-seed guarded reruns for the learning-curve evidence and the corresponding narrative in the DDPG stability section. The updated shaded learning-curve figure has been regenerated and copied into `elsarticle/fig/fig6.png`.

## Remaining High-Impact Blockers

1. Run additional NSGA-II seeds or a pooled-budget NSGA-II comparison to close the archive-budget fairness gap.
2. Add physical-stack or empirical validation for at least a small representative candidate set.
3. Add a learning-curve or coverage diagnostic for surrogate adequacy if no new physical validation can be obtained.

## 2026-04-11 Local Writing Improvement Loop

### Round Summary

This loop was run as a local writing-only improvement pass after the latest result-to-claim consistency gate. No new experiments were launched. The goal was to tighten claim boundaries, improve the self-containedness of the revised benchmark-fragility story, and make the newly added four-method appendix figure read as a bounded validation aid rather than as a silent benchmark expansion.

### Round 1: Structural Claim Tightening

Main edits:

1. Revised the abstract so the checkpoint-sensitivity audit, `CMA-ES` follow-up, and representative physical probe are described explicitly as bounded supporting evidence rather than as a broader optimizer leaderboard.
2. Tightened the contribution list in `elsarticle/manuscript.tex` so the paper now claims:
   - benchmark-fragility interpretation rather than optimizer superiority;
   - checkpoint-sensitivity and `CMA-ES` as bounding evidence;
   - bounded physical probing as validation-mode support, not full closure.
3. Added a new limitation in the Discussion clarifying that the four-method physical probe is still a representative-candidate comparison, not a fully symmetric benchmark.
4. Clarified in `elsarticle/appendix.tex` that the four-method physical figure mixes balanced-scenario representatives (`DDPG`, `CMA-ES`, `RandomSearch`) with the selected `NSGA-II` benchmark representative by design.

Output:

- `elsarticle/manuscript_round1.pdf`

### Round 2: Final Framing Cleanup

Main edits:

1. Replaced the remaining `decision-support bias` wording in the Introduction with the narrower `preference-conditioned search bias` framing.
2. Strengthened the final concluding paragraph so it now states more explicitly that the checkpoint audit, `CMA-ES` follow-up, and representative physical probe do not overturn the main `NSGA-II > tested DDPG` result; instead, they explain why that result must be interpreted conditionally.

Output:

- `elsarticle/manuscript_round2.pdf`
- `elsarticle/manuscript.pdf` (current final)

### Compile / Format Status

- Final PDF: `elsarticle/manuscript.pdf`
- Page count: `27`
- Undefined references: `0`
- Undefined citations: `0`
- Remaining LaTeX warnings: only `Underfull \\hbox` warnings in appendix table text; no blocking compile errors

### Files Preserved

- `elsarticle/manuscript_round0_original.pdf`
- `elsarticle/manuscript_round1.pdf`
- `elsarticle/manuscript_round2.pdf`
