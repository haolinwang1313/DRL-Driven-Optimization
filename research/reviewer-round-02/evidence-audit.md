# Paper02DRL Reviewer Round 2 Evidence Audit

Scope: pre-revision fact and evidence audit for the Applied Energy second-round revision. This file only records local evidence already present in the repository, Git history, ignored artifacts, and lightweight derived audit tables under `research/reviewer-round-02/`.

Hard limits observed: no manuscript edits, no response-letter edits, no package/tool/config edits, no experiment reruns, no physical-stack jobs, no remote access, no repository visibility changes, no ignored artifact upload, and no secret-value inspection.

## Executive Summary

The current manuscript result pathway is mainly tied to `artifacts/server_runs/20260405_highest_precision_2000_compare`, not to the older 500-row `artifacts/publication/data/simulated_samples.csv` bundle. The current 2000-row dataset is a deterministic `fallback_analytic` simulation pool, not a verified Grasshopper/Honeybee/EnergyPlus/Radiance physical-simulation export.

The strongest pre-revision risks are:

- The manuscript reward equations do not match the implemented DDPG reward scale.
- Fig. 9(d) uses post-hoc utility, not the training reward.
- DDPG and NSGA-II share the same per-seed evaluation budget, but the reported archives are size-asymmetric: 20 DDPG retained candidates per scenario versus 2000 NSGA-II candidates.
- Physical-probe claims are not closed by local artifacts: the expected physical-stack CSVs are absent in this checkout.
- Fig. 10 and Fig. 11 are intentionally pinned from tracked PDFs rather than regenerated from the current result bundle.
- Fig. 1 has extractable typo/formula issues, and no editable Fig. 1/Fig. 3 source file was found.

The safe revision stance is to frame the paper as a surrogate-conditioned benchmark and design-support study, not as evidence of general DRL superiority or physically certified urban-energy optimality.

## Status Matrix

| Section | Status | Main conclusion | Primary local evidence |
| --- | --- | --- | --- |
| A. Data lineage | Confirmed / Unknown | Current results use a 2000-row `fallback_analytic` pool; the historical Excel dataset exists only in Git history and has unverified physical origin. | `audit-facts.json`, `evidence-manifest.json`, `data/catalog.yaml`, Git history |
| B. Sampling and design space | Confirmed | The 12 columns are descriptors with deterministic dependencies, not 12 independent design variables; sampling is random morphology generation, not LHS/grid. | `paper_repro/morphology.py`, `dataset_feature_summary.csv`, `dataset_correlation_matrix.csv` |
| C. Result provenance | Confirmed / Likely | Fig. 4-9 and Fig. 12 are linked to current server-run artifacts; Fig. 10/11 are pinned tracked PDFs. | `tools/build_manuscript_result_figures.py`, `evidence-manifest.json` |
| D. Reward, utility, Fig. 9 | Confirmed | Implemented reward is `1 - normalized_distance`; manuscript Eq. (6) says `10^6 - d_weighted`; Fig. 9(d) is post-hoc utility. | `paper_repro/optimizers.py`, `paper_repro/metrics.py`, `fig9d_utility_recalc.csv` |
| E. RL semantics | Confirmed | The DDPG loop is a serialized surrogate-query policy search over absolute morphology descriptors, not a physical continuous-control process. | `paper_repro/optimizers.py` |
| F. Optimizer fairness | Confirmed / Partial | Per-seed query budget is comparable, but retained archive size is not comparable; HV/IGD should disclose full archive and equal-size sensitivity. | `configs/revision.yaml`, `hv_igd_full_archive.csv`, `hv_igd_downsample_summary.csv` |
| G. Physical probe | Unknown / High risk | Local physical-stack evidence is incomplete; expected physical probe CSVs are absent. | `audit-facts.json`, `tools/build_four_method_physprobe_figure.py`, `paper_repro/physical_stack.py` |
| H. Climate and energy-system scope | Confirmed limitation | Current evidence does not support broad climate-transfer or integrated energy-system claims. | `configs/revision.yaml`, `paper_repro/simulation.py`, artifact metadata |
| I. "Best" terminology | Confirmed ambiguity | "Best", "winner", and "selected" refer to multiple incompatible criteria. | Manuscript/appendix text, optimizer outputs, surrogate selection JSON |
| J. GitHub release readiness | Confirmed gap | Repository is private and current evidence-critical artifacts are ignored; a public release needs a separate artifact-release plan. | `gh repo view`, `.gitignore`, `release_scan` |
| K. Fig. 1/Fig. 3 sources | Confirmed / Unknown | Only tracked PDFs were found; editable sources were not found. Fig. 1 text extraction exposes typos and stale reward formula text. | `fig1_text_extract.txt`, `fig3_text_extract.txt`, `git ls-files` |

## A. Data Lineage

Status: Confirmed for local artifact identity; Unknown for the physical origin of the historical Excel dataset.

Evidence:

- Current figure rebuild logic prioritizes `artifacts/server_runs/20260405_highest_precision_2000_compare` and `artifacts/server_runs/20260405_surrogate_rebenchmark`, with `artifacts/publication` only as fallback.
- Current main dataset: `artifacts/server_runs/20260405_highest_precision_2000_compare/data/simulated_samples.csv`.
- Current dataset rows: 2000.
- Current dataset SHA-256: `b8bc287ad3d9c8db9f7e090630fd3fe1f2276e0d3f8072f3fa13adcee8cba5cc`.
- Current dataset `simulation_mode`: `fallback_analytic`.
- Older publication dataset: `artifacts/publication/data/simulated_samples.csv`.
- Older publication dataset rows: 500.
- Older publication dataset SHA-256: `db9b53a87f0e9c3b9292e2292833ab39ed5d160aee613270bc96f304b06ede4d`.
- The 500-row publication dataset matches `artifacts/publication/imported/data/simulated_samples.csv`, but does not match the current 2000-row dataset.
- Rebenchmark scale datasets are nested prefixes: 500, 1000, and 1500 rows are exact prefixes of the 2000-row pool.
- `artifacts/server_runs/20260405_surrogate_rebenchmark/data/selected_dataset.json` points to an active 1500-row alias, while the current comparison bundle uses the strict 2000-row highest-accuracy override.
- `artifacts/server_runs/20260405_highest_precision_2000_compare/models/selected_surrogate.json` records `scale_selection.mode = highest_accuracy_override`, `selected_dataset_scale = 2000`, and `selected_candidate = tuned_standard`.
- `data/external/benchmark/dataset.xlsx` is absent from this checkout and cataloged as `benchmark_only`, not tracked by Git, with unknown license and redistributability.
- Historical `initial_paper/Dataset.xlsx` is absent from the working tree but recoverable from Git commit `94b37f026e298d3c7ee30f865ffa09331dd09383`.
- Historical Excel blob SHA-1: `9b575e4cc1550c450a7ce2c64bc734cc0a525d52`.
- Historical Excel SHA-256: `562502611d4a805662d11fe8f89903ee6c61bec5a44113e20ee15ded8c9bed79`.
- Historical Excel has one sheet, 286 rows, and columns `Method`, `EUlt (kWh/m²/y)`, `EG (10⁶ kWh/y)`, `H (h)`, `Unnamed: 4`.

Audit conclusion:

The 2000-row dataset can be treated as the current local result dataset, but only as a fallback-analytic dataset. No local evidence verifies that the historical Excel dataset directly came from Grasshopper, Honeybee, EnergyPlus, or Radiance. Do not describe it as a certified physical-simulation source unless independent provenance is added.

## B. Sampling Coverage And Design-Space Interpretation

Status: Confirmed.

Evidence:

- `paper_repro/morphology.py` uses `random_block()`, not Latin hypercube sampling or a grid.
- `open_space_index = rng.integers(0, 9)`.
- `theta_deg = rng.uniform(-45, 45)`.
- Non-open cells randomly choose prototype and floor count.
- Current 2000-row OSLI counts:
  - 0: 191
  - 1: 261
  - 2: 232
  - 3: 217
  - 4: 212
  - 5: 217
  - 6: 222
  - 7: 199
  - 8: 249
- No exact duplicate feature rows were found in the current 2000-row dataset.
- Normalized nearest-neighbor distance:
  - minimum: 0.06767783412087773
  - median: 0.1991756882074244
  - 95th percentile: 0.31346758347323345
- PCA needs 6 components for 95% variance.
- PCA participation ratio is approximately 5.0212.
- Deterministic relation check:
  - `FAR - BD * AF` max absolute residual: `1.3322676295501878e-15`.
  - `OSR - (1 - BD) / FAR` max absolute residual: `2.220446049250313e-16`.
- `OSLI` is integer-valued in the dataset, but optimizer actions are continuous normalized vectors and can map to non-integer OSLI.
- `OptimizationEnvironment.evaluate_batch()` clips normalized actions to `[0, 1]`, maps them by feature min/max, and applies distance/extrapolation guardrails. It is not a strict constructive morphology decoder.

Audit conclusion:

The paper should describe the 12 fields as morphology descriptors or surrogate input descriptors, not as 12 independent design degrees of freedom. Feasibility is bounded by surrogate training-range guardrails and nearest-sample distance, not by a full urban morphology validity decoder.

## C. Results, Figures, Tables, And Provenance

Status: Confirmed for the main rebuild script; Likely for some manuscript/table links that depend on TeX inclusion and artifact naming.

Evidence:

- `tools/build_manuscript_result_figures.py` defines:
  - `CURRENT_COMPARE_RUN = 20260405_highest_precision_2000_compare`.
  - `CURRENT_SELECTION_RUN = 20260405_surrogate_rebenchmark`.
- Fig. 4 and Fig. 5 use `compare_root/models/cv_predictions.csv`.
- Fig. 6 uses DDPG logs.
- Fig. 7, Fig. 8, and Fig. 9 use `compare_root/optimization/ddpg_results.csv`, `compare_root/optimization/nsga2_results.csv`, and `compare_root/optimization/optimization_results.csv`.
- Fig. 12 uses `compare_root/models/surrogate.pt` and `compare_root/data/simulated_samples.csv`.
- Fig. 10 and Fig. 11 are intentionally pinned: the builder restores tracked PDFs from `HEAD:paper/manuscript/figures/fig10.pdf` and `HEAD:paper/manuscript/figures/fig11.pdf` instead of regenerating them from the current bundle.
- `evidence-manifest.json` records figure PDFs, scripts, configs, data files, model files, hashes, row counts, and public-release readiness where discoverable.

Audit conclusion:

The current manuscript figure provenance is not uniform. Fig. 4-9 and Fig. 12 are tied to current server-run artifacts. Fig. 10/11 should be described as pinned legacy/curated figure artifacts unless regenerated and closed against the current result bundle.

## D. Reward, Utility, And Fig. 9

Status: Confirmed.

Implemented DDPG reward:

```text
state = (outputs - target_min) / target_range
utopia = [0, 1, 1]
weighted_distance = sqrt(sum((w * (state - utopia))^2))
normalized_distance = weighted_distance / sqrt(sum(w^2))
reward = 1 - normalized_distance
```

Manuscript equation mismatch:

- Manuscript Eq. (5) omits the implementation's division by `sqrt(sum(w^2))`.
- Manuscript Eq. (6) states `R = 10^6 - d_weighted`.
- The implementation does not use a `10^6` offset; it uses a `1 - normalized_distance` reward scale.

Fig. 9(d) utility recalculation:

| Scenario | DDPG best utility | NSGA-II best utility | DDPG minus NSGA-II |
| --- | ---: | ---: | ---: |
| Balanced_Performance | 0.733125514077819 | 0.9999995323705076 | -0.26687401829268864 |
| Energy_Saving_Focus | 0.8597227209533328 | 0.9999997194823047 | -0.1402769985289719 |
| Energy_Generation_Focus | 0.9535704056035622 | 0.9999997194823047 | -0.046429313878742495 |

Evidence:

- `paper_repro/optimizers.py` contains the reward and vectorized `reward_batch()` formula.
- `paper_repro/metrics.py` contains `normalized_benefit_frame()` and utility logic.
- `fig9d_utility_recalc.csv` records the recalculated utility rows.
- Utility direction is larger-is-better.
- Normalization min/max source is `artifacts/server_runs/20260405_highest_precision_2000_compare/optimization/optimization_results.csv`.

Audit conclusion:

Fig. 9(d) still supports the direction that NSGA-II is stronger than DDPG under the post-hoc utility calculation. It does not validate the manuscript reward equations. The reward equations should be corrected before revision.

## E. RL Sequence Semantics

Status: Confirmed.

Evidence:

- Each episode starts from a random normalized action.
- The surrogate maps that action to outputs, which are normalized into a 3-dimensional performance state.
- The actor outputs a 12-dimensional normalized action.
- Gaussian noise is added during exploration.
- Actions are clipped to `[0, 1]`.
- The action is an absolute morphology descriptor vector, not an incremental edit.
- `next_state` is the surrogate output of the current action, normalized over the 3 target metrics.
- `done` becomes true only at the fixed horizon step.
- The 40-step episode horizon serializes repeated surrogate queries; it does not represent a physical time sequence.
- State only contains `EUIt`, `EG`, and `H`, so different morphology descriptors can share similar states.

Audit conclusion:

The DDPG setup is best described as sequentialized static black-box search or context-conditioned policy search over surrogate descriptors. Calling it physical continuous control would overstate the evidence.

Future change blast radius if the semantics are changed:

- `OptimizationEnvironment.evaluate_batch()`.
- DDPG actor/action interpretation.
- Replay buffer state/action contracts.
- Morphology decoder and feasibility constraints.
- NSGA-II, CMA-ES, RandomSearch comparison interfaces.
- Result manifests and figure builders.
- Tests for feasible descriptor decoding and optimizer budget accounting.

## F. DDPG Versus NSGA-II Fairness

Status: Confirmed for budget and archive counts; Partial for fairness interpretation.

Evidence:

- `configs/revision.yaml` sets DDPG to 600 episodes x 40 steps = 24000 surrogate queries per seed.
- NSGA-II fair-budget mode uses an evaluation budget of 24000 per run.
- Both DDPG and NSGA-II use 20 seeds/runs.
- DDPG retains one best-reward candidate per seed, yielding 20 rows per scenario.
- NSGA-II retains final populations, yielding 2000 rows across 20 runs.

Full-archive HV/IGD:

| Group | HV | IGD | Archive rows | Non-dominated rows |
| --- | ---: | ---: | ---: | ---: |
| Balanced_Performance | 0.347404240428601 | 0.7196336732419489 | 20 | 6 |
| Energy_Generation_Focus | 1.0107424885251182 | 0.22048366927452373 | 20 | 3 |
| Energy_Saving_Focus | 0.6252282421179792 | 0.5490857458951972 | 20 | 5 |
| NSGA-II | 1.3310000000000002 | 0.10676879393889152 | 2000 | 100 |

NSGA-II downsample sensitivity:

| NSGA-II sample size | HV mean | IGD mean | IGD 5-95% |
| ---: | ---: | ---: | ---: |
| 20 | 1.3309976498846339 | 0.58358 | 0.369-0.761 |
| 60 | 1.3309999004217636 | 0.57565 | 0.461-0.716 |
| 100 | 1.3309999663577017 | 0.51740 | 0.386-0.676 |

Audit conclusion:

The per-seed query-budget comparison is defensible. The archive-based HV/IGD comparison is not size-symmetric and should be reported with archive counts, per-seed logic, and equal-size downsampling sensitivity. NSGA-II remains strong in HV even when downsampled, but IGD is sensitive to archive sampling.

## G. Physical Probe Evidence

Status: Unknown / High risk.

Evidence:

- Local `artifacts/publication/reevaluation/` contains `top_candidate_reevaluation.csv`.
- Expected physical probe CSVs are absent:
  - `physical_stack_candidate_probe_asynccheck30.csv`
  - `physical_stack_candidate_probe_asynccheck29.csv`
  - `physical_stack_candidate_probe_asynccheck28.csv`
  - `physical_stack_candidate_probe_asynccheck27.csv`
  - `physical_stack_candidate_probe_physprobe_methods_v2.csv`
- `tools/build_four_method_physprobe_figure.py` expects physical probe CSVs for two-method and four-method figures.
- `paper_repro/physical_stack.py` projects candidates to nearest sampled blocks before simulation and records `matched_sample_id` and `projection_distance`.
- Physical stack EUIt uses Honeybee/EnergyPlus commands when available.
- Physical stack EG is a simplified PV proxy using EPW annual GHI, roof area, 0.8 coverage, 0.2 efficiency, and 0.75 performance ratio.
- Physical stack H uses a Radiance direct-sun/window-sensor workflow or fallback sampled sky.
- `server.local.yaml` exists and is ignored; secret values were not read.
- `.env` is absent.

Audit conclusion:

The local repository does not contain a complete physical-probe artifact chain for the current appendix/manuscript claims. Treat physical-probe claims as unverified locally unless the missing CSVs/result JSONs and command logs are supplied or regenerated under a controlled run.

## H. Cross-Climate And Energy-System Scope

Status: Confirmed limitation.

Evidence:

- `configs/revision.yaml` includes Dongtai and Nanjing weather URLs.
- Current fallback-analytic metadata has `weather_epw = null`.
- `paper_repro/simulation.py` applies station-dependent analytic bias; it does not prove EPW time-series simulation for the current dataset.
- The current optimization model does not include storage dispatch, EV charging load, distributed-energy coordination, grid interaction, or time-resolved supply-demand balance.
- The physical stack can theoretically swap EPW files, but this audit did not read secrets, SSH, or run remote/physical jobs.

Audit conclusion:

Climate transfer and integrated energy-system implications should stay in limitations or future work. They are not supported as current empirical findings.

## I. "Best", "Winner", And "Selected" Terminology

Status: Confirmed ambiguity.

Observed meanings:

- Best reward: optimizer-internal single-seed or single-scenario best-reward candidate.
- Best utility: post-hoc normalized linear utility maximum.
- Best HV/IGD: group-level front quality.
- Selected surrogate: `selected_surrogate.json` highest-accuracy override, 2000 rows, `tuned_standard`.
- Physical selected candidate: representative candidate after nearest-block projection in the physical stack.

Audit conclusion:

Revision text should qualify every use of "best", "winner", and "selected" with its criterion. Unqualified wording invites reviewer objections because the criteria are not interchangeable.

## J. GitHub Public-Release Readiness

Status: Confirmed gap.

Evidence:

- `gh repo view haolinwang1313/DRL-Driven-Optimization --json visibility,nameWithOwner,url` returned `visibility = PRIVATE`.
- `.gitignore` excludes `artifacts/`, `data/external/benchmark/*.xlsx`, `server.local.yaml`, and `configs/*.local.yaml`.
- Ignored local items include `artifacts/`, `configs/revision.local.yaml`, and `server.local.yaml`.
- Release scan:
  - `artifacts/publication`: exists, ignored, about 79 MB.
  - `artifacts/server_runs/20260405_highest_precision_2000_compare`: exists, ignored, about 25 MB.
  - `artifacts/server_runs/20260405_surrogate_rebenchmark`: exists, ignored, about 31 MB.
  - `data/external/benchmark/dataset.xlsx`: absent.
  - `configs/revision.local.yaml`: exists, ignored, high risk, do not publish.
  - `server.local.yaml`: exists, ignored, high risk, do not publish.

Audit conclusion:

The repository is not ready for public reproducibility by GitHub visibility change alone. A public release needs a separate artifact channel, rights clarification for benchmark data, sanitized configs, and a manifest that links current 2000-row data, blocks, metadata, selected surrogate checkpoint, optimization archives/log summaries, and any physical-probe evidence.

## K. Fig. 1 And Fig. 3 Editable Sources

Status: Confirmed for tracked PDFs; Unknown for editable sources.

Evidence:

- Tracked PDFs exist:
  - `paper/manuscript/figures/fig1.pdf`
  - `paper/manuscript/figures/fig3.pdf`
- No editable source file was found for Fig. 1 or Fig. 3 in the local tracked/untracked search.
- `fig1_text_extract.txt` includes:
  - `street loactions`
  - `Acotr netword`
  - `netword`
  - `R = 10^6 - dweighted`
- `fig3_text_extract.txt` was extractable but does not close editable-source provenance.

Audit conclusion:

Fig. 1 needs typo and formula synchronization before submission. Fig. 1/Fig. 3 should be treated as PDF-only in this checkout unless editable source files are recovered.

## Public Release Checklist Before Visibility Change

- Publish or archive the current 2000-row data, block JSONL, and metadata.
- Publish or archive the selected surrogate checkpoint, surrogate summary, and CV predictions.
- Publish or archive optimization archives/log summaries used by manuscript figures.
- Supply or regenerate missing physical-probe CSVs/result JSONs before making physical-probe claims.
- Clarify rights for `data/external/benchmark/dataset.xlsx` before redistribution.
- Exclude `server.local.yaml`, `.env`, private keys, local override YAML files, credentials, and personal paths.
- Include exact artifact hashes and artifact-channel URLs in a reproducibility manifest.

## Files Added By This Audit

- `audit-facts.json`: structured local facts, hashes, row counts, and audit calculations.
- `evidence-manifest.json`: evidence inventory with source class and release-readiness flags.
- `dataset_feature_summary.csv`: current 2000-row feature summary.
- `dataset_correlation_matrix.csv`: descriptor correlation matrix.
- `fig9d_utility_recalc.csv`: Fig. 9(d) utility recalculation.
- `hv_igd_full_archive.csv`: full-archive HV/IGD.
- `hv_igd_downsample_repetitions.csv`: repeated NSGA-II downsampling.
- `hv_igd_downsample_summary.csv`: downsampling summary.
- `fig1_text_extract.txt`: Fig. 1 PDF text extraction.
- `fig3_text_extract.txt`: Fig. 3 PDF text extraction.
- `decision-input.md`: short decision input for revision planning.
