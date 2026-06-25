# Revision Tracker

This tracker maps reviewer comments to implementation status, evidence artifacts, and remaining blockers.

## Reviewer #2

| Item | Status | Evidence |
|------|--------|----------|
| External / independent validation | Partially resolved | `paper_repro/simulation.py`, `paper_repro/pipeline.py`, `paper/manuscript/appendix.tex` |
| Fairness of DRL vs NSGA-II | Partially resolved | `paper_repro/optimizers.py`, `paper_repro/metrics.py`, `paper/manuscript/manuscript.tex` |
| Dataset size and coverage transparency | Resolved | `paper/manuscript/manuscript.tex`, `artifacts/publication/diagnostics/dataset_coverage_summary.json` |
| Nonlinear landscape claim | Resolved | `artifacts/publication/diagnostics/nonlinear_response_profiles.png`, `paper/manuscript/appendix.tex` |
| Convergence evidence | Partially resolved | `paper_repro/metrics.py`, `paper_repro/pipeline.py`, `paper/manuscript/manuscript.tex` |
| Graphical abstract quality | Resolved | `paper/manuscript/figures/fig1.pdf` |
| AF / OSR consistency | Resolved | `paper/manuscript/manuscript.tex`, `paper/manuscript/appendix.tex` |
| $R^2$ formatting and missing supplementary caption | Resolved | `paper/manuscript/manuscript.tex`, `paper/manuscript/appendix.tex` |
| Figure / table labels, units, legends | Partially resolved | `paper_repro/figures.py`, `paper/manuscript/manuscript.tex` |
| Abbreviation list | Resolved | `paper/manuscript/manuscript.tex` |

## Reviewer #4

| Item | Status | Evidence |
|------|--------|----------|
| Overstated abstract / claims | Resolved | `paper/manuscript/manuscript.tex` |
| DDPG positioning vs broader baselines | Partially resolved | `paper/manuscript/manuscript.tex` |
| Surrogate reliability at optima | Resolved | `artifacts/publication/reevaluation/top_candidate_reevaluation.csv`, `paper/manuscript/appendix.tex` |
| RL formulation / scalarization justification | Resolved | `paper/manuscript/manuscript.tex` |
| Reward bound explanation | Resolved | `paper/manuscript/manuscript.tex` |
| Benchmark transparency | Partially resolved | `paper_repro/optimizers.py`, `configs/revision.yaml`, `paper_repro/publication.py`, `tools/run_ddpg_revision_batch.sh` |
| Discussion of limitations | Resolved | `paper/manuscript/manuscript.tex` |

## Remaining blockers

- Server-side revision DDPG reruns are running, but final outputs have not yet been harvested into the worktree.
- The current publication pipeline can sync and validate server results, but the locally available imported data still carries `fallback_analytic`.
