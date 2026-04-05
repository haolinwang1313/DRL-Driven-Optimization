# Revision Tracker

This tracker maps reviewer comments to implementation status, evidence artifacts, and remaining blockers.

## Reviewer #2

| Item | Status | Evidence |
|------|--------|----------|
| External / independent validation | Partially resolved | `paper_repro/simulation.py`, `paper_repro/pipeline.py`, `elsarticle/supplement.tex` |
| Fairness of DRL vs NSGA-II | Partially resolved | `paper_repro/optimizers.py`, `paper_repro/metrics.py`, `elsarticle/manuscript.tex` |
| Dataset size and coverage transparency | Resolved | `elsarticle/manuscript.tex`, `artifacts/publication/diagnostics/dataset_coverage_summary.json` |
| Nonlinear landscape claim | Resolved | `artifacts/publication/diagnostics/nonlinear_response_profiles.png`, `elsarticle/supplement.tex` |
| Convergence evidence | Partially resolved | `paper_repro/metrics.py`, `paper_repro/pipeline.py`, `elsarticle/manuscript.tex` |
| Graphical abstract quality | Resolved | `elsarticle/graphical_abstract.png` |
| AF / OSR consistency | Resolved | `elsarticle/manuscript.tex`, `elsarticle/supplement.tex` |
| $R^2$ formatting and missing supplementary caption | Resolved | `elsarticle/manuscript.tex`, `elsarticle/supplement.tex` |
| Figure / table labels, units, legends | Partially resolved | `paper_repro/figures.py`, `elsarticle/manuscript.tex` |
| Abbreviation list | Resolved | `elsarticle/manuscript.tex` |

## Reviewer #4

| Item | Status | Evidence |
|------|--------|----------|
| Overstated abstract / claims | Resolved | `elsarticle/manuscript.tex` |
| DDPG positioning vs broader baselines | Partially resolved | `elsarticle/manuscript.tex` |
| Surrogate reliability at optima | Resolved | `artifacts/publication/reevaluation/top_candidate_reevaluation.csv`, `elsarticle/supplement.tex` |
| RL formulation / scalarization justification | Resolved | `elsarticle/manuscript.tex` |
| Reward bound explanation | Resolved | `elsarticle/manuscript.tex` |
| Benchmark transparency | Partially resolved | `paper_repro/optimizers.py`, `configs/revision.yaml`, `paper_repro/publication.py`, `tools/run_ddpg_revision_batch.sh` |
| Discussion of limitations | Resolved | `elsarticle/manuscript.tex` |

## Remaining blockers

- Server-side revision DDPG reruns are running, but final outputs have not yet been harvested into the worktree.
- The current publication pipeline can sync and validate server results, but the locally available imported data still carries `fallback_analytic`.
