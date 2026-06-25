# Decision Input For Reviewer Round 2

## Core Decision

Proceed with the second-round revision only as a bounded, evidence-first manuscript revision. The current evidence supports a surrogate-conditioned benchmark and design-support study. It does not support unqualified claims of DRL superiority, physical-stack certification, broad climate transfer, or integrated energy-system operation.

## Direct Answers

### Which dataset is the current manuscript using?

The current result-building pathway uses:

`artifacts/server_runs/20260405_highest_precision_2000_compare/data/simulated_samples.csv`

This file has 2000 rows and SHA-256:

`b8bc287ad3d9c8db9f7e090630fd3fe1f2276e0d3f8072f3fa13adcee8cba5cc`

It is not the older 500-row `artifacts/publication/data/simulated_samples.csv`.

### Is the 2000-row dataset physically simulated?

Not by local evidence. The metadata marks it as `simulation_mode = fallback_analytic`. The generator is deterministic and nested by prefixes across 500, 1000, 1500, and 2000 rows.

Safe wording: "fallback analytic surrogate-training dataset" or "analytic simulation pool".

Unsafe wording: "EnergyPlus/Radiance simulated dataset" unless new provenance is supplied.

### Is the old Excel benchmark usable as physical ground truth?

Unknown. `initial_paper/Dataset.xlsx` is recoverable from Git history and has 286 rows, but this audit found no metadata or logs proving Grasshopper/Honeybee/EnergyPlus/Radiance origin. `data/external/benchmark/dataset.xlsx` is absent and cataloged with unknown license and redistributability.

### What does Fig. 9(d) currently say?

The recalculated Fig. 9(d) post-hoc utility still favors NSGA-II over DDPG:

| Scenario | DDPG | NSGA-II | DDPG minus NSGA-II |
| --- | ---: | ---: | ---: |
| Balanced_Performance | 0.733125514077819 | 0.9999995323705076 | -0.26687401829268864 |
| Energy_Saving_Focus | 0.8597227209533328 | 0.9999997194823047 | -0.1402769985289719 |
| Energy_Generation_Focus | 0.9535704056035622 | 0.9999997194823047 | -0.046429313878742495 |

Utility is larger-is-better and is not the same as the training reward.

### What is the implemented reward?

Implemented reward:

```text
state = (outputs - target_min) / target_range
utopia = [0, 1, 1]
d = sqrt(sum((w * (state - utopia))^2)) / sqrt(sum(w^2))
reward = 1 - d
```

The manuscript equation `R = 10^6 - d_weighted` is not the implemented reward.

### Are the 12 inputs independent design variables?

No. They are morphology descriptors. At least two deterministic relations are exact to numerical tolerance:

- `FAR = BD * AF`
- `OSR = (1 - BD) / FAR`

Also, dataset OSLI is integer-valued, while optimizer actions are continuous and can map to non-integer OSLI.

### Is the DDPG versus NSGA-II comparison fair?

Partly. The per-seed surrogate-query budget is comparable: DDPG uses 600 x 40 = 24000 queries per seed, and NSGA-II fair-budget mode uses 24000 evaluations per run.

The retained archives are not size-symmetric:

- DDPG: 20 retained candidates per scenario.
- NSGA-II: 2000 retained candidates across 20 runs.

Report archive counts and add equal-size downsampling sensitivity if HV/IGD remains in the revision.

### Is there local closure for physical-probe claims?

No. The expected physical-stack probe CSVs are absent from `artifacts/publication/reevaluation/`. The local code indicates that physical probing projects optimizer candidates to nearest sampled blocks and then runs or approximates physical metrics, but the current artifact chain is incomplete.

Safe action: mark physical-probe claims as unverified locally or supply the missing CSVs/result JSONs and logs.

### Is the repository ready to make public?

No. The GitHub repository is currently private, and evidence-critical artifacts are ignored:

- current 2000-row data, blocks, and metadata
- selected surrogate checkpoint and summaries
- optimization archives and logs
- physical-probe artifacts
- external benchmark dataset rights

`server.local.yaml` and local config overrides are high-risk and must not be published.

### What must be fixed before the second-round revision?

Minimum evidence-first fixes:

- Correct the reward equation and any reward text/figure labels.
- Qualify Fig. 9(d) as post-hoc utility, not training reward.
- State that the 12 fields are morphology descriptors, not independent design variables.
- Disclose DDPG/NSGA-II archive-size asymmetry in HV/IGD reporting.
- Treat physical-probe claims as unverified unless missing artifacts are supplied.
- Remove or correct Fig. 1 typos and stale reward formula text.
- Recover editable Fig. 1/Fig. 3 sources or document them as PDF-only.
- Keep climate/energy-system expansion as limitation or future work.
- Define every "best", "winner", and "selected" by criterion.
- Prepare a separate artifact-release plan before any public GitHub release.

## Recommended Revision Posture

Use conservative framing:

"The study evaluates surrogate-conditioned optimization behavior under a controlled morphology descriptor space and finite benchmark artifacts."

Avoid broad framing:

"DDPG is superior for urban energy design" or "the selected candidates are physically validated optimal urban forms."
