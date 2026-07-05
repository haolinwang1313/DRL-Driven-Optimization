# Usage

Run commands from the repository root after installing the package.

## Load the Canonical Dataset

```python
import pandas as pd

samples = pd.read_csv("data/generated/canonical_2000/simulated_samples.csv")
print(samples.shape)
print(samples[["FAR", "SVF", "EUIt", "EG", "H"]].head())
```

Expected shape: `(2000, 18)`.

## Load Optimizer Result Tables

```python
import pandas as pd

ddpg = pd.read_csv("results/optimization/ddpg_results.csv")
nsga2 = pd.read_csv("results/optimization/nsga2_results.csv")

print(len(ddpg), len(nsga2))
print(ddpg.groupby("scenario")[["EUIt", "EG", "H", "reward"]].mean())
```

Expected retained rows: 60 for DDPG and 2000 for NSGA-II.

## Inspect Surrogate Summaries

```python
import json
import pandas as pd

scale = pd.read_csv("results/surrogate/scale_study.csv")
with open("results/surrogate/surrogate_summary.public.json", encoding="utf-8") as f:
    summary = json.load(f)

print(scale[["dataset_scale", "candidate", "selection_objective", "is_selected"]])
print(summary["dataset_rows"])
```

## Inspect Projection and Physical Summaries

```python
import pandas as pd

projection = pd.read_csv("results/projection/feasible_projection_summary.csv")
physical = pd.read_csv("results/physical_stress/physical_stress_cases.csv")

print(projection.head())
print(physical[["subset", "matched_sample_id", "physical_EUIt", "physical_EG_GHI_proxy", "physical_H"]].head())
```

## Inspect Climate Sensitivity

```python
import pandas as pd

climate = pd.read_csv("results/climate_sensitivity/climate_case_results.csv")
rank = pd.read_csv("results/climate_sensitivity/climate_rank_stability.csv")

print(climate.groupby("station")[["delta_EUIt_vs_baseline", "delta_EG_vs_baseline", "delta_H_vs_baseline"]].mean())
print(rank)
```

## Validate the Public Package

```bash
python scripts/validate_public_release.py
python -m pytest -q
```

The validator checks catalog paths, SHA-256 hashes, row counts, required public files, canonical `fallback_analytic` metadata, and disallowed local-machine markers.
