# Reproducibility

## Scope

This public package supports lightweight reproduction and inspection of the APEN benchmark evidence from committed files:

- Load the canonical 2000-row analytic-response dataset.
- Inspect morphology descriptors and analytic targets.
- Read DDPG and NSGA-II retained optimizer result tables.
- Read feasible-projection, physical-stress, and climate-sensitivity summaries.
- Run public tests and validate hashes, row counts, and release boundaries.

The package does not rerun long optimizer training, surrogate selection, or physical evaluation workflows.

## Data Boundary

`data/generated/canonical_2000/simulated_samples.csv` uses `fallback_analytic` response generation. It is not a direct annual EnergyPlus/Radiance dataset for all 2000 samples.

Selected physical stress-test outputs are released as processed summaries under `results/physical_stress/`. Climate sensitivity summaries are released under `results/climate_sensitivity/`. Raw execution logs, machine-specific configs, and model checkpoints are outside this Git package.

The older external spreadsheet path `data/external/benchmark/dataset.xlsx` is recorded as `not_included` in `data/catalog.yaml` and is not required for the current analytic-response benchmark package.

## Installation

```bash
python -m venv .venv
python -m pip install -e ".[test]"
```

With `uv`:

```bash
uv sync
```

## Quick Verification

```bash
python -m pytest -q
python scripts/validate_public_release.py
python -m compileall paper_repro scripts tests
```

With `uv`:

```bash
uv run pytest -q
uv run python scripts/validate_public_release.py
uv run python -m compileall paper_repro scripts tests
```

Quick data check:

```bash
python -c "import pandas as pd; df = pd.read_csv('data/generated/canonical_2000/simulated_samples.csv'); print(df.shape); print(df[['EUIt','EG','H']].describe())"
```

Expected headline counts:

- 2000 canonical samples.
- 2000 generated block records.
- 60 retained DDPG rows.
- 2000 retained NSGA-II rows.
- 24 physical stress-test cases.
- 12 climate sensitivity case rows.

## Reproducing Summaries

The committed CSV and JSON files support direct inspection of the result summaries used by the public package:

- `results/optimization/`: retained optimizer result tables.
- `results/surrogate/`: surrogate validation and scale-study summaries.
- `results/projection/`: feasible-projection summaries.
- `results/physical_stress/`: processed physical stress-test summaries.
- `results/climate_sensitivity/`: processed climate sensitivity summaries.
- `results/figure_data/`: final figure-data tables and manifest.

Heavy reruns require non-committed execution context and are outside this public package. The released tables are the reproducibility surface for the APEN public branch.

## Files Outside This Git Package

- `data/external/benchmark/dataset.xlsx`: source and redistribution rights are not verified, and the file is not required for the current public analytic-response package.
- `surrogate.pt`: trained checkpoint is omitted; released validation tables and processed summaries are committed instead.
- Raw per-episode logs: summarized into public DDPG training and seed-diagnostic tables.
- Raw EPW files: not redistributed; public climate tables use station-level summaries.
- Machine-specific execution configs: not needed for the committed lightweight checks.
