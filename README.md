# Surrogate-Conditioned Benchmark Fragility in Block-Scale Urban Energy Design

This repository is a major-revision workspace for an `Applied Energy` manuscript. It is not a complete public data release. The current claim boundary is methodological: the workflow studies surrogate-conditioned optimizer benchmarking for block-scale urban morphology design, not general DRL superiority.

## Canonical Files

- Revision config: `configs/revision.yaml`
- Python package: `paper_repro/`
- Manuscript: `paper/manuscript/manuscript.tex`
- Response letter: `paper/response/round-01/letter.tex`
- Revision tracker: `paper/response/round-01/tracker/revision-tracker.json`
- Artifact root: `artifacts/publication`

## Repository Layout

- `paper_repro/`: reusable pipeline code for simulation fallback, surrogate training, optimization, diagnostics, publication validation, and reviewer utilities.
- `tools/`: helper entrypoints for figure rebuilding, result merging, checkpoint inspection, and batch orchestration.
- `tests/`: lightweight regression tests restored from project history.
- `configs/`: experiment and publication-mode configuration.
- `data/`: local input data catalog and benchmark-data placement.
- `experiments/`: historical experiment logbook and run definitions.
- `research/`: research notes and findings.
- `paper/manuscript/`: TeX manuscript, appendix, references, class file, and tracked figure PDFs.
- `paper/response/round-01/`: formal first-round journal response package and migrated tracker/review state.
- `paper/snapshots/`: working-draft PDF snapshots.
- `paper/submission/`: frozen files only after confirmed journal submission.
- `artifacts/`: generated experiment outputs; ignored by Git.

## Install

Run from the repository root:

```bash
uv sync
```

## Fast Verification

```bash
uv run pytest -q
uv run python -m paper_repro.cli --help
uv run python -m compileall paper_repro tools
uv run python -c "from paper_repro.config import Config; Config.from_yaml('configs/revision.yaml'); print('config ok')"
uv run python tools/build_manuscript_result_figures.py --help
```

## Data Preparation

The benchmark spreadsheet is expected at:

```text
data/external/benchmark/dataset.xlsx
```

The file is intentionally ignored until redistribution rights are clear. Record local copies in `data/catalog.yaml`. The benchmark dataset is used only for benchmark comparison and is not used during surrogate training.

## Core Pipeline Commands

These commands can be expensive and may depend on local artifacts or available simulation support:

```bash
uv run python -m paper_repro.cli --config configs/revision.yaml build-dataset
uv run python -m paper_repro.cli --config configs/revision.yaml select-surrogate
uv run python -m paper_repro.cli --config configs/revision.yaml run-optimizers
uv run python -m paper_repro.cli --config configs/revision.yaml publication-diagnostics
```

Do not run long DDPG, NSGA-II, CMA-ES, random-search, remote-sync, or physical-probe workflows unless the task explicitly requires them.

## Figure And TeX Builds

Figure helper:

```bash
uv run python tools/build_manuscript_result_figures.py --help
```

Manuscript:

```bash
cd paper/manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build manuscript.tex
```

Response:

```bash
cd paper/response/round-01
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build letter.tex
```

Generated build products go under ignored `build/` directories. The tracked PDFs under `paper/snapshots/` are synchronized working snapshots, not proof of journal submission.
