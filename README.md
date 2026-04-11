# Surrogate-Conditioned Benchmark Fragility in Block-Scale Urban Energy Design

This repository contains the code and manuscript package for our block-scale urban morphology optimization study based on surrogate modeling, DDPG, and NSGA-II.

The current version is maintained for an article under peer review. The repository is therefore kept intentionally concise and does not yet serve as a full public release of all workflows and artifacts.

## Overview

The study focuses on optimizing urban morphology factors to improve three block-scale energy indicators:

- `EUIt`: Energy Use Intensity
- `EG`: Energy Generation
- `H`: Sunlight Hours

The current workflow combines:

- parametric morphology generation
- surrogate-based performance prediction
- optimizer comparison between DDPG and fair-budget NSGA-II
- benchmark-diagnostic analysis under different surrogate checkpoints

## Repository Structure

The main directories are:

- `paper_repro/`: core pipeline, surrogate modeling, optimization, diagnostics, and reviewer utilities
- `configs/`: runtime and revision configurations
- `tools/`: helper scripts for reruns, figure generation, and result processing
- `elsarticle/`: current manuscript source and figure files

## Environment

Recommended environment:

- Python `3.10+`
- local virtual environment managed with `uv` or standard `venv`

Core Python dependencies include:

- `torch`
- `optuna`
- `pymoo`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `PyYAML`

## Minimal Workflow

Run all commands from the repository root.

### 1. Build the revision dataset

```bash
python -m paper_repro.cli --config configs/revision.yaml build-dataset
```

### 2. Select the surrogate

```bash
python -m paper_repro.cli --config configs/revision.yaml select-surrogate
```

### 3. Run the optimizers

```bash
python -m paper_repro.cli --config configs/revision.yaml run-optimizers
```

### 4. Rebuild the manuscript figures

```bash
uv run python tools/build_manuscript_result_figures.py --compile-manuscript
```

## Notes

- The repository supports a documented fallback analytic simulator when a full physical stack is unavailable.
- The manuscript currently reflects the revision workflow centered on `configs/revision.yaml`.
- The figure and manuscript package in `elsarticle/` represents the current submission version.
