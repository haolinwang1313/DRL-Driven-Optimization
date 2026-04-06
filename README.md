# Paper02 Reproduction

This repository reconstructs the workflow described in:

- `manuscript1105_clean.pdf`
- `Supplementary Information.pdf`

The implementation is configuration-driven and provides CLI entry points for:

- `bootstrap-sim-stack`
- `build-dataset`
- `train-surrogate`
- `select-surrogate`
- `run-optimizers`
- `make-paper-figures`
- `full-reproduce`

The code attempts to bootstrap a Ladybug Tools style simulation stack, but it also
supports a documented fallback simulator so the full pipeline remains executable on
machines without Rhino/Grasshopper, EnergyPlus, or Radiance.

For the current manuscript figure set used in `elsarticle/`, rebuild with:

- `uv run python tools/build_manuscript_result_figures.py --compile-manuscript`

This script is pinned to the current strict-highest-accuracy result bundle
`artifacts/server_runs/20260405_highest_precision_2000_compare`, while keeping
`fig10.pdf` and `fig11.pdf` on the approved committed version.

For revision-mode benchmarking, the expected sequence is:

1. `python -m paper_repro.cli --config configs/revision.yaml build-dataset`
2. `python -m paper_repro.cli --config configs/revision.yaml select-surrogate`
3. `python -m paper_repro.cli --config configs/revision.yaml run-optimizers`
