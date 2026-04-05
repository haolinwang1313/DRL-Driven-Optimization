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

For revision-mode benchmarking, the expected sequence is:

1. `python -m paper_repro.cli --config configs/revision.yaml build-dataset`
2. `python -m paper_repro.cli --config configs/revision.yaml select-surrogate`
3. `python -m paper_repro.cli --config configs/revision.yaml run-optimizers`
