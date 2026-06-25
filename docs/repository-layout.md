# Repository Layout

This repository separates reusable code, local inputs, generated outputs, research notes, and journal-facing paper assets.

## Root

- `README.md`: operational entrypoint for humans.
- `AGENTS.md`: stable agent instructions and integrity constraints.
- `PROJECT.yaml`: machine-readable canonical paths.
- `pyproject.toml` and `uv.lock`: Python package and dependency lock.
- `.gitignore`: explicit local-state and generated-output exclusions.

## Code And Configuration

- `paper_repro/`: reusable package code. The package name and location remain stable for this revision.
- `tools/`: auxiliary scripts and batch entrypoints. Scripts may rebuild figures or inspect artifacts, but should not contain separate scientific logic when package code already provides it.
- `tests/`: regression checks for metrics, morphology, simulation scale handling, and surrogate selection.
- `configs/`: declared experiment and publication settings. `configs/revision.yaml` is the canonical revision config.

## Data And Artifacts

- `data/`: input data placement and catalog records.
- `data/external/benchmark/`: local benchmark spreadsheet location. Binary benchmark data remains ignored unless redistribution rights are explicit.
- `artifacts/`: machine-generated experiment outputs. This directory is ignored and keeps the existing `artifacts/publication/` semantics.

## Research And Experiments

- `experiments/`: historical experiment logbook and run notes.
- `research/`: research findings and decision notes.

## Paper

- `paper/manuscript/`: TeX manuscript source, appendix, references, class file, and tracked figure PDFs.
- `paper/manuscript/figures/`: tracked figure PDFs referenced by the current manuscript.
- `paper/response/round-01/`: first formal journal response round. Internal automated-review rounds are not journal rounds.
- `paper/snapshots/`: working PDF snapshots that have not been proven to be frozen journal-submission files.
- `paper/submission/`: files actually submitted to the journal, organized by true submission stage.

## Trellis

Stable project guidance may live under `.trellis/spec/`. Runtime state, cache, workspace journals, and generated scripts remain local-only.
