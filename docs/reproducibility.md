# Reproducibility

This note defines what can be verified quickly from a clean clone and what requires local inputs, expensive experiments, server access, or physical simulation tools.

## Clean Clone Setup

```bash
uv sync
uv run pytest -q
uv run python -m paper_repro.cli --help
uv run python -m compileall paper_repro tools
uv run python -c "from paper_repro.config import Config; Config.from_yaml('configs/revision.yaml'); print('config ok')"
```

## Local Inputs

- Benchmark spreadsheet: `data/external/benchmark/dataset.xlsx`
- Optional original submission PDFs: `paper/submission/initial/`
- Publication artifacts: `artifacts/publication/`
- Optional local server config: `server.local.yaml`

The benchmark spreadsheet and local server config are ignored by Git.

## Fast Checks

- Unit tests: `uv run pytest -q`
- CLI import/help: `uv run python -m paper_repro.cli --help`
- Syntax/bytecode check: `uv run python -m compileall paper_repro tools`
- Config load: `uv run python -c "from paper_repro.config import Config; Config.from_yaml('configs/revision.yaml'); print('config ok')"`
- Figure script help: `uv run python tools/build_manuscript_result_figures.py --help`

## Expensive Or Environment-Dependent Commands

The following may run long experiments or require local artifacts:

- `build-dataset`
- `select-surrogate`
- `run-optimizers`
- DDPG, NSGA-II, CMA-ES, and random-search batch scripts
- physical-stack or remote synchronization workflows

Do not run them during low-risk structure work unless the task explicitly requests it.

## TeX

If `latexmk` is installed:

```bash
cd paper/manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build manuscript.tex

cd ../response/round-01
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build letter.tex
```

If `latexmk` is unavailable, record the environment limitation and continue Python-side verification.

## Migration Boundary

The 2026-06-25 structure migration did not rerun expensive experiments, did not connect to remote servers, and did not change scientific algorithms, hyperparameters, optimization budgets, manuscript claims, or substantive reviewer-response content.
