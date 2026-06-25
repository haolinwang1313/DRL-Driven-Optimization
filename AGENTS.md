# AGENTS.md

<!-- BEGIN TRELLIS MANAGED BLOCK -->
Trellis status: project-local `.trellis/spec/` guidance is available; runtime state, workspace journals, cache files, and generated Trellis scripts remain local-only unless explicitly promoted.
<!-- END TRELLIS MANAGED BLOCK -->

## Project Objective

This repository supports an `Applied Energy` major-revision package for a methodological cautionary study on surrogate-conditioned benchmark fragility in block-scale urban energy design.

Keep the paper framed as a surrogate-assisted benchmarking and design-support study, not as a universal DRL-superiority claim. Comparative statements must remain bounded by surrogate reliability, benchmark fairness, and the current validation mode.

## Canonical Entrypoints

- Project map: `PROJECT.yaml`
- Runtime config: `configs/revision.yaml`
- Python package: `paper_repro/`
- Manuscript source: `paper/manuscript/manuscript.tex`
- Appendix source: `paper/manuscript/appendix.tex`
- Reviewer response: `paper/response/round-01/letter.tex`
- Revision tracker: `paper/response/round-01/tracker/revision-tracker.json`
- Response status: `paper/response/round-01/status.md`
- Artifact root: `artifacts/publication`
- Layout guide: `docs/repository-layout.md`
- Reproducibility guide: `docs/reproducibility.md`

## Editable Scope

- Code: `paper_repro/`, `tools/`, `tests/`
- Config: `configs/*.yaml`, except local overrides
- Current paper source: `paper/manuscript/`
- Current response package: `paper/response/round-01/`
- Project docs: `README.md`, `docs/`, `PROJECT.yaml`, this file

Generated artifacts under `artifacts/` stay out of Git. Current work-in-progress PDFs belong under `paper/snapshots/`; only confirmed journal-submitted files belong under `paper/submission/`.

## Research Integrity Rules

Do not change scientific results by restructuring the repository. In routine maintenance, do not modify numerical algorithms, random seeds, model hyperparameters, optimization budgets, imported result artifacts, manuscript scientific claims, or substantive reviewer-response content.

Do not invent empirical validation, physical-stack closure, server results, or submission history. If evidence is missing, record it as missing or unverified.

Do not commit secrets, `.env` files, SSH keys, API keys, `server.local.yaml`, `.venv/`, `artifacts/`, `.agents/`, `.codex/`, or Trellis runtime/workspace state.

## Default Commands

Run from the repository root:

```bash
uv run pytest -q
uv run python -m paper_repro.cli --help
uv run python -m compileall paper_repro tools
uv run python -c "from paper_repro.config import Config; Config.from_yaml('configs/revision.yaml'); print('config ok')"
```

For TeX checks, compile into ignored build directories:

```bash
cd paper/manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build manuscript.tex

cd ../response/round-01
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build letter.tex
```

If a task requires unavailable remote compute, physical simulation software, missing imported artifacts, or unsourced claims, stop and report the blocker instead of guessing.
