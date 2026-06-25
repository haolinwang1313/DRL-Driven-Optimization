# Repository Layout Spec

The canonical repository layout is documented in `docs/repository-layout.md` and summarized here for Trellis-aware agents.

- Keep `paper_repro/` at the repository root; do not rename it or migrate to `src/` during the current revision.
- Keep reusable code in `paper_repro/`, helper entrypoints in `tools/`, tests in `tests/`, and experiment declarations in `configs/`.
- Keep manuscript source in `paper/manuscript/` and formal reviewer-response material in `paper/response/<journal-round>/`.
- Keep generated experiment outputs under ignored `artifacts/` paths.
- Keep local benchmark input data under `data/external/benchmark/`, with metadata in `data/catalog.yaml`.
- Keep work-in-progress PDFs in `paper/snapshots/`; only confirmed submitted files belong in `paper/submission/`.
- Keep Trellis runtime, cache, workspace, and generated scripts out of Git unless explicitly promoted.
