# Migration Map

## Baseline

- Original HEAD: `f780136d413a78c4c48205ae7dcbe9df2cd7c33a`
- Origin main at fetch: `f780136d413a78c4c48205ae7dcbe9df2cd7c33a`
- Branch: `chore/restructure-revision-workspace`
- Audit time: `2026-06-25T10:54:51.1473180+08:00`
- Pre-migration tracked-file count: 79

## Pre-Migration Tracked Roots

- Root state files: `AUTO_REVIEW.md`, `EXPERIMENT_LOG.md`, `PAPER_IMPROVEMENT_LOG.md`, `PAPER_IMPROVEMENT_STATE.json`, `REVIEWER_MEMORY.md`, `REVIEW_STATE.json`, `REVISION_TRACKER.md`, `REVISION_TRACKER.json`
- Code and config: `configs/`, `paper_repro/`, `tools/`, `pyproject.toml`, `uv.lock`
- Paper and response: `elsarticle/`, `letter-2-reviewers/`

## Ignored But Existing Key Local Content

- `.agents/`, `.codex/`, `.trellis/`, `.venv/`
- `artifacts/`
- `configs/revision.local.yaml`
- `server.local.yaml`
- LaTeX build side files such as `*.aux`, `*.log`, `*.fls`, and `*.fdb_latexmk`
- `paper02_repro.egg-info/`

## PDF Baseline Hashes

- `elsarticle/manuscript.pdf` -> `paper/snapshots/2026-06-25-synced/manuscript.pdf`
  - SHA256: `9d982224e7267a29a5a674ce4d528d690498c67e2472e475f72c2f94056da5df`
- `letter-2-reviewers/letter.pdf` -> `paper/snapshots/2026-06-25-synced/letter.pdf`
  - SHA256: `6432dbfb761e31cc60fac4ca311c55cd122c4862979e07d3dad0b1b7cad48aee`
- Baseline copies for text comparison were saved outside the repository at `C:\Users\13262\AppData\Local\Temp\paper02-restructure-baseline-f780136`.

## Restored From Git History

- From `e2668085a0044f910f465619d1bd581f28c08155`:
  - `tests/conftest.py`
  - `tests/test_metrics.py`
  - `tests/test_morphology.py`
  - `tests/test_simulation_scale_study.py`
  - `tests/test_surrogate_selection.py`
- From `1feb553e98db3013e8eba937e530d1bb9311d0a9`:
  - `tools/build_manuscript_result_figures.py`
- From existing local historical tool state, retained because current `fig13.pdf` is the four-method physical-probe figure:
  - `tools/build_four_method_physprobe_figure.py`

`tools/build_publication_result_figures.py` was not restored because it was not present in the inspected history path and would duplicate the restored canonical figure workflow.

## Path Mapping

| Old path | New path |
|---|---|
| `elsarticle/` | `paper/manuscript/` |
| `elsarticle/fig*.pdf` | `paper/manuscript/figures/fig*.pdf` |
| `elsarticle/manuscript.pdf` | `paper/snapshots/2026-06-25-synced/manuscript.pdf` |
| `letter-2-reviewers/` | `paper/response/round-01/` |
| `letter-2-reviewers/letter.pdf` | `paper/snapshots/2026-06-25-synced/letter.pdf` |
| `REVISION_TRACKER.md` | `paper/response/round-01/tracker/revision-tracker.md` |
| `REVISION_TRACKER.json` | `paper/response/round-01/tracker/revision-tracker.json` |
| `AUTO_REVIEW.md` | `paper/response/round-01/reviews/automated-review-log.md` |
| `REVIEWER_MEMORY.md` | `paper/response/round-01/reviews/reviewer-memory.md` |
| `PAPER_IMPROVEMENT_LOG.md` | `paper/response/round-01/logs/paper-improvement-log.md` |
| `PAPER_IMPROVEMENT_STATE.json` | `paper/response/round-01/state/paper-improvement-state.json` |
| `REVIEW_STATE.json` | `paper/response/round-01/state/review-state.json` |
| `EXPERIMENT_LOG.md` | `experiments/logbook.md` |
| `findings.md` | `research/findings.md` |
| `Dataset.xlsx` | `data/external/benchmark/dataset.xlsx` |
| `manuscript1105_clean.pdf` | `paper/submission/initial/manuscript-original.pdf` |
| `Supplementary Information.pdf` | `paper/submission/initial/supplementary-information-original.pdf` |

`Dataset.xlsx`, `manuscript1105_clean.pdf`, and `Supplementary Information.pdf` are not present in the current worktree, so no binary files were moved for those paths.

## Figure Hashes After Migration

```text
509d77838d0314277cbce5ddae25c7ca39c7859a9ec42193e47d05067b7d6446  paper/manuscript/figures/fig1.pdf
9a05c202b0757c51cd3945cbff52bb5dcce50ca81f17d992536ce782a730f098  paper/manuscript/figures/fig2.pdf
3cae0393fad59dd15229ae0df59545d4fa2f6ba6e30d710701cec9d4297bf952  paper/manuscript/figures/fig3.pdf
bda4f4e5bde21dceb97af09ebe4938b4d952406582211b9c0a3c6d83d004e7b0  paper/manuscript/figures/fig4.pdf
f47b8d9082acfa95021e9b359d95817869f5ff2ec394ddc21dd2cc368284e21c  paper/manuscript/figures/fig5.pdf
6cc65560b84e2d6e59924982e5c587816ca7fea60d181a8638ab685387a1461a  paper/manuscript/figures/fig6.pdf
ec9039dcfcdeb98eb1c0db1059b7c37d8db8b09a02a75c8c69d558b6bb6f8d42  paper/manuscript/figures/fig7.pdf
12b559f68dfccbd3686060c5928ff990a81f3278578bcbd02090c80fc52a71ce  paper/manuscript/figures/fig8.pdf
aed4f7877b86efec667d22f607ed9381786539c74591d9bec544ca05a74ea291  paper/manuscript/figures/fig9.pdf
ed0d52af2b6a5b8a2f7f8f1a0b649e15e86fcd4c0020cfe96a84b379c795b359  paper/manuscript/figures/fig10.pdf
34431aa639c911e739135b3208472c1e60c5723adb188d565b965d5ca01324e3  paper/manuscript/figures/fig11.pdf
0565030ba4a7807ae11bccbc834a941e040a3aa4ce1bc3513e30fc9545f395b2  paper/manuscript/figures/fig12.pdf
2d0b3d69df7c531e78be1290fe29e5b17bfe55b99a98de4ba02490c07d686233  paper/manuscript/figures/fig13.pdf
```

No exact duplicate target files were found by SHA256 among the searched target names.

## Validation Notes

The pre-migration baseline check passed:

- `uv run pytest -q`: 8 passed
- `uv run python -m paper_repro.cli --help`: passed
- `uv run python -m compileall paper_repro tools`: passed

Post-migration validation results are recorded in the final task report.
