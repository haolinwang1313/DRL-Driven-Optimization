# AGENTS.md

<!-- BEGIN TRELLIS MANAGED BLOCK -->
Trellis status: no `.trellis/` directory, `workflow.md`, or `spec/` files are present in the current worktree as of 2026-04-06. Preserve this block for future Trellis-managed updates. If Trellis files are added later, update this block in place rather than replacing the rest of this document.
<!-- END TRELLIS MANAGED BLOCK -->

## Project Objective

Drive this repository toward a reject-and-resubmit package for `Applied Energy` built around a methodological cautionary-study narrative, not a universal-DRL-superiority narrative.

Current paper objective:
- position the work as a surrogate-conditioned benchmarking and design-support study for block-scale urban morphology optimization;
- keep all claims bounded by surrogate reliability, benchmark fairness, and the current lack of physical-stack / empirical external validation;
- improve the manuscript, experiments, diagnostics, and reviewer-facing evidence without inventing unavailable infrastructure or results.

## Read First

Read these files before making substantial changes:
- `README.md`
- `elsarticle/manuscript.tex`
- `elsarticle/appendix.tex`
- `REVISION_TRACKER.md`
- `REVISION_TRACKER.json`
- `REVIEWER_MEMORY.md`
- `PAPER_IMPROVEMENT_LOG.md`
- `paper_repro/reviewer.py`
- `configs/revision.yaml`

Read as needed for task-specific work:
- `letter-2-reviewers/revision-notes/reviewer1.tex`
- `letter-2-reviewers/revision-notes/reviewer2.tex`
- `tools/*.py`
- `tools/*.sh`
- `tests/`
- `initial_paper/manuscript1105_clean.pdf`
- `initial_paper/Supplementary Information.pdf`

## Main Code Root

Primary implementation and paper roots:
- `paper_repro/`: main pipeline, simulation fallback, surrogate training, optimization, publication diagnostics, reviewer utilities
- `configs/`: runtime and publication-mode configuration
- `tools/`: batch helpers for remote runs, figure rebuilds, diagnostics, and result merging
- `elsarticle/`: manuscript source, appendix, references, figures, journal assets
- `tests/`: regression checks for metrics, morphology logic, simulation scaling, and surrogate selection

## Editable Scope

Files and directories that are normally safe to modify for project work:
- `paper_repro/`
- `configs/`
- `tools/`
- `tests/`
- `elsarticle/manuscript.tex`
- `elsarticle/appendix.tex`
- `elsarticle/references.tex`
- `letter-2-reviewers/` only when the task explicitly includes reviewer-response package updates
- root-level project state and planning files such as `REVISION_TRACKER.*`, `REVIEWER_MEMORY.md`, `PAPER_IMPROVEMENT_LOG.md`, and this `AGENTS.md`

Generated files may be regenerated, but should not be hand-edited unless the task explicitly requires it:
- figures under `elsarticle/fig/`
- generated PDFs
- artifact summaries under `artifacts/` when present

## Non-Editable Scope

Do not modify these by default:
- `initial_paper/`: original source submission artifacts and historical reference PDFs
- `elsarticle/elsarticle.cls`
- `.git/`
- any secrets, SSH keys, or files under user home outside this repo
- global Codex / Claude / MCP configuration such as `~/.codex/config.toml`, global reviewer settings, or global model selection

Treat these as read-only source-of-truth unless the user explicitly asks to rebuild or replace them:
- imported result bundles under `artifacts/publication/imported/` when present
- synced remote result snapshots under `artifacts/server_runs/` when present
- committed historical response-letter PDFs

## Success Metrics

Use these as default project-level success metrics for long-running automation:
- manuscript claims remain aligned with evidence and do not revert to unsupported DRL-superiority language;
- publication-mode pipeline remains centered on `configs/revision.yaml`;
- benchmark evidence is traceable to concrete artifacts, configs, and diagnostics;
- methodological limitations remain explicit in manuscript and reviewer memory;
- new experiments improve the methodological-cautionary story rather than broadening claims beyond evidence;
- every major automation run leaves behind updated state files or artifacts that the next run can inspect.

Task-level success should usually mean one of:
- a reproducible new artifact bundle under `artifacts/publication/`;
- a manuscript revision with consistent text, tables, captions, and tracker updates;
- a reviewer-oriented evidence update recorded in `REVISION_TRACKER.*`, `REVIEWER_MEMORY.md`, or `PAPER_IMPROVEMENT_LOG.md`.

## Stop Criteria

Stop and report instead of guessing when any of the following happens:
- the task requires GPU details, SSH access, remote hosts, or server paths that are not configured in the current worktree;
- the task requires physical-stack validation, but Rhino / Grasshopper / EnergyPlus / Radiance access is unavailable;
- the task would require changing global reviewer wiring, global MCP setup, or global model defaults;
- the task depends on missing imported artifacts under `artifacts/publication/` or missing remote sync outputs;
- the requested conclusion would require inventing missing experimental results, unavailable datasets, or unsourced reviewer claims;
- the task would require editing read-only source-of-truth files listed above.

## Experiment / Runtime Environment

Known project runtime facts from repository configuration:
- Python requirement: `>=3.10`
- Main package: `paper02-repro`
- Core dependencies include `torch`, `optuna`, `pymoo`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `PyYAML`, `openpyxl`
- Default publication config: `configs/revision.yaml`
- Default publication artifact root: `artifacts/publication`
- Default DDPG publication setting: `600` episodes, `40` steps per episode, `20` seeds per scenario
- Default NSGA-II publication setting: `fair_budget`, `20` seeds, `24000` evaluations
- Weather download is enabled in config and targets Dongtai with Nanjing fallback
- Simulation mode is designed to use physical stack if available, but current manuscript and tracker state still indicate reliance on the documented fallback analytic simulator

Execution defaults:
- prefer `python -m paper_repro.cli --config configs/revision.yaml <command>` for pipeline work;
- prefer `uv run python tools/build_manuscript_result_figures.py --compile-manuscript` for rebuilding current manuscript figure set;
- do not assume `artifacts/` is already present in every worktree clone;
- do not assume remote execution is available unless project-local server config is supplied.

## Reviewer Backend Policy

Project-local reviewer wiring discovered during initialization:
- `paper_repro/reviewer.py` is the repo-owned review entrypoint;
- it reads `CLAUDE_REVIEW_API_KEY`, `CLAUDE_REVIEW_BASE_URL`, and `CLAUDE_REVIEW_MODEL` from `~/.claude/.env`;
- it calls an OpenAI-compatible `/chat/completions` endpoint and writes outputs under `artifacts/.../reports/reviews/`;
- the current Codex runtime also exposes `claude-review` MCP tools.

Policy:
- treat the existing reviewer environment as fixed for this repo;
- do not switch reviewer backend from within this project;
- do not modify global MCP wiring, global reviewer settings, or global model defaults;
- if ARIS needs review, use the existing repo-local reviewer path or the already-available runtime MCP path, but do not introduce a new reviewer stack.

Current project convention check:
- no project-local evidence of `Kimi via llm-chat` was found in this worktree;
- no `.trellis/` reviewer configuration was found in this worktree.

## Automation Defaults

Default ARIS-style execution assumptions for this repo:
- operate in single-run batch mode, not interactive multi-turn mode;
- start from the current repo state and inspect trackers before changing manuscript or code;
- prefer small, resumable units of work that update explicit state files;
- keep the publication pipeline anchored to `configs/revision.yaml` unless the task explicitly targets another config;
- update project-local state files after meaningful progress so the next batch run can resume cleanly;
- do not clean or rewrite generated result directories unless the task explicitly requires regeneration;
- if a task depends on remote compute, stop after preparing commands or scripts unless valid project-local remote config is present.

Suggested default run order for long workflows:
1. inspect `REVISION_TRACKER.*`, `REVIEWER_MEMORY.md`, and `PAPER_IMPROVEMENT_LOG.md`;
2. inspect relevant manuscript sections and config;
3. run the smallest pipeline step that produces the missing evidence;
4. sync or summarize outputs into project state files;
5. only then revise manuscript text that depends on those outputs.

## Output / State Files

Primary state files already in use:
- `REVISION_TRACKER.md`
- `REVISION_TRACKER.json`
- `REVIEWER_MEMORY.md`
- `REVIEW_STATE.json`
- `PAPER_IMPROVEMENT_LOG.md`

Primary paper outputs:
- `elsarticle/manuscript.tex`
- `elsarticle/appendix.tex`
- `elsarticle/manuscript.pdf`
- `letter-2-reviewers/letter.tex`
- `letter-2-reviewers/letter.pdf`

Expected generated artifact roots from project config:
- `artifacts/publication/data`
- `artifacts/publication/models`
- `artifacts/publication/optimization`
- `artifacts/publication/diagnostics`
- `artifacts/publication/reevaluation`
- `artifacts/publication/reports`
- `artifacts/publication/figures`
- `artifacts/publication/imported`

If these artifact directories are absent in a worktree clone, treat that as a missing-input condition rather than silently fabricating replacements.

## Missing Information

Do not guess these. Fill them in only when discovered from project-local files or explicit user input.

- Canonical ARIS workflow phase name for this repo beyond the current revision / resubmission state
- Whether ARIS should treat `paper_repro/reviewer.py` or runtime `claude-review` MCP as the canonical reviewer entrypoint
- Any project-specific reviewer requirement for Kimi via `llm-chat`
- Local GPU model, CUDA availability, VRAM limits, and whether `runtime.device=auto` resolves to CPU or GPU here
- Actual remote SSH host, username, key path, and remote project directories; only `server.local.example.yaml` is present
- Whether a real `server.local.yaml` exists outside the repo and is intended for this worktree
- Availability of Rhino / Grasshopper / Ladybug Tools / EnergyPlus / Radiance for physical-stack validation
- Presence and freshness of `artifacts/publication/` bundles in this worktree clone
- Presence and location of any external empirical validation dataset
- Approved destination journal package requirements if this repo is retargeted beyond the current `Applied Energy` framing
