# Round 2 Figure Plan

## Main manuscript lock
- Manual Fig. 1 is user supplied and is included for gallery/QA only; the automated workflow must not edit or re-export it.
- Fig. 2 is the round3 workflow flowchart candidate.
- Fig. 3 is the round3 DDPG learning architecture candidate.
- Superseded round2 manual candidates remain on disk for audit but are not preferred gallery entries.
- Main Fig. 4-10 and Supplementary Fig. S1-S9 are rebuilt through the round-2 figure builder.

### Manual Fig. 1 candidate (Fig1 manual_fig1_candidate)
- Source files: paper/manuscript/figures/round2_candidate/manual/fig1.pdf
- Claim boundary: Manual Fig. 1 is included for visual QA and gallery context only; the automated workflow must not edit or re-export it.
- Revision note: User-supplied manual Fig. 1 candidate; included for gallery and QA only, with no automatic edits.

### Fig. 2 workflow round3 candidate (Fig2 workflow_round3)
- Source files: paper/manuscript/figures/source/fig2_workflow_round3.tex, paper/manuscript/figures/source/round3_figure_style.tex
- Claim boundary: The figure describes the DDPG-based surrogate-search workflow and single-query surrogate interaction; actor-critic learning mechanics are reserved for Fig. 3.
- Revision note: Redrawn as a round3 workflow flowchart with a separate single-query callout.

### Fig. 3 DDPG architecture round3 candidate (Fig3 ddpg_architecture_round3)
- Source files: paper/manuscript/figures/source/fig3_ddpg_architecture_round3.tex, paper/manuscript/figures/source/round3_figure_style.tex
- Claim boundary: The figure documents the DDPG learning architecture used in surrogate-assisted optimization and does not describe workflow termination or episode sequencing.
- Revision note: Redrawn as a round3 DDPG architecture diagram with interaction, replay, online, target, and update blocks.

### Main Fig. 4 (M1 data_and_surrogate_validation)
- Source files: descriptor_coverage.csv, surrogate_parity_mean_predictions.csv
- Claim boundary: Supports descriptor-space coverage and analytic-target surrogate fidelity only.
- Revision note: Carried forward from the canonical round-2 candidate set without a layout change in this task.

### Main Fig. 5 (M2 surrogate_robustness)
- Source files: surrogate_validation_regimes.csv
- Claim boundary: All panels remain analytic-target surrogate-validation evidence, not physical validation.
- Revision note: Carried forward from the canonical round-2 candidate set without a layout change in this task.

### Main Fig. 6 (M3 ddpg_training_dynamics)
- Source files: ddpg_training_curves_summary.csv, ddpg_seed_diagnostics.csv
- Claim boundary: Training dynamics describe serialized surrogate-query search only.
- Revision note: Carried forward from the canonical round-2 candidate set without a layout change in this task.

### Main Fig. 7 (M4 benchmark_fairness)
- Source files: benchmark_utility.csv, benchmark_equal_size_20.csv, benchmark_output_contract_counts.csv
- Claim boundary: Equal-size metrics remain benchmark-reference-v2 evidence only; broader contract and diagnostic baselines are deferred to Supplementary Information.
- Revision note: Simplified the main benchmark figure to fixed-domain utility plus matched-size HV/IGD for DDPG and NSGA-II only.

### Main Fig. 8 (M5 feasible_projection)
- Source files: feasible_projection_summary.csv, feasible_projection_metrics.csv
- Claim boundary: Projection remains a nearest-neighbour representation diagnostic rather than physical validation.
- Revision note: Reduced the projection figure to representation compression plus projection-distance diagnostics for the main-text comparison.

### Main Fig. 9 (M6 physical_cross_model_stress_test)
- Source files: physical_direct_cases.csv, physical_stress_metrics.csv
- Claim boundary: This figure is limited to the direct-case physics-based cross-model stress test and does not support optimizer-superiority claims.
- Revision note: Kept the same 18 direct cases while tightening axis wording, typography, marker size, and statistics placement.

### Main Fig. 10 (M7 cross_climate_sensitivity)
- Source files: climate_case_results.csv, climate_summary.csv, climate_rank_stability.csv
- Claim boundary: This figure is a limited four-block cross-climate physical sensitivity analysis only.
- Revision note: Kept the same climate data while switching to the muted climate palette and a zero-centered low-saturation heatmap.

## Supplementary Information candidates

### Supplementary Fig. S1 (S1 A1_descriptor_distributions)
- Source files: descriptor_coverage.csv
- Claim boundary: Descriptive coverage diagnostics only.
- Revision note: Carried forward from the canonical round-2 candidate set and relabeled for the Supplementary Information split.

### Supplementary Fig. S2 (S2 A2_residual_diagnostics)
- Source files: surrogate_parity_mean_predictions.csv
- Claim boundary: Residual diagnostics describe analytic-target surrogate error only.
- Revision note: Carried forward from the canonical round-2 candidate set and relabeled for the Supplementary Information split.

### Supplementary Fig. S3 (S3 A3_scale_study)
- Source files: scale_study.csv
- Claim boundary: Scale-study rows support the surrogate-selection rationale only.
- Revision note: Carried forward from the canonical round-2 candidate set and relabeled for the Supplementary Information split.

### Supplementary Fig. S4 (S4 B1_seed_diagnostics)
- Source files: ddpg_seed_diagnostics.csv
- Claim boundary: Seed diagnostics remain Supplementary Information training evidence only.
- Revision note: Carried forward from the canonical round-2 candidate set and relabeled for the Supplementary Information split.

### Supplementary Fig. S5 (S5 B2_morphology_signatures)
- Source files: morphology_signatures.csv
- Claim boundary: Descriptor signatures are descriptive summaries, not stable design rules.
- Revision note: Carried forward from the canonical round-2 candidate set and relabeled for the Supplementary Information split.

### Supplementary Fig. S6 (S6 B3_hv_ceiling_diagnostics)
- Source files: benchmark_hv_ceiling.csv
- Claim boundary: HV ceiling and tuple-collapse panels remain supplementary diagnostics only.
- Revision note: Moved ceiling and duplicate-collapse diagnostics into a readable two-panel Supplementary Information layout with short labels.

### Supplementary Fig. S7 (S7 B4_optimizer_linked_gap_decomposition)
- Source files: optimizer_linked_physical_gaps.csv
- Claim boundary: These optimizer-linked cases are representative bridge diagnostics rather than a global optimizer benchmark.
- Revision note: Rebuilt the optimizer-linked bridge diagnostics as horizontal gap bars with an out-of-panel legend and short case labels.

### Supplementary Fig. S8 (S8 B5_nonlinear_response_profiles)
- Source files: scale_study.csv
- Claim boundary: Selected surrogate response profiles illustrate local trends only.
- Revision note: Carried forward from the canonical round-2 candidate set and relabeled for the Supplementary Information split.

### Supplementary Fig. S9 (S9 B6_climate_case_detail)
- Source files: climate_case_results.csv
- Claim boundary: Case-level climate details remain limited to four blocks and three additional climates.
- Revision note: Carried forward from the canonical round-2 candidate set and relabeled for the Supplementary Information split.
