# Round 2 Figure Gallery

## Part I - Main manuscript candidates

### Manual Fig. 1 candidate (Fig1 manual_fig1_candidate)
- Source files: `paper/manuscript/figures/round2_candidate/manual/fig1.pdf` (c46c7c3c2eaffcd316843e5eda70cc28ec03b64acb2919ab70054c74f26a2332)
- Claim boundary: Manual Fig. 1 is included for visual QA and gallery context only; the automated workflow must not edit or re-export it.
- Revision note: User-supplied manual Fig. 1 candidate; included for gallery and QA only, with no automatic edits.
- Unresolved visual concerns: None

### Fig. 2 simplified candidate (Fig2 serialized_surrogate_query_process)
- Source files: `paper/manuscript/figures/source/fig2_serialized_search_round2.tex` (6d002851d50b588fb8d22b33d72f5b4c013fb18469d9576c00b9985e846ad4dc), `paper/manuscript/figures/source/round2_figure_style.tex` (396b9da979eef3e79363bc0d12277e8af916ab4905ebba39c2e6c65eb2148354)
- Claim boundary: The sequence represents repeated black-box queries to a static guarded surrogate and not physical-time evolution.
- Revision note: Simplified the serialized surrogate-query episode into four process nodes plus a compact episode strip.
- Unresolved visual concerns: None

### Fig. 3 simplified candidate (Fig3 actor_critic_architecture_and_learning)
- Source files: `paper/manuscript/figures/source/fig3_actor_critic_round2.tex` (4996867a2238c06475686b04a8c938b834b6ddff344912aa4ebcade633a55911), `paper/manuscript/figures/source/round2_figure_style.tex` (396b9da979eef3e79363bc0d12277e8af916ab4905ebba39c2e6c65eb2148354)
- Claim boundary: The figure documents DDPG learning mechanics only and does not describe the surrogate environment or episode sequence.
- Revision note: Simplified the actor-critic diagram into network architecture and four learning-equation groups.
- Unresolved visual concerns: None

### Main Fig. 4 (M1 data_and_surrogate_validation)
- Source files: `descriptor_coverage.csv` (cefa7377e3fb108e17439db483bf0d3e25758c7fcd2a1aaa5079b49caf4af098), `surrogate_parity_mean_predictions.csv` (5837b986d33d904cf641a57584e85718cbc64d64bffdb762665adf8886fe8c07)
- Claim boundary: Supports descriptor-space coverage and analytic-target surrogate fidelity only.
- Revision note: Carried forward from the canonical round-2 candidate set without a layout change in this task.
- Unresolved visual concerns: None

### Main Fig. 5 (M2 surrogate_robustness)
- Source files: `surrogate_validation_regimes.csv` (4a35f8c39487082357292a0d50ae4b7f4061b284a820e5ac298598e89528b016)
- Claim boundary: All panels remain analytic-target surrogate-validation evidence, not physical validation.
- Revision note: Carried forward from the canonical round-2 candidate set without a layout change in this task.
- Unresolved visual concerns: None

### Main Fig. 6 (M3 ddpg_training_dynamics)
- Source files: `ddpg_training_curves_summary.csv` (563e9c7b4738aab80c6e24ac9671eb8955e031cae015f776c96a037cd6242c03), `ddpg_seed_diagnostics.csv` (ceace85452fba6a4f6b3169870626bd0830ef359db49c24750af607744af202f)
- Claim boundary: Training dynamics describe serialized surrogate-query search only.
- Revision note: Carried forward from the canonical round-2 candidate set without a layout change in this task.
- Unresolved visual concerns: None

### Main Fig. 7 (M4 benchmark_fairness)
- Source files: `benchmark_utility.csv` (c8dafd40dd57df9bdf21fbf5d5f8af3a798b1f82a03b414c9b1fe0e5c9be13de), `benchmark_equal_size_20.csv` (f60b07256e7b64ea2f3197c7f428c999857666ae8cb04558971f8269092c7168), `benchmark_output_contract_counts.csv` (83c911df9c51878df891b3ab7ca635f3a6e1822fede8caab09e51013cd756c65)
- Claim boundary: Equal-size metrics remain benchmark-reference-v2 evidence only; broader contract and diagnostic baselines are deferred to Supplementary Information.
- Revision note: Simplified the main benchmark figure to fixed-domain utility plus matched-size HV/IGD for DDPG and NSGA-II only.
- Unresolved visual concerns: None

### Main Fig. 8 (M5 feasible_projection)
- Source files: `feasible_projection_summary.csv` (16f00a0bdf86e0a5e726af8826cbce84445959d8635b1a52e7ebf8b2a0b7e9bd), `feasible_projection_metrics.csv` (1132a919d3db4aae0454a314b4a981f6045762acf5b8c48a111426ab4f99b75b)
- Claim boundary: Projection remains a nearest-neighbour representation diagnostic rather than physical validation.
- Revision note: Reduced the projection figure to representation compression plus projection-distance diagnostics for the main-text comparison.
- Unresolved visual concerns: None

### Main Fig. 9 (M6 physical_cross_model_stress_test)
- Source files: `physical_direct_cases.csv` (0403f94ea9bd6d4d089b930723e08f860cdf3da726f99dcaaee038fa6b7f39d3), `physical_stress_metrics.csv` (cf9ec8e791d55efe5edafe4a1f1cd779523801e0d6bce8bbbdcadd03bd7f4303)
- Claim boundary: This figure is limited to the direct-case physics-based cross-model stress test and does not support optimizer-superiority claims.
- Revision note: Kept the same 18 direct cases while tightening axis wording, typography, marker size, and statistics placement.
- Unresolved visual concerns: None

### Main Fig. 10 (M7 cross_climate_sensitivity)
- Source files: `climate_case_results.csv` (f624253730cdf63ce60f4e774516b4854f0c3dc6a33c4b2580a02b5df4bc3536), `climate_summary.csv` (7a03bcf3714f398a72ad99f172320caac79dc865bc8989fc65d53a8262feb1e6), `climate_rank_stability.csv` (ebd3f1d2ec549a98ab0173e104910fbcdf88df91a899446f8466dc6de30680dc)
- Claim boundary: This figure is a limited four-block cross-climate physical sensitivity analysis only.
- Revision note: Kept the same climate data while switching to the muted climate palette and a zero-centered low-saturation heatmap.
- Palette note: Beijing `#539F97`, Guangzhou `#6C7AAD`, Harbin `#BE7A7A`.
- Unresolved visual concerns: None

## Part II - Supplementary Information candidates

### Supplementary Fig. S1 (S1 A1_descriptor_distributions)
- Source files: `descriptor_coverage.csv` (cefa7377e3fb108e17439db483bf0d3e25758c7fcd2a1aaa5079b49caf4af098)
- Claim boundary: Descriptive coverage diagnostics only.
- Revision note: Carried forward from the canonical round-2 candidate set and relabeled for the Supplementary Information split.
- Unresolved visual concerns: None

### Supplementary Fig. S2 (S2 A2_residual_diagnostics)
- Source files: `surrogate_parity_mean_predictions.csv` (5837b986d33d904cf641a57584e85718cbc64d64bffdb762665adf8886fe8c07)
- Claim boundary: Residual diagnostics describe analytic-target surrogate error only.
- Revision note: Carried forward from the canonical round-2 candidate set and relabeled for the Supplementary Information split.
- Unresolved visual concerns: None

### Supplementary Fig. S3 (S3 A3_scale_study)
- Source files: `scale_study.csv` (87120cdcbdb9cdd09560f4778f0351398933affa460de2884913fdb6554e9bf7)
- Claim boundary: Scale-study rows support the surrogate-selection rationale only.
- Revision note: Carried forward from the canonical round-2 candidate set and relabeled for the Supplementary Information split.
- Unresolved visual concerns: None

### Supplementary Fig. S4 (S4 B1_seed_diagnostics)
- Source files: `ddpg_seed_diagnostics.csv` (ceace85452fba6a4f6b3169870626bd0830ef359db49c24750af607744af202f)
- Claim boundary: Seed diagnostics remain Supplementary Information training evidence only.
- Revision note: Carried forward from the canonical round-2 candidate set and relabeled for the Supplementary Information split.
- Unresolved visual concerns: None

### Supplementary Fig. S5 (S5 B2_morphology_signatures)
- Source files: `morphology_signatures.csv` (aa6df073123507e16a287b9407fb657b2875118f2be839926c4e4b4009287acd)
- Claim boundary: Descriptor signatures are descriptive summaries, not stable design rules.
- Revision note: Carried forward from the canonical round-2 candidate set and relabeled for the Supplementary Information split.
- Unresolved visual concerns: None

### Supplementary Fig. S6 (S6 B3_hv_ceiling_diagnostics)
- Source files: `benchmark_hv_ceiling.csv` (fa862502f1966a89702ac730992db9f642cf304b3a6949d879a9207b370c4f4a)
- Claim boundary: HV ceiling and tuple-collapse panels remain supplementary diagnostics only.
- Revision note: Moved ceiling and duplicate-collapse diagnostics into a readable two-panel Supplementary Information layout with short labels.
- Unresolved visual concerns: None

### Supplementary Fig. S7 (S7 B4_optimizer_linked_gap_decomposition)
- Source files: `optimizer_linked_physical_gaps.csv` (79d6291c83058eab43693346035444a30e2b97f3bac28ed993a11affe53e3383)
- Claim boundary: These optimizer-linked cases are representative bridge diagnostics rather than a global optimizer benchmark.
- Revision note: Rebuilt the optimizer-linked bridge diagnostics as horizontal gap bars with an out-of-panel legend and short case labels.
- Unresolved visual concerns: None

### Supplementary Fig. S8 (S8 B5_nonlinear_response_profiles)
- Source files: `scale_study.csv` (87120cdcbdb9cdd09560f4778f0351398933affa460de2884913fdb6554e9bf7)
- Claim boundary: Selected surrogate response profiles illustrate local trends only.
- Revision note: Carried forward from the canonical round-2 candidate set and relabeled for the Supplementary Information split.
- Unresolved visual concerns: None

### Supplementary Fig. S9 (S9 B6_climate_case_detail)
- Source files: `climate_case_results.csv` (f624253730cdf63ce60f4e774516b4854f0c3dc6a33c4b2580a02b5df4bc3536)
- Claim boundary: Case-level climate details remain limited to four blocks and three additional climates.
- Revision note: Carried forward from the canonical round-2 candidate set and relabeled for the Supplementary Information split.
- Unresolved visual concerns: None
