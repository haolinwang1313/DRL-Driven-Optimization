# Round 2 Figure Gallery

## M1 data_and_surrogate_validation
- Planned manuscript location: Main Fig. 4 candidate
- Source files: `descriptor_coverage.csv` (c9c77a2738ec0cb359d6a6801b67689b8d2c4e46c632d7e21cfbe4fefbfe8897), `surrogate_parity_mean_predictions.csv` (5837b986d33d904cf641a57584e85718cbc64d64bffdb762665adf8886fe8c07)
- Panel descriptions: PCA cumulative explained variance for the 12 morphology descriptors.; EUIt repeated-CV parity using sample-level mean out-of-fold predictions.; EG repeated-CV parity using sample-level mean out-of-fold predictions.; H repeated-CV parity using sample-level mean out-of-fold predictions.
- Claim boundary: Supports descriptor-space coverage and analytic-target surrogate fidelity only.
- Unresolved visual concerns: None

## M2 surrogate_robustness
- Planned manuscript location: Main Fig. 5 candidate
- Source files: `surrogate_validation_regimes.csv` (4a35f8c39487082357292a0d50ae4b7f4061b284a820e5ac298598e89528b016)
- Panel descriptions: nMAE heatmap across repeated CV, leave-one-OSLI-out, outer-shell, and feature-tail regimes.; Spearman heatmap across the same surrogate-validation regimes.
- Claim boundary: All panels remain analytic-target surrogate-validation evidence, not physical validation.
- Unresolved visual concerns: None

## M3 ddpg_training_dynamics
- Planned manuscript location: Main Fig. 6 candidate
- Source files: `ddpg_training_curves_summary.csv` (563e9c7b4738aab80c6e24ac9671eb8955e031cae015f776c96a037cd6242c03), `ddpg_seed_diagnostics.csv` (ceace85452fba6a4f6b3169870626bd0830ef359db49c24750af607744af202f)
- Panel descriptions: Episode cumulative reward across 20 seeds.; Episode-end EUIt across 20 seeds.; Episode-end EG across 20 seeds.; Episode-end H across 20 seeds.
- Claim boundary: Training dynamics describe serialized surrogate-query search only.
- Unresolved visual concerns: None

## M4 benchmark_fairness
- Planned manuscript location: Main Fig. 7 candidate
- Source files: `benchmark_utility.csv` (c8dafd40dd57df9bdf21fbf5d5f8af3a798b1f82a03b414c9b1fe0e5c9be13de), `benchmark_equal_size_20.csv` (f60b07256e7b64ea2f3197c7f428c999857666ae8cb04558971f8269092c7168), `benchmark_output_contract_counts.csv` (83c911df9c51878df891b3ab7ca635f3a6e1822fede8caab09e51013cd756c65)
- Panel descriptions: Fixed-domain post-hoc utility for DDPG and NSGA-II across the three scalarization scenarios.; Equal-size-20 HV with 5–95% intervals under benchmark-reference-v2.; Equal-size-20 IGD with 5–95% intervals under benchmark-reference-v2.; Output-contract asymmetry across retained rows, unique objective tuples, and unique feasible blocks.
- Claim boundary: Equal-size metrics are canonical only under benchmark-reference-v2 and must stay separate from asymmetric full-archive diagnostics.
- Unresolved visual concerns: None

## M5 feasible_projection
- Planned manuscript location: Main Fig. 8 candidate
- Source files: `feasible_projection_summary.csv` (16f00a0bdf86e0a5e726af8826cbce84445959d8635b1a52e7ebf8b2a0b7e9bd), `feasible_projection_metrics.csv` (1132a919d3db4aae0454a314b4a981f6045762acf5b8c48a111426ab4f99b75b)
- Panel descriptions: Projection-distance distribution from descriptor candidates to feasible blocks.; Duplicate-collapse rate with unique feasible-block counts.; HV before and after projection under benchmark-reference-v2.; IGD before and after projection under benchmark-reference-v2.
- Claim boundary: Projection panels are representation-sensitivity diagnostics rather than physical validation.
- Unresolved visual concerns: None

## M6 physical_cross_model_stress_test
- Planned manuscript location: Main Fig. 9 candidate
- Source files: `physical_direct_cases.csv` (0403f94ea9bd6d4d089b930723e08f860cdf3da726f99dcaaee038fa6b7f39d3), `physical_stress_metrics.csv` (cf9ec8e791d55efe5edafe4a1f1cd779523801e0d6bce8bbbdcadd03bd7f4303)
- Panel descriptions: EUIt parity for the 18 direct feasible cases.; Simplified rooftop-PV proxy parity for the 18 direct feasible cases.; January 20 windowsill direct-sun-hours parity for the 18 direct feasible cases.
- Claim boundary: This figure is limited to the direct-case physics-based cross-model stress test and does not support optimizer-superiority claims.
- Unresolved visual concerns: None

## M7 cross_climate_sensitivity
- Planned manuscript location: Main Fig. 10 candidate
- Source files: `climate_case_results.csv` (f624253730cdf63ce60f4e774516b4854f0c3dc6a33c4b2580a02b5df4bc3536), `climate_summary.csv` (7a03bcf3714f398a72ad99f172320caac79dc865bc8989fc65d53a8262feb1e6), `climate_rank_stability.csv` (ebd3f1d2ec549a98ab0173e104910fbcdf88df91a899446f8466dc6de30680dc)
- Panel descriptions: Mean ΔEUIt relative to Dongtai with four-block spread.; Mean ΔEG relative to Dongtai with four-block spread.; Mean ΔH relative to Dongtai with four-block spread.; Rank-stability heatmap across Beijing, Guangzhou, and Harbin.
- Claim boundary: This figure is a limited four-block cross-climate physical sensitivity analysis only.
- Unresolved visual concerns: None

## A1 A1_descriptor_distributions
- Planned manuscript location: Appendix Fig. A1 candidate
- Source files: `descriptor_coverage.csv` (c9c77a2738ec0cb359d6a6801b67689b8d2c4e46c632d7e21cfbe4fefbfe8897)
- Panel descriptions: Descriptor interquartile summaries for the 12 morphology descriptors.; OSLI frequency distribution.; Normalized nearest-neighbor distance distribution.
- Claim boundary: Descriptive coverage diagnostics only.
- Unresolved visual concerns: None

## A2 A2_residual_diagnostics
- Planned manuscript location: Appendix Fig. A2 candidate
- Source files: `surrogate_parity_mean_predictions.csv` (5837b986d33d904cf641a57584e85718cbc64d64bffdb762665adf8886fe8c07)
- Panel descriptions: EUIt residual distribution.; EG residual distribution.; H residual distribution.
- Claim boundary: Residual diagnostics describe analytic-target surrogate error only.
- Unresolved visual concerns: None

## A3 A3_scale_study
- Planned manuscript location: Appendix Fig. A3 candidate
- Source files: `scale_study.csv` (87120cdcbdb9cdd09560f4778f0351398933affa460de2884913fdb6554e9bf7)
- Panel descriptions: Mean target nMAE across dataset scales.; Mean tail nMAE across dataset scales.; Mean R² across dataset scales.; Selection objective across dataset scales.
- Claim boundary: Scale-study rows support the surrogate-selection rationale only.
- Unresolved visual concerns: None

## B1 B1_seed_diagnostics
- Planned manuscript location: Appendix Fig. B1 candidate
- Source files: `ddpg_seed_diagnostics.csv` (ceace85452fba6a4f6b3169870626bd0830ef359db49c24750af607744af202f)
- Panel descriptions: Best reward by scenario.; Final reward by scenario.; Plateau episode by scenario.; Best-to-final regression ratio by scenario.
- Claim boundary: Seed diagnostics are appendix-only training evidence.
- Unresolved visual concerns: None

## B2 B2_morphology_signatures
- Planned manuscript location: Appendix Fig. B2 candidate
- Source files: `morphology_signatures.csv` (aa6df073123507e16a287b9407fb657b2875118f2be839926c4e4b4009287acd)
- Panel descriptions: Median morphology descriptor signatures for representative retained-output groups.
- Claim boundary: Descriptor signatures are descriptive summaries, not stable design rules.
- Unresolved visual concerns: None

## B3 B3_hv_ceiling_diagnostics
- Planned manuscript location: Appendix Fig. B3 candidate
- Source files: `benchmark_hv_ceiling.csv` (fa862502f1966a89702ac730992db9f642cf304b3a6949d879a9207b370c4f4a)
- Panel descriptions: HV fraction of the theoretical ceiling.; Clipped-utopia fraction.; Unique objective tuple count.; Unique non-dominated tuple count.
- Claim boundary: HV ceiling panels explain saturation and duplicate collapse only.
- Unresolved visual concerns: None

## B4 B4_optimizer_linked_gap_decomposition
- Planned manuscript location: Appendix Fig. B4 candidate
- Source files: `optimizer_linked_physical_gaps.csv` (79d6291c83058eab43693346035444a30e2b97f3bac28ed993a11affe53e3383)
- Panel descriptions: EUIt projection and cross-model gap decomposition for optimizer-linked cases.; EG projection and cross-model gap decomposition for optimizer-linked cases.; H projection and cross-model gap decomposition for optimizer-linked cases.
- Claim boundary: Optimizer-linked cases remain appendix-only bridge diagnostics.
- Unresolved visual concerns: None

## B5 B5_nonlinear_response_profiles
- Planned manuscript location: Appendix Fig. B5 candidate
- Source files: `scale_study.csv` (87120cdcbdb9cdd09560f4778f0351398933affa460de2884913fdb6554e9bf7)
- Panel descriptions: OSR → EUIt surrogate response profile.; FAR → EG surrogate response profile.; SVF → H surrogate response profile.; θ → H surrogate response profile.
- Claim boundary: Selected surrogate response profiles illustrate local trends only.
- Unresolved visual concerns: None

## B6 B6_climate_case_detail
- Planned manuscript location: Appendix Fig. B6 candidate
- Source files: `climate_case_results.csv` (f624253730cdf63ce60f4e774516b4854f0c3dc6a33c4b2580a02b5df4bc3536)
- Panel descriptions: Per-block ΔEUIt heatmap across Beijing, Guangzhou, and Harbin.; Per-block ΔEG heatmap across Beijing, Guangzhou, and Harbin.; Per-block ΔH heatmap across Beijing, Guangzhou, and Harbin.
- Claim boundary: Case-level climate details remain limited to four blocks and three additional climates.
- Unresolved visual concerns: None
