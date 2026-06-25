# Round 2 Caption Drafts

## M1 data_and_surrogate_validation
This candidate figure summarizes PCA cumulative explained variance for the 12 morphology descriptors.; EUIt repeated-CV parity using sample-level mean out-of-fold predictions.; EG repeated-CV parity using sample-level mean out-of-fold predictions.; H repeated-CV parity using sample-level mean out-of-fold predictions.. It uses descriptor_coverage.csv, surrogate_parity_mean_predictions.csv and should be interpreted within the following boundary: Supports descriptor-space coverage and analytic-target surrogate fidelity only..

## M2 surrogate_robustness
This candidate figure summarizes nMAE heatmap across repeated CV, leave-one-OSLI-out, outer-shell, and feature-tail regimes.; Spearman heatmap across the same surrogate-validation regimes.. It uses surrogate_validation_regimes.csv and should be interpreted within the following boundary: All panels remain analytic-target surrogate-validation evidence, not physical validation..

## M3 ddpg_training_dynamics
This candidate figure summarizes Episode cumulative reward across 20 seeds.; Episode-end EUIt across 20 seeds.; Episode-end EG across 20 seeds.; Episode-end H across 20 seeds.. It uses ddpg_training_curves_summary.csv, ddpg_seed_diagnostics.csv and should be interpreted within the following boundary: Training dynamics describe serialized surrogate-query search only..

## M4 benchmark_fairness
This candidate figure summarizes Fixed-domain post-hoc utility for DDPG and NSGA-II across the three scalarization scenarios.; Equal-size-20 HV with 5–95% intervals under benchmark-reference-v2.; Equal-size-20 IGD with 5–95% intervals under benchmark-reference-v2.; Output-contract asymmetry across retained rows, unique objective tuples, and unique feasible blocks.. It uses benchmark_utility.csv, benchmark_equal_size_20.csv, benchmark_output_contract_counts.csv and should be interpreted within the following boundary: Equal-size metrics are canonical only under benchmark-reference-v2 and must stay separate from asymmetric full-archive diagnostics..

## M5 feasible_projection
This candidate figure summarizes Projection-distance distribution from descriptor candidates to feasible blocks.; Duplicate-collapse rate with unique feasible-block counts.; HV before and after projection under benchmark-reference-v2.; IGD before and after projection under benchmark-reference-v2.. It uses feasible_projection_summary.csv, feasible_projection_metrics.csv and should be interpreted within the following boundary: Projection panels are representation-sensitivity diagnostics rather than physical validation..

## M6 physical_cross_model_stress_test
This candidate figure summarizes EUIt parity for the 18 direct feasible cases.; Simplified rooftop-PV proxy parity for the 18 direct feasible cases.; January 20 windowsill direct-sun-hours parity for the 18 direct feasible cases.. It uses physical_direct_cases.csv, physical_stress_metrics.csv and should be interpreted within the following boundary: This figure is limited to the direct-case physics-based cross-model stress test and does not support optimizer-superiority claims..

## M7 cross_climate_sensitivity
This candidate figure summarizes Mean ΔEUIt relative to Dongtai with four-block spread.; Mean ΔEG relative to Dongtai with four-block spread.; Mean ΔH relative to Dongtai with four-block spread.; Rank-stability heatmap across Beijing, Guangzhou, and Harbin.. It uses climate_case_results.csv, climate_summary.csv, climate_rank_stability.csv and should be interpreted within the following boundary: This figure is a limited four-block cross-climate physical sensitivity analysis only..

## A1 A1_descriptor_distributions
This candidate figure summarizes Descriptor interquartile summaries for the 12 morphology descriptors.; OSLI frequency distribution.; Normalized nearest-neighbor distance distribution.. It uses descriptor_coverage.csv and should be interpreted within the following boundary: Descriptive coverage diagnostics only..

## A2 A2_residual_diagnostics
This candidate figure summarizes EUIt residual distribution.; EG residual distribution.; H residual distribution.. It uses surrogate_parity_mean_predictions.csv and should be interpreted within the following boundary: Residual diagnostics describe analytic-target surrogate error only..

## A3 A3_scale_study
This candidate figure summarizes Mean target nMAE across dataset scales.; Mean tail nMAE across dataset scales.; Mean R² across dataset scales.; Selection objective across dataset scales.. It uses scale_study.csv and should be interpreted within the following boundary: Scale-study rows support the surrogate-selection rationale only..

## B1 B1_seed_diagnostics
This candidate figure summarizes Best reward by scenario.; Final reward by scenario.; Plateau episode by scenario.; Best-to-final regression ratio by scenario.. It uses ddpg_seed_diagnostics.csv and should be interpreted within the following boundary: Seed diagnostics are appendix-only training evidence..

## B2 B2_morphology_signatures
This candidate figure summarizes Median morphology descriptor signatures for representative retained-output groups.. It uses morphology_signatures.csv and should be interpreted within the following boundary: Descriptor signatures are descriptive summaries, not stable design rules..

## B3 B3_hv_ceiling_diagnostics
This candidate figure summarizes HV fraction of the theoretical ceiling.; Clipped-utopia fraction.; Unique objective tuple count.; Unique non-dominated tuple count.. It uses benchmark_hv_ceiling.csv and should be interpreted within the following boundary: HV ceiling panels explain saturation and duplicate collapse only..

## B4 B4_optimizer_linked_gap_decomposition
This candidate figure summarizes EUIt projection and cross-model gap decomposition for optimizer-linked cases.; EG projection and cross-model gap decomposition for optimizer-linked cases.; H projection and cross-model gap decomposition for optimizer-linked cases.. It uses optimizer_linked_physical_gaps.csv and should be interpreted within the following boundary: Optimizer-linked cases remain appendix-only bridge diagnostics..

## B5 B5_nonlinear_response_profiles
This candidate figure summarizes OSR → EUIt surrogate response profile.; FAR → EG surrogate response profile.; SVF → H surrogate response profile.; θ → H surrogate response profile.. It uses scale_study.csv and should be interpreted within the following boundary: Selected surrogate response profiles illustrate local trends only..

## B6 B6_climate_case_detail
This candidate figure summarizes Per-block ΔEUIt heatmap across Beijing, Guangzhou, and Harbin.; Per-block ΔEG heatmap across Beijing, Guangzhou, and Harbin.; Per-block ΔH heatmap across Beijing, Guangzhou, and Harbin.. It uses climate_case_results.csv and should be interpreted within the following boundary: Case-level climate details remain limited to four blocks and three additional climates..
