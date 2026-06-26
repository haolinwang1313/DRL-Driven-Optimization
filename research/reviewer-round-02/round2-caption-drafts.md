# Round 2 Caption Drafts

## Main manuscript candidates

### Manual Fig. 1 candidate (Fig1 manual_fig1_candidate)
This candidate figure summarizes User-supplied manual overview of morphology generation, surrogate modeling, descriptor-space optimization, and assessment flow.. It uses paper/manuscript/figures/round2_candidate/manual/fig1.pdf and should be interpreted within the following boundary: Manual Fig. 1 is included for visual QA and gallery context only; the automated workflow must not edit or re-export it..
Revision note: User-supplied manual Fig. 1 candidate; included for gallery and QA only, with no automatic edits.

### Fig. 2 workflow round3 candidate (Fig2 workflow_round3)
Figure 2. Workflow of the DDPG-based surrogate search. The optimization starts by initializing the surrogate-based environment and selecting one preference scenario. Each episode begins from a random descriptor query and then proceeds through repeated surrogate-query steps, in which the actor produces an absolute descriptor query and the guarded surrogate returns the next state and reward. The episode terminates after a fixed query horizon, and the process continues until all episodes and seeds are completed.
Revision note: Redrawn as a round3 workflow flowchart with a separate single-query callout.

### Fig. 3 DDPG architecture round3 candidate (Fig3 ddpg_architecture_round3)
Figure 3. DDPG learning architecture used in the surrogate-assisted optimization. The environment provides state transitions that are stored in the experience replay buffer. The online actor generates actions, the online critic evaluates state--action pairs, and the target networks provide the temporal-difference target for critic training. The lower equation blocks summarize the temporal-difference target, critic loss, actor objective, and soft target-network update.
Revision note: Redrawn as a round3 DDPG architecture diagram with interaction, replay, online, target, and update blocks.

### Main Fig. 4 (M1 data_and_surrogate_validation)
This candidate figure summarizes PCA cumulative explained variance for the 12 morphology descriptors.; EUIt repeated-CV parity using sample-level mean out-of-fold predictions.; EG repeated-CV parity using sample-level mean out-of-fold predictions.; H repeated-CV parity using sample-level mean out-of-fold predictions.. It uses descriptor_coverage.csv, surrogate_parity_mean_predictions.csv and should be interpreted within the following boundary: Supports descriptor-space coverage and analytic-target surrogate fidelity only..
Revision note: Carried forward from the canonical round-2 candidate set without a layout change in this task.

### Main Fig. 5 (M2 surrogate_robustness)
This candidate figure summarizes nMAE heatmap across repeated CV, leave-one-OSLI-out, outer-shell, and feature-tail regimes.; Spearman heatmap across the same surrogate-validation regimes.. It uses surrogate_validation_regimes.csv and should be interpreted within the following boundary: All panels remain analytic-target surrogate-validation evidence, not physical validation..
Revision note: Carried forward from the canonical round-2 candidate set without a layout change in this task.

### Main Fig. 6 (M3 ddpg_training_dynamics)
This candidate figure summarizes Episode cumulative reward across 20 seeds.; Episode-end EUIt across 20 seeds.; Episode-end EG across 20 seeds.; Episode-end H across 20 seeds.. It uses ddpg_training_curves_summary.csv, ddpg_seed_diagnostics.csv and should be interpreted within the following boundary: Training dynamics describe serialized surrogate-query search only..
Revision note: Carried forward from the canonical round-2 candidate set without a layout change in this task.

### Main Fig. 7 (M4 benchmark_fairness)
This candidate figure summarizes Fixed-domain post-hoc utility for matched DDPG scenarios and NSGA-II.; Equal-size-20 HV under benchmark-reference-v2.; Equal-size-20 IGD under benchmark-reference-v2.. It uses benchmark_utility.csv, benchmark_equal_size_20.csv, benchmark_output_contract_counts.csv and should be interpreted within the following boundary: Equal-size metrics remain benchmark-reference-v2 evidence only; broader contract and diagnostic baselines are deferred to Supplementary Information..
Revision note: Simplified the main benchmark figure to fixed-domain utility plus matched-size HV/IGD for DDPG and NSGA-II only.

### Main Fig. 8 (M5 feasible_projection)
This candidate figure summarizes Descriptor-space compression from retained candidates to unique projected feasible blocks.; Projection-distance distribution from descriptor candidates to feasible blocks.. It uses feasible_projection_summary.csv, feasible_projection_metrics.csv and should be interpreted within the following boundary: Projection remains a nearest-neighbour representation diagnostic rather than physical validation..
Revision note: Reduced the projection figure to representation compression plus projection-distance diagnostics for the main-text comparison.

### Main Fig. 9 (M6 physical_cross_model_stress_test)
This candidate figure summarizes EUIt parity for the 18 direct feasible cases.; GHI-based rooftop-PV proxy parity for the 18 direct feasible cases.; Direct-sun-hours parity for the 18 direct feasible cases.. It uses physical_direct_cases.csv, physical_stress_metrics.csv and should be interpreted within the following boundary: This figure is limited to the direct-case physics-based cross-model stress test and does not support optimizer-superiority claims..
Revision note: Kept the same 18 direct cases while tightening axis wording, typography, marker size, and statistics placement.

### Main Fig. 10 (M7 cross_climate_sensitivity)
This candidate figure summarizes Mean ΔEUIt relative to Dongtai with four-block spread.; Mean ΔEG relative to Dongtai with four-block spread.; Mean ΔH relative to Dongtai with four-block spread.; Rank-stability heatmap across Beijing, Guangzhou, and Harbin.. It uses climate_case_results.csv, climate_summary.csv, climate_rank_stability.csv and should be interpreted within the following boundary: This figure is a limited four-block cross-climate physical sensitivity analysis only..
Revision note: Kept the same climate data while switching to the muted climate palette and a zero-centered low-saturation heatmap.

## Supplementary Information candidates

### Supplementary Fig. S1 (S1 A1_descriptor_distributions)
This candidate figure summarizes Descriptor interquartile summaries for the 12 morphology descriptors.; OSLI frequency distribution.; Normalized nearest-neighbor distance distribution.. It uses descriptor_coverage.csv and should be interpreted within the following boundary: Descriptive coverage diagnostics only..
Revision note: Carried forward from the canonical round-2 candidate set and relabeled for the Supplementary Information split.

### Supplementary Fig. S2 (S2 A2_residual_diagnostics)
This candidate figure summarizes EUIt residual distribution.; EG residual distribution.; H residual distribution.. It uses surrogate_parity_mean_predictions.csv and should be interpreted within the following boundary: Residual diagnostics describe analytic-target surrogate error only..
Revision note: Carried forward from the canonical round-2 candidate set and relabeled for the Supplementary Information split.

### Supplementary Fig. S3 (S3 A3_scale_study)
This candidate figure summarizes Mean target nMAE across dataset scales.; Mean tail nMAE across dataset scales.; Mean R² across dataset scales.; Selection objective across dataset scales.. It uses scale_study.csv and should be interpreted within the following boundary: Scale-study rows support the surrogate-selection rationale only..
Revision note: Carried forward from the canonical round-2 candidate set and relabeled for the Supplementary Information split.

### Supplementary Fig. S4 (S4 B1_seed_diagnostics)
This candidate figure summarizes Best reward by scenario.; Final reward by scenario.; Plateau episode by scenario.; Best-to-final regression ratio by scenario.. It uses ddpg_seed_diagnostics.csv and should be interpreted within the following boundary: Seed diagnostics remain Supplementary Information training evidence only..
Revision note: Carried forward from the canonical round-2 candidate set and relabeled for the Supplementary Information split.

### Supplementary Fig. S5 (S5 B2_morphology_signatures)
This candidate figure summarizes Median morphology descriptor signatures for representative retained-output groups.. It uses morphology_signatures.csv and should be interpreted within the following boundary: Descriptor signatures are descriptive summaries, not stable design rules..
Revision note: Carried forward from the canonical round-2 candidate set and relabeled for the Supplementary Information split.

### Supplementary Fig. S6 (S6 B3_hv_ceiling_diagnostics)
This candidate figure summarizes HV fraction of the theoretical ceiling alongside clipped-utopia fraction.; Unique clipped objective tuples and unique non-dominated tuples on a log-scaled count axis.. It uses benchmark_hv_ceiling.csv and should be interpreted within the following boundary: HV ceiling and tuple-collapse panels remain supplementary diagnostics only..
Revision note: Moved ceiling and duplicate-collapse diagnostics into a readable two-panel Supplementary Information layout with short labels.

### Supplementary Fig. S7 (S7 B4_optimizer_linked_gap_decomposition)
This candidate figure summarizes EUIt projection and cross-model gap decomposition for optimizer-linked cases.; EG projection and cross-model gap decomposition for optimizer-linked cases.; H projection and cross-model gap decomposition for optimizer-linked cases.. It uses optimizer_linked_physical_gaps.csv and should be interpreted within the following boundary: These optimizer-linked cases are representative bridge diagnostics rather than a global optimizer benchmark..
Revision note: Rebuilt the optimizer-linked bridge diagnostics as horizontal gap bars with an out-of-panel legend and short case labels.

### Supplementary Fig. S8 (S8 B5_nonlinear_response_profiles)
This candidate figure summarizes OSR to EUIt surrogate response profile.; FAR to EG surrogate response profile.; SVF to H surrogate response profile.; Theta to H surrogate response profile.. It uses scale_study.csv and should be interpreted within the following boundary: Selected surrogate response profiles illustrate local trends only..
Revision note: Carried forward from the canonical round-2 candidate set and relabeled for the Supplementary Information split.

### Supplementary Fig. S9 (S9 B6_climate_case_detail)
This candidate figure summarizes Per-block ΔEUIt heatmap across Beijing, Guangzhou, and Harbin.; Per-block ΔEG heatmap across Beijing, Guangzhou, and Harbin.; Per-block ΔH heatmap across Beijing, Guangzhou, and Harbin.. It uses climate_case_results.csv and should be interpreted within the following boundary: Case-level climate details remain limited to four blocks and three additional climates..
Revision note: Carried forward from the canonical round-2 candidate set and relabeled for the Supplementary Information split.
