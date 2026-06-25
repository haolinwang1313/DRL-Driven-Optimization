# HV Ceiling Interpretation

- Fixed reference point: `[1.1, 1.1, 1.1]`.
- Theoretical maximum HV: `1.331000`.
- HV values near the ceiling indicate reference-volume saturation, not archive richness by themselves.

## Group diagnostics
- `DDPG::Balanced_Performance`
  method/scenario: `DDPG` / `Balanced_Performance`
  HV = `0.661912`, distance_to_hv_ceiling = `0.669088`, fraction_of_theoretical_max = `0.497304`
  clipped_utopia_rows = `0` / `20`
  unique_objective_tuples = `20`
  unique_non_dominated_tuples = `5`
- `CMA-ES::Balanced_Performance`
  method/scenario: `CMA-ES` / `Balanced_Performance`
  HV = `1.331000`, distance_to_hv_ceiling = `0.000000`, fraction_of_theoretical_max = `1.000000`
  clipped_utopia_rows = `1998` / `2000`
  unique_objective_tuples = `2`
  unique_non_dominated_tuples = `1`
- `RandomSearch::Balanced_Performance`
  method/scenario: `RandomSearch` / `Balanced_Performance`
  HV = `1.200068`, distance_to_hv_ceiling = `0.130932`, fraction_of_theoretical_max = `0.901629`
  clipped_utopia_rows = `0` / `2000`
  unique_objective_tuples = `2000`
  unique_non_dominated_tuples = `11`
- `FeasiblePoolRandom::Balanced_Performance`
  method/scenario: `FeasiblePoolRandom` / `Balanced_Performance`
  HV = `1.307244`, distance_to_hv_ceiling = `0.023756`, fraction_of_theoretical_max = `0.982152`
  clipped_utopia_rows = `0` / `20`
  unique_objective_tuples = `1`
  unique_non_dominated_tuples = `1`
- `DDPG::Energy_Saving_Focus`
  method/scenario: `DDPG` / `Energy_Saving_Focus`
  HV = `1.002757`, distance_to_hv_ceiling = `0.328243`, fraction_of_theoretical_max = `0.753386`
  clipped_utopia_rows = `0` / `20`
  unique_objective_tuples = `20`
  unique_non_dominated_tuples = `8`
- `CMA-ES::Energy_Saving_Focus`
  method/scenario: `CMA-ES` / `Energy_Saving_Focus`
  HV = `1.331000`, distance_to_hv_ceiling = `0.000000`, fraction_of_theoretical_max = `1.000000`
  clipped_utopia_rows = `2000` / `2000`
  unique_objective_tuples = `1`
  unique_non_dominated_tuples = `1`
- `RandomSearch::Energy_Saving_Focus`
  method/scenario: `RandomSearch` / `Energy_Saving_Focus`
  HV = `1.107954`, distance_to_hv_ceiling = `0.223046`, fraction_of_theoretical_max = `0.832422`
  clipped_utopia_rows = `0` / `2000`
  unique_objective_tuples = `2000`
  unique_non_dominated_tuples = `14`
- `FeasiblePoolRandom::Energy_Saving_Focus`
  method/scenario: `FeasiblePoolRandom` / `Energy_Saving_Focus`
  HV = `1.307244`, distance_to_hv_ceiling = `0.023756`, fraction_of_theoretical_max = `0.982152`
  clipped_utopia_rows = `0` / `20`
  unique_objective_tuples = `1`
  unique_non_dominated_tuples = `1`
- `DDPG::Energy_Generation_Focus`
  method/scenario: `DDPG` / `Energy_Generation_Focus`
  HV = `1.170428`, distance_to_hv_ceiling = `0.160572`, fraction_of_theoretical_max = `0.879360`
  clipped_utopia_rows = `0` / `20`
  unique_objective_tuples = `20`
  unique_non_dominated_tuples = `7`
- `CMA-ES::Energy_Generation_Focus`
  method/scenario: `CMA-ES` / `Energy_Generation_Focus`
  HV = `1.331000`, distance_to_hv_ceiling = `0.000000`, fraction_of_theoretical_max = `1.000000`
  clipped_utopia_rows = `1999` / `2000`
  unique_objective_tuples = `2`
  unique_non_dominated_tuples = `1`
- `RandomSearch::Energy_Generation_Focus`
  method/scenario: `RandomSearch` / `Energy_Generation_Focus`
  HV = `1.192443`, distance_to_hv_ceiling = `0.138557`, fraction_of_theoretical_max = `0.895900`
  clipped_utopia_rows = `0` / `2000`
  unique_objective_tuples = `2000`
  unique_non_dominated_tuples = `14`
- `FeasiblePoolRandom::Energy_Generation_Focus`
  method/scenario: `FeasiblePoolRandom` / `Energy_Generation_Focus`
  HV = `1.307244`, distance_to_hv_ceiling = `0.023756`, fraction_of_theoretical_max = `0.982152`
  clipped_utopia_rows = `0` / `20`
  unique_objective_tuples = `1`
  unique_non_dominated_tuples = `1`
- `NSGA-II`
  method/scenario: `NSGA-II` / `NSGA-II`
  HV = `1.330999`, distance_to_hv_ceiling = `0.000001`, fraction_of_theoretical_max = `1.000000`
  clipped_utopia_rows = `0` / `2000`
  unique_objective_tuples = `1761`
  unique_non_dominated_tuples = `17`

## Interpretation rules
- CMA-ES ceiling-hitting rows mostly collapse onto one or two clipped objective tuples, so they do not imply a richer Pareto archive.
- NSGA-II stays near the ceiling because it covers the normalized reference box well; diversity still needs separate tuple-count or spread diagnostics.