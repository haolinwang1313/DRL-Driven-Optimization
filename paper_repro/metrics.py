from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sklearn.metrics import mean_absolute_error, r2_score

from paper_repro.constants import MORPHOLOGY_FEATURES, PERFORMANCE_TARGETS
from paper_repro.contracts import write_json


def load_benchmark_results(path: str | Path) -> pd.DataFrame:
    frame = pd.read_excel(path)
    frame = frame.rename(
        columns={
            "Method": "method",
            "EUlt (kWh/m²/y)": "EUIt",
            "EG (10⁶ kWh/y)": "EG",
            "H (h)": "H",
        }
    )
    frame["scenario"] = frame.iloc[:, 4].fillna("NSGA-II")
    return frame[["method", "scenario", "EUIt", "EG", "H"]]


def summarize_objectives(frame: pd.DataFrame, group_column: str) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for group_name, group in frame.groupby(group_column):
        record = {"count": float(len(group))}
        for target in PERFORMANCE_TARGETS:
            record[f"{target}_mean"] = float(group[target].mean())
            record[f"{target}_median"] = float(group[target].median())
            record[f"{target}_std"] = float(group[target].std(ddof=0))
        summary[str(group_name)] = record
    return summary


def summarize_surrogate_predictions(
    cv_predictions: pd.DataFrame,
    *,
    low_quantile: float = 0.1,
    high_quantile: float = 0.9,
) -> dict[str, object]:
    per_target_rows: list[dict[str, float | str]] = []
    quantile_payload = {"low": float(low_quantile), "high": float(high_quantile)}
    for target in PERFORMANCE_TARGETS:
        truth = cv_predictions[f"true_{target}"].to_numpy(dtype=float)
        pred = cv_predictions[f"pred_{target}"].to_numpy(dtype=float)
        abs_error = np.abs(pred - truth)
        target_range = max(float(np.max(truth) - np.min(truth)), 1e-8)
        q_low = float(np.quantile(truth, low_quantile))
        q_high = float(np.quantile(truth, high_quantile))
        low_mask = truth <= q_low
        high_mask = truth >= q_high
        low_mae = float(abs_error[low_mask].mean()) if np.any(low_mask) else float(abs_error.mean())
        high_mae = float(abs_error[high_mask].mean()) if np.any(high_mask) else float(abs_error.mean())
        per_target_rows.append(
            {
                "target": target,
                "mae": float(mean_absolute_error(truth, pred)),
                "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
                "r2": float(r2_score(truth, pred)),
                "nmae": float(mean_absolute_error(truth, pred) / target_range),
                "q_low": q_low,
                "q_high": q_high,
                "low_tail_mae": low_mae,
                "high_tail_mae": high_mae,
                "tail_mae": float(np.mean([low_mae, high_mae])),
                "tail_nmae": float(np.mean([low_mae, high_mae]) / target_range),
            }
        )
    summary_frame = pd.DataFrame(per_target_rows)
    aggregate = {
        "mean_target_mae": float(summary_frame["mae"].mean()),
        "mean_target_nmae": float(summary_frame["nmae"].mean()),
        "mean_tail_mae": float(summary_frame["tail_mae"].mean()),
        "mean_tail_nmae": float(summary_frame["tail_nmae"].mean()),
        "mean_r2": float(summary_frame["r2"].mean()),
        "worst_target_nmae": float(summary_frame["nmae"].max()),
    }
    return {
        "quantiles": quantile_payload,
        "per_target": summary_frame.to_dict(orient="records"),
        "aggregate": aggregate,
    }


def _minimization_array(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([frame["EUIt"].to_numpy(), -frame["EG"].to_numpy(), -frame["H"].to_numpy()])


def compute_hv_igd_by_method(frame: pd.DataFrame) -> pd.DataFrame:
    matrix = _minimization_array(frame)
    groups = np.where(frame["method"].to_numpy() == "NSGA-II", "NSGA-II", frame["scenario"].to_numpy())
    group_fronts: dict[str, np.ndarray] = {}
    nds = NonDominatedSorting()
    for group_name in pd.unique(groups):
        indices = np.where(groups == group_name)[0]
        group_matrix = matrix[indices]
        front = nds.do(group_matrix, only_non_dominated_front=True)
        group_fronts[str(group_name)] = group_matrix[front]

    combined_front = np.vstack(list(group_fronts.values()))
    ideal = combined_front.min(axis=0)
    nadir = combined_front.max(axis=0)
    reference_front = (combined_front - ideal) / np.maximum(nadir - ideal, 1e-8)
    reference_point = np.array([1.1, 1.1, 1.1])
    rows = []
    for group_name, group_matrix in group_fronts.items():
        normalized_group = (group_matrix - ideal) / np.maximum(nadir - ideal, 1e-8)
        rows.append(
            {
                "method": group_name,
                "HV": float(HV(ref_point=reference_point)(normalized_group)),
                "IGD": float(IGD(reference_front)(normalized_group)),
            }
        )
    return pd.DataFrame(rows)


def normalized_benefit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["EUIt_score"] = 1.0 - (frame["EUIt"] - frame["EUIt"].min()) / max(frame["EUIt"].max() - frame["EUIt"].min(), 1e-8)
    normalized["EG_score"] = (frame["EG"] - frame["EG"].min()) / max(frame["EG"].max() - frame["EG"].min(), 1e-8)
    normalized["H_score"] = (frame["H"] - frame["H"].min()) / max(frame["H"].max() - frame["H"].min(), 1e-8)
    return normalized


def select_preference_aligned_candidates(frame: pd.DataFrame, scenario_weights: dict[str, list[float]]) -> pd.DataFrame:
    normalized = normalized_benefit_frame(frame)
    rows = []
    for scenario_name, weights in scenario_weights.items():
        weighted = np.asarray(weights, dtype=float)
        scenario_frame = normalized.copy()
        scenario_frame["utility"] = (
            weighted[0] * scenario_frame["EUIt_score"]
            + weighted[1] * scenario_frame["EG_score"]
            + weighted[2] * scenario_frame["H_score"]
        )
        for method_name, method_frame in scenario_frame.groupby("method"):
            best_index = method_frame["utility"].idxmax()
            record = scenario_frame.loc[
                best_index,
                ["method", "scenario", "seed", *MORPHOLOGY_FEATURES, "EUIt", "EG", "H", "utility"],
            ].to_dict()
            record["selection_scenario"] = scenario_name
            rows.append(record)
    return pd.DataFrame(rows)


def summarize_preference_utilities(frame: pd.DataFrame, scenario_weights: dict[str, list[float]]) -> dict[str, list[dict[str, float | str]]]:
    selected = select_preference_aligned_candidates(frame, scenario_weights)
    payload: dict[str, list[dict[str, float | str]]] = {}
    for scenario_name, group in selected.groupby("selection_scenario"):
        payload[str(scenario_name)] = group.to_dict(orient="records")
    return payload


def _plateau_episode_from_reward(frame: pd.DataFrame) -> tuple[float, float, float]:
    reward = frame["cumulative_reward"].to_numpy(dtype=float)
    rolling_window = min(20, len(reward))
    rolling = pd.Series(reward).rolling(rolling_window, min_periods=rolling_window).mean().dropna().to_numpy()
    if len(rolling) == 0:
        return float(frame["episode"].iloc[-1]), float(rolling_window), float(reward[-1])
    best_rolling = float(np.max(rolling))
    plateau_threshold = best_rolling * 0.95 if best_rolling >= 0.0 else best_rolling * 1.05
    plateau_episode = float(frame["episode"].iloc[-1])
    for idx, value in enumerate(rolling, start=rolling_window - 1):
        if value >= plateau_threshold:
            plateau_episode = float(frame["episode"].iloc[min(idx, len(frame) - 1)])
            break
    return plateau_episode, float(rolling_window), float(plateau_threshold)


def compute_convergence_diagnostics(ddpg_logs: dict[str, list[dict[str, float]]]) -> dict[str, dict[str, float]]:
    diagnostics: dict[str, dict[str, float]] = {}
    for scenario_name, rows in ddpg_logs.items():
        if not rows:
            diagnostics[scenario_name] = {"episodes": 0.0}
            continue
        frame = pd.DataFrame(rows)
        reward = frame["cumulative_reward"].to_numpy(dtype=float)
        plateau_episode, rolling_window, plateau_threshold = _plateau_episode_from_reward(frame)
        diagnostics[scenario_name] = {
            "episodes": float(len(frame)),
            "reward_initial": float(reward[0]),
            "reward_final": float(reward[-1]),
            "reward_gain": float(reward[-1] - reward[0]),
            "reward_std": float(np.std(reward)),
            "plateau_episode": plateau_episode,
            "rolling_window": float(rolling_window),
            "plateau_threshold": float(plateau_threshold),
        }
    return diagnostics


def compute_seeded_convergence_diagnostics(
    ddpg_all_logs: dict[str, dict[str, list[dict[str, float]]]],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    rows: list[dict[str, float | str]] = []
    for scenario_name, seed_logs in ddpg_all_logs.items():
        for seed_name, entries in seed_logs.items():
            if not entries:
                continue
            frame = pd.DataFrame(entries)
            reward = frame["cumulative_reward"].to_numpy(dtype=float)
            best_index = int(np.argmax(reward))
            best_reward = float(reward[best_index])
            final_reward = float(reward[-1])
            plateau_episode, _, _ = _plateau_episode_from_reward(frame)
            rows.append(
                {
                    "scenario": str(scenario_name),
                    "seed": str(seed_name),
                    "episodes": float(len(frame)),
                    "reward_initial": float(reward[0]),
                    "reward_final": final_reward,
                    "reward_best": best_reward,
                    "reward_gain": final_reward - float(reward[0]),
                    "best_episode": float(frame["episode"].iloc[best_index]),
                    "plateau_episode": plateau_episode,
                    "best_final_gap": best_reward - final_reward,
                    "best_final_gap_ratio": (best_reward - final_reward) / max(abs(best_reward), 1e-8),
                }
            )
    seeded = pd.DataFrame(rows)
    if seeded.empty:
        return seeded, {}

    summary: dict[str, dict[str, float]] = {}
    for scenario_name, group in seeded.groupby("scenario"):
        summary[str(scenario_name)] = {
            "n_seeds": float(len(group)),
            "reward_best_mean": float(group["reward_best"].mean()),
            "reward_final_mean": float(group["reward_final"].mean()),
            "best_final_gap_mean": float(group["best_final_gap"].mean()),
            "best_final_gap_ratio_mean": float(group["best_final_gap_ratio"].mean()),
            "plateau_episode_mean": float(group["plateau_episode"].mean()),
            "late_regression_seed_fraction": float((group["best_final_gap"] > 0.2 * group["reward_best"].abs().clip(lower=1e-8)).mean()),
        }
    return seeded, summary


def write_metrics_report(payload: dict, path: str | Path) -> Path:
    return write_json(payload, path)
