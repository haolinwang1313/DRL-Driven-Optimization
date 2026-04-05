from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from paper_repro.config import Config
from paper_repro.constants import MORPHOLOGY_FEATURES, PERFORMANCE_TARGETS
from paper_repro.metrics import compute_hv_igd_by_method, normalized_benefit_frame

sns.set_theme(style="whitegrid")

TARGET_LABELS = {
    "EUIt": "EUIt (kWh/m$^2$/y)",
    "EG": "EG ($10^6$ kWh/y)",
    "H": "H (h)",
}

FEATURE_LABELS = {
    "FAR": "FAR",
    "SD": "SD (m)",
    "AF": "AF (floors)",
    "AR_ew": "AR$_{e-w}$",
    "AR_ns": "AR$_{n-s}$",
    "SVF": "SVF",
    "BD": "BD",
    "OSR": "OSR",
    "SC": "SC",
    "PAR": "PAR",
    "theta": r"$\theta$ (deg)",
    "OSLI": "OSLI",
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_prediction_figures(cv_predictions: pd.DataFrame, figures_dir: Path) -> None:
    representative = cv_predictions.loc[cv_predictions["fold"] == cv_predictions["fold"].min()].copy()
    representative = representative.sort_values("sample_id")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for axis, target in zip(axes, PERFORMANCE_TARGETS, strict=True):
        axis.plot(representative["sample_id"], representative[f"true_{target}"], label="Simulated", color="#6B7280", linewidth=1.8)
        axis.plot(representative["sample_id"], representative[f"pred_{target}"], label="Predicted", color="#0F766E", linewidth=1.2)
        axis.set_ylabel(TARGET_LABELS[target])
    axes[0].legend()
    axes[-1].set_xlabel("Sample ID")
    fig.suptitle("DNN Prediction on Representative Fold")
    _save(fig, figures_dir / "fig4_dnn_predictions.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, target in zip(axes, PERFORMANCE_TARGETS, strict=True):
        errors = representative[f"pred_{target}"] - representative[f"true_{target}"]
        sns.histplot(errors, kde=True, ax=axis, color="#B45309")
        axis.set_title(f"{target} Error")
        axis.set_xlabel(f"Prediction error in {TARGET_LABELS[target]}")
        axis.set_ylabel("Count")
    fig.suptitle("Prediction Error Distribution")
    _save(fig, figures_dir / "fig5_dnn_error_distribution.png")


def make_learning_curve_figure(ddpg_logs: dict[str, list[dict[str, float]]], optimization_dir: Path, figures_dir: Path) -> None:
    all_log_path = optimization_dir / "ddpg_logs_all.json"
    palette = {
        "Balanced_Performance": "#1D4ED8",
        "Energy_Saving_Focus": "#D97706",
        "Energy_Generation_Focus": "#059669",
    }
    mapping = [
        ("cumulative_reward", "Cumulative reward", "Reward (unitless)"),
        ("EUIt", "EUIt", TARGET_LABELS["EUIt"]),
        ("EG", "EG", TARGET_LABELS["EG"]),
        ("H", "H", TARGET_LABELS["H"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True)
    if all_log_path.exists():
        all_logs = json.loads(all_log_path.read_text(encoding="utf-8"))
        for axis, (column, title, ylabel) in zip(axes.flatten(), mapping, strict=True):
            for scenario, seed_logs in all_logs.items():
                seed_frames = []
                for seed_id, rows in seed_logs.items():
                    frame = pd.DataFrame(rows)
                    if frame.empty:
                        continue
                    seed_frames.append(frame[["episode", column]].rename(columns={column: f"seed_{seed_id}"}).set_index("episode"))
                if not seed_frames:
                    continue
                merged = pd.concat(seed_frames, axis=1).sort_index()
                mean_curve = merged.mean(axis=1)
                std_curve = merged.std(axis=1).fillna(0.0)
                color = palette.get(scenario, "#1D4ED8")
                axis.plot(mean_curve.index, mean_curve.values, color=color, linewidth=1.6, label=scenario.replace("_", " "))
                axis.fill_between(mean_curve.index, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.18)
            axis.set_title(title)
            axis.set_xlabel("Episode")
            axis.set_ylabel(ylabel)
            if column == "cumulative_reward":
                axis.legend(frameon=False, fontsize=8)
    else:
        scenario = next(iter(ddpg_logs))
        log = pd.DataFrame(ddpg_logs[scenario])
        for axis, (column, title, ylabel) in zip(axes.flatten(), mapping, strict=True):
            axis.plot(log["episode"], log[column], color="#1D4ED8", linewidth=1.2)
            axis.set_title(title)
            axis.set_xlabel("Episode")
            axis.set_ylabel(ylabel)
    fig.suptitle("DDPG learning curves across scenarios (mean $\pm$ standard deviation across seeds)")
    _save(fig, figures_dir / "fig6_ddpg_learning_curve.png")


def make_objective_distribution_figure(results: pd.DataFrame, figures_dir: Path) -> None:
    order = ["NSGA-II", "Balanced_Performance", "Energy_Saving_Focus", "Energy_Generation_Focus"]
    display = results.copy()
    display["method_label"] = np.where(display["method"] == "NSGA-II", "NSGA-II", display["scenario"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, target in zip(axes, PERFORMANCE_TARGETS, strict=True):
        sns.violinplot(data=display, x="method_label", y=target, order=order, inner="box", ax=axis, cut=0, palette="Set2")
        axis.set_xlabel("")
        axis.set_ylabel(TARGET_LABELS[target])
        axis.tick_params(axis="x", rotation=20)
        axis.set_title(TARGET_LABELS[target])
    fig.suptitle("Objective Distributions Across Optimization Methods")
    _save(fig, figures_dir / "fig7_objective_distributions.png")


def make_parallel_and_radar_figures(results: pd.DataFrame, figures_dir: Path) -> None:
    display = results.copy()
    display["method_label"] = np.where(display["method"] == "NSGA-II", "NSGA-II", display["scenario"])
    medians = display.groupby("method_label")[MORPHOLOGY_FEATURES].median().reset_index()
    scaled = medians.copy()
    for feature in MORPHOLOGY_FEATURES:
        column = scaled[feature]
        scaled[feature] = (column - column.min()) / max(column.max() - column.min(), 1e-8)

    fig, ax = plt.subplots(figsize=(13, 5))
    for _, row in scaled.iterrows():
        ax.plot(range(len(MORPHOLOGY_FEATURES)), row[MORPHOLOGY_FEATURES], label=row["method_label"], linewidth=1.8)
    ax.set_xticks(range(len(MORPHOLOGY_FEATURES)))
    ax.set_xticklabels([FEATURE_LABELS.get(feature, feature) for feature in MORPHOLOGY_FEATURES], rotation=35, ha="right")
    ax.set_ylabel("Normalized median (unitless)")
    ax.legend()
    ax.set_title("Parallel Coordinate Style Comparison of Learned Strategies")
    _save(fig, figures_dir / "fig8_parallel_coordinates.png")

    labels = MORPHOLOGY_FEATURES
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, axes = plt.subplots(2, 2, subplot_kw={"polar": True}, figsize=(11, 9))
    for axis, (_, row) in zip(axes.flatten(), scaled.iterrows(), strict=True):
        values = row[MORPHOLOGY_FEATURES].tolist()
        values += values[:1]
        axis.plot(angles, values, linewidth=2.0, color="#0F766E")
        axis.fill(angles, values, alpha=0.25, color="#0F766E")
        axis.set_xticks(angles[:-1])
        axis.set_xticklabels([FEATURE_LABELS.get(feature, feature) for feature in labels], fontsize=8)
        axis.set_title(row["method_label"])
        axis.set_yticklabels([])
        axis.set_ylim(0.0, 1.0)
    fig.suptitle("Morphological Strategy Radar")
    _save(fig, figures_dir / "fig9_strategy_radar.png")


def make_metrics_figures(results: pd.DataFrame, figures_dir: Path) -> pd.DataFrame:
    hv_igd = compute_hv_igd_by_method(results)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(data=hv_igd, x="method", y="HV", ax=axes[0], palette="crest")
    sns.barplot(data=hv_igd, x="method", y="IGD", ax=axes[1], palette="flare")
    axes[0].set_ylabel("Hypervolume (unitless)")
    axes[1].set_ylabel("IGD (unitless)")
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
    fig.suptitle("Solution Set Quality Metrics")
    _save(fig, figures_dir / "fig10_hv_igd.png")

    normalized = normalized_benefit_frame(results)
    box_frame = normalized.melt(
        id_vars=["method", "scenario", "seed", "reward"],
        value_vars=["EUIt_score", "EG_score", "H_score"],
        var_name="metric",
        value_name="score",
    )
    box_frame["method_label"] = np.where(box_frame["method"] == "NSGA-II", "NSGA-II", box_frame["scenario"])
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=box_frame, x="metric", y="score", hue="method_label", ax=ax)
    ax.set_xlabel("")
    ax.set_xticklabels(["EUIt score", "EG score", "H score"])
    ax.set_ylabel("Normalized score (unitless)")
    ax.set_title("Normalized Objective Comparison")
    _save(fig, figures_dir / "fig11_normalized_boxplots.png")
    return hv_igd


def make_correlation_figures(results: pd.DataFrame, figures_dir: Path) -> None:
    combined = results[MORPHOLOGY_FEATURES + PERFORMANCE_TARGETS]
    corr = combined.corr(numeric_only=True).loc[PERFORMANCE_TARGETS, MORPHOLOGY_FEATURES]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    sns.heatmap(corr, cmap="coolwarm", center=0.0, ax=ax)
    ax.set_xlabel("Urban morphology factors")
    ax.set_ylabel("Performance targets")
    ax.set_xticklabels([FEATURE_LABELS.get(feature, feature) for feature in MORPHOLOGY_FEATURES], rotation=35, ha="right")
    ax.set_yticklabels([TARGET_LABELS.get(target, target) for target in PERFORMANCE_TARGETS], rotation=0)
    ax.set_title("Correlation Between UMFs and Performance Targets")
    _save(fig, figures_dir / "fig12_correlation_heatmap.png")

    pairs = [("OSR", "EUIt"), ("FAR", "EG"), ("SVF", "H"), ("theta", "H")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, (feature, target) in zip(axes.flatten(), pairs, strict=True):
        sns.regplot(data=results, x=feature, y=target, scatter_kws={"s": 15, "alpha": 0.6}, line_kws={"color": "#B91C1C"}, ax=axis)
        axis.set_title(f"{feature} vs {target}")
        axis.set_xlabel(FEATURE_LABELS.get(feature, feature))
        axis.set_ylabel(TARGET_LABELS.get(target, target))
    fig.suptitle("Key Joint Distributions")
    _save(fig, figures_dir / "fig13_joint_plots.png")


def make_supplementary_factor_distribution(results: pd.DataFrame, figures_dir: Path) -> None:
    supplementary = results.copy()
    supplementary["group"] = np.where(supplementary["method"] == "NSGA-II", "NSGA-II", "DDPG-Combined")
    melted = supplementary.melt(id_vars=["group"], value_vars=MORPHOLOGY_FEATURES, var_name="feature", value_name="value")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.violinplot(data=melted, x="feature", y="value", hue="group", split=True, inner="quart", cut=0, ax=ax)
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Supplementary Distribution of 12 UMFs")
    _save(fig, figures_dir / "supplementary_umf_distributions.png")


def generate_all_figures(config: Config) -> dict[str, str]:
    figures_dir = Path(config["report"]["figures_dir"])
    model_dir = Path(config["report"]["models_dir"])
    optimization_dir = Path(config["report"]["optimization_dir"])

    cv_predictions = pd.read_csv(model_dir / "cv_predictions.csv")
    ddpg_logs = json.loads((optimization_dir / "ddpg_logs.json").read_text(encoding="utf-8"))
    ddpg_results = pd.read_csv(optimization_dir / "ddpg_results.csv")
    nsga_results = pd.read_csv(optimization_dir / "nsga2_results.csv")
    results = pd.concat([ddpg_results, nsga_results], ignore_index=True)

    make_prediction_figures(cv_predictions, figures_dir)
    make_learning_curve_figure(ddpg_logs, optimization_dir, figures_dir)
    make_objective_distribution_figure(results, figures_dir)
    make_parallel_and_radar_figures(results, figures_dir)
    hv_igd = make_metrics_figures(results, figures_dir)
    make_correlation_figures(results, figures_dir)
    make_supplementary_factor_distribution(results, figures_dir)
    return {"figures_dir": str(figures_dir), "hv_igd_rows": str(len(hv_igd))}
