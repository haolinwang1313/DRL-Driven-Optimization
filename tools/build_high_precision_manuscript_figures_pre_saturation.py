from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd

from paper_repro.config import Config
from paper_repro.constants import MORPHOLOGY_FEATURES, PERFORMANCE_TARGETS
from paper_repro.metrics import normalized_benefit_frame, compute_seeded_convergence_diagnostics
from paper_repro.simulation import reevaluate_candidates
from paper_repro.surrogate import load_surrogate


CM_TO_IN = 1 / 2.54
SINGLE_COL_IN = 8.5 * CM_TO_IN
DOUBLE_COL_IN = 17.5 * CM_TO_IN

PALETTE = {
    "NSGA-II": "#4D4D4D",
    "Balanced_Performance": "#1F77B4",
    "Energy_Saving_Focus": "#009E73",
    "Energy_Generation_Focus": "#D55E00",
}

DIVERGING_CMAP = "RdBu_r"

MARKERS = {
    "NSGA-II": "o",
    "Balanced_Performance": "o",
    "Energy_Saving_Focus": "^",
    "Energy_Generation_Focus": "s",
}

FEATURE_LABELS = {
    "FAR": "FAR",
    "SD": "SD (m)",
    "AF": "AF (floors)",
    "AR_ew": "AR_e-w",
    "AR_ns": "AR_n-s",
    "SVF": "SVF",
    "BD": "BD",
    "OSR": "OSR",
    "SC": "SC",
    "PAR": "PAR",
    "theta": "theta (deg)",
    "OSLI": "OSLI",
}

TARGET_LABELS = {
    "EUIt": "EUIt (kWh/m$^2$/y)",
    "EG": "EG ($10^6$ kWh/y)",
    "H": "H (h)",
}


def set_journal_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def blend_with_white(color: str, amount: float) -> tuple[float, float, float]:
    rgb = np.array(mcolors.to_rgb(color), dtype=float)
    return tuple((1.0 - amount) * rgb + amount * np.ones(3))


def blend_with_black(color: str, amount: float) -> tuple[float, float, float]:
    rgb = np.array(mcolors.to_rgb(color), dtype=float)
    return tuple((1.0 - amount) * rgb)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", direction="in", top=False, right=False)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(False)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.03,
        0.97,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )


def annotate_bar_values(ax: plt.Axes, x: np.ndarray, vals: list[float], fmt: str = "{:.2f}") -> None:
    y_max = max(vals) if vals else 1.0
    offset = max(y_max * 0.025, 0.01)
    for xpos, val in zip(x, vals, strict=True):
        ax.text(xpos, val + offset, fmt.format(val), ha="center", va="bottom", fontsize=8.5)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _run_root(root: Path, run_id: str) -> Path:
    return root / "artifacts" / "server_runs" / run_id


def build_fig4(cv: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_IN, 5.6 * CM_TO_IN))
    for ax, target, tag in zip(axes, PERFORMANCE_TARGETS, ["a", "b", "c"], strict=True):
        x = cv[f"true_{target}"].to_numpy()
        y = cv[f"pred_{target}"].to_numpy()
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax.scatter(x, y, s=14, c=PALETTE["Balanced_Performance"], alpha=0.7, edgecolors="white", linewidths=0.25)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color=PALETTE["NSGA-II"], linewidth=1.1)
        style_axis(ax)
        panel_label(ax, tag)
        ax.set_xlabel(f"True {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    fig.tight_layout(pad=0.5, w_pad=0.7)
    _save(fig, out_path)


def build_fig5(cv: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_IN, 5.6 * CM_TO_IN))
    for ax, target, tag in zip(axes, PERFORMANCE_TARGETS, ["a", "b", "c"], strict=True):
        residual = cv[f"pred_{target}"] - cv[f"true_{target}"]
        sigma = float(residual.std(ddof=0))
        ax.hist(
            residual,
            bins=18,
            color="#4C78A8",
            alpha=0.75,
            edgecolor="white",
            linewidth=0.35,
            density=True,
        )
        ax.axvline(0.0, color=PALETTE["NSGA-II"], linewidth=1.0)
        ax.axvline(2.0 * sigma, color=PALETTE["Energy_Generation_Focus"], linewidth=1.0, linestyle="--")
        ax.axvline(-2.0 * sigma, color=PALETTE["Energy_Generation_Focus"], linewidth=1.0, linestyle="--")
        style_axis(ax)
        panel_label(ax, tag)
        ax.set_xlabel(f"{target} residual")
        ax.set_ylabel("Density")
    fig.tight_layout(pad=0.5, w_pad=0.7)
    _save(fig, out_path)


def build_fig6(ddpg_logs_all: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 10.2 * CM_TO_IN), sharex=True)
    mapping = [
        ("cumulative_reward", "Cumulative reward"),
        ("EUIt", TARGET_LABELS["EUIt"]),
        ("EG", TARGET_LABELS["EG"]),
        ("H", TARGET_LABELS["H"]),
    ]
    for ax, (metric, ylabel), tag in zip(axes.flatten(), mapping, ["a", "b", "c", "d"], strict=True):
        for scenario in ["Balanced_Performance", "Energy_Saving_Focus", "Energy_Generation_Focus"]:
            seed_map = ddpg_logs_all[scenario]
            frames = []
            for seed, rows in sorted(seed_map.items(), key=lambda item: int(item[0])):
                frame = pd.DataFrame(rows)[["episode", metric]].rename(columns={metric: seed}).set_index("episode")
                frames.append(frame)
            merged = pd.concat(frames, axis=1).sort_index()
            mean_curve = merged.mean(axis=1)
            std_curve = merged.std(axis=1).fillna(0.0)
            ax.plot(mean_curve.index, mean_curve.values, color=PALETTE[scenario], linewidth=1.4)
            ax.fill_between(mean_curve.index, mean_curve - std_curve, mean_curve + std_curve, color=PALETTE[scenario], alpha=0.18, linewidth=0)
        style_axis(ax)
        panel_label(ax, tag)
        ax.set_ylabel(ylabel)
        if ax in axes[1]:
            ax.set_xlabel("Episode")
    handles = [
        Line2D([0], [0], color=PALETTE[s], lw=1.4, marker=MARKERS[s], markersize=4.5, label=s.replace("_", " "))
        for s in ["Balanced_Performance", "Energy_Saving_Focus", "Energy_Generation_Focus"]
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, handletextpad=0.5, columnspacing=1.2)
    fig.tight_layout(pad=0.5, w_pad=0.8, h_pad=0.8, rect=(0, 0, 1, 0.93))
    _save(fig, out_path)


def build_fig7(combined: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_IN, 5.6 * CM_TO_IN))
    pairs = [("EG", "EUIt"), ("H", "EUIt"), ("EG", "H")]
    order = ["NSGA-II", "Balanced_Performance", "Energy_Saving_Focus", "Energy_Generation_Focus"]
    for ax, (x_col, y_col), tag in zip(axes, pairs, ["a", "b", "c"], strict=True):
        for method in order:
            subset = combined.loc[combined["method"] == "NSGA-II"] if method == "NSGA-II" else combined.loc[combined["scenario"] == method]
            if method == "NSGA-II":
                ax.scatter(
                    subset[x_col],
                    subset[y_col],
                    s=18,
                    marker=MARKERS[method],
                    facecolors="none",
                    edgecolors=PALETTE[method],
                    linewidths=0.8,
                    alpha=0.75,
                )
            else:
                ax.scatter(
                    subset[x_col],
                    subset[y_col],
                    s=18,
                    marker=MARKERS[method],
                    c=PALETTE[method],
                    edgecolors="white",
                    linewidths=0.25,
                    alpha=0.7,
                )
        style_axis(ax)
        panel_label(ax, tag)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    handles = []
    for method in order:
        if method == "NSGA-II":
            handles.append(
                Line2D([0], [0], marker=MARKERS[method], linestyle="", markerfacecolor="none", markeredgecolor=PALETTE[method], markeredgewidth=0.8, markersize=6, label="NSGA-II")
            )
        else:
            handles.append(
                Line2D([0], [0], marker=MARKERS[method], linestyle="", markerfacecolor=PALETTE[method], markeredgecolor="white", markeredgewidth=0.25, markersize=6, label=method.replace("_", " "))
            )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=4, frameon=False, handletextpad=0.5, columnspacing=1.0)
    fig.tight_layout(pad=0.5, w_pad=0.8, rect=(0, 0, 1, 0.90))
    _save(fig, out_path)


def build_fig8(combined: pd.DataFrame, out_path: Path) -> None:
    order = ["NSGA-II", "Balanced_Performance", "Energy_Saving_Focus", "Energy_Generation_Focus"]
    display = combined.copy()
    display["method_label"] = np.where(display["method"] == "NSGA-II", "NSGA-II", display["scenario"])
    medians = display.groupby("method_label")[MORPHOLOGY_FEATURES].median().reindex(order)
    scaled = medians.copy()
    for feature in MORPHOLOGY_FEATURES:
        col = medians[feature]
        denom = max(float(col.max() - col.min()), 1e-8)
        scaled[feature] = 2.0 * ((col - col.min()) / denom) - 1.0
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN, 4.8 * CM_TO_IN))
    im = ax.imshow(scaled.to_numpy(), cmap=DIVERGING_CMAP, vmin=-1.0, vmax=1.0, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(MORPHOLOGY_FEATURES)))
    ax.set_xticklabels([FEATURE_LABELS.get(f, f) for f in MORPHOLOGY_FEATURES], rotation=35, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(["NSGA-II", "Balanced", "Saving", "Generation"])
    panel_label(ax, "a")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Relative median", fontsize=8)
    cbar.ax.tick_params(labelsize=8, direction="in")
    fig.tight_layout(pad=0.5)
    _save(fig, out_path)


def build_fig9(ddpg: pd.DataFrame, nsga: pd.DataFrame, combined: pd.DataFrame, utility_weights: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 10.4 * CM_TO_IN), sharex=True)
    scenarios = ["Balanced_Performance", "Energy_Saving_Focus", "Energy_Generation_Focus"]
    scenario_labels = ["Balanced", "Saving", "Generation"]
    x = np.arange(len(scenarios))
    for ax, metric, tag in zip(axes.flatten()[:3], ["EUIt", "EG", "H"], ["a", "b", "c"], strict=True):
        summary = ddpg.groupby("scenario")[metric].agg(["mean", "std"]).reindex(scenarios)
        ax.errorbar(
            x,
            summary["mean"],
            yerr=summary["std"],
            color=PALETTE["Balanced_Performance"],
            marker="o",
            markersize=4.2,
            linewidth=1.3,
            capsize=2.2,
        )
        ax.axhline(float(nsga[metric].mean()), color=PALETTE["NSGA-II"], linewidth=1.1, linestyle="--")
        style_axis(ax)
        panel_label(ax, tag)
        ax.set_ylabel(TARGET_LABELS[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels)
        if ax in axes[1]:
            ax.set_xlabel("Scenario")

    normalized = normalized_benefit_frame(combined.copy())
    ddpg_utilities = []
    nsga_utilities = []
    for scenario in scenarios:
        weights = utility_weights[scenario]
        subset = normalized.copy()
        subset["utility"] = (
            weights[0] * subset["EUIt_score"]
            + weights[1] * subset["EG_score"]
            + weights[2] * subset["H_score"]
        )
        ddpg_best = subset[(subset["method"] == "DDPG") & (subset["scenario"] == scenario)].sort_values("utility", ascending=False).iloc[0]
        nsga_best = subset[subset["method"] == "NSGA-II"].sort_values("utility", ascending=False).iloc[0]
        ddpg_utilities.append(float(ddpg_best["utility"]))
        nsga_utilities.append(float(nsga_best["utility"]))

    ax = axes[1, 1]
    ax.plot(x, ddpg_utilities, color=PALETTE["Balanced_Performance"], marker="o", linewidth=1.3, label="DDPG")
    ax.plot(x, nsga_utilities, color=PALETTE["NSGA-II"], marker="s", linewidth=1.1, linestyle="--", label="NSGA-II")
    style_axis(ax)
    panel_label(ax, "d")
    ax.set_ylabel("Utility")
    ax.set_xlabel("Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)

    handles = [
        Line2D([0], [0], color=PALETTE["Balanced_Performance"], marker="o", lw=1.3, markersize=4.2, label="DDPG"),
        Line2D([0], [0], color=PALETTE["NSGA-II"], marker="s", lw=1.1, linestyle="--", markersize=4.0, label="NSGA-II"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, handletextpad=0.5, columnspacing=1.2)
    fig.tight_layout(pad=0.5, w_pad=0.8, h_pad=0.8, rect=(0, 0, 1, 0.93))
    _save(fig, out_path)


def build_fig10(ddpg_logs_all: dict, out_path: Path) -> None:
    _, summary = compute_seeded_convergence_diagnostics(ddpg_logs_all)
    scenarios = ["Balanced_Performance", "Energy_Saving_Focus", "Energy_Generation_Focus"]
    labels = ["Balanced", "Saving", "Generation"]
    x = np.arange(len(scenarios))
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 10.0 * CM_TO_IN))
    metrics = [
        ("reward_best_mean", "Best reward mean"),
        ("reward_final_mean", "Final reward mean"),
        ("plateau_episode_mean", "Plateau episode mean"),
        ("late_regression_seed_fraction", "Late regression fraction"),
    ]
    for ax, (metric, ylabel), tag in zip(axes.flatten(), metrics, ["a", "b", "c", "d"], strict=True):
        vals = [summary[s][metric] for s in scenarios]
        ax.bar(x, vals, color=[PALETTE[s] for s in scenarios], width=0.62)
        annotate_bar_values(ax, x, vals, "{:.2f}")
        style_axis(ax)
        panel_label(ax, tag)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    fig.tight_layout(pad=0.5, w_pad=0.8, h_pad=0.8)
    _save(fig, out_path)


def build_fig11(regime_winners: pd.DataFrame, out_path: Path) -> None:
    regime = regime_winners.sort_values("dataset_scale")
    scales = regime["dataset_scale"].tolist()
    x = np.arange(len(scales))
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 10.0 * CM_TO_IN))
    metrics = [
        ("mean_target_nmae", "Mean target nMAE"),
        ("mean_tail_nmae", "Mean tail nMAE"),
        ("mean_r2", "Mean $R^2$"),
        ("selection_objective", "Selection objective"),
    ]
    for ax, (metric, ylabel), tag in zip(axes.flatten(), metrics, ["a", "b", "c", "d"], strict=True):
        vals = regime[metric].to_numpy(dtype=float)
        colors = ["#4C78A8", "#4C78A8", "#4C78A8", "#D55E00"]
        ax.bar(x, vals, color=colors, width=0.62)
        annotate_bar_values(ax, x, list(vals), "{:.4f}" if max(vals) < 0.2 else "{:.3f}")
        style_axis(ax)
        panel_label(ax, tag)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in scales])
        ax.set_xlabel("Dataset scale")
    fig.tight_layout(pad=0.5, w_pad=0.8, h_pad=0.8)
    _save(fig, out_path)


def build_nonlinear_response(bundle, dataset: pd.DataFrame, out_pdf: Path) -> None:
    base_point = dataset[MORPHOLOGY_FEATURES].median().to_numpy(dtype=float)
    pairs = [("OSR", "EUIt"), ("FAR", "EG"), ("SVF", "H"), ("theta", "H")]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 10.2 * CM_TO_IN))
    for tag, ax, (feature, target) in zip(["a", "b", "c", "d"], axes.flatten(), pairs, strict=True):
        values = np.linspace(dataset[feature].quantile(0.05), dataset[feature].quantile(0.95), 100)
        responses = []
        feature_idx = MORPHOLOGY_FEATURES.index(feature)
        for value in values:
            probe = base_point.copy()
            probe[feature_idx] = value
            responses.append(bundle.predict_action(probe)[PERFORMANCE_TARGETS.index(target)])
        ax.plot(values, responses, color="#1D4ED8", linewidth=1.4)
        style_axis(ax)
        panel_label(ax, tag)
        ax.set_xlabel(FEATURE_LABELS.get(feature, feature))
        ax.set_ylabel(TARGET_LABELS[target])
    fig.tight_layout(pad=0.5, w_pad=0.8, h_pad=0.8)
    _save(fig, out_pdf)


def build_reevaluation(config: Config, combined: pd.DataFrame, out_csv: Path) -> None:
    normalized = normalized_benefit_frame(combined.copy())
    scenarios = ["Balanced_Performance", "Energy_Saving_Focus", "Energy_Generation_Focus"]
    rows = []
    for scenario in scenarios:
        weights = config["optimization"]["utility_weights"][scenario]
        subset = normalized.copy()
        subset["utility"] = (
            weights[0] * subset["EUIt_score"]
            + weights[1] * subset["EG_score"]
            + weights[2] * subset["H_score"]
        )
        ddpg_best = subset[(subset["method"] == "DDPG") & (subset["scenario"] == scenario)].sort_values("utility", ascending=False).iloc[0]
        nsga_best = subset[subset["method"] == "NSGA-II"].sort_values("utility", ascending=False).iloc[0]
        rows.append(ddpg_best[["method", "scenario", "seed", *MORPHOLOGY_FEATURES, "EUIt", "EG", "H"]].to_dict())
        rows.append(nsga_best[["method", "scenario", "seed", *MORPHOLOGY_FEATURES, "EUIt", "EG", "H"]].to_dict())
    selected = pd.DataFrame(rows).drop_duplicates(subset=["method", "scenario", "seed"]).reset_index(drop=True)
    reevaluated = reevaluate_candidates(config, selected[MORPHOLOGY_FEATURES].reset_index(drop=True), deterministic=True)
    reevaluated["method"] = selected["method"].to_numpy()
    reevaluated["scenario"] = selected["scenario"].to_numpy()
    reevaluated["seed"] = selected["seed"].to_numpy()
    for target in PERFORMANCE_TARGETS:
        reevaluated[f"surrogate_{target}"] = selected[target].to_numpy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    reevaluated.to_csv(out_csv, index=False)


def main() -> None:
    set_journal_style()
    root = Path.cwd()
    run_id = "20260405_highest_precision_2000_compare"
    run_root = _run_root(root, run_id)
    fig_dir = root / "elsarticle" / "fig"
    report_dir = run_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    cv = pd.read_csv(run_root / "models" / "cv_predictions.csv")
    ddpg = pd.read_csv(run_root / "optimization" / "ddpg_results.csv")
    nsga = pd.read_csv(run_root / "optimization" / "nsga2_results.csv")
    combined = pd.read_csv(run_root / "optimization" / "optimization_results.csv")
    logs_all = json.loads((run_root / "optimization" / "ddpg_logs_all.json").read_text(encoding="utf-8"))
    regime_winners = pd.read_csv(root / "artifacts" / "server_runs" / "20260405_surrogate_rebenchmark" / "models" / "surrogate_regime_winners.csv")
    config = Config.from_yaml(root / "configs" / f"revision.server_{run_id}.yaml")
    bundle = load_surrogate(run_root / "models" / "surrogate.pt")
    dataset = pd.read_csv(run_root / "data" / "simulated_samples.csv")

    build_fig4(cv, fig_dir / "fig4.pdf")
    build_fig5(cv, fig_dir / "fig5.pdf")
    build_fig6(logs_all, fig_dir / "fig6.pdf")
    build_fig7(combined, fig_dir / "fig7.pdf")
    build_fig8(combined, fig_dir / "fig8.pdf")
    build_fig9(ddpg, nsga, combined, config["optimization"]["utility_weights"], fig_dir / "fig9.pdf")
    build_fig10(logs_all, fig_dir / "fig10.pdf")
    build_fig11(regime_winners, fig_dir / "fig11.pdf")
    build_nonlinear_response(bundle, dataset, fig_dir / "nonlinear_response_profiles.pdf")
    build_reevaluation(config, combined, report_dir / "top_candidate_reevaluation_2000.csv")
    print(json.dumps({"status": "ok", "figure_dir": str(fig_dir), "report_dir": str(report_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
