from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd

from paper_repro.constants import MORPHOLOGY_FEATURES


CM_TO_IN = 1 / 2.54
DOUBLE_COL_IN = 17.5 * CM_TO_IN

METHOD_STYLES = {
    "NSGA-II": {"color": "#4D4D4D", "marker": "o", "label": "NSGA-II", "facecolors": "none", "alpha": 0.75, "linewidths": 0.8},
    "Balanced_Performance": {"color": "#1F77B4", "marker": "o", "label": "Balanced", "alpha": 0.72},
    "Energy_Saving_Focus": {"color": "#009E73", "marker": "^", "label": "Saving", "alpha": 0.72},
    "Energy_Generation_Focus": {"color": "#D55E00", "marker": "s", "label": "Generation", "alpha": 0.72},
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


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", direction="in", top=False, right=False)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(False)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )


def load_imported_results(root: Path) -> pd.DataFrame:
    ddpg = pd.read_csv(root / "artifacts/publication/imported/optimization/ddpg_results.csv")
    nsga = pd.read_csv(root / "artifacts/publication/imported/optimization/nsga2_results.csv")
    return pd.concat([ddpg, nsga], ignore_index=True)


def _concat_csv_glob(glob_pattern: str, directory: Path, output_path: Path) -> pd.DataFrame:
    parts = [pd.read_csv(path) for path in sorted(directory.glob(glob_pattern))]
    if not parts:
        raise FileNotFoundError(f"No files matched {glob_pattern} in {directory}")
    combined = pd.concat(parts, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined


def _combine_seed_logs(prefix_map: dict[str, str], directory: Path, output_path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    payload: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for scenario, prefix in prefix_map.items():
        scenario_payload: dict[str, list[dict[str, Any]]] = {}
        for path in sorted(directory.glob(f"ddpg_logs_all_{prefix}_*.json")):
            shard = json.loads(path.read_text(encoding="utf-8"))
            if scenario not in shard:
                continue
            for seed, rows in shard[scenario].items():
                scenario_payload[str(seed)] = rows
        if scenario_payload:
            payload[scenario] = scenario_payload
    if not payload:
        raise FileNotFoundError(f"No log shards found for {prefix_map}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_matched_results(root: Path) -> pd.DataFrame:
    optimization_dir = root / "artifacts/publication/optimization"
    ddpg_path = optimization_dir / "ddpg_results_remote_match.csv"
    if ddpg_path.exists():
        ddpg = pd.read_csv(ddpg_path)
    else:
        ddpg = _concat_csv_glob("ddpg_results_match_*.csv", optimization_dir, ddpg_path)
    nsga = pd.read_csv(optimization_dir / "nsga2_results.csv")
    return pd.concat([ddpg, nsga], ignore_index=True)


def load_matched_random(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "artifacts/publication/optimization/random_search_results_remote_match.csv")


def load_earlystop_results(root: Path) -> pd.DataFrame:
    optimization_dir = root / "artifacts/publication/optimization"
    earlystop_path = optimization_dir / "ddpg_results_earlystop_all.csv"
    if earlystop_path.exists():
        return pd.read_csv(earlystop_path)
    return _concat_csv_glob("ddpg_results_stop_*.csv", optimization_dir, earlystop_path)


def load_checkpoint_analysis(root: Path) -> dict:
    path = root / "artifacts/publication/reports/surrogate_checkpoint_analysis.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_guarded_logs(root: Path) -> dict:
    optimization_dir = root / "artifacts/publication/optimization"
    path = optimization_dir / "ddpg_logs_all_guardrail_full.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return _combine_seed_logs(
        {
            "Balanced_Performance": "guard_bp",
            "Energy_Saving_Focus": "guard_es",
            "Energy_Generation_Focus": "guard_eg",
        },
        optimization_dir,
        path,
    )


def load_earlystop_logs(root: Path) -> dict:
    optimization_dir = root / "artifacts/publication/optimization"
    path = optimization_dir / "ddpg_logs_all_earlystop_all.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return _combine_seed_logs(
        {
            "Balanced_Performance": "stop_bal",
            "Energy_Saving_Focus": "stop_es",
            "Energy_Generation_Focus": "stop_eg",
        },
        optimization_dir,
        path,
    )


def load_matched_logs(root: Path) -> dict:
    optimization_dir = root / "artifacts/publication/optimization"
    path = optimization_dir / "ddpg_logs_all_remote_match.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return _combine_seed_logs(
        {
            "Balanced_Performance": "match_bal",
            "Energy_Saving_Focus": "match_es",
            "Energy_Generation_Focus": "match_eg",
        },
        optimization_dir,
        path,
    )


def build_fig6_learning_curve(root: Path, out_path: Path) -> None:
    try:
        logs = load_guarded_logs(root)
    except Exception:
        logs = load_matched_logs(root)
    palette = {
        "Balanced_Performance": "#1F77B4",
        "Energy_Saving_Focus": "#009E73",
        "Energy_Generation_Focus": "#D55E00",
    }
    mapping = [
        ("cumulative_reward", "Cumulative reward (unitless)"),
        ("EUIt", "EUIt (kWh/m$^2$/y)"),
        ("EG", "EG ($10^6$ kWh/y)"),
        ("H", "H (h)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 10.6 * CM_TO_IN), sharex=True)
    for ax, (metric, ylabel), tag in zip(axes.flatten(), mapping, ["a", "b", "c", "d"], strict=True):
        for scenario, seed_map in logs.items():
            frames = []
            for seed, rows in sorted(seed_map.items(), key=lambda item: int(item[0])):
                frame = pd.DataFrame(rows)[["episode", metric]].rename(columns={metric: seed}).set_index("episode")
                frames.append(frame)
            merged = pd.concat(frames, axis=1).sort_index()
            mean_curve = merged.mean(axis=1)
            std_curve = merged.std(axis=1).fillna(0.0)
            color = palette[scenario]
            # Highlight the scenario mean and use the shaded band to summarize seed variability.
            ax.plot(mean_curve.index, mean_curve.values, color=color, linewidth=1.6, label=METHOD_STYLES[scenario]["label"], zorder=3)
            ax.fill_between(mean_curve.index, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.18, linewidth=0, zorder=2)
        style_axis(ax)
        panel_label(ax, tag)
        ax.set_ylabel(ylabel)
        if ax in axes[1]:
            ax.set_xlabel("Episode")
    handles = [Line2D([0], [0], color=palette[key], lw=1.6, label=METHOD_STYLES[key]["label"]) for key in ["Balanced_Performance", "Energy_Saving_Focus", "Energy_Generation_Focus"]]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.2,
    )
    fig.tight_layout(pad=0.5, w_pad=0.8, h_pad=0.8, rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_fig4_parity(root: Path, out_path: Path) -> None:
    cv = pd.read_csv(root / "artifacts/publication/models/cv_predictions.csv")
    targets = [
        ("EUIt", "EUIt true (kWh/m$^2$/y)", "EUIt predicted (kWh/m$^2$/y)"),
        ("EG", "EG true ($10^6$ kWh/y)", "EG predicted ($10^6$ kWh/y)"),
        ("H", "H true (h)", "H predicted (h)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_IN, 5.8 * CM_TO_IN))
    for ax, (target, xlabel, ylabel), tag in zip(axes, targets, ["a", "b", "c"], strict=True):
        x = cv[f"true_{target}"].to_numpy()
        y = cv[f"pred_{target}"].to_numpy()
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax.scatter(
            x,
            y,
            s=16,
            c="#1F77B4",
            alpha=0.7,
            edgecolors="white",
            linewidths=0.25,
        )
        ax.plot([lo, hi], [lo, hi], color="#4D4D4D", linewidth=1.1, linestyle="--")
        style_axis(ax)
        panel_label(ax, tag)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    fig.tight_layout(pad=0.5, w_pad=0.8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_fig5_residuals(root: Path, out_path: Path) -> None:
    cv = pd.read_csv(root / "artifacts/publication/models/cv_predictions.csv")
    targets = [
        ("EUIt", "Residual (kWh/m$^2$/y)"),
        ("EG", "Residual ($10^6$ kWh/y)"),
        ("H", "Residual (h)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_IN, 5.8 * CM_TO_IN))
    for ax, (target, xlabel), tag in zip(axes, targets, ["a", "b", "c"], strict=True):
        residual = cv[f"pred_{target}"] - cv[f"true_{target}"]
        sigma = float(residual.std(ddof=0))
        ax.hist(
            residual,
            bins=18,
            color="#4C78A8",
            alpha=0.75,
            edgecolor="white",
            linewidth=0.4,
            density=True,
        )
        ax.axvline(0.0, color="#4D4D4D", linewidth=1.0)
        ax.axvline(2.0 * sigma, color="#D55E00", linewidth=1.0, linestyle="--")
        ax.axvline(-2.0 * sigma, color="#D55E00", linewidth=1.0, linestyle="--")
        style_axis(ax)
        panel_label(ax, tag)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density (unitless)")
    handles = [
        Line2D([0], [0], color="#4D4D4D", lw=1.0, label="0"),
        Line2D([0], [0], color="#D55E00", lw=1.0, linestyle="--", label=r"$\pm 2\sigma$"),
    ]
    axes[2].legend(handles=handles, frameon=False, loc="upper right")
    fig.tight_layout(pad=0.5, w_pad=0.8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_fig7_objective_space(root: Path, out_path: Path) -> None:
    results = load_matched_results(root)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_IN, 5.8 * CM_TO_IN))
    pairs = [
        ("EG", "EUIt", "EG ($10^6$ kWh/y)", "EUIt (kWh/m$^2$/y)"),
        ("H", "EUIt", "H (h)", "EUIt (kWh/m$^2$/y)"),
        ("EG", "H", "EG ($10^6$ kWh/y)", "H (h)"),
    ]
    order = ["NSGA-II", "Balanced_Performance", "Energy_Saving_Focus", "Energy_Generation_Focus"]
    for ax, (x_col, y_col, xlabel, ylabel), tag in zip(axes, pairs, ["a", "b", "c"], strict=True):
        for method in order:
            if method == "NSGA-II":
                subset = results.loc[results["method"] == "NSGA-II"]
            else:
                subset = results.loc[results["scenario"] == method]
            style = METHOD_STYLES[method]
            scatter_kwargs = {
                "s": 18,
                "marker": style["marker"],
                "alpha": style.get("alpha", 0.72),
                "label": style["label"],
            }
            if method == "NSGA-II":
                scatter_kwargs.update({"facecolors": "none", "edgecolors": style["color"], "linewidths": style.get("linewidths", 0.8)})
            else:
                scatter_kwargs.update({"c": style["color"], "edgecolors": "white", "linewidths": 0.25})
            ax.scatter(subset[x_col], subset[y_col], **scatter_kwargs)
        style_axis(ax)
        panel_label(ax, tag)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    handles = []
    for method in order:
        style = METHOD_STYLES[method]
        if method == "NSGA-II":
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=style["marker"],
                    linestyle="",
                    markerfacecolor="none",
                    markeredgecolor=style["color"],
                    markeredgewidth=style.get("linewidths", 0.8),
                    markersize=6,
                    label=style["label"],
                )
            )
        else:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=style["marker"],
                    linestyle="",
                    markerfacecolor=style["color"],
                    markeredgecolor="white",
                    markeredgewidth=0.25,
                    markersize=6,
                    label=style["label"],
                )
            )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    fig.tight_layout(pad=0.5, w_pad=0.8, rect=(0, 0, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_fig8_signature_heatmap(root: Path, out_path: Path) -> None:
    results = load_matched_results(root)
    results["method_label"] = np.where(results["method"] == "NSGA-II", "NSGA-II", results["scenario"])
    order = ["NSGA-II", "Balanced_Performance", "Energy_Saving_Focus", "Energy_Generation_Focus"]
    medians = results.groupby("method_label")[MORPHOLOGY_FEATURES].median().reindex(order)
    scaled = medians.copy()
    for feature in MORPHOLOGY_FEATURES:
        col = medians[feature]
        denom = max(col.max() - col.min(), 1e-8)
        scaled[feature] = 2.0 * ((col - col.min()) / denom) - 1.0
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN, 4.8 * CM_TO_IN))
    im = ax.imshow(scaled.to_numpy(), cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(MORPHOLOGY_FEATURES)))
    ax.set_xticklabels([FEATURE_LABELS.get(feature, feature) for feature in MORPHOLOGY_FEATURES], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(["NSGA-II", "Balanced", "Saving", "Generation"])
    ax.tick_params(which="both", direction="in", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_label("Relative median (unitless)", fontsize=10)
    cbar.ax.tick_params(labelsize=9, direction="in")
    panel_label(ax, "a")
    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_fig9_matched_benchmark(root: Path, out_path: Path) -> None:
    ddpg = pd.read_csv(root / "artifacts/publication/optimization/ddpg_results_remote_match.csv")
    random = load_matched_random(root)
    nsga = pd.read_csv(root / "artifacts/publication/optimization/nsga2_results.csv")

    scenarios = ["Balanced_Performance", "Energy_Saving_Focus", "Energy_Generation_Focus"]
    scenario_labels = ["Balanced", "Saving", "Generation"]
    palette = {
        "DDPG": "#1F77B4",
        "Random Search": "#D55E00",
        "NSGA-II": "#4D4D4D",
    }

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 10.8 * CM_TO_IN), sharex=True)
    metric_specs = [
        ("EUIt", "EUIt (kWh/m$^2$/y)", "a"),
        ("EG", "EG ($10^6$ kWh/y)", "b"),
        ("H", "H (h)", "c"),
        ("reward", "Reward (unitless)", "d"),
    ]

    x = np.arange(len(scenarios))
    for ax, (metric, ylabel, tag) in zip(axes.flatten(), metric_specs, strict=True):
        ddpg_summary = ddpg.groupby("scenario")[metric].agg(["mean", "std"]).reindex(scenarios)
        rand_summary = random.groupby("scenario")[metric].agg(["mean", "std"]).reindex(scenarios)
        ax.errorbar(
            x - 0.04,
            ddpg_summary["mean"],
            yerr=ddpg_summary["std"],
            color=palette["DDPG"],
            marker="o",
            markersize=4.5,
            linewidth=1.4,
            capsize=2.5,
            label="DDPG",
        )
        ax.errorbar(
            x + 0.04,
            rand_summary["mean"],
            yerr=rand_summary["std"],
            color=palette["Random Search"],
            marker="s",
            markersize=4.2,
            linewidth=1.3,
            capsize=2.5,
            label="Random search",
        )
        if metric != "reward":
            ref_value = float(nsga[metric].mean())
            ax.axhline(ref_value, color=palette["NSGA-II"], linewidth=1.2, linestyle="--")
        style_axis(ax)
        panel_label(ax, tag)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels)
        if ax in axes[1]:
            ax.set_xlabel("Scenario")
    handles = [
        Line2D([0], [0], color=palette["DDPG"], marker="o", lw=1.4, markersize=4.5, label="DDPG"),
        Line2D([0], [0], color=palette["Random Search"], marker="s", lw=1.3, markersize=4.2, label="Random search"),
        Line2D([0], [0], color=palette["NSGA-II"], lw=1.2, linestyle="--", label="NSGA-II mean"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.2,
    )
    fig.tight_layout(pad=0.5, w_pad=0.8, h_pad=0.8, rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_fig10_earlystop(root: Path, out_path: Path) -> None:
    full = pd.read_csv(root / "artifacts/publication/optimization/ddpg_results_remote_match.csv")
    early = load_earlystop_results(root)
    full_logs = load_matched_logs(root)
    early_logs = load_earlystop_logs(root)

    scenarios = ["Balanced_Performance", "Energy_Saving_Focus", "Energy_Generation_Focus"]
    scenario_labels = ["Balanced", "Saving", "Generation"]
    x = np.arange(len(scenarios))
    colors = {"Full": "#1F77B4", "Early stop": "#D55E00"}

    def seeded_summary(log_payload: dict) -> pd.DataFrame:
        rows = []
        for scenario, seed_map in log_payload.items():
            for seed, entries in seed_map.items():
                frame = pd.DataFrame(entries)
                reward = frame["cumulative_reward"].to_numpy(dtype=float)
                best = float(reward.max())
                final = float(reward[-1])
                rows.append(
                    {
                        "scenario": scenario,
                        "best_final_gap_ratio": (best - final) / max(abs(best), 1e-8),
                        "late_regression": float((best - final) > 0.2 * max(abs(best), 1e-8)),
                    }
                )
        return pd.DataFrame(rows)

    full_seeded = seeded_summary(full_logs)
    early_seeded = seeded_summary(early_logs)

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 10.8 * CM_TO_IN), sharex=True)
    metric_specs = [
        ("EUIt", "EUIt (kWh/m$^2$/y)", "a"),
        ("EG", "EG ($10^6$ kWh/y)", "b"),
        ("H", "H (h)", "c"),
    ]
    for ax, (metric, ylabel, tag) in zip(axes.flatten()[:3], metric_specs, strict=True):
        full_summary = full.groupby("scenario")[metric].agg(["mean", "std"]).reindex(scenarios)
        early_summary = early.groupby("scenario")[metric].agg(["mean", "std"]).reindex(scenarios)
        ax.errorbar(x - 0.04, full_summary["mean"], yerr=full_summary["std"], color=colors["Full"], marker="o", markersize=4.5, linewidth=1.4, capsize=2.5, label="Full")
        ax.errorbar(x + 0.04, early_summary["mean"], yerr=early_summary["std"], color=colors["Early stop"], marker="s", markersize=4.2, linewidth=1.3, capsize=2.5, label="Early stop")
        style_axis(ax)
        panel_label(ax, tag)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels)
        if ax in axes[1]:
            ax.set_xlabel("Scenario")

    ax = axes[1, 1]
    full_gap = full_seeded.groupby("scenario")["best_final_gap_ratio"].mean().reindex(scenarios)
    early_gap = early_seeded.groupby("scenario")["best_final_gap_ratio"].mean().reindex(scenarios)
    full_reg = full_seeded.groupby("scenario")["late_regression"].mean().reindex(scenarios)
    early_reg = early_seeded.groupby("scenario")["late_regression"].mean().reindex(scenarios)
    ax.plot(x, full_gap, color=colors["Full"], marker="o", linewidth=1.4, label="Gap ratio (full)")
    ax.plot(x, early_gap, color=colors["Early stop"], marker="s", linewidth=1.3, label="Gap ratio (early stop)")
    ax.plot(x, full_reg, color=colors["Full"], marker="^", linewidth=1.2, linestyle="--", label="Late regression (full)")
    ax.plot(x, early_reg, color=colors["Early stop"], marker="D", linewidth=1.1, linestyle="--", label="Late regression (early stop)")
    style_axis(ax)
    panel_label(ax, "d")
    ax.set_ylabel("Fraction / ratio (unitless)")
    ax.set_xlabel("Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    if ax.legend_ is not None:
        ax.legend_.remove()
    handles = [
        Line2D([0], [0], color=colors["Full"], marker="o", lw=1.4, markersize=4.5, label="Full"),
        Line2D([0], [0], color=colors["Early stop"], marker="s", lw=1.3, markersize=4.2, label="Early stop"),
        Line2D([0], [0], color=colors["Full"], marker="^", lw=1.2, linestyle="--", markersize=4.2, label="Late regression (full)"),
        Line2D([0], [0], color=colors["Early stop"], marker="D", lw=1.1, linestyle="--", markersize=4.0, label="Late regression (early stop)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    fig.tight_layout(pad=0.5, w_pad=0.8, h_pad=0.8, rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_fig11_checkpoint_sensitivity(root: Path, out_path: Path) -> None:
    analysis = load_checkpoint_analysis(root)
    targets = ["EUIt", "EG", "H"]
    target_labels = ["EUIt", "EG", "H"]
    # Match the manuscript palette more closely: blue for the local/publication
    # checkpoint and teal for the remote/matched checkpoint.
    colors = {"checkpoint_a": "#1F77B4", "checkpoint_b": "#009E73"}
    x = np.arange(len(targets))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COL_IN, 10.0 * CM_TO_IN),
        gridspec_kw={"width_ratios": [1.1, 1.0], "height_ratios": [1.0, 0.95]},
    )

    ax = axes[0, 0]
    width = 0.34
    mae_a = [analysis["checkpoint_a"]["training_fit"][t]["mae"] for t in targets]
    mae_b = [analysis["checkpoint_b"]["training_fit"][t]["mae"] for t in targets]
    ax.bar(x - width / 2, mae_a, width=width, color=colors["checkpoint_a"], label="Local checkpoint")
    ax.bar(x + width / 2, mae_b, width=width, color=colors["checkpoint_b"], label="Remote checkpoint")
    style_axis(ax)
    panel_label(ax, "a")
    ax.set_ylabel("Training MAE (unitless)")
    ax.set_xticks(x)
    ax.set_xticklabels(target_labels)

    ax = axes[0, 1]
    categories = ["EUIt < min", "EUIt > max", "EG < min", "EG > max", "H < min", "H > max"]
    x2 = np.arange(len(categories))
    vals_a = [
        analysis["checkpoint_a"]["random_raw"]["EUIt"]["below_min_frac"],
        analysis["checkpoint_a"]["random_raw"]["EUIt"]["above_max_frac"],
        analysis["checkpoint_a"]["random_raw"]["EG"]["below_min_frac"],
        analysis["checkpoint_a"]["random_raw"]["EG"]["above_max_frac"],
        analysis["checkpoint_a"]["random_raw"]["H"]["below_min_frac"],
        analysis["checkpoint_a"]["random_raw"]["H"]["above_max_frac"],
    ]
    vals_b = [
        analysis["checkpoint_b"]["random_raw"]["EUIt"]["below_min_frac"],
        analysis["checkpoint_b"]["random_raw"]["EUIt"]["above_max_frac"],
        analysis["checkpoint_b"]["random_raw"]["EG"]["below_min_frac"],
        analysis["checkpoint_b"]["random_raw"]["EG"]["above_max_frac"],
        analysis["checkpoint_b"]["random_raw"]["H"]["below_min_frac"],
        analysis["checkpoint_b"]["random_raw"]["H"]["above_max_frac"],
    ]
    ax.bar(x2 - width / 2, vals_a, width=width, color=colors["checkpoint_a"])
    ax.bar(x2 + width / 2, vals_b, width=width, color=colors["checkpoint_b"])
    style_axis(ax)
    panel_label(ax, "b")
    ax.set_ylabel("Out-of-bound fraction (unitless)")
    ax.set_xticks(x2)
    ax.set_xticklabels(categories, rotation=24, ha="right")
    ax.tick_params(axis="x", labelsize=8)

    ax = axes[1, 0]
    triple_a = analysis["checkpoint_a"]["random_raw"]["triple_better_than_bounds_frac"]
    triple_b = analysis["checkpoint_b"]["random_raw"]["triple_better_than_bounds_frac"]
    ax.bar([0, 1], [triple_a, triple_b], color=[colors["checkpoint_a"], colors["checkpoint_b"]], width=0.55)
    style_axis(ax)
    panel_label(ax, "c")
    ax.set_ylabel("Triple-tail fraction (unitless)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Local", "Remote"])
    for idx, value in enumerate([triple_a, triple_b]):
        ax.text(idx, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 1]
    metric_labels = ["EUIt < min", "H < min", "Triple tail"]
    local_vals = [
        analysis["checkpoint_a"]["random_raw"]["EUIt"]["below_min_frac"],
        analysis["checkpoint_a"]["random_raw"]["H"]["below_min_frac"],
        analysis["checkpoint_a"]["random_raw"]["triple_better_than_bounds_frac"],
    ]
    remote_vals = [
        analysis["checkpoint_b"]["random_raw"]["EUIt"]["below_min_frac"],
        analysis["checkpoint_b"]["random_raw"]["H"]["below_min_frac"],
        analysis["checkpoint_b"]["random_raw"]["triple_better_than_bounds_frac"],
    ]
    x3 = np.arange(len(metric_labels))
    ax.plot(x3, local_vals, color=colors["checkpoint_a"], marker="o", linewidth=1.4, markersize=4.5, label="Local checkpoint")
    ax.plot(x3, remote_vals, color=colors["checkpoint_b"], marker="s", linewidth=1.3, markersize=4.2, label="Remote checkpoint")
    style_axis(ax)
    panel_label(ax, "d")
    ax.set_ylabel("Tail diagnostic (unitless)")
    ax.set_xticks(x3)
    ax.set_xticklabels(metric_labels)

    handles = [
        Line2D([0], [0], color=colors["checkpoint_a"], marker="s", lw=0, markersize=7, label="Local checkpoint"),
        Line2D([0], [0], color=colors["checkpoint_b"], marker="s", lw=0, markersize=7, label="Remote checkpoint"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.4,
    )

    fig.tight_layout(pad=0.5, w_pad=0.9, h_pad=0.9, rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_publication_style()
    root = Path.cwd()
    fig_dir = root / "elsarticle" / "fig"
    if (root / "artifacts/publication/models/cv_predictions.csv").exists():
        build_fig4_parity(root, fig_dir / "fig4.pdf")
        build_fig5_residuals(root, fig_dir / "fig5.pdf")
    if (root / "artifacts/publication/optimization/ddpg_logs_all_guardrail_full.json").exists():
        build_fig6_learning_curve(root, fig_dir / "fig6.pdf")
    build_fig7_objective_space(root, fig_dir / "fig7.pdf")
    build_fig8_signature_heatmap(root, fig_dir / "fig8.pdf")
    build_fig9_matched_benchmark(root, fig_dir / "fig9.pdf")
    build_fig10_earlystop(root, fig_dir / "fig10.pdf")
    build_fig11_checkpoint_sensitivity(root, fig_dir / "fig11.pdf")
    print(
        json.dumps(
            {
                "figures": [
                    "elsarticle/fig/fig4.pdf",
                    "elsarticle/fig/fig5.pdf",
                    "elsarticle/fig/fig6.pdf",
                    "elsarticle/fig/fig7.pdf",
                    "elsarticle/fig/fig8.pdf",
                    "elsarticle/fig/fig9.pdf",
                    "elsarticle/fig/fig10.pdf",
                    "elsarticle/fig/fig11.pdf",
                ]
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
