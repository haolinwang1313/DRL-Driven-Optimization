from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd

from paper_repro.config import Config
from paper_repro.constants import MORPHOLOGY_FEATURES, PERFORMANCE_TARGETS
from paper_repro.metrics import normalized_benefit_frame
from paper_repro.surrogate import load_surrogate


CURRENT_COMPARE_RUN = "20260405_highest_precision_2000_compare"
CURRENT_SELECTION_RUN = "20260405_surrogate_rebenchmark"
STATIC_FIG_COMMIT = "87b1e66eb217251587be94e95e73898ed4740859"
DDPG_LOG_SHARD_GROUPS = {
    "rev": ["balrev", "savrev", "genrev"],
    "match": ["match_bal", "match_es", "match_eg"],
    "guard": ["guard_bp", "guard_es", "guard_eg"],
    "stop": ["stop_bal", "stop_es", "stop_eg"],
}

CM_TO_IN = 1 / 2.54
DOUBLE_COL_IN = 17.5 * CM_TO_IN

PALETTE = {
    "NSGA-II": "#4D4D4D",
    "Balanced_Performance": "#1F77B4",
    "Energy_Saving_Focus": "#009E73",
    "Energy_Generation_Focus": "#D55E00",
}

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


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _restore_pdf_from_git(repo_root: Path, commit: str, repo_relative_path: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        subprocess.run(
            ["git", "show", f"{commit}:{repo_relative_path}"],
            cwd=repo_root,
            check=True,
            stdout=handle,
        )


def _compile_manuscript(repo_root: Path) -> None:
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "manuscript.tex"],
        cwd=repo_root / "elsarticle",
        check=True,
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_result_roots(repo_root: Path) -> tuple[Path, Path]:
    compare_root = repo_root / "artifacts" / "server_runs" / CURRENT_COMPARE_RUN
    selection_root = repo_root / "artifacts" / "server_runs" / CURRENT_SELECTION_RUN
    if compare_root.exists():
        return compare_root, selection_root

    publication_root = repo_root / "artifacts" / "publication"
    if publication_root.exists():
        return publication_root, publication_root

    raise FileNotFoundError(
        "Neither artifacts/server_runs/<current_run> nor artifacts/publication exists; cannot rebuild manuscript figures."
    )


def _merge_ddpg_logs_all_from_shards(optimization_dir: Path, prefixes: list[str]) -> tuple[pd.DataFrame, dict[str, dict[str, list[dict]]]]:
    result_frames: list[pd.DataFrame] = []
    merged_logs_all: dict[str, dict[str, list[dict]]] = {}

    for prefix in prefixes:
        result_frames.extend(pd.read_csv(path) for path in sorted(optimization_dir.glob(f"ddpg_results_{prefix}_*.csv")))
        for json_path in sorted(optimization_dir.glob(f"ddpg_logs_all_{prefix}_*.json")):
            payload = _load_json(json_path)
            for scenario, seed_map in payload.items():
                scenario_bucket = merged_logs_all.setdefault(scenario, {})
                for seed, rows in seed_map.items():
                    scenario_bucket[str(seed)] = rows

    if not result_frames:
        raise FileNotFoundError(f"No DDPG shard results found for prefixes: {prefixes}")

    merged_frame = pd.concat(result_frames, ignore_index=True).sort_values(["scenario", "seed"]).reset_index(drop=True)
    return merged_frame, merged_logs_all


def _load_ddpg_logs_all(compare_root: Path) -> dict[str, dict[str, list[dict]]]:
    optimization_dir = compare_root / "optimization"
    logs_all_path = optimization_dir / "ddpg_logs_all.json"
    if logs_all_path.exists():
        return _load_json(logs_all_path)

    base_results = pd.read_csv(optimization_dir / "ddpg_results.csv").sort_values(["scenario", "seed"]).reset_index(drop=True)
    base_columns = list(base_results.columns)

    for prefixes in DDPG_LOG_SHARD_GROUPS.values():
        try:
            merged_frame, merged_logs_all = _merge_ddpg_logs_all_from_shards(optimization_dir, prefixes)
        except FileNotFoundError:
            continue
        if list(merged_frame.columns) != base_columns or len(merged_frame) != len(base_results):
            continue
        try:
            pd.testing.assert_frame_equal(
                base_results[base_columns],
                merged_frame[base_columns],
                check_exact=False,
                atol=1e-8,
                rtol=1e-8,
            )
            return merged_logs_all
        except AssertionError:
            continue

    raise FileNotFoundError(
        f"Could not infer ddpg_logs_all.json from shard outputs under {optimization_dir}."
    )


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


def build_fig6(ddpg_logs_all: dict[str, dict[str, list[dict[str, float]]]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 10.2 * CM_TO_IN), sharex=True)
    mapping = [
        ("cumulative_reward", "Cumulative reward"),
        ("EUIt", TARGET_LABELS["EUIt"]),
        ("EG", TARGET_LABELS["EG"]),
        ("H", TARGET_LABELS["H"]),
    ]
    scenarios = ["Balanced_Performance", "Energy_Saving_Focus", "Energy_Generation_Focus"]
    for ax, (metric, ylabel), tag in zip(axes.flatten(), mapping, ["a", "b", "c", "d"], strict=True):
        for scenario in scenarios:
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
        for s in scenarios
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
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=PALETTE["NSGA-II"], markeredgewidth=0.8, markersize=6, label="NSGA-II"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=PALETTE["Balanced_Performance"], markeredgecolor="white", markeredgewidth=0.25, markersize=6, label="Balanced Performance"),
        Line2D([0], [0], marker="^", linestyle="", markerfacecolor=PALETTE["Energy_Saving_Focus"], markeredgecolor="white", markeredgewidth=0.25, markersize=6, label="Energy Saving Focus"),
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor=PALETTE["Energy_Generation_Focus"], markeredgecolor="white", markeredgewidth=0.25, markersize=6, label="Energy Generation Focus"),
    ]
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
    im = ax.imshow(scaled.to_numpy(), cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto", interpolation="nearest")
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


def build_fig9(ddpg: pd.DataFrame, nsga: pd.DataFrame, combined: pd.DataFrame, utility_weights: dict[str, list[float]], out_path: Path) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the current manuscript figure set.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root containing elsarticle/ and artifacts/server_runs/.")
    parser.add_argument("--static-fig-commit", default=STATIC_FIG_COMMIT, help="Commit that provides the kept fig10/fig11 PDFs.")
    parser.add_argument("--compile-manuscript", action="store_true", help="Run latexmk after rebuilding the figure PDFs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    compare_root, selection_root = _resolve_result_roots(root)
    fig_dir = root / "elsarticle" / "fig"

    set_journal_style()
    cv = pd.read_csv(compare_root / "models" / "cv_predictions.csv")
    ddpg = pd.read_csv(compare_root / "optimization" / "ddpg_results.csv")
    nsga = pd.read_csv(compare_root / "optimization" / "nsga2_results.csv")
    combined = pd.read_csv(compare_root / "optimization" / "optimization_results.csv")
    logs_all = _load_ddpg_logs_all(compare_root)
    config = Config.from_yaml(root / "configs" / "revision.yaml")
    bundle = load_surrogate(compare_root / "models" / "surrogate.pt")
    dataset = pd.read_csv(compare_root / "data" / "simulated_samples.csv")

    build_fig4(cv, fig_dir / "fig4.pdf")
    build_fig5(cv, fig_dir / "fig5.pdf")
    build_fig6(logs_all, fig_dir / "fig6.pdf")
    build_fig7(combined, fig_dir / "fig7.pdf")
    build_fig8(combined, fig_dir / "fig8.pdf")
    build_fig9(ddpg, nsga, combined, config["optimization"]["utility_weights"], fig_dir / "fig9.pdf")
    build_nonlinear_response(bundle, dataset, fig_dir / "nonlinear_response_profiles.pdf")

    # fig10/fig11 are intentionally pinned to the approved manuscript version.
    _restore_pdf_from_git(root, args.static_fig_commit, "elsarticle/fig/fig10.pdf", fig_dir / "fig10.pdf")
    _restore_pdf_from_git(root, args.static_fig_commit, "elsarticle/fig/fig11.pdf", fig_dir / "fig11.pdf")

    if args.compile_manuscript:
        _compile_manuscript(root)

    print(
        json.dumps(
            {
                "status": "ok",
                "compare_run": str(compare_root),
                "selection_run": str(selection_root),
                "figure_dir": str(fig_dir),
                "static_fig_commit": args.static_fig_commit,
                "compiled": args.compile_manuscript,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
