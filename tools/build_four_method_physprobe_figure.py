from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


CM_TO_IN = 1 / 2.54
DOUBLE_COL_IN = 17.5 * CM_TO_IN

METHOD_ORDER = ["DDPG", "NSGA-II", "CMA-ES", "RandomSearch"]
DISPLAY_LABELS = {
    "DDPG": "DDPG",
    "NSGA-II": "NSGA-II",
    "CMA-ES": "CMA-ES",
    "RandomSearch": "Random",
}
COLORS = {
    "DDPG": "#1F77B4",
    "NSGA-II": "#4D4D4D",
    "CMA-ES": "#D55E00",
    "RandomSearch": "#009E73",
}
TARGET_LABELS = {
    "EUIt": "EUIt (kWh/m$^2$/y)",
    "EG": "EG ($10^6$ kWh/y)",
    "H": "H (h)",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.03,
        0.97,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.2},
    )


def load_plot_frame(repo_root: Path) -> pd.DataFrame:
    reevaluation_dir = repo_root / "artifacts" / "publication" / "reevaluation"
    diagnostics_dir = repo_root / "artifacts" / "publication" / "diagnostics"

    physical_rows = pd.DataFrame(
        json.loads((diagnostics_dir / "physical_stack_result_edff06a1d365.json").read_text(encoding="utf-8"))
    )[["method", "scenario", "seed", "physical_EUIt", "physical_EG_total_production", "physical_H_proxy"]]

    ddpg = pd.read_csv(reevaluation_dir / "physical_stack_candidate_probe_asynccheck30.csv")
    nsga = pd.read_csv(reevaluation_dir / "physical_stack_candidate_probe_asynccheck29.csv")
    cmaes = pd.read_csv(repo_root / "artifacts" / "publication" / "optimization" / "cmaes_results_audit_cmaes.csv")
    random_search = pd.read_csv(repo_root / "artifacts" / "publication" / "optimization" / "random_search_results_remote_match.csv")

    physical_overrides = pd.concat(
        [
            ddpg[["method", "scenario", "seed", "physical_EUIt", "physical_EG_total_production", "physical_H_proxy"]],
            nsga[["method", "scenario", "seed", "physical_EUIt", "physical_EG_total_production", "physical_H_proxy"]],
        ],
        ignore_index=True,
    )
    physical_rows = pd.concat(
        [
            physical_overrides,
            physical_rows[~physical_rows["method"].isin(["DDPG", "NSGA-II"])],
        ],
        ignore_index=True,
    )

    surrogate_rows = pd.concat(
        [
            ddpg[["method", "scenario", "seed", "surrogate_EUIt", "surrogate_EG", "surrogate_H"]],
            nsga[["method", "scenario", "seed", "surrogate_EUIt", "surrogate_EG", "surrogate_H"]],
            cmaes.rename(columns={"EUIt": "surrogate_EUIt", "EG": "surrogate_EG", "H": "surrogate_H"})[
                ["method", "scenario", "seed", "surrogate_EUIt", "surrogate_EG", "surrogate_H"]
            ],
            random_search.rename(columns={"EUIt": "surrogate_EUIt", "EG": "surrogate_EG", "H": "surrogate_H"})[
                ["method", "scenario", "seed", "surrogate_EUIt", "surrogate_EG", "surrogate_H"]
            ].query("scenario == 'Balanced_Performance' and seed == 11"),
        ],
        ignore_index=True,
    )

    frame = physical_rows.merge(surrogate_rows, on=["method", "scenario", "seed"], how="left", validate="one_to_one")
    frame["method"] = pd.Categorical(frame["method"], categories=METHOD_ORDER, ordered=True)
    frame = frame.sort_values("method").reset_index(drop=True)
    return frame


def build_figure(frame: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_IN, 5.8 * CM_TO_IN))
    metrics = [
        ("surrogate_EUIt", "physical_EUIt", TARGET_LABELS["EUIt"]),
        ("surrogate_EG", "physical_EG_total_production", TARGET_LABELS["EG"]),
        ("surrogate_H", "physical_H_proxy", TARGET_LABELS["H"]),
    ]
    y_positions = list(range(len(frame)))[::-1]

    for ax, (sur_col, phys_col, xlabel), tag in zip(axes, metrics, ["a", "b", "c"], strict=True):
        for y, (_, row) in zip(y_positions, frame.iterrows(), strict=True):
            color = COLORS[str(row["method"])]
            ax.plot(
                [float(row[sur_col]), float(row[phys_col])],
                [y, y],
                color=color,
                linewidth=1.4,
                alpha=0.85,
                zorder=1,
            )
            ax.scatter(
                [float(row[sur_col])],
                [y],
                s=34,
                facecolors="white",
                edgecolors=color,
                linewidths=1.1,
                marker="s",
                zorder=3,
            )
            ax.scatter(
                [float(row[phys_col])],
                [y],
                s=34,
                c=color,
                edgecolors="white",
                linewidths=0.35,
                marker="o",
                zorder=4,
            )
        ax.set_xlabel(xlabel)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([DISPLAY_LABELS[m] for m in frame["method"]])
        style_axis(ax)
        panel_label(ax, tag)

    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor="white", markeredgecolor="#444444", markeredgewidth=1.0, markersize=5.5, label="Surrogate"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#444444", markeredgecolor="white", markeredgewidth=0.35, markersize=5.5, label="Physical probe"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.5,
    )
    fig.tight_layout(pad=0.5, w_pad=0.8, rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    set_style()
    frame = load_plot_frame(repo_root)
    build_figure(frame, repo_root / "elsarticle" / "fig" / "fig12_four_method_physprobe.pdf")


if __name__ == "__main__":
    main()
