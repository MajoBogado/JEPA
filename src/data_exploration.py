"""
src/data_exploration.py

Step 2: Exploratory analysis of the OASIS-2 longitudinal dataset.

Covers:
- Dataset shape and column inventory
- Visit count distribution per participant
- Converter identification and conversion timing
- MMSE trajectory plots (converters vs non-converters)
- Missing value audit
- Group column value verification
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# ── Colour palette used across all plots ────────────────────────────────────
GROUP_COLOUR_MAP = {
    "Nondemented": "#2196F3",   # blue
    "Demented": "#F44336",      # red
    "Converted": "#FF9800",     # amber
}


# ── I/O helpers ─────────────────────────────────────────────────────────────

def load_raw_dataset(raw_data_path: Path) -> pd.DataFrame:
    """Load the raw OASIS-2 CSV and normalise column names."""
    oasis_df = pd.read_csv(raw_data_path)
    oasis_df.columns = oasis_df.columns.str.strip()
    print(f"[exploration] Loaded dataset: {oasis_df.shape[0]} rows × {oasis_df.shape[1]} columns")
    return oasis_df


# ── Analysis functions ───────────────────────────────────────────────────────

def summarise_dataset_structure(oasis_df: pd.DataFrame) -> None:
    """Print shape, dtypes, and per-column null counts."""
    print("\n── Dataset structure ──────────────────────────────────")
    print(oasis_df.dtypes.to_string())

    null_counts = oasis_df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if null_cols.empty:
        print("\nNo missing values found.")
    else:
        print(f"\nColumns with missing values:\n{null_cols.to_string()}")

    print(f"\nGroup distribution (rows):\n{oasis_df['Group'].value_counts().to_string()}")

    participant_group = (
        oasis_df.groupby("Subject ID")["Group"]
        .apply(lambda groups: "Converted" if "Converted" in groups.values else groups.iloc[0])
    )
    print(f"\nGroup distribution (participants):\n{participant_group.value_counts().to_string()}")


def compute_visit_counts_per_participant(oasis_df: pd.DataFrame) -> pd.Series:
    """Return a Series mapping Subject ID → number of visits."""
    visit_counts = oasis_df.groupby("Subject ID")["Visit"].count()
    visit_counts.name = "num_visits"
    return visit_counts


def summarise_visit_distribution(visit_counts: pd.Series) -> None:
    """Print how many participants have 2, 3, 4, 5 visits."""
    print("\n── Visit count distribution ───────────────────────────")
    visit_freq = visit_counts.value_counts().sort_index()
    for n_visits, n_participants in visit_freq.items():
        print(f"  {n_visits} visits : {n_participants} participants")
    print(f"  Total participants : {len(visit_counts)}")


def identify_converters(oasis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame of converter participants with baseline and final metrics.
    Conversion visit is not tracked here — the 'Converted' group label is applied
    to all visits of a converter participant in OASIS-2, so it cannot be used to
    pinpoint timing. What matters for modelling is that the participant is a converter
    and whether their CDR worsened after visit 2.
    """
    converter_subject_ids = (
        oasis_df[oasis_df["Group"] == "Converted"]["Subject ID"].unique()
    )

    converter_records = []
    for subject_id in converter_subject_ids:
        subject_visits = oasis_df[oasis_df["Subject ID"] == subject_id].sort_values("Visit")

        baseline_row = subject_visits.iloc[0]
        final_row = subject_visits.iloc[-1]

        converter_records.append({
            "subject_id": subject_id,
            "total_visits": len(subject_visits),
            "baseline_mmse": baseline_row["MMSE"],
            "baseline_cdr": baseline_row["CDR"],
            "final_mmse": final_row["MMSE"],
            "final_cdr": final_row["CDR"],
            "mmse_drop_baseline_to_final": baseline_row["MMSE"] - final_row["MMSE"],
        })

    converters_df = pd.DataFrame(converter_records)
    return converters_df


def summarise_converters(converters_df: pd.DataFrame) -> None:
    """Print converter summary to stdout."""
    print("\n── Converter participants ─────────────────────────────")
    print(f"  Total converters : {len(converters_df)}")
    print(f"\n  Converter details:")
    print(converters_df.to_string(index=False))


def audit_missing_values(oasis_df: pd.DataFrame) -> pd.DataFrame:
    """Return and print a DataFrame summarising null counts and % per column."""
    total_rows = len(oasis_df)
    null_summary = pd.DataFrame({
        "null_count": oasis_df.isnull().sum(),
        "null_pct": (oasis_df.isnull().sum() / total_rows * 100).round(2),
    })
    null_summary = null_summary[null_summary["null_count"] > 0]

    print("\n── Missing value audit ────────────────────────────────")
    if null_summary.empty:
        print("  No missing values detected.")
    else:
        print(null_summary.to_string())

    return null_summary


def label_participant_group(oasis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each participant a single group label for plotting.
    Participants who appear as 'Converted' at any visit are labelled Converted,
    regardless of their other visit labels.
    """
    participant_group_label = (
        oasis_df.groupby("Subject ID")["Group"]
        .apply(
            lambda groups: "Converted" if "Converted" in groups.values else groups.iloc[0]
        )
        .reset_index()
        .rename(columns={"Group": "participant_group"})
    )
    return oasis_df.merge(participant_group_label, on="Subject ID")


# ── Plot functions ───────────────────────────────────────────────────────────

def plot_visit_count_distribution(visit_counts: pd.Series, figures_dir: Path) -> None:
    """Bar chart: number of participants by visit count."""
    visit_freq = visit_counts.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        visit_freq.index.astype(str),
        visit_freq.values,
        color="#455A64",
        width=0.55,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_xlabel("Number of visits", fontsize=11)
    ax.set_ylabel("Number of participants", fontsize=11)
    ax.set_title("Visit count distribution — OASIS-2", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for bar_patch, count in zip(ax.patches, visit_freq.values):
        ax.text(
            bar_patch.get_x() + bar_patch.get_width() / 2,
            bar_patch.get_height() + 0.3,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    output_path = figures_dir / "visit_count_distribution.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[exploration] Saved: {output_path}")


def plot_mmse_trajectories_by_group(oasis_df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Line plot of MMSE over visit number, one line per participant,
    coloured by their dominant group label. Converters drawn on top.
    """
    oasis_labelled = label_participant_group(oasis_df)

    draw_order = ["Nondemented", "Demented", "Converted"]
    alpha_by_group = {"Nondemented": 0.25, "Demented": 0.25, "Converted": 0.85}
    linewidth_by_group = {"Nondemented": 0.8, "Demented": 0.8, "Converted": 1.8}

    fig, ax = plt.subplots(figsize=(10, 6))

    for group_label in draw_order:
        group_subjects = oasis_labelled[oasis_labelled["participant_group"] == group_label]
        for subject_id, subject_visits in group_subjects.groupby("Subject ID"):
            subject_sorted = subject_visits.sort_values("Visit")
            ax.plot(
                subject_sorted["Visit"],
                subject_sorted["MMSE"],
                color=GROUP_COLOUR_MAP[group_label],
                alpha=alpha_by_group[group_label],
                linewidth=linewidth_by_group[group_label],
            )

    legend_handles = [
        plt.Line2D([0], [0], color=GROUP_COLOUR_MAP[g], linewidth=2, label=g)
        for g in draw_order
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc="lower left")
    ax.set_xlabel("Study visit number", fontsize=11)
    ax.set_ylabel("MMSE score", fontsize=11)
    ax.set_title("MMSE trajectories by participant group — OASIS-2", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 32)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    output_path = figures_dir / "mmse_trajectories_by_group.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[exploration] Saved: {output_path}")


def plot_converter_mmse_trajectories(
    oasis_df: pd.DataFrame,
    converters_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    One subplot per converter showing their MMSE trajectory.
    A vertical dashed line marks the visit at which conversion was recorded.
    """
    converter_ids = converters_df["subject_id"].tolist()
    n_converters = len(converter_ids)
    n_cols = 4
    n_rows = int(np.ceil(n_converters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.2), sharey=True)
    axes_flat = axes.flatten()

    for plot_idx, subject_id in enumerate(converter_ids):
        ax = axes_flat[plot_idx]
        subject_visits = oasis_df[oasis_df["Subject ID"] == subject_id].sort_values("Visit")

        ax.plot(
            subject_visits["Visit"],
            subject_visits["MMSE"],
            marker="o",
            color=GROUP_COLOUR_MAP["Converted"],
            linewidth=1.8,
            markersize=5,
        )

        ax.set_title(subject_id, fontsize=8, fontweight="bold")
        ax.set_xlabel("Visit", fontsize=8)
        if plot_idx % n_cols == 0:
            ax.set_ylabel("MMSE", fontsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for unused_idx in range(n_converters, len(axes_flat)):
        axes_flat[unused_idx].set_visible(False)

    fig.suptitle(
        "MMSE trajectories — converter participants\n(dashed line = first visit labelled Converted)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    output_path = figures_dir / "converter_mmse_trajectories.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[exploration] Saved: {output_path}")


def plot_nwbv_trajectories_by_group(oasis_df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Line plot of normalised whole brain volume (nWBV) over visit number.
    Brain atrophy signal that an LLM cannot reason about from text alone.
    """
    oasis_labelled = label_participant_group(oasis_df)

    draw_order = ["Nondemented", "Demented", "Converted"]
    alpha_by_group = {"Nondemented": 0.25, "Demented": 0.25, "Converted": 0.85}
    linewidth_by_group = {"Nondemented": 0.8, "Demented": 0.8, "Converted": 1.8}

    fig, ax = plt.subplots(figsize=(10, 6))

    for group_label in draw_order:
        group_subjects = oasis_labelled[oasis_labelled["participant_group"] == group_label]
        for subject_id, subject_visits in group_subjects.groupby("Subject ID"):
            subject_sorted = subject_visits.sort_values("Visit")
            ax.plot(
                subject_sorted["Visit"],
                subject_sorted["nWBV"],
                color=GROUP_COLOUR_MAP[group_label],
                alpha=alpha_by_group[group_label],
                linewidth=linewidth_by_group[group_label],
            )

    legend_handles = [
        plt.Line2D([0], [0], color=GROUP_COLOUR_MAP[g], linewidth=2, label=g)
        for g in draw_order
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc="upper right")
    ax.set_xlabel("Study visit number", fontsize=11)
    ax.set_ylabel("Normalised Whole Brain Volume (nWBV)", fontsize=11)
    ax.set_title("Brain volume trajectories by participant group — OASIS-2", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    output_path = figures_dir / "nwbv_trajectories_by_group.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[exploration] Saved: {output_path}")


def plot_baseline_distributions(oasis_df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Box plots of MMSE and nWBV at visit 1, split by group.
    Shows how similar converters look to non-demented participants at baseline.
    """
    oasis_labelled = label_participant_group(oasis_df)
    baseline_visits = oasis_labelled[oasis_labelled["Visit"] == 1].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, ["MMSE", "nWBV"]):
        group_data = [
            baseline_visits[baseline_visits["participant_group"] == g][metric].dropna()
            for g in ["Nondemented", "Converted", "Demented"]
        ]
        box_plot = ax.boxplot(
            group_data,
            patch_artist=True,
            medianprops={"color": "white", "linewidth": 2},
            widths=0.45,
        )
        group_colours = [
            GROUP_COLOUR_MAP["Nondemented"],
            GROUP_COLOUR_MAP["Converted"],
            GROUP_COLOUR_MAP["Demented"],
        ]
        for patch, colour in zip(box_plot["boxes"], group_colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.75)

        ax.set_xticklabels(["Nondemented", "Converted", "Demented"], fontsize=10)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f"{metric} at visit 1 (baseline) by group", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Baseline distributions — how similar converters look to non-demented participants",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    output_path = figures_dir / "baseline_distributions_by_group.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[exploration] Saved: {output_path}")


# ── Entry point called from main.py ─────────────────────────────────────────

def run_full_exploration(raw_data_path: Path, output_dir: Path) -> None:
    """
    Orchestrates all exploration steps in sequence.
    Called by main.py --step exploration.
    """
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    oasis_df = load_raw_dataset(raw_data_path)

    summarise_dataset_structure(oasis_df)

    visit_counts = compute_visit_counts_per_participant(oasis_df)
    summarise_visit_distribution(visit_counts)

    converters_df = identify_converters(oasis_df)
    summarise_converters(converters_df)

    audit_missing_values(oasis_df)

    plot_visit_count_distribution(visit_counts, figures_dir)
    plot_mmse_trajectories_by_group(oasis_df, figures_dir)
    plot_converter_mmse_trajectories(oasis_df, converters_df, figures_dir)
    plot_nwbv_trajectories_by_group(oasis_df, figures_dir)
    plot_baseline_distributions(oasis_df, figures_dir)

    print("\n[exploration] All exploration steps complete.")