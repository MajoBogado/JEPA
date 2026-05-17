"""
src/evaluation.py

Step 6: Compare all three models side by side.

Covers:
- Per-participant prediction comparison table (LLM vs XGBoost vs JEPA)
- Metrics summary table across all three models
- ROC curves on the same chart
- MMSE trajectory plots for participants each model handled differently
- Qualitative failure case analysis
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


# ── Colour palette ────────────────────────────────────────────────────────────

MODEL_COLOUR_MAP = {
    "LLM (gpt-4o-mini)": "#FF9800",
    "XGBoost": "#2196F3",
    "JEPA": "#4CAF50",
}


# ── Load results ──────────────────────────────────────────────────────────────

def load_all_predictions(results_dir: Path) -> pd.DataFrame:
    """
    Merge LLM, XGBoost and JEPA predictions into a single participant-level
    DataFrame for side-by-side comparison.
    """
    llm_predictions_df = pd.read_csv(results_dir / "llm_predictions.csv")
    xgb_predictions_df = pd.read_csv(results_dir / "xgb_predictions.csv")
    jepa_predictions_df = pd.read_csv(results_dir / "jepa_predictions.csv")

    comparison_df = llm_predictions_df[
        ["subject_id", "participant_group", "cdr_worsened_after_v2", "llm_prediction"]
    ].merge(
        xgb_predictions_df[["subject_id", "xgb_prediction", "xgb_prediction_proba"]],
        on="subject_id",
    ).merge(
        jepa_predictions_df[["subject_id", "jepa_prediction", "jepa_prediction_proba"]],
        on="subject_id",
    )

    return comparison_df


def load_all_metrics(results_dir: Path) -> pd.DataFrame:
    """Load and combine metrics from all three models into one summary table."""
    xgb_metrics_df  = pd.read_csv(results_dir / "xgb_metrics.csv")
    jepa_metrics_df = pd.read_csv(results_dir / "jepa_metrics.csv")

    # Compute LLM metrics inline from predictions
    llm_predictions_df = pd.read_csv(results_dir / "llm_predictions.csv")
    from src.ml_baseline import evaluate_predictions
    llm_metrics = evaluate_predictions(
        y_true=llm_predictions_df["cdr_worsened_after_v2"],
        y_pred=llm_predictions_df["llm_prediction"],
        y_pred_proba=llm_predictions_df["llm_prediction"].astype(float),
    )
    llm_metrics_df = pd.DataFrame([llm_metrics])
    llm_metrics_df.insert(0, "model", "llm")

    metrics_df = pd.concat([llm_metrics_df, xgb_metrics_df, jepa_metrics_df], ignore_index=True)
    return metrics_df


# ── Comparison table ──────────────────────────────────────────────────────────

def print_prediction_comparison_table(comparison_df: pd.DataFrame) -> None:
    """
    Print a per-participant table showing actual outcome and each model's prediction.
    Highlights disagreements between models.
    """
    print("\n── Per-participant prediction comparison ──────────────")
    print(f"  {'Subject ID':<12} {'Group':<14} {'Actual':>6} {'LLM':>5} {'XGB':>5} {'JEPA':>5}  Notes")
    print(f"  {'-'*12} {'-'*14} {'-'*6} {'-'*5} {'-'*5} {'-'*5}  {'-'*30}")

    for _, row in comparison_df.sort_values("participant_group").iterrows():
        actual = int(row["cdr_worsened_after_v2"])
        llm_pred  = int(row["llm_prediction"])
        xgb_pred  = int(row["xgb_prediction"])
        jepa_pred = int(row["jepa_prediction"])

        # Flag cases where models disagree or all got wrong
        all_wrong = (llm_pred != actual) and (xgb_pred != actual) and (jepa_pred != actual)
        models_disagree = not (llm_pred == xgb_pred == jepa_pred)
        llm_wrong_others_right = (llm_pred != actual) and (xgb_pred == actual) and (jepa_pred == actual)

        note = ""
        if all_wrong:
            note = "← all models wrong"
        elif llm_wrong_others_right:
            note = "← LLM wrong, others right"
        elif models_disagree and actual == 1:
            note = "← models disagree on positive case"

        print(
            f"  {row['subject_id']:<12} {row['participant_group']:<14} "
            f"{actual:>6} {llm_pred:>5} {xgb_pred:>5} {jepa_pred:>5}  {note}"
        )


def print_metrics_summary_table(metrics_df: pd.DataFrame) -> None:
    """Print a clean metrics comparison table across all three models."""
    print("\n── Metrics summary ────────────────────────────────────")
    display_cols = ["model", "auc_roc", "accuracy", "sensitivity", "specificity", "f1"]
    print(metrics_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# ── ROC curve plot ────────────────────────────────────────────────────────────

def plot_roc_curves(
    comparison_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Plot ROC curves for all three models on the same chart."""
    fig, ax = plt.subplots(figsize=(8, 7))

    y_true = comparison_df["cdr_worsened_after_v2"]

    model_proba_map = {
        "LLM (gpt-4o-mini)": comparison_df["llm_prediction"].astype(float),
        "XGBoost":            comparison_df["xgb_prediction_proba"],
        "JEPA":               comparison_df["jepa_prediction_proba"],
    }

    for model_name, y_proba in model_proba_map.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = np.trapezoid(tpr, fpr)
        ax.plot(
            fpr, tpr,
            color=MODEL_COLOUR_MAP[model_name],
            linewidth=2,
            label=f"{model_name} (AUC = {auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], color="#BDBDBD", linestyle="--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("ROC curves — LLM vs XGBoost vs JEPA", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    output_path = figures_dir / "roc_curves_comparison.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[evaluation] Saved: {output_path}")


# ── MMSE trajectory plots ─────────────────────────────────────────────────────

def plot_mmse_trajectories_for_positive_cases(
    comparison_df: pd.DataFrame,
    raw_data_path: Path,
    figures_dir: Path,
) -> None:
    """
    For participants with actual CDR worsening (label=1), plot their full
    MMSE trajectory across all visits using raw (unnormalised) MMSE values.
    Annotates which models predicted correctly vs incorrectly.
    """
    augmented_path = Path("data/raw/oasis_augmented.csv")
    data_path = augmented_path if augmented_path.exists() else raw_data_path
    oasis_df = pd.read_csv(data_path)
    oasis_df.columns = oasis_df.columns.str.strip()

    positive_cases = comparison_df[comparison_df["cdr_worsened_after_v2"] == 1]
    n_positive = len(positive_cases)

    n_cols = 3
    n_rows = int(np.ceil(n_positive / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes_flat = np.array(axes).flatten()

    for plot_idx, (_, participant_row) in enumerate(positive_cases.iterrows()):
        ax = axes_flat[plot_idx]
        subject_id = participant_row["subject_id"]
        subject_col = "Subject ID" if "Subject ID" in oasis_df.columns else "subject_id"
        subject_visits = oasis_df[oasis_df[subject_col] == subject_id].sort_values("Visit")

        if subject_visits.empty or subject_visits["MMSE"].isnull().all():
            ax.set_title(f"{subject_id}\n(no raw data)", fontsize=9)
            ax.axis("off")
            continue

        ax.plot(
            subject_visits["Visit"],
            subject_visits["MMSE"],
            marker="o",
            color="#455A64",
            linewidth=2,
            markersize=6,
        )

        ax.axvline(x=2, color="#E53935", linestyle="--", linewidth=1.2, alpha=0.7)

        y_min = subject_visits["MMSE"].min()
        ax.text(2.05, y_min, "cutoff", fontsize=7, color="#E53935", va="bottom")

        llm_correct  = "✓" if int(participant_row["llm_prediction"])  == 1 else "✗"
        xgb_correct  = "✓" if int(participant_row["xgb_prediction"])  == 1 else "✗"
        jepa_correct = "✓" if int(participant_row["jepa_prediction"]) == 1 else "✗"

        annotation = f"LLM:  {llm_correct}\nXGB:  {xgb_correct}\nJEPA: {jepa_correct}"
        ax.text(
            0.97, 0.97, annotation,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#BDBDBD"),
            fontfamily="monospace",
        )

        ax.set_title(
            f"{subject_id}\n({participant_row['participant_group']})",
            fontsize=9,
            fontweight="bold",
        )
        ax.set_xlabel("Visit", fontsize=8)
        ax.set_ylabel("MMSE", fontsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Hide unused subplots
    for unused_idx in range(n_positive, len(axes_flat)):
        axes_flat[unused_idx].set_visible(False)

    fig.suptitle(
        "MMSE trajectories — participants with actual CDR worsening\n"
        "(dashed line = prediction cutoff at visit 2)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path = figures_dir / "mmse_trajectories_positive_cases.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluation] Saved: {output_path}")


# ── Failure case analysis ─────────────────────────────────────────────────────

def plot_missed_cases_full_signals(
    comparison_df: pd.DataFrame,
    raw_data_path: Path,
    figures_dir: Path,
) -> None:
    """
    For participants that ALL three models missed (FN across the board),
    plot MMSE, nWBV and CDR trajectories side by side across all visits.
    This visualises why the cases were hard — the signal was not yet
    visible at the visit 2 prediction cutoff.
    """
    augmented_path = Path("data/raw/oasis_augmented.csv")
    data_path = augmented_path if augmented_path.exists() else raw_data_path
    oasis_df = pd.read_csv(data_path)
    oasis_df.columns = oasis_df.columns.str.strip()

    # Find participants all three models missed
    all_missed = comparison_df[
        (comparison_df["cdr_worsened_after_v2"] == 1) &
        (comparison_df["llm_prediction"] == 0) &
        (comparison_df["xgb_prediction"] == 0) &
        (comparison_df["jepa_prediction"] == 0)
    ]

    if all_missed.empty:
        print("[evaluation] No cases missed by all three models — skipping missed cases plot.")
        return

    n_participants = len(all_missed)
    signals = ["MMSE", "nWBV", "CDR"]
    signal_colours = {"MMSE": "#2196F3", "nWBV": "#4CAF50", "CDR": "#F44336"}

    fig, axes = plt.subplots(
        n_participants, 3,
        figsize=(14, 4.5 * n_participants),
    )

    # Ensure axes is always 2D
    if n_participants == 1:
        axes = axes.reshape(1, 3)

    for row_idx, (_, participant_row) in enumerate(all_missed.iterrows()):
        subject_id = participant_row["subject_id"]
        subject_col = "Subject ID" if "Subject ID" in oasis_df.columns else "subject_id"
        subject_visits = oasis_df[oasis_df[subject_col] == subject_id].sort_values("Visit")

        for col_idx, signal in enumerate(signals):
            ax = axes[row_idx][col_idx]

            if subject_visits.empty or signal not in subject_visits.columns:
                ax.axis("off")
                continue

            ax.plot(
                subject_visits["Visit"],
                subject_visits[signal],
                marker="o",
                color=signal_colours[signal],
                linewidth=2,
                markersize=7,
            )

            # Mark prediction cutoff
            ax.axvline(x=2, color="#E53935", linestyle="--", linewidth=1.2, alpha=0.7)
            y_min = subject_visits[signal].min()
            ax.text(2.05, y_min, "cutoff", fontsize=7, color="#E53935", va="bottom")

            # Annotate visit 1 and visit 2 values
            for visit_num in [1, 2]:
                visit_row = subject_visits[subject_visits["Visit"] == visit_num]
                if not visit_row.empty:
                    val = visit_row[signal].values[0]
                    ax.annotate(
                        f"{val}",
                        xy=(visit_num, val),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                        color=signal_colours[signal],
                    )

            if col_idx == 0:
                ax.set_ylabel(
                    f"{subject_id}\n({participant_row['participant_group']})\n\n{signal}",
                    fontsize=9,
                    fontweight="bold",
                )
            else:
                ax.set_ylabel(signal, fontsize=9)

            ax.set_xlabel("Visit", fontsize=8)
            ax.set_title(
                f"{signal} trajectory" + (" — all models missed" if col_idx == 1 else ""),
                fontsize=9,
            )
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle(
        "Cases missed by all three models — full signal view\n"
        "MMSE (cognitive), nWBV (brain volume), CDR (dementia rating)\n"
        "Dashed line = visit 2 prediction cutoff",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path = figures_dir / "missed_cases_full_signals.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluation] Saved: {output_path}")


def print_failure_case_analysis(comparison_df: pd.DataFrame) -> None:
    """
    Identify and describe cases where models disagree — especially where
    one model is right and another is wrong on positive cases.
    This is the centrepiece of the AMI Labs presentation narrative.
    """
    print("\n── Failure case analysis ──────────────────────────────")

    positive_cases = comparison_df[comparison_df["cdr_worsened_after_v2"] == 1]
    negative_cases = comparison_df[comparison_df["cdr_worsened_after_v2"] == 0]

    print("\n  Positive cases (actual CDR worsening):")
    for _, row in positive_cases.iterrows():
        llm_result  = "correct" if int(row["llm_prediction"])  == 1 else "MISSED"
        xgb_result  = "correct" if int(row["xgb_prediction"])  == 1 else "MISSED"
        jepa_result = "correct" if int(row["jepa_prediction"]) == 1 else "MISSED"
        print(
            f"    {row['subject_id']} ({row['participant_group']}) | "
            f"LLM: {llm_result:<8} XGB: {xgb_result:<8} JEPA: {jepa_result}"
        )

    print("\n  False alarms (predicted worsen, actually stable):")
    for _, row in negative_cases.iterrows():
        false_alarms = []
        if int(row["llm_prediction"])  == 1: false_alarms.append("LLM")
        if int(row["xgb_prediction"])  == 1: false_alarms.append("XGB")
        if int(row["jepa_prediction"]) == 1: false_alarms.append("JEPA")
        if false_alarms:
            print(f"    {row['subject_id']} ({row['participant_group']}) | False alarm from: {', '.join(false_alarms)}")

    # Cases where models disagree on positive cases — most interesting for narrative
    disagreements_on_positives = positive_cases[
        ~(
            (positive_cases["llm_prediction"] == positive_cases["xgb_prediction"]) &
            (positive_cases["xgb_prediction"] == positive_cases["jepa_prediction"])
        )
    ]
    if not disagreements_on_positives.empty:
        print(f"\n  Model disagreements on positive cases (most interesting for presentation):")
        for _, row in disagreements_on_positives.iterrows():
            print(f"    {row['subject_id']} — LLM={int(row['llm_prediction'])} XGB={int(row['xgb_prediction'])} JEPA={int(row['jepa_prediction'])} Actual=1")


# ── Save combined results ─────────────────────────────────────────────────────

def save_comparison_table(comparison_df: pd.DataFrame, results_dir: Path) -> None:
    """Save the full per-participant comparison table."""
    output_path = results_dir / "model_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    print(f"\n[evaluation] Saved: {output_path}")


# ── Entry point called from main.py ──────────────────────────────────────────

def run_full_comparison(output_dir: Path) -> None:
    """
    Orchestrates the full comparison step.
    Called by main.py --step comparison.
    """
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_data_path = Path("data/raw/oasis_longitudinal.csv")

    comparison_df = load_all_predictions(output_dir)
    metrics_df    = load_all_metrics(output_dir)

    print_prediction_comparison_table(comparison_df)
    print_metrics_summary_table(metrics_df)
    print_failure_case_analysis(comparison_df)

    plot_roc_curves(comparison_df, figures_dir)
    plot_mmse_trajectories_for_positive_cases(comparison_df, raw_data_path, figures_dir)
    plot_missed_cases_full_signals(comparison_df, raw_data_path, figures_dir)

    save_comparison_table(comparison_df, output_dir)

    print("\n[evaluation] Comparison step complete.")