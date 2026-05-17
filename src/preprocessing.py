"""
src/preprocessing.py

Step 3: Preprocessing the OASIS-2 longitudinal dataset.

Covers:
- Missing value imputation (median per group) for MMSE and SES
- Feature engineering: delta features V1 -> V2
- Ground truth label: CDR worsened at ANY visit after visit 2
- Stratified train/test split (120 train / 30 test)
- Save processed files to data/processed/
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROCESSED_DIR = Path("data/processed")
RANDOM_SEED = 42


# ── Imputation ───────────────────────────────────────────────────────────────

def impute_missing_values(oasis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing MMSE and SES values using median per participant group.
    Nondemented participants get the Nondemented median, etc.
    This avoids leaking signal across groups.
    """
    oasis_imputed = oasis_df.copy()

    for column in ["MMSE", "SES"]:
        group_medians = oasis_imputed.groupby("Group")[column].median()
        for group_label, median_value in group_medians.items():
            group_mask = oasis_imputed["Group"] == group_label
            missing_mask = oasis_imputed[column].isnull()
            oasis_imputed.loc[group_mask & missing_mask, column] = median_value

    remaining_nulls = oasis_imputed[["MMSE", "SES"]].isnull().sum().sum()
    print(f"[preprocessing] Imputation complete. Remaining nulls in MMSE/SES: {remaining_nulls}")
    return oasis_imputed


# ── Feature engineering ──────────────────────────────────────────────────────

def extract_visit_row(subject_visits: pd.DataFrame, visit_number: int) -> pd.Series:
    """Return the row for a specific visit number, or None if it doesn't exist."""
    visit_row = subject_visits[subject_visits["Visit"] == visit_number]
    if visit_row.empty:
        return None
    return visit_row.iloc[0]


def build_participant_feature_row(
    subject_id: str,
    visit_1_row: pd.Series,
    visit_2_row: pd.Series,
    participant_group: str,
) -> dict:
    """
    Build a flat feature dict for one participant using visit 1 and visit 2 data.
    Includes raw values for both visits and delta features (change V1 -> V2).
    """
    return {
        "subject_id": subject_id,
        "participant_group": participant_group,

        # Visit 1 features
        "age_v1": visit_1_row["Age"],
        "mmse_v1": visit_1_row["MMSE"],
        "cdr_v1": visit_1_row["CDR"],
        "nwbv_v1": visit_1_row["nWBV"],
        "etiv_v1": visit_1_row["eTIV"],
        "asf_v1": visit_1_row["ASF"],

        # Visit 2 features
        "age_v2": visit_2_row["Age"],
        "mmse_v2": visit_2_row["MMSE"],
        "cdr_v2": visit_2_row["CDR"],
        "nwbv_v2": visit_2_row["nWBV"],
        "etiv_v2": visit_2_row["eTIV"],
        "asf_v2": visit_2_row["ASF"],
        "mr_delay_v2": visit_2_row["MR Delay"],

        # Delta features — rate of change is more predictive than absolute values
        "mmse_delta_v1_v2": visit_2_row["MMSE"] - visit_1_row["MMSE"],
        "nwbv_delta_v1_v2": visit_2_row["nWBV"] - visit_1_row["nWBV"],
        "cdr_delta_v1_v2": visit_2_row["CDR"] - visit_1_row["CDR"],
        "age_delta_v1_v2": visit_2_row["Age"] - visit_1_row["Age"],

        # Static participant features
        "education_years": visit_1_row["EDUC"],
        "ses": visit_1_row["SES"],
        "sex": visit_1_row["M/F"],
        "handedness": visit_1_row["Hand"],
    }


def build_ground_truth_label(
    subject_visits: pd.DataFrame,
    visit_2_cdr: float,
) -> int | None:
    """
    Returns 1 if CDR worsened at ANY visit after visit 2, 0 if stable.
    Returns None if there are no visits after visit 2 — these participants
    are excluded from the dataset entirely since there is nothing to predict.
    """
    visits_after_cutoff = subject_visits[subject_visits["Visit"] > 2]
    if visits_after_cutoff.empty:
        return None

    max_cdr_after_cutoff = visits_after_cutoff["CDR"].max()
    return int(max_cdr_after_cutoff > visit_2_cdr)


def engineer_features_and_labels(oasis_imputed: pd.DataFrame) -> pd.DataFrame:
    """
    Build a participant-level DataFrame with:
    - Flat feature columns from visits 1 and 2
    - Delta features (change V1 -> V2)
    - Binary ground truth label (CDR worsened after visit 2)

    Participants with fewer than 2 visits are excluded.
    """
    feature_rows = []
    excluded_count = 0

    for subject_id, subject_visits in oasis_imputed.groupby("Subject ID"):
        subject_visits_sorted = subject_visits.sort_values("Visit")

        visit_1_row = extract_visit_row(subject_visits_sorted, visit_number=1)
        visit_2_row = extract_visit_row(subject_visits_sorted, visit_number=2)

        if visit_1_row is None or visit_2_row is None:
            excluded_count += 1
            continue

        # Assign participant group — converters are labelled Converted on all visits
        participant_group = (
            "Converted" if "Converted" in subject_visits_sorted["Group"].values
            else subject_visits_sorted.iloc[0]["Group"]
        )

        feature_row = build_participant_feature_row(
            subject_id=subject_id,
            visit_1_row=visit_1_row,
            visit_2_row=visit_2_row,
            participant_group=participant_group,
        )

        ground_truth_label = build_ground_truth_label(
            subject_visits=subject_visits_sorted,
            visit_2_cdr=visit_2_row["CDR"],
        )

        if ground_truth_label is None:
            excluded_count += 1
            continue

        feature_row["cdr_worsened_after_v2"] = ground_truth_label
        feature_rows.append(feature_row)

    participant_features_df = pd.DataFrame(feature_rows)
    print(f"[preprocessing] Feature engineering complete.")
    print(f"  Participants included  : {len(participant_features_df)}")
    print(f"  Excluded (< 2 visits)  : {excluded_count} (no valid ground truth — nothing to predict)")
    print(f"  CDR worsened (label=1) : {participant_features_df['cdr_worsened_after_v2'].sum()}")
    print(f"  CDR stable   (label=0) : {(participant_features_df['cdr_worsened_after_v2'] == 0).sum()}")
    return participant_features_df


# ── Train / test split ───────────────────────────────────────────────────────

def split_train_test(participant_features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified 80/20 split preserving group proportions.
    Real participants with positive labels are forced into the test set
    to ensure at least some genuine clinical cases appear in evaluation.
    Returns (train_df, test_df).
    """
    # Force real positive cases into test set — there are only 3 and the
    # stratified split sent all of them to train by chance
    real_positive_mask = (
        ~participant_features_df["subject_id"].str.startswith("SYN") &
        (participant_features_df["cdr_worsened_after_v2"] == 1)
    )
    forced_test_df = participant_features_df[real_positive_mask]
    remaining_df = participant_features_df[~real_positive_mask]

    print(f"[preprocessing] Forcing {len(forced_test_df)} real positive participants into test set.")

    # Stratified split on the remaining participants
    remaining_train_df, remaining_test_df = train_test_split(
        remaining_df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=remaining_df["participant_group"],
    )

    test_df = pd.concat([forced_test_df, remaining_test_df], ignore_index=True)
    train_df = remaining_train_df.reset_index(drop=True)

    print(f"\n[preprocessing] Train/test split complete.")
    print(f"  Train : {len(train_df)} participants")
    print(f"  Test  : {len(test_df)} participants")

    for split_label, split_df in [("Train", train_df), ("Test", test_df)]:
        group_counts = split_df["participant_group"].value_counts()
        label_counts = split_df["cdr_worsened_after_v2"].value_counts()
        print(f"\n  {split_label} group breakdown:")
        for group, count in group_counts.items():
            print(f"    {group} : {count}")
        print(f"  {split_label} label breakdown:")
        print(f"    CDR worsened (1) : {label_counts.get(1, 0)}")
        print(f"    CDR stable   (0) : {label_counts.get(0, 0)}")

    return train_df, test_df


# ── Save processed files ─────────────────────────────────────────────────────

def save_processed_splits(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save train and test splits to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DIR / "train_participants.csv"
    test_path = PROCESSED_DIR / "test_participants.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n[preprocessing] Saved: {train_path}")
    print(f"[preprocessing] Saved: {test_path}")


# ── Entry point called from main.py ─────────────────────────────────────────

def run_full_preprocessing(raw_data_path: Path, output_dir: Path) -> None:
    """
    Orchestrates all preprocessing steps in sequence.
    Called by main.py --step preprocessing.
    """
    oasis_df = pd.read_csv(raw_data_path)
    oasis_df.columns = oasis_df.columns.str.strip()

    oasis_imputed = impute_missing_values(oasis_df)
    participant_features_df = engineer_features_and_labels(oasis_imputed)
    train_df, test_df = split_train_test(participant_features_df)
    save_processed_splits(train_df, test_df)

    print("\n[preprocessing] All preprocessing steps complete.")