"""
src/synthetic_data.py

Generates synthetic longitudinal participants to augment the OASIS-2 dataset.

Why synthetic data:
    The real OASIS-2 dataset yields only 52 usable participants under correct
    methodology (requiring 3+ visits for a valid forward-looking ground truth).
    Of those, only 3 have positive labels. This is insufficient for a meaningful
    train/test split. Synthetic participants are generated from the real data's
    statistical properties to make the exercise executable.

What is generated:
    - ~150 synthetic participants with 3-5 visits each
    - ~20% converters (oversampled vs real 9%) to ensure enough positive cases
    - Trajectories follow real OASIS-2 distributions per group
    - Longitudinal correlations preserved: MMSE declines over time in converters,
      nWBV declines gradually in all groups, CDR progression follows clinical logic

Transparency note:
    Synthetic participants are flagged with a 'synthetic' column = True.
    All results and README explicitly state the augmentation approach and rationale.
"""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
N_SYNTHETIC_PARTICIPANTS = 150
CONVERTER_FRACTION = 0.20      # 20% converters vs 9% in real data
DEMENTED_FRACTION = 0.40       # keep roughly similar to real data
VISIT_INTERVAL_DAYS_MEAN = 730  # ~2 years between visits
VISIT_INTERVAL_DAYS_STD = 180


# ── Real data statistics per group ───────────────────────────────────────────
# These are derived from the real OASIS-2 dataset and used as generation priors.
# Values confirmed from exploration step outputs.

GROUP_STATS = {
    "Nondemented": {
        "age_mean": 74.0, "age_std": 8.0,
        "mmse_v1_mean": 29.1, "mmse_v1_std": 1.0,
        "mmse_annual_delta_mean": 0.0, "mmse_annual_delta_std": 0.8,
        "nwbv_v1_mean": 0.738, "nwbv_v1_std": 0.038,
        "nwbv_annual_delta_mean": -0.003, "nwbv_annual_delta_std": 0.003,
        "etiv_mean": 1490, "etiv_std": 150,
        "cdr_v1_options": [0.0],
        "cdr_v1_probs": [1.0],
        "educ_mean": 14.5, "educ_std": 2.5,
        "ses_mean": 2.5, "ses_std": 1.0,
        "female_prob": 0.65,
    },
    "Demented": {
        "age_mean": 76.0, "age_std": 7.0,
        "mmse_v1_mean": 24.0, "mmse_v1_std": 4.5,
        "mmse_annual_delta_mean": -1.5, "mmse_annual_delta_std": 2.0,
        "nwbv_v1_mean": 0.713, "nwbv_v1_std": 0.033,
        "nwbv_annual_delta_mean": -0.004, "nwbv_annual_delta_std": 0.003,
        "etiv_mean": 1480, "etiv_std": 150,
        "cdr_v1_options": [0.5, 1.0, 2.0],
        "cdr_v1_probs": [0.65, 0.30, 0.05],
        "educ_mean": 13.5, "educ_std": 3.0,
        "ses_mean": 2.8, "ses_std": 1.0,
        "female_prob": 0.60,
    },
    "Converted": {
        "age_mean": 75.0, "age_std": 7.0,
        "mmse_v1_mean": 29.0, "mmse_v1_std": 1.2,
        "mmse_annual_delta_mean": -1.8, "mmse_annual_delta_std": 1.5,
        "nwbv_v1_mean": 0.728, "nwbv_v1_std": 0.036,
        "nwbv_annual_delta_mean": -0.004, "nwbv_annual_delta_std": 0.003,
        "etiv_mean": 1485, "etiv_std": 150,
        "cdr_v1_options": [0.0],
        "cdr_v1_probs": [1.0],
        "educ_mean": 14.0, "educ_std": 2.5,
        "ses_mean": 2.5, "ses_std": 1.0,
        "female_prob": 0.65,
    },
}


# ── Visit generation ──────────────────────────────────────────────────────────

def generate_visit_sequence(
    rng: np.random.Generator,
    group: str,
    n_visits: int,
    subject_id: str,
) -> list[dict]:
    """
    Generate a realistic sequence of visits for one synthetic participant.
    Longitudinal correlations are preserved:
    - MMSE declines over time at a group-appropriate rate
    - nWBV declines gradually in all groups
    - CDR progression follows clinical logic (can only stay same or worsen
      except in rare cases)
    - Age increases naturally between visits
    """
    stats = GROUP_STATS[group]

    # Baseline values at visit 1
    baseline_age = int(np.clip(rng.normal(stats["age_mean"], stats["age_std"]), 60, 96))
    baseline_mmse = float(np.clip(rng.normal(stats["mmse_v1_mean"], stats["mmse_v1_std"]), 0, 30))
    baseline_nwbv = float(np.clip(rng.normal(stats["nwbv_v1_mean"], stats["nwbv_v1_std"]), 0.60, 0.85))
    baseline_etiv = int(np.clip(rng.normal(stats["etiv_mean"], stats["etiv_std"]), 1100, 1900))
    baseline_asf = round(1430 / baseline_etiv, 3)  # ASF is derived from eTIV
    baseline_cdr = float(rng.choice(stats["cdr_v1_options"], p=stats["cdr_v1_probs"]))
    education = int(np.clip(rng.normal(stats["educ_mean"], stats["educ_std"]), 6, 23))
    ses = float(np.clip(round(rng.normal(stats["ses_mean"], stats["ses_std"])), 1, 5))
    sex = "F" if rng.random() < stats["female_prob"] else "M"

    visits = []
    cumulative_days = 0
    current_mmse = baseline_mmse
    current_nwbv = baseline_nwbv
    current_cdr = baseline_cdr
    current_age = baseline_age

    for visit_num in range(1, n_visits + 1):
        if visit_num > 1:
            # Days since previous visit (~1-3 years)
            interval_days = int(np.clip(
                rng.normal(VISIT_INTERVAL_DAYS_MEAN, VISIT_INTERVAL_DAYS_STD),
                365, 1460,
            ))
            cumulative_days += interval_days
            years_elapsed = interval_days / 365.0

            # MMSE trajectory — group-specific annual decline rate
            mmse_change = rng.normal(
                stats["mmse_annual_delta_mean"] * years_elapsed,
                stats["mmse_annual_delta_std"] * np.sqrt(years_elapsed),
            )
            current_mmse = float(np.clip(current_mmse + mmse_change, 0, 30))

            # nWBV trajectory — gradual atrophy in all groups
            nwbv_change = rng.normal(
                stats["nwbv_annual_delta_mean"] * years_elapsed,
                stats["nwbv_annual_delta_std"] * np.sqrt(years_elapsed),
            )
            current_nwbv = float(np.clip(current_nwbv + nwbv_change, 0.60, 0.85))

            # CDR progression — clinical logic: can worsen, rarely improves
            current_cdr = advance_cdr(rng, group, current_cdr, visit_num, n_visits)

            # Age increases naturally
            current_age = baseline_age + int(cumulative_days / 365)

        visit_group = determine_visit_group_label(group, current_cdr, baseline_cdr)

        visits.append({
            "Subject ID": subject_id,
            "MRI ID": f"{subject_id}_MR{visit_num}",
            "Group": visit_group,
            "Visit": visit_num,
            "MR Delay": cumulative_days,
            "M/F": sex,
            "Hand": "R",  # OASIS-2 is predominantly right-handed
            "Age": current_age,
            "EDUC": education,
            "SES": ses,
            "MMSE": round(current_mmse, 1),
            "CDR": current_cdr,
            "eTIV": baseline_etiv,  # eTIV is stable across visits
            "nWBV": round(current_nwbv, 3),
            "ASF": baseline_asf,
            "synthetic": True,
        })

    return visits


def advance_cdr(
    rng: np.random.Generator,
    group: str,
    current_cdr: float,
    visit_num: int,
    total_visits: int,
) -> float:
    """
    Advance CDR rating following clinical progression logic.
    CDR scale: 0, 0.5, 1, 2, 3
    Converters start at 0 and must reach 0.5 by some visit.
    Demented participants stay at or above their baseline CDR.
    """
    cdr_scale = [0.0, 0.5, 1.0, 2.0]
    current_idx = cdr_scale.index(current_cdr) if current_cdr in cdr_scale else 1

    if group == "Converted":
        # Converters must reach CDR=0.5 by final visit
        # Increase probability of worsening as visits progress
        worsen_prob = 0.4 + (visit_num / total_visits) * 0.4
        if current_cdr == 0.0 and rng.random() < worsen_prob:
            return 0.5
        elif current_cdr >= 0.5:
            # Small chance of further progression
            if rng.random() < 0.15 and current_idx < len(cdr_scale) - 1:
                return cdr_scale[current_idx + 1]
        return current_cdr

    elif group == "Demented":
        # Demented participants can worsen, rarely improve
        if rng.random() < 0.20 and current_idx < len(cdr_scale) - 1:
            return cdr_scale[current_idx + 1]
        return current_cdr

    else:  # Nondemented
        # Very small chance of CDR increase (catching borderline cases)
        if rng.random() < 0.03:
            return 0.5
        return current_cdr


def determine_visit_group_label(
    true_group: str,
    current_cdr: float,
    baseline_cdr: float,
) -> str:
    """
    Assign the Group label for a visit, mirroring OASIS-2 conventions.
    Converted participants are labelled Converted on all visits.
    """
    if true_group == "Converted":
        return "Converted"
    elif true_group == "Demented":
        return "Demented"
    else:
        return "Nondemented"


# ── Dataset generation ────────────────────────────────────────────────────────

def generate_group_counts(n_total: int) -> dict[str, int]:
    """
    Calculate how many participants to generate per group.
    Converters are oversampled to 20% to ensure enough positive cases.
    """
    n_converters = int(n_total * CONVERTER_FRACTION)
    n_demented = int(n_total * DEMENTED_FRACTION)
    n_nondemented = n_total - n_converters - n_demented
    return {
        "Nondemented": n_nondemented,
        "Demented": n_demented,
        "Converted": n_converters,
    }


def generate_synthetic_dataset(
    n_participants: int = N_SYNTHETIC_PARTICIPANTS,
) -> pd.DataFrame:
    """
    Generate a full synthetic longitudinal dataset with the same structure
    as the real OASIS-2 CSV.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    group_counts = generate_group_counts(n_participants)

    all_visits = []
    participant_counter = 1000  # Start IDs at 1000 to avoid collisions with real IDs

    for group, count in group_counts.items():
        for _ in range(count):
            subject_id = f"SYN_{participant_counter:04d}"
            participant_counter += 1

            # Converters and Demented get 3-5 visits, Nondemented get 2-4
            if group in ("Converted", "Demented"):
                n_visits = int(rng.choice([3, 4, 5], p=[0.5, 0.35, 0.15]))
            else:
                n_visits = int(rng.choice([3, 4], p=[0.7, 0.3]))

            visit_rows = generate_visit_sequence(
                rng=rng,
                group=group,
                n_visits=n_visits,
                subject_id=subject_id,
            )
            all_visits.extend(visit_rows)

    synthetic_df = pd.DataFrame(all_visits)
    print(f"[synthetic] Generated {n_participants} synthetic participants")
    print(f"  {synthetic_df['Subject ID'].nunique()} unique participants")
    print(f"  {len(synthetic_df)} total visit rows")
    print(f"  Group distribution (rows):\n{synthetic_df['Group'].value_counts().to_string()}")
    return synthetic_df


# ── Combine with real data ────────────────────────────────────────────────────

def combine_real_and_synthetic(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine real and synthetic datasets into one DataFrame.
    Real participants get synthetic=False, synthetic get synthetic=True.
    """
    real_flagged = real_df.copy()
    real_flagged["synthetic"] = False

    combined_df = pd.concat([real_flagged, synthetic_df], ignore_index=True)
    print(f"\n[synthetic] Combined dataset:")
    print(f"  Real participants      : {real_df['Subject ID'].nunique()}")
    print(f"  Synthetic participants : {synthetic_df['Subject ID'].nunique()}")
    print(f"  Total participants     : {combined_df['Subject ID'].nunique()}")
    print(f"  Total visit rows       : {len(combined_df)}")
    return combined_df


# ── Save ──────────────────────────────────────────────────────────────────────

def save_augmented_dataset(combined_df: pd.DataFrame, output_path: Path) -> None:
    """Save the combined real + synthetic dataset to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    print(f"[synthetic] Saved augmented dataset: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_synthetic_augmentation(raw_data_path: Path, output_dir: Path) -> None:
    """
    Generate synthetic participants and combine with real OASIS-2 data.
    Saves augmented dataset to data/raw/oasis_augmented.csv.
    Called by main.py --step synthetic.
    """
    real_df = pd.read_csv(raw_data_path)
    real_df.columns = real_df.columns.str.strip()

    synthetic_df = generate_synthetic_dataset(n_participants=N_SYNTHETIC_PARTICIPANTS)
    combined_df = combine_real_and_synthetic(real_df, synthetic_df)

    augmented_path = Path("data/raw/oasis_augmented.csv")
    save_augmented_dataset(combined_df, augmented_path)

    print("\n[synthetic] Augmentation complete.")
    print("  Next step: run preprocessing with --data-path data/raw/oasis_augmented.csv")