"""
src/llm_predictor.py

Step 3: LLM baseline using OpenAI gpt-4o-mini.

Each test participant's visit 1 and visit 2 data is serialised into a
structured text prompt. The model is asked to predict whether CDR will
worsen by the final study visit.

This is the baseline that demonstrates what a language model can and
cannot do with longitudinal clinical data presented as text.
"""

import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_MODEL = "gpt-4o-mini"
PREDICTION_SLEEP_SECONDS = 0.5  # Avoid hitting rate limits


# ── Prompt builder ───────────────────────────────────────────────────────────

def build_clinical_prompt(participant_row: pd.Series) -> str:
    """
    Serialise a participant's visit 1 and visit 2 data into a structured
    text prompt for the LLM. The model sees exactly what a clinician would
    read — but as text, with no internal model of physiological trajectories.
    """
    return f"""You are a clinical AI assistant supporting an Alzheimer's disease trial.
A participant has completed two study visits. Based on their longitudinal
data below, predict whether their Clinical Dementia Rating (CDR) will
WORSEN by their final study visit.

Respond with only: "1" (CDR will worsen) or "0" (CDR will remain stable).

Participant data:

VISIT 1:
- Age: {participant_row['age_v1']}
- MMSE: {participant_row['mmse_v1']}
- CDR: {participant_row['cdr_v1']}
- Normalised Brain Volume (nWBV): {participant_row['nwbv_v1']:.4f}
- Education (years): {participant_row['education_years']}
- Sex: {participant_row['sex']}

VISIT 2 (approx. {participant_row['mr_delay_v2']} days later):
- Age: {participant_row['age_v2']}
- MMSE: {participant_row['mmse_v2']}
- CDR: {participant_row['cdr_v2']}
- Normalised Brain Volume (nWBV): {participant_row['nwbv_v2']:.4f}

Change from visit 1 to visit 2:
- MMSE change: {participant_row['mmse_delta_v1_v2']:+.1f}
- nWBV change: {participant_row['nwbv_delta_v1_v2']:+.4f}
- CDR change: {participant_row['cdr_delta_v1_v2']:+.1f}

Prediction (1 = will worsen, 0 = will remain stable):"""


# ── API call ─────────────────────────────────────────────────────────────────

def call_openai_for_prediction(
    openai_client: OpenAI,
    prompt: str,
    subject_id: str,
) -> int:
    """
    Send a single prompt to gpt-4o-mini and parse the binary prediction.
    Returns 1 (CDR will worsen) or 0 (CDR will remain stable).
    Falls back to 0 if the response cannot be parsed.
    """
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0,  # Deterministic — clinical prediction should not vary
    )

    raw_response = response.choices[0].message.content.strip()

    if raw_response in ("0", "1"):
        return int(raw_response)

    # Handle cases where the model adds punctuation or extra text
    if "1" in raw_response:
        print(f"  [llm] {subject_id}: non-clean response '{raw_response}' — parsed as 1")
        return 1
    if "0" in raw_response:
        print(f"  [llm] {subject_id}: non-clean response '{raw_response}' — parsed as 0")
        return 0

    print(f"  [llm] {subject_id}: unparseable response '{raw_response}' — defaulting to 0")
    return 0


# ── Batch prediction ─────────────────────────────────────────────────────────

def predict_all_test_participants(
    test_df: pd.DataFrame,
    openai_client: OpenAI,
) -> pd.DataFrame:
    """
    Run LLM predictions for all test participants.
    Returns the test DataFrame with an added llm_prediction column.
    """
    predictions = []

    for _, participant_row in test_df.iterrows():
        subject_id = participant_row["subject_id"]
        prompt = build_clinical_prompt(participant_row)

        prediction = call_openai_for_prediction(
            openai_client=openai_client,
            prompt=prompt,
            subject_id=subject_id,
        )
        predictions.append(prediction)
        print(
            f"  [llm] {subject_id} | "
            f"group={participant_row['participant_group']} | "
            f"predicted={prediction} | "
            f"actual={int(participant_row['cdr_worsened_after_v2'])}"
        )
        time.sleep(PREDICTION_SLEEP_SECONDS)

    test_with_predictions = test_df.copy()
    test_with_predictions["llm_prediction"] = predictions
    return test_with_predictions


# ── Summary ──────────────────────────────────────────────────────────────────

def summarise_llm_predictions(test_with_predictions: pd.DataFrame) -> None:
    """Print a quick accuracy summary broken down by group."""
    print("\n── LLM prediction summary ─────────────────────────────")

    total = len(test_with_predictions)
    correct = (
        test_with_predictions["llm_prediction"] == test_with_predictions["cdr_worsened_after_v2"]
    ).sum()
    print(f"  Overall accuracy : {correct}/{total} ({correct/total*100:.1f}%)")

    print(f"\n  Breakdown by group:")
    for group_label, group_df in test_with_predictions.groupby("participant_group"):
        group_correct = (
            group_df["llm_prediction"] == group_df["cdr_worsened_after_v2"]
        ).sum()
        print(f"    {group_label} : {group_correct}/{len(group_df)} correct")


# ── Save results ─────────────────────────────────────────────────────────────

def save_llm_predictions(test_with_predictions: pd.DataFrame, output_dir: Path) -> None:
    """Save the test set with LLM predictions appended."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "llm_predictions.csv"
    test_with_predictions.to_csv(output_path, index=False)
    print(f"\n[llm] Saved: {output_path}")


# ── Entry point called from main.py ─────────────────────────────────────────

def run_llm_baseline(output_dir: Path) -> None:
    """
    Orchestrates the LLM baseline prediction step.
    Called by main.py --step llm_baseline.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found. Add it to your .env file."
        )

    test_df = pd.read_csv("data/processed/test_participants.csv")
    print(f"[llm] Loaded test set: {len(test_df)} participants")

    openai_client = OpenAI(api_key=api_key)

    print(f"[llm] Running predictions with {OPENAI_MODEL}...")
    test_with_predictions = predict_all_test_participants(test_df, openai_client)

    summarise_llm_predictions(test_with_predictions)
    save_llm_predictions(test_with_predictions, output_dir)

    print("\n[llm] LLM baseline complete.")