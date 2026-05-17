"""
src/ml_baseline.py

Step 4: XGBoost baseline model.

Trains on the 115 train participants using flat tabular features from
visits 1 and 2 plus delta features. Evaluates on the 29 test participants.

This is the honest benchmark — both the LLM and the joint embedding model
need to be compared against this to make any meaningful claim.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


RANDOM_SEED = 42

FEATURE_COLUMNS = [
    # Visit 1 values
    "age_v1", "mmse_v1", "cdr_v1", "nwbv_v1", "etiv_v1", "asf_v1",
    # Visit 2 values
    "age_v2", "mmse_v2", "cdr_v2", "nwbv_v2", "etiv_v2", "asf_v2",
    "mr_delay_v2",
    # Delta features — rate of change V1 -> V2
    "mmse_delta_v1_v2", "nwbv_delta_v1_v2", "cdr_delta_v1_v2", "age_delta_v1_v2",
    # Static participant features
    "education_years", "ses",
]

TARGET_COLUMN = "cdr_worsened_after_v2"


# ── Feature preparation ──────────────────────────────────────────────────────

def encode_categorical_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode sex and handedness as integers.
    Fit the encoder on train only to avoid leaking test distribution.
    """
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    for categorical_col in ["sex", "handedness"]:
        encoder = LabelEncoder()
        train_encoded[categorical_col] = encoder.fit_transform(train_df[categorical_col])
        test_encoded[categorical_col] = encoder.transform(test_df[categorical_col])

    return train_encoded, test_encoded


def extract_features_and_labels(
    df: pd.DataFrame,
    include_categorical: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and label vector y from a participant DataFrame."""
    feature_cols = FEATURE_COLUMNS.copy()
    if include_categorical:
        feature_cols += ["sex", "handedness"]

    X = df[feature_cols]
    y = df[TARGET_COLUMN]
    return X, y


# ── Model training ───────────────────────────────────────────────────────────

def train_xgboost_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBClassifier:
    """
    Train an XGBoost classifier on the training set.
    scale_pos_weight handles class imbalance by upweighting the positive class.
    """
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    positive_class_weight = n_negative / n_positive

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=positive_class_weight,
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train)
    print(f"[ml_baseline] XGBoost trained on {len(X_train)} participants.")
    print(f"  Class weight applied: {positive_class_weight:.2f} (negative/positive ratio)")
    return xgb_model


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict:
    """
    Compute evaluation metrics for binary CDR worsening prediction.
    Sensitivity (recall) is the most important metric — missing a converter
    is the costly clinical error.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall — converters caught
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Stable correctly identified

    metrics = {
        "auc_roc": roc_auc_score(y_true, y_pred_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }
    return metrics


def summarise_metrics(metrics: dict, model_name: str) -> None:
    """Print evaluation metrics to stdout."""
    print(f"\n── {model_name} evaluation ─────────────────────────────")
    print(f"  AUC-ROC     : {metrics['auc_roc']:.3f}")
    print(f"  Accuracy    : {metrics['accuracy']:.3f}")
    print(f"  F1 score    : {metrics['f1']:.3f}")
    print(f"  Sensitivity : {metrics['sensitivity']:.3f}  (converters caught)")
    print(f"  Specificity : {metrics['specificity']:.3f}  (stable correctly identified)")
    print(f"\n  Confusion matrix:")
    print(f"    True  Positives  (TP) — predicted worsen,  actually worsened : {metrics['true_positives']}")
    print(f"    False Positives  (FP) — predicted worsen,  actually stable   : {metrics['false_positives']}")
    print(f"    True  Negatives  (TN) — predicted stable,  actually stable   : {metrics['true_negatives']}")
    print(f"    False Negatives  (FN) — predicted stable,  actually worsened : {metrics['false_negatives']}")


def print_feature_importance(xgb_model: XGBClassifier, feature_columns: list) -> None:
    """Print top features by importance score."""
    importance_scores = xgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": importance_scores,
    }).sort_values("importance", ascending=False)

    print("\n── Feature importance (top 10) ────────────────────────")
    print(feature_importance_df.head(10).to_string(index=False))


# ── Save results ─────────────────────────────────────────────────────────────

def save_xgboost_predictions(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    metrics: dict,
    output_dir: Path,
) -> None:
    """Save test predictions and metrics to results/."""
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = test_df[["subject_id", "participant_group", TARGET_COLUMN]].copy()
    predictions_df["xgb_prediction"] = y_pred
    predictions_df["xgb_prediction_proba"] = y_pred_proba
    predictions_path = output_dir / "xgb_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.insert(0, "model", "xgboost")
    metrics_path = output_dir / "xgb_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"\n[ml_baseline] Saved: {predictions_path}")
    print(f"[ml_baseline] Saved: {metrics_path}")


# ── Entry point called from main.py ─────────────────────────────────────────

def run_ml_baseline(output_dir: Path) -> None:
    """
    Orchestrates the XGBoost baseline training and evaluation.
    Called by main.py --step ml_baseline.
    """
    train_df = pd.read_csv("data/processed/train_participants.csv")
    test_df = pd.read_csv("data/processed/test_participants.csv")
    print(f"[ml_baseline] Loaded train: {len(train_df)}, test: {len(test_df)}")

    train_encoded, test_encoded = encode_categorical_features(train_df, test_df)

    feature_cols = FEATURE_COLUMNS + ["sex", "handedness"]
    X_train, y_train = extract_features_and_labels(train_encoded)
    X_test, y_test = extract_features_and_labels(test_encoded)

    xgb_model = train_xgboost_classifier(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

    metrics = evaluate_predictions(y_test, y_pred, y_pred_proba)
    summarise_metrics(metrics, model_name="XGBoost")
    print_feature_importance(xgb_model, feature_cols)

    save_xgboost_predictions(test_df, y_pred, y_pred_proba, metrics, output_dir)

    print("\n[ml_baseline] XGBoost baseline complete.")