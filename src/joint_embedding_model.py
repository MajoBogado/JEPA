"""
src/joint_embedding_model.py

Step 5: JEPA-inspired joint embedding model using PyTorch.

Architecture:
- Three modality-specific encoders: cognitive, brain structure, demographic
- Each visit is encoded separately with shared encoder weights
- The two visit embeddings are combined and passed to a predictor
- Prediction happens in embedding space, not from raw values

This follows the core JEPA principle: learn representations of states,
then predict across states. The self-supervised pretraining objective
(predicting visit 2 embedding from visit 1) is the next layer beyond
this supervised implementation — omitted here due to dataset size.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

from src.ml_baseline import evaluate_predictions, summarise_metrics


RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architecture hyperparameters
COGNITIVE_INPUT_DIM = 2       # MMSE, CDR
BRAIN_STRUCTURE_INPUT_DIM = 3 # nWBV, eTIV, ASF
DEMOGRAPHIC_INPUT_DIM = 4     # Age, Education, SES, Sex (encoded)
ENCODER_HIDDEN_DIM = 32
ENCODER_OUTPUT_DIM = 16
JOINT_EMBEDDING_DIM = ENCODER_OUTPUT_DIM * 3  # One per modality stream

# Training hyperparameters
LEARNING_RATE = 1e-3
N_EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 15
VALIDATION_SPLIT = 0.15


# ── Dataset ──────────────────────────────────────────────────────────────────

class LongitudinalParticipantDataset(Dataset):
    """
    PyTorch Dataset wrapping participant-level features for two visits.
    Returns tensors for each modality stream at visit 1 and visit 2,
    plus the binary ground truth label.
    """

    def __init__(self, participant_df: pd.DataFrame):
        self.cognitive_v1 = torch.tensor(
            participant_df[["mmse_v1", "cdr_v1"]].values, dtype=torch.float32
        )
        self.cognitive_v2 = torch.tensor(
            participant_df[["mmse_v2", "cdr_v2"]].values, dtype=torch.float32
        )
        self.brain_structure_v1 = torch.tensor(
            participant_df[["nwbv_v1", "etiv_v1", "asf_v1"]].values, dtype=torch.float32
        )
        self.brain_structure_v2 = torch.tensor(
            participant_df[["nwbv_v2", "etiv_v2", "asf_v2"]].values, dtype=torch.float32
        )
        self.demographic_v1 = torch.tensor(
            participant_df[["age_v1", "education_years", "ses", "sex_encoded"]].values,
            dtype=torch.float32,
        )
        self.demographic_v2 = torch.tensor(
            participant_df[["age_v2", "education_years", "ses", "sex_encoded"]].values,
            dtype=torch.float32,
        )
        self.labels = torch.tensor(
            participant_df["cdr_worsened_after_v2"].values, dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        return (
            self.cognitive_v1[idx],
            self.cognitive_v2[idx],
            self.brain_structure_v1[idx],
            self.brain_structure_v2[idx],
            self.demographic_v1[idx],
            self.demographic_v2[idx],
            self.labels[idx],
        )


# ── Model architecture ────────────────────────────────────────────────────────

class ModalityEncoder(nn.Module):
    """
    Encodes a single modality (cognitive, brain structure, or demographic)
    into a fixed-size embedding vector.

    Shared weights are used across visits — the encoder learns what a
    visit state looks like independently of which visit it is.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class JointEmbeddingPredictor(nn.Module):
    """
    JEPA-inspired joint embedding model for longitudinal patient trajectories.

    Three modality encoders (cognitive, brain structure, demographic) each
    encode visit 1 and visit 2 separately with shared weights. The resulting
    embeddings are concatenated into a joint representation and passed to
    a predictor that outputs the probability of CDR worsening.

    Prediction happens in embedding space — the model reasons about the
    relationship between visit states, not raw feature values.
    """

    def __init__(self):
        super().__init__()

        # Shared-weight encoders — same encoder processes both visits
        self.cognitive_encoder = ModalityEncoder(
            input_dim=COGNITIVE_INPUT_DIM,
            hidden_dim=ENCODER_HIDDEN_DIM,
            output_dim=ENCODER_OUTPUT_DIM,
        )
        self.brain_structure_encoder = ModalityEncoder(
            input_dim=BRAIN_STRUCTURE_INPUT_DIM,
            hidden_dim=ENCODER_HIDDEN_DIM,
            output_dim=ENCODER_OUTPUT_DIM,
        )
        self.demographic_encoder = ModalityEncoder(
            input_dim=DEMOGRAPHIC_INPUT_DIM,
            hidden_dim=ENCODER_HIDDEN_DIM,
            output_dim=ENCODER_OUTPUT_DIM,
        )

        # Predictor operates on the joint embedding from both visits
        # Input: 2 visits × 3 modalities × ENCODER_OUTPUT_DIM
        predictor_input_dim = 2 * JOINT_EMBEDDING_DIM
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # No Sigmoid here — BCEWithLogitsLoss applies it internally,
            # which is numerically more stable and supports pos_weight
        )

    def encode_visit(
        self,
        cognitive_features: torch.Tensor,
        brain_structure_features: torch.Tensor,
        demographic_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a single visit across all three modality streams.
        Returns the concatenated joint embedding for that visit.
        """
        cognitive_embedding = self.cognitive_encoder(cognitive_features)
        brain_structure_embedding = self.brain_structure_encoder(brain_structure_features)
        demographic_embedding = self.demographic_encoder(demographic_features)
        return torch.cat([cognitive_embedding, brain_structure_embedding, demographic_embedding], dim=1)

    def forward(
        self,
        cognitive_v1: torch.Tensor,
        cognitive_v2: torch.Tensor,
        brain_structure_v1: torch.Tensor,
        brain_structure_v2: torch.Tensor,
        demographic_v1: torch.Tensor,
        demographic_v2: torch.Tensor,
    ) -> torch.Tensor:
        visit_1_embedding = self.encode_visit(cognitive_v1, brain_structure_v1, demographic_v1)
        visit_2_embedding = self.encode_visit(cognitive_v2, brain_structure_v2, demographic_v2)

        # Concatenate visit embeddings — predictor reasons about the trajectory
        trajectory_embedding = torch.cat([visit_1_embedding, visit_2_embedding], dim=1)
        return self.predictor(trajectory_embedding).squeeze(1)


# ── Feature preparation ───────────────────────────────────────────────────────

def prepare_jepa_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode sex as integer and normalise continuous features.
    Fit on train only to avoid leaking test distribution.
    """
    train_prepared = train_df.copy()
    test_prepared = test_df.copy()

    sex_encoder = LabelEncoder()
    train_prepared["sex_encoded"] = sex_encoder.fit_transform(train_df["sex"])
    test_prepared["sex_encoded"] = sex_encoder.transform(test_df["sex"])

    # Normalise continuous features using train mean and std
    continuous_cols = [
        "mmse_v1", "mmse_v2", "cdr_v1", "cdr_v2",
        "nwbv_v1", "nwbv_v2", "etiv_v1", "etiv_v2",
        "asf_v1", "asf_v2",
        "age_v1", "age_v2", "education_years", "ses",
    ]
    for col in continuous_cols:
        col_mean = train_df[col].mean()
        col_std = train_df[col].std()
        if col_std > 0:
            train_prepared[col] = (train_df[col] - col_mean) / col_std
            test_prepared[col] = (test_df[col] - col_mean) / col_std

    return train_prepared, test_prepared


def split_train_validation(
    train_df: pd.DataFrame,
    validation_fraction: float = VALIDATION_SPLIT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the training set into train and validation.
    Stratified on participant group to preserve class balance.
    """
    from sklearn.model_selection import train_test_split
    train_split, val_split = train_test_split(
        train_df,
        test_size=validation_fraction,
        random_state=RANDOM_SEED,
        stratify=train_df["participant_group"],
    )
    return train_split, val_split


# ── Training loop ─────────────────────────────────────────────────────────────

def compute_positive_class_weight(labels: pd.Series) -> torch.Tensor:
    """Compute weight for positive class to handle class imbalance."""
    n_negative = (labels == 0).sum()
    n_positive = (labels == 1).sum()
    positive_weight = torch.tensor([n_negative / n_positive], dtype=torch.float32)
    return positive_weight.to(DEVICE)


def train_one_epoch(
    jepa_model: JointEmbeddingPredictor,
    train_loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    loss_fn: nn.BCELoss,
) -> float:
    """Run one training epoch and return average loss."""
    jepa_model.train()
    total_loss = 0.0

    for batch in train_loader:
        cog_v1, cog_v2, brain_v1, brain_v2, demo_v1, demo_v2, labels = [
            tensor.to(DEVICE) for tensor in batch
        ]
        optimiser.zero_grad()
        predictions = jepa_model(cog_v1, cog_v2, brain_v1, brain_v2, demo_v1, demo_v2)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate_validation_loss(
    jepa_model: JointEmbeddingPredictor,
    val_loader: DataLoader,
    loss_fn: nn.BCELoss,
) -> float:
    """Compute validation loss without updating weights."""
    jepa_model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            cog_v1, cog_v2, brain_v1, brain_v2, demo_v1, demo_v2, labels = [
                tensor.to(DEVICE) for tensor in batch
            ]
            predictions = jepa_model(cog_v1, cog_v2, brain_v1, brain_v2, demo_v1, demo_v2)
            loss = loss_fn(predictions, labels)
            total_val_loss += loss.item()

    return total_val_loss / len(val_loader)


def train_jepa_model(
    train_split_df: pd.DataFrame,
    val_split_df: pd.DataFrame,
) -> JointEmbeddingPredictor:
    """
    Full training loop with early stopping on validation loss.
    Returns the best model checkpoint.
    """
    torch.manual_seed(RANDOM_SEED)

    train_dataset = LongitudinalParticipantDataset(train_split_df)
    val_dataset = LongitudinalParticipantDataset(val_split_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    jepa_model = JointEmbeddingPredictor().to(DEVICE)
    optimiser = torch.optim.Adam(jepa_model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=5, factor=0.5
    )

    positive_class_weight = compute_positive_class_weight(train_split_df["cdr_worsened_after_v2"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_class_weight)
    print(f"[jepa] Positive class weight: {positive_class_weight.item():.2f}")

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    print(f"[jepa] Training on {DEVICE} | {len(train_split_df)} train, {len(val_split_df)} val")

    for epoch in range(N_EPOCHS):
        train_loss = train_one_epoch(jepa_model, train_loader, optimiser, loss_fn)
        val_loss = evaluate_validation_loss(jepa_model, val_loader, loss_fn)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in jepa_model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1:3d}/{N_EPOCHS} | "
                f"train loss: {train_loss:.4f} | "
                f"val loss: {val_loss:.4f} | "
                f"best val: {best_val_loss:.4f}"
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} — no improvement for {EARLY_STOPPING_PATIENCE} epochs.")
            break

    jepa_model.load_state_dict(best_model_state)
    print(f"[jepa] Training complete. Best validation loss: {best_val_loss:.4f}")
    return jepa_model


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_test_participants(
    jepa_model: JointEmbeddingPredictor,
    test_df: pd.DataFrame,
    decision_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on the test set.
    Returns (hard predictions, probability scores).
    """
    jepa_model.eval()
    test_dataset = LongitudinalParticipantDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=len(test_df), shuffle=False)

    with torch.no_grad():
        for batch in test_loader:
            cog_v1, cog_v2, brain_v1, brain_v2, demo_v1, demo_v2, _ = [
                tensor.to(DEVICE) for tensor in batch
            ]
            predicted_probas = torch.sigmoid(
                jepa_model(cog_v1, cog_v2, brain_v1, brain_v2, demo_v1, demo_v2)
            ).cpu().numpy()

    predicted_labels = (predicted_probas >= decision_threshold).astype(int)
    return predicted_labels, predicted_probas


# ── Save results ──────────────────────────────────────────────────────────────

def save_jepa_predictions(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    metrics: dict,
    output_dir: Path,
) -> None:
    """Save test predictions and metrics to results/."""
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = test_df[["subject_id", "participant_group", "cdr_worsened_after_v2"]].copy()
    predictions_df["jepa_prediction"] = y_pred
    predictions_df["jepa_prediction_proba"] = y_pred_proba
    predictions_path = output_dir / "jepa_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.insert(0, "model", "jepa")
    metrics_path = output_dir / "jepa_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"\n[jepa] Saved: {predictions_path}")
    print(f"[jepa] Saved: {metrics_path}")


# ── Entry point called from main.py ──────────────────────────────────────────

def run_joint_embedding(output_dir: Path) -> None:
    """
    Orchestrates JEPA-inspired model training and evaluation.
    Called by main.py --step joint_embedding.
    """
    train_df = pd.read_csv("data/processed/train_participants.csv")
    test_df = pd.read_csv("data/processed/test_participants.csv")
    print(f"[jepa] Loaded train: {len(train_df)}, test: {len(test_df)}")

    train_prepared, test_prepared = prepare_jepa_features(train_df, test_df)
    train_split_df, val_split_df = split_train_validation(train_prepared)

    jepa_model = train_jepa_model(train_split_df, val_split_df)

    y_pred, y_pred_proba = predict_test_participants(jepa_model, test_prepared)

    metrics = evaluate_predictions(
        y_true=test_df["cdr_worsened_after_v2"],
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
    )
    summarise_metrics(metrics, model_name="JEPA-inspired Joint Embedding")

    save_jepa_predictions(test_df, y_pred, y_pred_proba, metrics, output_dir)

    print("\n[jepa] Joint embedding step complete.")