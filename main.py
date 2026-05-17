"""
main.py — Orchestrator for the JEPA vs LLM clinical trial prediction exercise.

Usage:
    python main.py --step exploration
    python main.py --step synthetic
    python main.py --step preprocessing
    python main.py --step preprocessing --data-path data/raw/oasis_augmented.csv
    python main.py --step llm_baseline
    python main.py --step ml_baseline
    python main.py --step joint_embedding
    python main.py --step comparison

Each step maps to a module in src/. No logic lives here.
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JEPA vs LLM — OASIS-2 longitudinal prediction exercise"
    )
    parser.add_argument(
        "--step",
        required=True,
        choices=["exploration", "synthetic", "preprocessing", "llm_baseline", "ml_baseline", "joint_embedding", "comparison"],
        help="Which pipeline step to run",
    )
    parser.add_argument(
        "--data-path",
        default="data/raw/oasis_longitudinal.csv",
        help="Path to the raw OASIS-2 CSV file (use oasis_augmented.csv after synthetic step)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for saving figures and metrics",
    )
    return parser.parse_args()


def run_exploration(raw_data_path: Path, output_dir: Path) -> None:
    from src.data_exploration import run_full_exploration
    run_full_exploration(raw_data_path=raw_data_path, output_dir=output_dir)


def run_synthetic(raw_data_path: Path, output_dir: Path) -> None:
    from src.synthetic_data import run_synthetic_augmentation
    run_synthetic_augmentation(raw_data_path=raw_data_path, output_dir=output_dir)


def run_preprocessing(raw_data_path: Path, output_dir: Path) -> None:
    from src.preprocessing import run_full_preprocessing
    run_full_preprocessing(raw_data_path=raw_data_path, output_dir=output_dir)


def run_llm_baseline(output_dir: Path) -> None:
    from src.llm_predictor import run_llm_baseline
    run_llm_baseline(output_dir=output_dir)


def run_ml_baseline(output_dir: Path) -> None:
    from src.ml_baseline import run_ml_baseline
    run_ml_baseline(output_dir=output_dir)


def run_joint_embedding(output_dir: Path) -> None:
    from src.joint_embedding_model import run_joint_embedding
    run_joint_embedding(output_dir=output_dir)


def run_comparison(output_dir: Path) -> None:
    from src.evaluation import run_full_comparison
    run_full_comparison(output_dir=output_dir)


STEP_RUNNERS = {
    "exploration":     run_exploration,
    "synthetic":       run_synthetic,
    "preprocessing":   run_preprocessing,
    "llm_baseline":    run_llm_baseline,
    "ml_baseline":     run_ml_baseline,
    "joint_embedding": run_joint_embedding,
    "comparison":      run_comparison,
}

DATA_DEPENDENT_STEPS = {"exploration", "synthetic", "preprocessing"}


if __name__ == "__main__":
    args = parse_args()

    raw_data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step_runner = STEP_RUNNERS[args.step]

    if args.step in DATA_DEPENDENT_STEPS:
        step_runner(raw_data_path=raw_data_path, output_dir=output_dir)
    else:
        step_runner(output_dir=output_dir)

    print(f"\n[main] Step '{args.step}' completed.")