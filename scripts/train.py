"""Training Script: Trains PCA Eigenfaces and all three classifiers (Euclidean, KNN, SVM)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.evaluation.experiments import ExperimentRunner

logger = setup_logger("TrainScript")


def main():
    config = load_config()
    raw_dir = PROJECT_ROOT / config["paths"]["raw_data_dir"]
    label_path = PROJECT_ROOT / config["paths"]["label_file"]
    models_dir = PROJECT_ROOT / "models"
    figures_dir = PROJECT_ROOT / config["paths"]["figures_dir"]
    target_size = tuple(config["preprocessing"]["target_size"])
    candidate_k = config["eigenfaces"]["candidate_k_values"]

    logger.info("Initializing ExperimentRunner...")
    runner = ExperimentRunner(
        raw_data_dir=raw_dir,
        label_path=label_path,
        models_dir=models_dir,
        figures_dir=figures_dir,
        target_size=target_size
    )

    logger.info("Partitioning dataset into Train, Val, and Test splits...")
    runner.prepare_dataset_splits(train_ratio=0.70, val_ratio=0.15, seed=42)

    logger.info(f"Training models across candidate K values: {candidate_k}...")
    results = runner.run_experiments(candidate_k_values=candidate_k)

    logger.info(f"Training completed successfully! Optimal K: {results['best_k']}")
    for model_name, metrics in results["final_evaluation"].items():
        logger.info(f"  Model: {model_name:10s} | Test Acc: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | FAR: {metrics['FAR']:.4f} | FRR: {metrics['FRR']:.4f}")


if __name__ == "__main__":
    main()

