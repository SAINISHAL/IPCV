"""Evaluation metrics calculation module."""
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


class EvaluationMetrics:
    """Calculates classification, per-class, and unknown face metrics."""

    @staticmethod
    def compute_all_metrics(y_true: List[str], y_pred: List[str],
                            enrolled_classes: List[str]) -> Dict[str, Any]:
        """Compute accuracy, macro precision, recall, F1, and per-class statistics."""
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        # Standard accuracy
        acc = float(accuracy_score(y_true_arr, y_pred_arr))

        # Precision, recall, macro F1
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true_arr, y_pred_arr, average="macro", zero_division=0
        )
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true_arr, y_pred_arr, average="weighted", zero_division=0
        )

        # Per-class accuracy
        per_class_acc = {}
        unique_classes = sorted(list(set(y_true)))
        for cls in unique_classes:
            mask = (y_true_arr == cls)
            if np.sum(mask) > 0:
                cls_acc = float(np.mean(y_pred_arr[mask] == cls))
                per_class_acc[cls] = cls_acc

        return {
            "accuracy": acc,
            "macro_precision": float(p_macro),
            "macro_recall": float(r_macro),
            "macro_f1": float(f1_macro),
            "weighted_f1": float(f1_weighted),
            "per_class_accuracy": per_class_acc,
            "total_samples": len(y_true)
        }

