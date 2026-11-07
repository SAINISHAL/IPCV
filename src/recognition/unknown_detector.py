"""Unknown Face Detector and ROC/FAR/FRR Metrics Calculator."""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("UnknownDetector")


class UnknownDetector:
    """Manages rejection calibration and calculates False Acceptance & False Rejection Rates."""

    UNKNOWN_LABEL = "UNKNOWN"

    @staticmethod
    def evaluate_unknown_rejection(true_labels: List[str], predicted_labels: List[str],
                                   enrolled_set: set) -> Dict[str, float]:
        """Evaluate unknown rejection performance.
        
        Definitions:
          - False Acceptance (FA): An un-enrolled / unknown person is classified as an enrolled student.
          - False Rejection (FR): An enrolled student is rejected as UNKNOWN.
          - True Acceptance (TA): An enrolled student is accepted as an enrolled student.
          - True Rejection (TR): An un-enrolled / unknown person is rejected as UNKNOWN.
        """
        fa_count = 0
        fr_count = 0
        total_unknown = 0
        total_known = 0
        correct_known = 0
        correct_unknown = 0

        for true_lbl, pred_lbl in zip(true_labels, predicted_labels):
            is_known = (true_lbl in enrolled_set and true_lbl != UnknownDetector.UNKNOWN_LABEL)
            
            if is_known:
                total_known += 1
                if pred_lbl == UnknownDetector.UNKNOWN_LABEL:
                    fr_count += 1
                elif pred_lbl == true_lbl:
                    correct_known += 1
            else:
                total_unknown += 1
                if pred_lbl != UnknownDetector.UNKNOWN_LABEL:
                    fa_count += 1
                else:
                    correct_unknown += 1

        far = float(fa_count / total_unknown) if total_unknown > 0 else 0.0
        frr = float(fr_count / total_known) if total_known > 0 else 0.0
        known_acc = float(correct_known / total_known) if total_known > 0 else 0.0
        unknown_acc = float(correct_unknown / total_unknown) if total_unknown > 0 else 1.0

        return {
            "FAR": far,
            "FRR": frr,
            "known_accuracy": known_acc,
            "unknown_accuracy": unknown_acc,
            "total_known_eval": total_known,
            "total_unknown_eval": total_unknown
        }

