"""
Probability Calibration Evaluation Script.

Evaluates probability calibration on the held-out test set for:
1. Category Classification Model
2. Urgency Classification Model

Metrics calculated:
- Multi-Class Brier Score
- Expected Calibration Error (ECE) across confidence bins
- Maximum Calibration Error (MCE)
- Reliability Diagrams (Calibration Curves)

Calibration Principle:
"If a model outputs 0.80 confidence, calibration checks whether predictions
 in that confidence range are correct at approximately the expected frequency (80%)."

Saves:
- docs/images/category_calibration.png
- docs/images/urgency_calibration.png
- ml/evaluation/calibration_metrics.json
"""

import sys
import json
import logging
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def compute_multiclass_brier(y_true, y_prob, classes):
    """
    Compute multi-class Brier score:
    Brier = (1 / N) * sum_{i=1}^N sum_{k=1}^K (y_{ik} - p_{ik})^2
    """
    lb = LabelBinarizer()
    lb.fit(classes)
    y_true_bin = lb.transform(y_true)
    if y_true_bin.shape[1] == 1:
        # Binary case fallback
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    
    brier = np.mean(np.sum((y_prob - y_true_bin) ** 2, axis=1))
    return float(brier)


def compute_ece(y_true, y_pred, confidences, n_bins=10):
    """
    Compute Top-Label Expected Calibration Error (ECE) and bin statistics.
    
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    is_correct = (np.array(y_true) == np.array(y_pred)).astype(float)
    confidences = np.array(confidences)

    ece = 0.0
    mce = 0.0
    bin_data = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if in_bin.sum() > 0:
            accuracy_in_bin = np.mean(is_correct[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            calibration_gap = abs(accuracy_in_bin - avg_confidence_in_bin)
            ece += prop_in_bin * calibration_gap
            mce = max(mce, calibration_gap)

            bin_data.append({
                'bin_range': f"({bin_lower:.2f}, {bin_upper:.2f}]",
                'count': int(in_bin.sum()),
                'avg_confidence': float(avg_confidence_in_bin),
                'accuracy': float(accuracy_in_bin),
                'gap': float(calibration_gap)
            })
        else:
            bin_data.append({
                'bin_range': f"({bin_lower:.2f}, {bin_upper:.2f}]",
                'count': 0,
                'avg_confidence': float((bin_lower + bin_upper) / 2.0),
                'accuracy': 0.0,
                'gap': 0.0
            })

    return float(ece), float(mce), bin_data


def plot_reliability_diagram(bin_data, ece, mce, brier, title, save_path):
    """Generate and save publication-quality reliability diagram / calibration curve."""
    valid_bins = [b for b in bin_data if b['count'] > 0]
    
    confs = [b['avg_confidence'] for b in valid_bins]
    accs = [b['accuracy'] for b in valid_bins]
    counts = [b['count'] for b in valid_bins]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Reliability curve
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    ax1.plot(confs, accs, marker='o', linewidth=2, color='#0284c7', label='Model Calibration')
    
    # Gap bars
    for b in valid_bins:
        c, a = b['avg_confidence'], b['accuracy']
        ax1.plot([c, c], [c, a], color='#f43f5e', linewidth=1.5, alpha=0.7)

    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xlabel('Mean Predicted Confidence')
    ax1.set_ylabel('Observed Accuracy')
    ax1.set_title(f"{title}\nECE: {ece:.4f} | MCE: {mce:.4f} | Brier Score: {brier:.4f}", fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Confidence distribution histogram
    bin_centers = [b['avg_confidence'] for b in valid_bins]
    ax2.bar(bin_centers, counts, width=0.08, color='#6366f1', edgecolor='black', alpha=0.7)
    ax2.set_xlim([0, 1])
    ax2.set_xlabel('Confidence Bin')
    ax2.set_ylabel('Sample Count')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved calibration diagram to {save_path}")


def evaluate_calibration():
    """Run probability calibration evaluation on the test set."""
    test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found at {test_path}")

    test = pd.read_csv(test_path)
    X_test_text = test['cleaned_text'].fillna('')
    y_test_cat = test['category']
    y_test_urg = test['urgency']

    models_dir = PROJECT_ROOT / "ml" / "models"
    images_dir = PROJECT_ROOT / "docs" / "images"
    eval_dir = PROJECT_ROOT / "ml" / "evaluation"
    images_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. Evaluate Category Model Calibration
    logger.info("Evaluating Category model calibration...")
    cat_model = joblib.load(models_dir / 'category_model.joblib')
    cat_vec = joblib.load(models_dir / 'tfidf_vectorizer.joblib')
    X_test_cat = cat_vec.transform(X_test_text)

    cat_probs = cat_model.predict_proba(X_test_cat)
    cat_preds = cat_model.predict(X_test_cat)
    cat_confs = np.max(cat_probs, axis=1)

    cat_classes = list(cat_model.classes_)
    cat_brier = compute_multiclass_brier(y_test_cat, cat_probs, cat_classes)
    cat_ece, cat_mce, cat_bins = compute_ece(y_test_cat, cat_preds, cat_confs, n_bins=10)

    plot_reliability_diagram(
        cat_bins, cat_ece, cat_mce, cat_brier,
        "Category Classifier Reliability Diagram (Held-Out Test Set)",
        images_dir / 'category_calibration.png'
    )

    # 2. Evaluate Urgency Model Calibration
    logger.info("Evaluating Urgency model calibration...")
    urg_model = joblib.load(models_dir / 'urgency_model.joblib')
    urg_vec = joblib.load(models_dir / 'urgency_tfidf_vectorizer.joblib')
    X_test_urg = urg_vec.transform(X_test_text)

    urg_probs = urg_model.predict_proba(X_test_urg)
    urg_preds = urg_model.predict(X_test_urg)
    urg_confs = np.max(urg_probs, axis=1)

    urg_classes = list(urg_model.classes_)
    urg_brier = compute_multiclass_brier(y_test_urg, urg_probs, urg_classes)
    urg_ece, urg_mce, urg_bins = compute_ece(y_test_urg, urg_preds, urg_confs, n_bins=10)

    plot_reliability_diagram(
        urg_bins, urg_ece, urg_mce, urg_brier,
        "Urgency Classifier Reliability Diagram (Held-Out Test Set)",
        images_dir / 'urgency_calibration.png'
    )

    report = {
        'principle': "If a model outputs 0.80 confidence, calibration checks whether predictions in that range are correct ~80% of the time.",
        'category_calibration': {
            'model': 'CalibratedClassifierCV (RandomForest, sigmoid, 5-fold CV)',
            'test_samples': len(test),
            'brier_score': cat_brier,
            'expected_calibration_error_ece': cat_ece,
            'maximum_calibration_error_mce': cat_mce,
            'bin_breakdown': cat_bins
        },
        'urgency_calibration': {
            'model': 'CalibratedClassifierCV (RandomForest, sigmoid, 5-fold CV)',
            'test_samples': len(test),
            'brier_score': urg_brier,
            'expected_calibration_error_ece': urg_ece,
            'maximum_calibration_error_mce': urg_mce,
            'bin_breakdown': urg_bins
        }
    }

    with open(eval_dir / 'calibration_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("PROBABILITY CALIBRATION EVALUATION RESULTS (HELD-OUT TEST SET)")
    print("=" * 80)
    print(f"\n--- Category Classifier ---")
    print(f"  Multi-Class Brier Score: {cat_brier:.4f}")
    print(f"  Expected Calibration Error (ECE): {cat_ece:.4f} ({cat_ece * 100:.2f}%)")
    print(f"  Maximum Calibration Error (MCE):  {cat_mce:.4f} ({cat_mce * 100:.2f}%)")

    print(f"\n--- Urgency Classifier ---")
    print(f"  Multi-Class Brier Score: {urg_brier:.4f}")
    print(f"  Expected Calibration Error (ECE): {urg_ece:.4f} ({urg_ece * 100:.2f}%)")
    print(f"  Maximum Calibration Error (MCE):  {urg_mce:.4f} ({urg_mce * 100:.2f}%)")
    print("=" * 80 + "\n")

    return report


if __name__ == "__main__":
    evaluate_calibration()
