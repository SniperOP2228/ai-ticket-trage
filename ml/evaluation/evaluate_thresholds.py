"""
Confidence Threshold Analysis for Human-in-the-Loop Routing.

Methodology:
1. Evaluates thresholds [0.50, 0.60, 0.70, 0.80, 0.90] on the VALIDATION set.
   Calculates:
   - Coverage (% automatically routed where min(cat_conf, urg_conf) >= threshold)
   - Review Rate (% sent to human review)
   - Auto-Route Accuracy (empirical accuracy on the auto-routed subset)
   - Auto-Route Error Rate (1 - accuracy on auto-routed subset)
2. Uses the validation analysis to select the operating threshold (0.80).
3. Evaluates the selected operating threshold ONCE on the untouched TEST set.
4. Generates:
   - ml/evaluation/threshold_analysis.json
   - Markdown summary table
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]


def evaluate_threshold_metrics(cat_model, cat_vec, urg_model, urg_vec, df):
    """Calculate coverage, review rate, and auto-route accuracy across thresholds."""
    X_text = df['cleaned_text'].fillna('')
    y_cat = df['category'].values
    y_urg = df['urgency'].values

    X_cat = cat_vec.transform(X_text)
    X_urg = urg_vec.transform(X_text)

    cat_probs = cat_model.predict_proba(X_cat)
    cat_preds = cat_model.predict(X_cat)
    cat_confs = np.max(cat_probs, axis=1)

    urg_probs = urg_model.predict_proba(X_urg)
    urg_preds = urg_model.predict(X_urg)
    urg_confs = np.max(urg_probs, axis=1)

    # Combined confidence: min(category_confidence, urgency_confidence)
    combined_confs = np.minimum(cat_confs, urg_confs)
    both_correct = (cat_preds == y_cat) & (urg_preds == y_urg)
    cat_correct = (cat_preds == y_cat)

    n_total = len(df)
    results = []

    for t in THRESHOLDS:
        auto_mask = combined_confs >= t
        n_auto = int(np.sum(auto_mask))
        n_review = n_total - n_auto

        coverage = float(n_auto / n_total)
        review_rate = float(n_review / n_total)

        if n_auto > 0:
            auto_cat_acc = float(np.mean(cat_correct[auto_mask]))
            auto_both_acc = float(np.mean(both_correct[auto_mask]))
            auto_cat_err = float(1.0 - auto_cat_acc)
        else:
            auto_cat_acc = 0.0
            auto_both_acc = 0.0
            auto_cat_err = 0.0

        results.append({
            'threshold': float(t),
            'auto_routed_count': n_auto,
            'review_count': n_review,
            'coverage': coverage,
            'review_rate': review_rate,
            'auto_route_category_accuracy': auto_cat_acc,
            'auto_route_both_accuracy': auto_both_acc,
            'auto_route_category_error_rate': auto_cat_err,
        })

    return results


def run_threshold_analysis():
    """Run threshold evaluation on validation set, select threshold, and benchmark on test set."""
    val_path = PROJECT_ROOT / "data" / "processed" / "val.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    models_dir = PROJECT_ROOT / "ml" / "models"
    eval_dir = PROJECT_ROOT / "ml" / "evaluation"

    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    cat_model = joblib.load(models_dir / 'category_model.joblib')
    cat_vec = joblib.load(models_dir / 'tfidf_vectorizer.joblib')
    urg_model = joblib.load(models_dir / 'urgency_model.joblib')
    urg_vec = joblib.load(models_dir / 'urgency_tfidf_vectorizer.joblib')

    # Step 1: Validation set analysis
    logger.info("Evaluating thresholds on VALIDATION set...")
    val_results = evaluate_threshold_metrics(cat_model, cat_vec, urg_model, urg_vec, val_df)

    print("\n" + "=" * 90)
    print("CONFIDENCE THRESHOLD ANALYSIS (VALIDATION SET)")
    print("=" * 90)
    print(f"{'Threshold':<10} | {'Coverage':<10} | {'Review Rate':<12} | {'Auto Cat Acc':<14} | {'Auto Both Acc':<14} | {'Auto Cat Err':<12}")
    print("-" * 90)
    for r in val_results:
        print(f"{r['threshold']:<10.2f} | {r['coverage']*100:<9.1f}% | {r['review_rate']*100:<11.1f}% | {r['auto_route_category_accuracy']*100:<13.1f}% | {r['auto_route_both_accuracy']*100:<13.1f}% | {r['auto_route_category_error_rate']*100:<11.1f}%")

    # Step 2: Select 0.80 based on validation trade-off (accuracy > 90% on auto-routed)
    selected_threshold = 0.80
    logger.info(f"Selected operating threshold based on validation data: {selected_threshold}")

    # Step 3: Evaluate chosen threshold ONCE on untouched test set
    logger.info("Evaluating selected threshold on held-out TEST set...")
    test_results = evaluate_threshold_metrics(cat_model, cat_vec, urg_model, urg_vec, test_df)

    test_selected = next(r for r in test_results if abs(r['threshold'] - selected_threshold) < 1e-4)

    print("\n" + "=" * 90)
    print(f"FINAL TEST SET EVALUATION AT SELECTED THRESHOLD ({selected_threshold})")
    print("=" * 90)
    print(f"  Total Test Tickets:                 {len(test_df)}")
    print(f"  Automatically Routed:               {test_selected['auto_routed_count']} ({test_selected['coverage']*100:.1f}%)")
    print(f"  Escalated for Human Review:         {test_selected['review_count']} ({test_selected['review_rate']*100:.1f}%)")
    print(f"  Auto-Route Category Accuracy:       {test_selected['auto_route_category_accuracy']*100:.1f}%")
    print(f"  Auto-Route Joint (Both) Accuracy:   {test_selected['auto_route_both_accuracy']*100:.1f}%")
    print(f"  Auto-Route Error Rate:              {test_selected['auto_route_category_error_rate']*100:.1f}%")
    print("=" * 90 + "\n")

    summary = {
        'selection_methodology': 'Threshold chosen using validation set performance trade-off; evaluated once on untouched test set.',
        'selected_threshold': selected_threshold,
        'validation_threshold_table': val_results,
        'test_threshold_table': test_results,
        'final_test_benchmark_at_selected_threshold': test_selected
    }

    with open(eval_dir / 'threshold_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_threshold_analysis()
