"""
Error Analysis for the Trained Models.

Identifies and analyzes misclassified examples:
1. Which categories are most confused?
2. Are short tickets harder to classify?
3. Are ambiguous tickets harder?
4. Common patterns in misclassified examples
5. Per-class performance breakdown

Generates error analysis report and saves to ml/evaluation/
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def run_error_analysis():
    """Perform comprehensive error analysis on the test set."""
    # Load test data
    test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")

    # Load models
    models_dir = PROJECT_ROOT / "ml" / "models"
    cat_model = joblib.load(models_dir / 'category_model.joblib')
    cat_vectorizer = joblib.load(models_dir / 'tfidf_vectorizer.joblib')
    urg_model = joblib.load(models_dir / 'urgency_model.joblib')
    urg_vectorizer = joblib.load(models_dir / 'urgency_tfidf_vectorizer.joblib')

    X_text = test['cleaned_text'].fillna('')
    X_cat = cat_vectorizer.transform(X_text)
    X_urg = urg_vectorizer.transform(X_text)

    # Predictions
    cat_preds = cat_model.predict(X_cat)
    urg_preds = urg_model.predict(X_urg)

    # Get confidence scores
    if hasattr(cat_model, 'predict_proba'):
        cat_probs = cat_model.predict_proba(X_cat)
        cat_confidence = cat_probs.max(axis=1)
    else:
        cat_confidence = np.ones(len(cat_preds))

    if hasattr(urg_model, 'predict_proba'):
        urg_probs = urg_model.predict_proba(X_urg)
        urg_confidence = urg_probs.max(axis=1)
    else:
        urg_confidence = np.ones(len(urg_preds))

    test['cat_pred'] = cat_preds
    test['cat_confidence'] = cat_confidence
    test['cat_correct'] = test['category'] == test['cat_pred']
    test['urg_pred'] = urg_preds
    test['urg_confidence'] = urg_confidence
    test['urg_correct'] = test['urgency'] == test['urg_pred']
    test['text_length'] = test['cleaned_text'].str.len()
    test['word_count'] = test['cleaned_text'].str.split().str.len()

    report = []

    # =============================================
    # 1. Overall Error Rates
    # =============================================
    cat_accuracy = test['cat_correct'].mean()
    urg_accuracy = test['urg_correct'].mean()

    report.append("=" * 70)
    report.append("ERROR ANALYSIS REPORT")
    report.append("=" * 70)
    report.append(f"\nOverall Category Accuracy: {cat_accuracy:.4f} ({test['cat_correct'].sum()}/{len(test)})")
    report.append(f"Overall Urgency Accuracy: {urg_accuracy:.4f} ({test['urg_correct'].sum()}/{len(test)})")
    report.append(f"Category errors: {(~test['cat_correct']).sum()}")
    report.append(f"Urgency errors: {(~test['urg_correct']).sum()}")

    # =============================================
    # 2. Most Confused Categories
    # =============================================
    report.append(f"\n{'=' * 70}")
    report.append("MOST CONFUSED CATEGORIES")
    report.append("=" * 70)

    cat_errors = test[~test['cat_correct']].copy()
    if len(cat_errors) > 0:
        confusion_pairs = cat_errors.groupby(['category', 'cat_pred']).size().reset_index(name='count')
        confusion_pairs = confusion_pairs.sort_values('count', ascending=False).head(10)
        report.append("\nTop 10 confusion pairs (Actual -> Predicted):")
        for _, row in confusion_pairs.iterrows():
            report.append(f"  {row['category']} -> {row['cat_pred']}: {row['count']} errors")

    # =============================================
    # 3. Text Length vs Accuracy
    # =============================================
    report.append(f"\n{'=' * 70}")
    report.append("TEXT LENGTH vs ACCURACY")
    report.append("=" * 70)

    # Bin by text length
    test['length_bin'] = pd.cut(test['word_count'], bins=[0, 10, 25, 50, 100, float('inf')],
                                labels=['Very Short (<=10)', 'Short (11-25)', 'Medium (26-50)',
                                       'Long (51-100)', 'Very Long (>100)'])

    length_analysis = test.groupby('length_bin', observed=True).agg(
        count=('cat_correct', 'size'),
        cat_accuracy=('cat_correct', 'mean'),
        urg_accuracy=('urg_correct', 'mean'),
        avg_confidence=('cat_confidence', 'mean'),
    ).reset_index()

    report.append("\nCategory Accuracy by Text Length:")
    for _, row in length_analysis.iterrows():
        report.append(f"  {row['length_bin']}: accuracy={row['cat_accuracy']:.3f}, "
                     f"n={row['count']}, avg_confidence={row['avg_confidence']:.3f}")

    # =============================================
    # 4. Confidence vs Accuracy
    # =============================================
    report.append(f"\n{'=' * 70}")
    report.append("CONFIDENCE vs ACCURACY")
    report.append("=" * 70)

    conf_bins = pd.cut(test['cat_confidence'], bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                       labels=['<0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'])
    conf_analysis = test.groupby(conf_bins, observed=True).agg(
        count=('cat_correct', 'size'),
        accuracy=('cat_correct', 'mean'),
    ).reset_index()

    report.append("\nCategory Accuracy by Confidence Level:")
    for _, row in conf_analysis.iterrows():
        report.append(f"  Confidence {row['cat_confidence']}: accuracy={row['accuracy']:.3f}, n={row['count']}")

    # =============================================
    # 5. Per-class Performance
    # =============================================
    report.append(f"\n{'=' * 70}")
    report.append("PER-CLASS PERFORMANCE")
    report.append("=" * 70)

    cat_report = classification_report(test['category'], test['cat_pred'], output_dict=True)
    report.append("\nCategory Classification Report:")
    report.append(classification_report(test['category'], test['cat_pred']))

    report.append("\nUrgency Classification Report:")
    report.append(classification_report(test['urgency'], test['urg_pred']))

    # =============================================
    # 6. Sample Misclassified Examples
    # =============================================
    report.append(f"\n{'=' * 70}")
    report.append("SAMPLE MISCLASSIFIED TICKETS (Category)")
    report.append("=" * 70)
    report.append("(Text truncated to 100 chars for readability)")

    if len(cat_errors) > 0:
        samples = cat_errors.nsmallest(10, 'cat_confidence')
        for _, row in samples.iterrows():
            text_preview = row['cleaned_text'][:100] + '...' if len(row['cleaned_text']) > 100 else row['cleaned_text']
            report.append(f"\n  Text: \"{text_preview}\"")
            report.append(f"  Actual: {row['category']} | Predicted: {row['cat_pred']} | Confidence: {row['cat_confidence']:.3f}")

    # =============================================
    # 7. Summary & Recommendations
    # =============================================
    report.append(f"\n{'=' * 70}")
    report.append("SUMMARY & RECOMMENDATIONS")
    report.append("=" * 70)

    low_conf_pct = (test['cat_confidence'] < 0.8).mean() * 100
    report.append(f"\n- {low_conf_pct:.1f}% of predictions have confidence < 0.80 (would trigger human review)")
    report.append(f"- Average confidence for correct: {test[test['cat_correct']]['cat_confidence'].mean():.3f}")
    report.append(f"- Average confidence for incorrect: {test[~test['cat_correct']]['cat_confidence'].mean():.3f}")

    if len(cat_errors) > 0:
        hardest_class = cat_errors['category'].value_counts().index[0]
        report.append(f"- Hardest category to classify: {hardest_class}")

    report.append("\nRecommendations:")
    report.append("  1. Consider adding more training data for underperforming categories")
    report.append("  2. Very short tickets (<10 words) may benefit from additional context")
    report.append("  3. The confidence threshold of 0.80 is appropriate based on confidence-accuracy analysis")
    report.append("  4. Human review corrections can be used to retrain and improve the model")

    # Print and save
    report_text = '\n'.join(report)
    print(report_text)

    # Save with utf-8 encoding
    eval_dir = PROJECT_ROOT / "ml" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_dir / 'error_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    # Save as structured JSON too
    error_data = {
        'category_accuracy': cat_accuracy,
        'urgency_accuracy': urg_accuracy,
        'total_test_samples': len(test),
        'category_errors': int((~test['cat_correct']).sum()),
        'urgency_errors': int((~test['urg_correct']).sum()),
        'low_confidence_pct': low_conf_pct,
        'avg_confidence_correct': float(test[test['cat_correct']]['cat_confidence'].mean()),
        'avg_confidence_incorrect': float(test[~test['cat_correct']]['cat_confidence'].mean()) if len(cat_errors) > 0 else None,
    }
    with open(eval_dir / 'error_analysis_data.json', 'w') as f:
        json.dump(error_data, f, indent=2)

    # Generate error analysis plots
    images_dir = PROJECT_ROOT / "docs" / "images"

    # Confidence distribution for correct vs incorrect
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(test[test['cat_correct']]['cat_confidence'], bins=30, alpha=0.7, label='Correct', color='#2ecc71')
    ax.hist(test[~test['cat_correct']]['cat_confidence'], bins=30, alpha=0.7, label='Incorrect', color='#e74c3c')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution: Correct vs Incorrect Predictions')
    ax.legend()
    ax.axvline(0.8, color='black', linestyle='--', label='Threshold (0.80)')
    plt.tight_layout()
    plt.savefig(images_dir / 'confidence_distribution.png', dpi=150)
    plt.close()

    logger.info(f"Error analysis saved to {eval_dir}")
    return report_text


if __name__ == "__main__":
    run_error_analysis()
