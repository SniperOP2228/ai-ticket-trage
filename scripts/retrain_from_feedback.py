"""
Reproducible Feedback Retraining Pipeline.

Methodology:
1. Loads validated human review feedback (data/processed/human_feedback.csv)
2. Combines feedback samples with original training split (data/processed/train.csv)
3. Retrains TF-IDF and calibrated classifiers
4. Evaluates the candidate model on the untouched test set (data/processed/test.csv)
5. Compares against the baseline production model metrics
6. DOES NOT overwrite production models by default!
   Requires an explicit '--promote' flag to promote candidate artifacts to production.

Usage:
  python scripts/retrain_from_feedback.py              # Evaluates candidate model
  python scripts/retrain_from_feedback.py --promote    # Evaluates and promotes to production
"""

import sys
import os
import json
import time
import argparse
import logging
import shutil
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RANDOM_SEED = 42


def retrain_from_feedback(promote: bool = False):
    """Retrain models with human feedback data, benchmark on test set, and optionally promote."""
    data_dir = PROJECT_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "ml" / "models"
    candidate_dir = models_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    feedback_path = data_dir / "human_feedback.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Base train or test dataset not found in data/processed/")

    orig_train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Check for feedback
    feedback = pd.DataFrame()
    if feedback_path.exists():
        feedback = pd.read_csv(feedback_path)

    n_feedback = len(feedback)
    logger.info(f"Loaded {len(orig_train)} base training tickets and {n_feedback} human feedback samples.")

    if n_feedback == 0:
        logger.warning("No human feedback samples available to augment training data. Retraining skipped. Review low-confidence tickets first.")
        return
    else:
        # Combine base training + feedback
        feedback_augmented = feedback[['cleaned_text', 'category', 'urgency']].dropna()
        combined_train = pd.concat([orig_train[['cleaned_text', 'category', 'urgency']], feedback_augmented], ignore_index=True)
        logger.info(f"Augmented training dataset size: {len(combined_train)} records.")

    # 1. Train Category Model Candidate
    logger.info("Training candidate Category model on augmented dataset...")
    cat_vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        strip_accents='unicode',
        token_pattern=r'\b[a-z][a-z]+\b'
    )
    X_train_cat = cat_vectorizer.fit_transform(combined_train['cleaned_text'].fillna(''))
    y_train_cat = combined_train['category']

    cat_candidate = CalibratedClassifierCV(
        estimator=RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1, max_depth=50
        ),
        method='sigmoid',
        cv=5
    )
    cat_candidate.fit(X_train_cat, y_train_cat)

    # Evaluate on held-out test set
    X_test_cat = cat_vectorizer.transform(test['cleaned_text'].fillna(''))
    cat_preds = cat_candidate.predict(X_test_cat)
    cat_test_acc = float(accuracy_score(test['category'], cat_preds))
    cat_test_f1 = float(f1_score(test['category'], cat_preds, average='macro', zero_division=0))

    # 2. Train Urgency Model Candidate
    logger.info("Training candidate Urgency model on augmented dataset...")
    urg_vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        strip_accents='unicode',
        token_pattern=r'\b[a-z][a-z]+\b'
    )
    X_train_urg = urg_vectorizer.fit_transform(combined_train['cleaned_text'].fillna(''))
    y_train_urg = combined_train['urgency']

    urg_candidate = CalibratedClassifierCV(
        estimator=RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1, max_depth=50
        ),
        method='sigmoid',
        cv=5
    )
    urg_candidate.fit(X_train_urg, y_train_urg)

    # Evaluate on held-out test set
    X_test_urg = urg_vectorizer.transform(test['cleaned_text'].fillna(''))
    urg_preds = urg_candidate.predict(X_test_urg)
    urg_test_acc = float(accuracy_score(test['urgency'], urg_preds))
    urg_test_f1 = float(f1_score(test['urgency'], urg_preds, average='macro', zero_division=0))

    # Compare with current baseline metadata
    baseline_cat_meta_path = models_dir / "category_model_metadata.json"
    baseline_cat_f1 = None
    if baseline_cat_meta_path.exists():
        with open(baseline_cat_meta_path) as f:
            meta = json.load(f)
            baseline_cat_f1 = meta.get('test_f1_macro')

    print("\n" + "=" * 80)
    print("FEEDBACK RETRAINING EVALUATION REPORT (HELD-OUT TEST SET)")
    print("=" * 80)
    print(f"Feedback Samples Incorporated: {n_feedback}")
    print(f"Category Candidate Model: Test Accuracy: {cat_test_acc:.4f} | Test Macro F1: {cat_test_f1:.4f}")
    if baseline_cat_f1 is not None:
        print(f"  (Baseline Production Macro F1: {baseline_cat_f1:.4f} -> Delta: {cat_test_f1 - baseline_cat_f1:+.4f})")
    print(f"Urgency Candidate Model:  Test Accuracy: {urg_test_acc:.4f} | Test Macro F1: {urg_test_f1:.4f}")
    print("=" * 80)

    # Save candidates to candidate_dir
    candidate_version = f"1.1.0-fb{n_feedback}"
    joblib.dump(cat_candidate, candidate_dir / 'category_model_candidate.joblib', compress=3)
    joblib.dump(cat_vectorizer, candidate_dir / 'tfidf_vectorizer_candidate.joblib', compress=3)
    joblib.dump(urg_candidate, candidate_dir / 'urgency_model_candidate.joblib', compress=3)
    joblib.dump(urg_vectorizer, candidate_dir / 'urgency_tfidf_vectorizer_candidate.joblib', compress=3)

    candidate_report = {
        'candidate_version': candidate_version,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        'feedback_samples': n_feedback,
        'category_metrics': {'test_accuracy': cat_test_acc, 'test_f1_macro': cat_test_f1},
        'urgency_metrics': {'test_accuracy': urg_test_acc, 'test_f1_macro': urg_test_f1},
        'promoted_to_production': promote
    }
    with open(candidate_dir / 'candidate_evaluation.json', 'w') as f:
        json.dump(candidate_report, f, indent=2)

    if promote:
        logger.info("\n[PROMOTING TO PRODUCTION] Overwriting production models with candidate artifacts...")
        shutil.copy2(candidate_dir / 'category_model_candidate.joblib', models_dir / 'category_model.joblib')
        shutil.copy2(candidate_dir / 'tfidf_vectorizer_candidate.joblib', models_dir / 'tfidf_vectorizer.joblib')
        shutil.copy2(candidate_dir / 'urgency_model_candidate.joblib', models_dir / 'urgency_model.joblib')
        shutil.copy2(candidate_dir / 'urgency_tfidf_vectorizer_candidate.joblib', models_dir / 'urgency_tfidf_vectorizer.joblib')

        # Update metadata
        with open(models_dir / 'category_model_metadata.json', 'w') as f:
            json.dump({
                'model_name': 'Random Forest (Calibrated with Feedback)',
                'model_version': candidate_version,
                'task': 'category_classification',
                'calibration_method': 'sigmoid (Platt scaling, 5-fold CV)',
                'is_calibrated': True,
                'feedback_samples_incorporated': n_feedback,
                'test_accuracy': cat_test_acc,
                'test_f1_macro': cat_test_f1,
                'promoted_at': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            }, f, indent=2)
        logger.info(f"Successfully promoted model version {candidate_version} to production!")
    else:
        logger.info("\nCandidate artifacts saved to ml/models/candidates/.")
        logger.info("To promote this candidate to production, rerun with: python scripts/retrain_from_feedback.py --promote")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain model on human feedback data.")
    parser.add_argument("--promote", action="store_true", help="Explicitly promote candidate to production model.")
    args = parser.parse_args()
    retrain_from_feedback(promote=args.promote)
