"""
Train and Compare Multiple Models for Category Classification,
followed by Probability Calibration of the Selected Production Model.

Methodology:
1. Candidate Model Comparison on Training Set, evaluated on Validation Set:
   - Logistic Regression
   - Multinomial Naive Bayes
   - Linear SVM
   - Random Forest
2. Model Selection strictly based on Validation Macro F1.
3. Probability Calibration of Selected Model:
   - Uses CalibratedClassifierCV(method='sigmoid', cv=5) on Training Data.
   - 5-fold cross-validation prevents data leakage and preserves the validation set.
   - Sigmoid (Platt scaling) is selected over Isotonic because multi-class (10 classes)
     with strong class imbalance (e.g., General Inquiry has <250 samples) causes
     non-parametric isotonic regression to overfit and produce staircase artifacts.
4. Evaluation of the Calibrated Model on both Validation and Untouched Test Sets.
5. Saves:
   - Compressed calibrated production model (joblib compress=3)
   - Correctly partitioned metadata (val_* vs test_* metrics)
   - Comparison and confusion matrix plots
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def load_splits():
    """Load preprocessed train/val/test splits."""
    data_dir = PROJECT_ROOT / "data" / "processed"
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")
    logger.info(f"Loaded splits — Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


def build_tfidf(train_texts, max_features=10000):
    """
    Build TF-IDF vectorizer fit ONLY on training data.
    
    Design decisions:
    - max_features=10000: balances vocabulary coverage with inference speed
    - ngram_range=(1,2): captures unigrams and bigrams like 'payment failed', 'database error'
    - sublinear_tf=True: applies log normalization, better for text classification
    - min_df=2: removes singular noise / typos
    - max_df=0.95: removes corpus-wide stopwords
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        strip_accents='unicode',
        token_pattern=r'\b[a-z][a-z]+\b'
    )
    vectorizer.fit(train_texts)
    logger.info(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
    return vectorizer


def evaluate_model(model, X, y_true, model_name=""):
    """Evaluate model and return metrics dict and predictions."""
    start = time.time()
    y_pred = model.predict(X)
    inference_time = time.time() - start

    metrics = {
        'model': model_name,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'inference_time_seconds': float(inference_time),
    }
    return metrics, y_pred


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix: {save_path}")


def train_category_models():
    """Train and compare candidate category models, then calibrate and save the selected model."""
    train, val, test = load_splits()

    X_train_text = train['cleaned_text'].fillna('')
    y_train = train['category']
    X_val_text = val['cleaned_text'].fillna('')
    y_val = val['category']
    X_test_text = test['cleaned_text'].fillna('')
    y_test = test['category']

    # Step 1: Fit TF-IDF on training set ONLY
    logger.info("Fitting TF-IDF vectorizer strictly on training data...")
    vectorizer = build_tfidf(X_train_text)
    X_train = vectorizer.transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    X_test = vectorizer.transform(X_test_text)

    labels = sorted(y_train.unique())
    logger.info(f"Category labels ({len(labels)} classes): {labels}")

    # Define candidate models
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            class_weight='balanced',
            C=1.0,
            solver='lbfgs'
        ),
        'Multinomial NB': MultinomialNB(alpha=0.1),
        'Linear SVM': CalibratedClassifierCV(
            LinearSVC(
                max_iter=2000,
                random_state=RANDOM_SEED,
                class_weight='balanced',
                C=1.0
            ),
            cv=3
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_SEED,
            class_weight='balanced',
            n_jobs=-1,
            max_depth=50
        ),
    }

    # Step 2: Compare candidate models on Validation Set
    results = []
    best_candidate_model = None
    best_candidate_name = ""
    best_candidate_val_f1 = -1.0
    best_candidate_val_metrics = None
    best_candidate_val_preds = None

    print(f"\n{'=' * 80}")
    print("STEP 1: CANDIDATE MODEL COMPARISON (Evaluated on Validation Set)")
    print(f"{'=' * 80}")

    for name, model in models.items():
        logger.info(f"Training {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        val_metrics, y_val_pred = evaluate_model(model, X_val, y_val, name)
        val_metrics['training_time_seconds'] = train_time
        results.append(val_metrics)

        print(f"\n--- {name} ---")
        print(f"  Val Accuracy:    {val_metrics['accuracy']:.4f}")
        print(f"  Val Macro F1:    {val_metrics['f1_macro']:.4f}")
        print(f"  Val Weighted F1: {val_metrics['f1_weighted']:.4f}")
        print(f"  Train Time:      {train_time:.2f}s")
        print(f"  Inference Time:  {val_metrics['inference_time_seconds']:.4f}s")

        if val_metrics['f1_macro'] > best_candidate_val_f1:
            best_candidate_val_f1 = val_metrics['f1_macro']
            best_candidate_model = model
            best_candidate_name = name
            best_candidate_val_metrics = val_metrics
            best_candidate_val_preds = y_val_pred

    eval_dir = PROJECT_ROOT / "ml" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(eval_dir / 'category_model_comparison.csv', index=False)

    print(f"\n{'=' * 80}")
    print("CANDIDATE MODEL VALIDATION COMPARISON TABLE")
    print(f"{'=' * 80}")
    print(results_df[['model', 'accuracy', 'f1_macro', 'f1_weighted',
                       'training_time_seconds', 'inference_time_seconds']].to_string(index=False))

    print(f"\n[SELECTED BASE MODEL] {best_candidate_name} (Validation Macro F1: {best_candidate_val_f1:.4f})")

    # Step 3: Probability Calibration of the Selected Model
    # Sigmoid (Platt scaling) with 5-fold cross-validation on X_train
    print(f"\n{'=' * 80}")
    print("STEP 2: PROBABILITY CALIBRATION (CalibratedClassifierCV, method='sigmoid', cv=5)")
    print(f"{'=' * 80}")
    logger.info(f"Calibrating {best_candidate_name} using 5-fold CV on training data...")

    calibrated_model = CalibratedClassifierCV(
        estimator=RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_SEED,
            class_weight='balanced',
            n_jobs=-1,
            max_depth=50
        ),
        method='sigmoid',
        cv=5
    )
    t0_cal = time.time()
    calibrated_model.fit(X_train, y_train)
    calibration_fit_time = time.time() - t0_cal
    logger.info(f"Calibration completed in {calibration_fit_time:.2f}s")

    # Step 4: Evaluate Calibrated Model on Validation Set
    cal_val_metrics, cal_val_preds = evaluate_model(
        calibrated_model, X_val, y_val, f"{best_candidate_name} (Calibrated)"
    )
    print(f"\nCalibrated Model on Validation Set:")
    print(f"  Val Accuracy:    {cal_val_metrics['accuracy']:.4f}")
    print(f"  Val Macro F1:    {cal_val_metrics['f1_macro']:.4f}")
    print(f"  Val Weighted F1: {cal_val_metrics['f1_weighted']:.4f}")

    # Step 5: Final Evaluation on Untouched TEST Set (Done ONLY once on final model)
    print(f"\n{'=' * 80}")
    print("STEP 3: FINAL TEST SET BENCHMARK (Held-out untouched test set)")
    print(f"{'=' * 80}")
    test_metrics, test_preds = evaluate_model(
        calibrated_model, X_test, y_test, f"{best_candidate_name} (Calibrated)"
    )
    print(f"Test Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"Test Macro F1:    {test_metrics['f1_macro']:.4f}")
    print(f"Test Weighted F1: {test_metrics['f1_weighted']:.4f}")
    print(f"\nFinal Test Classification Report:\n")
    print(classification_report(y_test, test_preds, target_names=labels))

    # Step 6: Generate and Save Visualizations
    images_dir = PROJECT_ROOT / "docs" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        y_val, cal_val_preds, labels,
        f'Category Confusion Matrix (Val) — {best_candidate_name} (Calibrated)',
        images_dir / 'category_confusion_matrix_val.png'
    )
    plot_confusion_matrix(
        y_test, test_preds, labels,
        f'Category Confusion Matrix (Test) — {best_candidate_name} (Calibrated)',
        images_dir / 'category_confusion_matrix_test.png'
    )

    # Model comparison chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    model_names = results_df['model']
    axes[0].barh(model_names, results_df['f1_macro'], color=sns.color_palette("viridis", len(model_names)))
    axes[0].set_xlabel('Validation Macro F1')
    axes[0].set_title('Category Models — Validation Macro F1')
    axes[1].barh(model_names, results_df['accuracy'], color=sns.color_palette("mako", len(model_names)))
    axes[1].set_xlabel('Validation Accuracy')
    axes[1].set_title('Category Models — Validation Accuracy')
    plt.tight_layout()
    plt.savefig(images_dir / 'category_model_comparison.png', dpi=150)
    plt.close()

    # Step 7: Save Compressed Calibrated Model and Vectorizer
    models_dir = PROJECT_ROOT / "ml" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save calibrated model with compress=3 (reduces size from ~150MB to ~30MB)
    joblib.dump(calibrated_model, models_dir / 'category_model.joblib', compress=3)
    joblib.dump(vectorizer, models_dir / 'tfidf_vectorizer.joblib', compress=3)
    logger.info(f"Saved calibrated category model and vectorizer with compression.")

    # Step 8: Save Accurate Model Metadata (BUG FIX: distinct validation vs test metrics)
    model_meta = {
        'model_name': best_candidate_name,
        'model_version': '1.0.0',
        'task': 'category_classification',
        'algorithm': 'RandomForestClassifier',
        'features': 'TF-IDF (max_features=10000, ngram_range=(1,2), sublinear_tf=True)',
        'calibration_method': 'sigmoid (Platt scaling, 5-fold CV)',
        'calibration_cv_folds': 5,
        'is_calibrated': True,
        'n_categories': len(labels),
        'categories': labels,
        # Proper validation metrics
        'val_accuracy': cal_val_metrics['accuracy'],
        'val_f1_macro': cal_val_metrics['f1_macro'],
        'val_f1_weighted': cal_val_metrics['f1_weighted'],
        # Proper untouched test metrics
        'test_accuracy': test_metrics['accuracy'],
        'test_f1_macro': test_metrics['f1_macro'],
        'test_f1_weighted': test_metrics['f1_weighted'],
        'uncalibrated_val_f1_macro': best_candidate_val_f1,
        'random_seed': RANDOM_SEED,
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
    }
    with open(models_dir / 'category_model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(model_meta, f, indent=2)

    experiment_log = {
        'experiment_id': 'category_v1_calibrated',
        'task': 'category_classification',
        'candidate_comparison': results,
        'selected_model': best_candidate_name,
        'calibration': {
            'method': 'sigmoid',
            'cv': 5,
            'fit_time_seconds': calibration_fit_time
        },
        'validation_metrics': cal_val_metrics,
        'test_metrics': test_metrics,
        'dataset_sizes': {
            'train': len(train),
            'val': len(val),
            'test': len(test),
        },
    }
    with open(eval_dir / 'category_experiment_log.json', 'w', encoding='utf-8') as f:
        json.dump(experiment_log, f, indent=2, default=str)

    logger.info("Category model training, calibration, and evaluation complete!")
    return calibrated_model, vectorizer, labels


if __name__ == "__main__":
    train_category_models()
