"""
Train and Compare Multiple Models for Category Classification.

Models compared:
1. Logistic Regression (baseline)
2. Multinomial Naive Bayes
3. Linear SVM (LinearSVC)
4. Random Forest

For each model, records:
- Accuracy, Precision, Recall, F1, Macro F1, Weighted F1
- Training time, Inference time

Uses TF-IDF features fit ONLY on training data.
Evaluates on validation set for model selection.
Final evaluation on test set only for the selected model.

Saves:
- Best model and vectorizer to ml/models/
- Comparison results to ml/evaluation/
- Confusion matrices to docs/images/
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

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
    - max_features=10000: balances vocabulary coverage with efficiency
    - ngram_range=(1,2): captures bigrams like "payment failed", "account blocked"
    - sublinear_tf=True: applies log normalization, better for text classification
    - min_df=2: removes very rare terms (noise)
    - max_df=0.95: removes overly common terms
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
    """Evaluate model and return metrics dict."""
    start = time.time()
    y_pred = model.predict(X)
    inference_time = time.time() - start

    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'inference_time_seconds': inference_time,
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
    """Train and compare all category classification models."""
    train, val, test = load_splits()

    X_train_text = train['cleaned_text'].fillna('')
    y_train = train['category']
    X_val_text = val['cleaned_text'].fillna('')
    y_val = val['category']
    X_test_text = test['cleaned_text'].fillna('')
    y_test = test['category']

    # Build TF-IDF on training data ONLY
    logger.info("Building TF-IDF vectorizer (fit on training data only)...")
    vectorizer = build_tfidf(X_train_text)
    X_train = vectorizer.transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    X_test = vectorizer.transform(X_test_text)

    # Get class labels
    labels = sorted(y_train.unique())
    logger.info(f"Category labels: {labels}")

    # Define models
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

    # Train and evaluate each model
    results = []
    best_model = None
    best_score = 0
    best_name = ""
    best_predictions = None

    print(f"\n{'=' * 80}")
    print("CATEGORY MODEL COMPARISON (evaluated on validation set)")
    print(f"{'=' * 80}")

    for name, model in models.items():
        logger.info(f"\nTraining {name}...")
        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start

        metrics, y_pred = evaluate_model(model, X_val, y_val, name)
        metrics['training_time_seconds'] = train_time
        results.append(metrics)

        print(f"\n--- {name} ---")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Macro F1:    {metrics['f1_macro']:.4f}")
        print(f"  Weighted F1: {metrics['f1_weighted']:.4f}")
        print(f"  Train time:  {train_time:.2f}s")
        print(f"  Inference:   {metrics['inference_time_seconds']:.4f}s")

        # Track best by macro F1 (better for imbalanced classes)
        if metrics['f1_macro'] > best_score:
            best_score = metrics['f1_macro']
            best_model = model
            best_name = name
            best_predictions = y_pred

    # Save comparison results
    eval_dir = PROJECT_ROOT / "ml" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(eval_dir / 'category_model_comparison.csv', index=False)

    print(f"\n{'=' * 80}")
    print("MODEL COMPARISON TABLE")
    print(f"{'=' * 80}")
    print(results_df[['model', 'accuracy', 'f1_macro', 'f1_weighted',
                       'training_time_seconds', 'inference_time_seconds']].to_string(index=False))

    print(f"\n[BEST MODEL] Best category model: {best_name} (Macro F1: {best_score:.4f})")

    # =============================================
    # Final evaluation on TEST set (ONLY for best model)
    # =============================================
    print(f"\n{'=' * 80}")
    print(f"FINAL TEST SET EVALUATION — {best_name}")
    print(f"{'=' * 80}")

    test_metrics, test_preds = evaluate_model(best_model, X_test, y_test, best_name)
    print(f"\nTest Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"Test Macro F1:    {test_metrics['f1_macro']:.4f}")
    print(f"Test Weighted F1: {test_metrics['f1_weighted']:.4f}")

    print(f"\nClassification Report:\n{classification_report(y_test, test_preds, target_names=labels)}")

    # Save confusion matrices
    images_dir = PROJECT_ROOT / "docs" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(y_val, best_predictions, labels,
                         f'Category Confusion Matrix (Val) — {best_name}',
                         images_dir / 'category_confusion_matrix_val.png')

    plot_confusion_matrix(y_test, test_preds, labels,
                         f'Category Confusion Matrix (Test) — {best_name}',
                         images_dir / 'category_confusion_matrix_test.png')

    # Plot model comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    model_names = results_df['model']

    axes[0].barh(model_names, results_df['f1_macro'], color=sns.color_palette("viridis", len(model_names)))
    axes[0].set_xlabel('Macro F1')
    axes[0].set_title('Category Models — Macro F1 Comparison')

    axes[1].barh(model_names, results_df['accuracy'], color=sns.color_palette("mako", len(model_names)))
    axes[1].set_xlabel('Accuracy')
    axes[1].set_title('Category Models — Accuracy Comparison')

    plt.tight_layout()
    plt.savefig(images_dir / 'category_model_comparison.png', dpi=150)
    plt.close()

    # =============================================
    # Save best model and vectorizer
    # =============================================
    models_dir = PROJECT_ROOT / "ml" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, models_dir / 'category_model.joblib')
    joblib.dump(vectorizer, models_dir / 'tfidf_vectorizer.joblib')
    logger.info(f"Saved best category model ({best_name}) and vectorizer")

    # Save model metadata
    model_meta = {
        'model_name': best_name,
        'model_version': '1.0.0',
        'task': 'category_classification',
        'features': 'TF-IDF (max_features=10000, ngram_range=(1,2))',
        'n_categories': len(labels),
        'categories': labels,
        'val_accuracy': test_metrics['accuracy'],
        'val_f1_macro': test_metrics['f1_macro'],
        'val_f1_weighted': test_metrics['f1_weighted'],
        'test_accuracy': test_metrics['accuracy'],
        'test_f1_macro': test_metrics['f1_macro'],
        'test_f1_weighted': test_metrics['f1_weighted'],
        'random_seed': RANDOM_SEED,
    }

    with open(models_dir / 'category_model_metadata.json', 'w') as f:
        json.dump(model_meta, f, indent=2)

    # Save full experiment log
    experiment_log = {
        'experiment_id': 'category_v1',
        'task': 'category_classification',
        'models': results,
        'best_model': best_name,
        'best_f1_macro': best_score,
        'test_metrics': test_metrics,
        'tfidf_params': {
            'max_features': 10000,
            'ngram_range': [1, 2],
            'sublinear_tf': True,
            'min_df': 2,
            'max_df': 0.95,
        },
        'data_split': {
            'train': len(train),
            'val': len(val),
            'test': len(test),
        },
    }

    with open(eval_dir / 'category_experiment_log.json', 'w') as f:
        json.dump(experiment_log, f, indent=2, default=str)

    return best_model, vectorizer, labels


if __name__ == "__main__":
    train_category_models()
