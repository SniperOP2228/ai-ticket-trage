"""
Train and Compare Models for Urgency/Priority Classification.

Separate pipeline from category classification.
Uses the same TF-IDF approach but trained independently.

Target: urgency (Low, Medium, High)

Models compared:
1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear SVM
4. Random Forest
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
    return train, val, test


def evaluate_model(model, X, y_true, model_name=""):
    """Evaluate model and return metrics."""
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
        'inference_time_seconds': inference_time,
    }
    return metrics, y_pred


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_urgency_models():
    """Train and compare urgency classification models."""
    train, val, test = load_splits()

    X_train_text = train['cleaned_text'].fillna('')
    y_train = train['urgency']
    X_val_text = val['cleaned_text'].fillna('')
    y_val = val['urgency']
    X_test_text = test['cleaned_text'].fillna('')
    y_test = test['urgency']

    # Build separate TF-IDF for urgency
    # Urgency may benefit from different features than category
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        strip_accents='unicode',
        token_pattern=r'\b[a-z][a-z]+\b'
    )
    vectorizer.fit(X_train_text)
    X_train = vectorizer.transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    X_test = vectorizer.transform(X_test_text)

    labels = ['Low', 'Medium', 'High']
    # Filter to only labels present in data
    present_labels = sorted(y_train.unique())
    logger.info(f"Urgency labels in data: {present_labels}")

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED,
            class_weight='balanced', solver='lbfgs'
        ),
        'Multinomial NB': MultinomialNB(alpha=0.1),
        'Linear SVM': CalibratedClassifierCV(
            LinearSVC(max_iter=2000, random_state=RANDOM_SEED, class_weight='balanced'),
            cv=3
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_SEED,
            class_weight='balanced', n_jobs=-1
        ),
    }

    results = []
    best_model = None
    best_score = 0
    best_name = ""

    print(f"\n{'=' * 80}")
    print("URGENCY MODEL COMPARISON (evaluated on validation set)")
    print(f"{'=' * 80}")

    for name, model in models.items():
        logger.info(f"Training {name} for urgency...")
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

        if metrics['f1_macro'] > best_score:
            best_score = metrics['f1_macro']
            best_model = model
            best_name = name

    # Test set evaluation for best model
    print(f"\n[BEST MODEL] Best urgency model: {best_name} (Macro F1: {best_score:.4f})")

    test_metrics, test_preds = evaluate_model(best_model, X_test, y_test, best_name)
    print(f"\n{'=' * 80}")
    print(f"FINAL TEST SET — {best_name}")
    print(f"{'=' * 80}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Macro F1: {test_metrics['f1_macro']:.4f}")
    print(f"\n{classification_report(y_test, test_preds, target_names=present_labels)}")

    # Save results
    eval_dir = PROJECT_ROOT / "ml" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(eval_dir / 'urgency_model_comparison.csv', index=False)

    images_dir = PROJECT_ROOT / "docs" / "images"
    plot_confusion_matrix(y_test, test_preds, present_labels,
                         f'Urgency Confusion Matrix (Test) — {best_name}',
                         images_dir / 'urgency_confusion_matrix_test.png')

    # Save model
    models_dir = PROJECT_ROOT / "ml" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, models_dir / 'urgency_model.joblib')
    joblib.dump(vectorizer, models_dir / 'urgency_tfidf_vectorizer.joblib')

    model_meta = {
        'model_name': best_name,
        'model_version': '1.0.0',
        'task': 'urgency_classification',
        'features': 'TF-IDF (max_features=8000, ngram_range=(1,2))',
        'labels': present_labels,
        'test_accuracy': test_metrics['accuracy'],
        'test_f1_macro': test_metrics['f1_macro'],
        'test_f1_weighted': test_metrics['f1_weighted'],
    }
    with open(models_dir / 'urgency_model_metadata.json', 'w') as f:
        json.dump(model_meta, f, indent=2)

    experiment_log = {
        'experiment_id': 'urgency_v1',
        'task': 'urgency_classification',
        'models': results,
        'best_model': best_name,
        'best_f1_macro': best_score,
        'test_metrics': test_metrics,
    }
    with open(eval_dir / 'urgency_experiment_log.json', 'w') as f:
        json.dump(experiment_log, f, indent=2, default=str)

    logger.info("Urgency model training complete!")
    return best_model, vectorizer


if __name__ == "__main__":
    train_urgency_models()
