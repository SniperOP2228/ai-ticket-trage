"""
Advanced NLP Model using Sentence Transformer Embeddings.

Compares TF-IDF baseline vs Sentence Transformer embeddings for:
- Category classification
- Urgency classification

Uses all-MiniLM-L6-v2 (small, fast, ~80MB):
- 384-dim embeddings
- Good balance of quality vs speed
- Feasible for deployment

This script generates embeddings, trains classifiers, and compares against TF-IDF.
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RANDOM_SEED = 42
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'


def load_splits():
    data_dir = PROJECT_ROOT / "data" / "processed"
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, val, test


def generate_embeddings(texts, model):
    """Generate sentence embeddings using SentenceTransformer."""
    logger.info(f"Generating embeddings for {len(texts)} texts...")
    start = time.time()
    embeddings = model.encode(texts.tolist(), show_progress_bar=True, batch_size=64)
    elapsed = time.time() - start
    logger.info(f"Embeddings generated in {elapsed:.1f}s ({len(texts)/elapsed:.0f} texts/sec)")
    return embeddings, elapsed


def train_advanced_models():
    """Train models with sentence transformer embeddings and compare to TF-IDF."""
    from sentence_transformers import SentenceTransformer

    train, val, test = load_splits()

    X_train_text = train['cleaned_text'].fillna('').values
    X_val_text = val['cleaned_text'].fillna('').values
    X_test_text = test['cleaned_text'].fillna('').values

    # Load sentence transformer
    logger.info(f"Loading SentenceTransformer: {EMBEDDING_MODEL}")
    st_model = SentenceTransformer(EMBEDDING_MODEL)

    # Generate embeddings
    X_train_emb, train_emb_time = generate_embeddings(pd.Series(X_train_text), st_model)
    X_val_emb, _ = generate_embeddings(pd.Series(X_val_text), st_model)
    X_test_emb, _ = generate_embeddings(pd.Series(X_test_text), st_model)

    logger.info(f"Embedding shape: {X_train_emb.shape}")

    # Save embeddings for reuse
    emb_dir = PROJECT_ROOT / "data" / "processed"
    np.save(emb_dir / 'train_embeddings.npy', X_train_emb)
    np.save(emb_dir / 'val_embeddings.npy', X_val_emb)
    np.save(emb_dir / 'test_embeddings.npy', X_test_emb)

    results = []

    # =============================================
    # Category Classification with Embeddings
    # =============================================
    print(f"\n{'=' * 80}")
    print("CATEGORY CLASSIFICATION — Embeddings vs TF-IDF")
    print(f"{'=' * 80}")

    y_train_cat = train['category']
    y_val_cat = val['category']
    y_test_cat = test['category']

    # Train on embeddings
    cat_model_emb = LogisticRegression(
        max_iter=1000, random_state=RANDOM_SEED,
        class_weight='balanced', solver='lbfgs'
    )
    t0 = time.time()
    cat_model_emb.fit(X_train_emb, y_train_cat)
    cat_train_time = time.time() - t0

    t0 = time.time()
    cat_preds_emb = cat_model_emb.predict(X_val_emb)
    cat_inf_time = time.time() - t0

    cat_acc_emb = accuracy_score(y_val_cat, cat_preds_emb)
    cat_f1_emb = f1_score(y_val_cat, cat_preds_emb, average='macro', zero_division=0)
    cat_f1w_emb = f1_score(y_val_cat, cat_preds_emb, average='weighted', zero_division=0)

    print(f"\nEmbedding (LR) — Val Accuracy: {cat_acc_emb:.4f}, Macro F1: {cat_f1_emb:.4f}, Weighted F1: {cat_f1w_emb:.4f}")
    print(f"  Train time: {cat_train_time:.2f}s, Inference: {cat_inf_time:.4f}s")
    print(f"  Embedding time: {train_emb_time:.1f}s")

    results.append({
        'task': 'category',
        'model': f'Sentence Transformer + LR ({EMBEDDING_MODEL})',
        'accuracy': cat_acc_emb,
        'f1_macro': cat_f1_emb,
        'f1_weighted': cat_f1w_emb,
        'train_time': cat_train_time,
        'inference_time': cat_inf_time,
        'embedding_time': train_emb_time,
    })

    # Load TF-IDF results for comparison
    eval_dir = PROJECT_ROOT / "ml" / "evaluation"
    tfidf_results_path = eval_dir / 'category_model_comparison.csv'
    if tfidf_results_path.exists():
        tfidf_df = pd.read_csv(tfidf_results_path)
        best_tfidf = tfidf_df.loc[tfidf_df['f1_macro'].idxmax()]
        print(f"\nBest TF-IDF model ({best_tfidf['model']}):")
        print(f"  Val Accuracy: {best_tfidf['accuracy']:.4f}, Macro F1: {best_tfidf['f1_macro']:.4f}")
        print(f"\nComparison: Embedding F1 vs TF-IDF F1: {cat_f1_emb:.4f} vs {best_tfidf['f1_macro']:.4f}")

    # =============================================
    # Urgency Classification with Embeddings
    # =============================================
    print(f"\n{'=' * 80}")
    print("URGENCY CLASSIFICATION — Embeddings vs TF-IDF")
    print(f"{'=' * 80}")

    y_train_urg = train['urgency']
    y_val_urg = val['urgency']

    urg_model_emb = LogisticRegression(
        max_iter=1000, random_state=RANDOM_SEED,
        class_weight='balanced', solver='lbfgs'
    )
    t0 = time.time()
    urg_model_emb.fit(X_train_emb, y_train_urg)
    urg_train_time = time.time() - t0

    urg_preds_emb = urg_model_emb.predict(X_val_emb)
    urg_acc_emb = accuracy_score(y_val_urg, urg_preds_emb)
    urg_f1_emb = f1_score(y_val_urg, urg_preds_emb, average='macro', zero_division=0)

    print(f"\nEmbedding (LR) — Val Accuracy: {urg_acc_emb:.4f}, Macro F1: {urg_f1_emb:.4f}")

    results.append({
        'task': 'urgency',
        'model': f'Sentence Transformer + LR ({EMBEDDING_MODEL})',
        'accuracy': urg_acc_emb,
        'f1_macro': urg_f1_emb,
        'train_time': urg_train_time,
    })

    # =============================================
    # Test Set Evaluation
    # =============================================
    print(f"\n{'=' * 80}")
    print("TEST SET EVALUATION — Embedding Models")
    print(f"{'=' * 80}")

    cat_test_preds = cat_model_emb.predict(X_test_emb)
    cat_test_acc = accuracy_score(y_test_cat, cat_test_preds)
    cat_test_f1 = f1_score(y_test_cat, cat_test_preds, average='macro', zero_division=0)
    print(f"\nCategory — Test Accuracy: {cat_test_acc:.4f}, Macro F1: {cat_test_f1:.4f}")
    print(classification_report(y_test_cat, cat_test_preds))

    y_test_urg = test['urgency']
    urg_test_preds = urg_model_emb.predict(X_test_emb)
    urg_test_acc = accuracy_score(y_test_urg, urg_test_preds)
    urg_test_f1 = f1_score(y_test_urg, urg_test_preds, average='macro', zero_division=0)
    print(f"\nUrgency — Test Accuracy: {urg_test_acc:.4f}, Macro F1: {urg_test_f1:.4f}")
    print(classification_report(y_test_urg, urg_test_preds))

    # =============================================
    # Save
    # =============================================
    models_dir = PROJECT_ROOT / "ml" / "models"
    joblib.dump(cat_model_emb, models_dir / 'category_model_embedding.joblib')
    joblib.dump(urg_model_emb, models_dir / 'urgency_model_embedding.joblib')

    # Save comparison
    with open(eval_dir / 'advanced_nlp_results.json', 'w') as f:
        json.dump({
            'embedding_model': EMBEDDING_MODEL,
            'embedding_dim': int(X_train_emb.shape[1]),
            'results': results,
            'category_test_accuracy': cat_test_acc,
            'category_test_f1_macro': cat_test_f1,
            'urgency_test_accuracy': urg_test_acc,
            'urgency_test_f1_macro': urg_test_f1,
            'comparison_notes': (
                'Sentence Transformer embeddings capture semantic meaning better than TF-IDF bag-of-words. '
                'However, they require more memory (~80MB model) and slower inference (embedding generation). '
                'TF-IDF is faster at inference time and requires no GPU. '
                'For deployment, TF-IDF is recommended unless accuracy improvement is significant.'
            ),
        }, f, indent=2, default=str)

    # Comparison chart
    images_dir = PROJECT_ROOT / "docs" / "images"
    if tfidf_results_path.exists():
        fig, ax = plt.subplots(figsize=(10, 5))
        comparison = pd.DataFrame({
            'Model': [f"TF-IDF + {best_tfidf['model']}", f'Embeddings + LR'],
            'Macro F1': [best_tfidf['f1_macro'], cat_f1_emb],
            'Accuracy': [best_tfidf['accuracy'], cat_acc_emb],
        })
        x = np.arange(len(comparison))
        width = 0.35
        ax.bar(x - width/2, comparison['Macro F1'], width, label='Macro F1', color='#3498db')
        ax.bar(x + width/2, comparison['Accuracy'], width, label='Accuracy', color='#2ecc71')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison['Model'])
        ax.set_ylabel('Score')
        ax.set_title('TF-IDF vs Sentence Embeddings — Category Classification')
        ax.legend()
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(images_dir / 'tfidf_vs_embeddings.png', dpi=150)
        plt.close()

    print(f"\n{'=' * 80}")
    print("ADVANCED NLP TRAINING COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    train_advanced_models()
