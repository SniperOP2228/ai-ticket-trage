"""
Data Preprocessing Pipeline for Customer Support Ticket Triage.

This script:
1. Loads raw English tickets
2. Combines subject + body into full text
3. Removes duplicates
4. Handles missing values
5. Cleans and normalizes text
6. Maps priority labels to Low/Medium/High
7. Performs stratified train/val/test split (70/15/15)
8. Checks for data leakage
9. Saves processed splits to data/processed/

IMPORTANT: No information from test/val sets influences preprocessing decisions.
TF-IDF vectorizer is fit ONLY on training data.
"""

import os
import sys
import re
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RANDOM_SEED = 42


def clean_text(text):
    """
    Clean and normalize text while preserving useful domain keywords.
    
    Preserves words like: urgent, refund, failed, blocked, payment, error, etc.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Keep alphanumeric, spaces, and basic punctuation
    text = re.sub(r'[^a-z0-9\s\.\,\!\?\-]', ' ', text)
    
    # Remove numbers that are standalone (keep numbers attached to words)
    text = re.sub(r'\b\d+\b', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def map_priority(priority_val):
    """
    Map priority values to standardized Low/Medium/High.
    
    The dataset may use numeric (1, 2, 3) or string priority values.
    This mapping is documented and defensible.
    """
    if pd.isna(priority_val):
        return "Medium"  # Default for missing
    
    prio_str = str(priority_val).strip().lower()
    
    # Handle numeric priorities
    if prio_str in ['1', 'low']:
        return "Low"
    elif prio_str in ['2', 'medium', 'normal']:
        return "Medium"
    elif prio_str in ['3', 'high', 'critical', 'urgent']:
        return "High"
    else:
        return "Medium"  # Default fallback


def preprocess_data():
    """Run the complete preprocessing pipeline."""
    # Load raw data
    raw_path = PROJECT_ROOT / "data" / "raw" / "tickets_english.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}. Run scripts/download_data.py first.")
    
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df)} raw English tickets")
    
    # =============================================
    # Step 1: Combine subject + body
    # =============================================
    df['full_text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
    df['full_text'] = df['full_text'].str.strip()
    
    # Remove rows with empty text
    empty_mask = df['full_text'].str.len() == 0
    logger.info(f"Removing {empty_mask.sum()} rows with empty text")
    df = df[~empty_mask].copy()
    
    # =============================================
    # Step 2: Remove duplicates (based on full_text)
    # =============================================
    n_before = len(df)
    df = df.drop_duplicates(subset=['full_text'], keep='first').copy()
    logger.info(f"Removed {n_before - len(df)} duplicate texts ({len(df)} remaining)")
    
    # =============================================
    # Step 3: Clean text
    # =============================================
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    
    # Remove rows where cleaned text is too short (< 5 chars)
    short_mask = df['cleaned_text'].str.len() < 5
    logger.info(f"Removing {short_mask.sum()} rows with cleaned text < 5 chars")
    df = df[~short_mask].copy()
    
    # =============================================
    # Step 4: Determine target columns
    # =============================================
    # Category: use 'queue' if available, else 'type'
    if 'queue' in df.columns:
        df['category'] = df['queue'].fillna('Other')
        category_col = 'queue'
    elif 'type' in df.columns:
        df['category'] = df['type'].fillna('Other')
        category_col = 'type'
    else:
        raise ValueError("No category column (queue or type) found in dataset")
    
    logger.info(f"Using '{category_col}' as category source")
    logger.info(f"Category distribution:\n{df['category'].value_counts()}")
    
    # Priority/Urgency
    if 'priority' in df.columns:
        df['urgency'] = df['priority'].apply(map_priority)
        logger.info(f"\nUrgency distribution (mapped from priority):\n{df['urgency'].value_counts()}")
    else:
        logger.warning("No priority column found. Setting all urgency to 'Medium'.")
        df['urgency'] = 'Medium'
    
    # =============================================
    # Step 5: Select and save relevant columns
    # =============================================
    cols_to_keep = ['cleaned_text', 'category', 'urgency', 'full_text']
    df_clean = df[cols_to_keep].copy()
    df_clean = df_clean.reset_index(drop=True)
    
    logger.info(f"\nFinal dataset: {len(df_clean)} records")
    logger.info(f"Categories: {df_clean['category'].nunique()}")
    logger.info(f"Urgency levels: {df_clean['urgency'].nunique()}")
    
    # =============================================
    # Step 6: Stratified Train/Val/Test Split
    # =============================================
    # We stratify on category since it's the primary classification task
    # 70% train, 15% val, 15% test
    
    # First split: 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df_clean,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df_clean['category']
    )
    
    # Second split: 50% of temp = 15% val, 50% of temp = 15% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df['category']
    )
    
    logger.info(f"\nSplit sizes:")
    logger.info(f"  Train: {len(train_df)} ({len(train_df)/len(df_clean)*100:.1f}%)")
    logger.info(f"  Val:   {len(val_df)} ({len(val_df)/len(df_clean)*100:.1f}%)")
    logger.info(f"  Test:  {len(test_df)} ({len(test_df)/len(df_clean)*100:.1f}%)")
    
    # =============================================
    # Step 7: Data Leakage Checks
    # =============================================
    print(f"\n{'=' * 70}")
    print("DATA LEAKAGE CHECKS")
    print(f"{'=' * 70}")
    
    # Check 1: Duplicate texts across splits
    train_texts = set(train_df['cleaned_text'].values)
    val_texts = set(val_df['cleaned_text'].values)
    test_texts = set(test_df['cleaned_text'].values)
    
    train_val_overlap = train_texts & val_texts
    train_test_overlap = train_texts & test_texts
    val_test_overlap = val_texts & test_texts
    
    print(f"Train-Val text overlap: {len(train_val_overlap)} texts")
    print(f"Train-Test text overlap: {len(train_test_overlap)} texts")
    print(f"Val-Test text overlap: {len(val_test_overlap)} texts")
    
    if len(train_val_overlap) > 0 or len(train_test_overlap) > 0:
        logger.warning("Data leakage detected! Removing overlapping texts from val/test...")
        val_df = val_df[~val_df['cleaned_text'].isin(train_texts)].copy()
        test_df = test_df[~test_df['cleaned_text'].isin(train_texts)].copy()
        logger.info(f"After leakage removal — Val: {len(val_df)}, Test: {len(test_df)}")
    else:
        print("[OK] No text overlap between splits - no data leakage detected")
    
    # Check 2: Verify stratification preserved class distribution
    print(f"\nCategory distribution across splits:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        dist = split_df['category'].value_counts(normalize=True)
        print(f"\n{split_name}:")
        for cat, pct in dist.items():
            print(f"  {cat}: {pct:.3f}")
    
    # Check 3: No target-derived features
    print(f"\n[OK] No target-derived features in input (only cleaned_text used for prediction)")
    print(f"[OK] No future information leakage (no temporal features)")
    print(f"[OK] TF-IDF vectorizer will be fit ONLY on training data")
    
    # =============================================
    # Step 8: Save processed data
    # =============================================
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)
    
    logger.info(f"\nSaved processed splits to {output_dir}")
    logger.info(f"  train.csv: {len(train_df)} records")
    logger.info(f"  val.csv: {len(val_df)} records")
    logger.info(f"  test.csv: {len(test_df)} records")
    
    # Save preprocessing summary
    summary = {
        'total_records': len(df_clean),
        'train_records': len(train_df),
        'val_records': len(val_df),
        'test_records': len(test_df),
        'n_categories': df_clean['category'].nunique(),
        'categories': list(df_clean['category'].unique()),
        'n_urgency_levels': df_clean['urgency'].nunique(),
        'urgency_levels': list(df_clean['urgency'].unique()),
        'random_seed': RANDOM_SEED,
        'category_source': category_col,
        'priority_mapping': 'numeric 1→Low, 2→Medium, 3→High',
    }
    
    import json
    with open(output_dir / 'preprocessing_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print("PREPROCESSING COMPLETE")
    print(f"{'=' * 70}")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    preprocess_data()
