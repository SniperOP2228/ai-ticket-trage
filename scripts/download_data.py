"""
Download the Customer Support Tickets dataset from HuggingFace.

Dataset: Tobi-Bueck/customer-support-tickets
License: CC-BY-NC-4.0
Source: https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets

This script:
1. Downloads the dataset using the HuggingFace datasets library
2. Filters to English-only tickets
3. Saves raw data to data/raw/
4. Prints basic dataset statistics
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_dataset():
    """Download and save the customer support tickets dataset."""
    from datasets import load_dataset
    import pandas as pd

    output_dir = PROJECT_ROOT / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Tobi-Bueck/customer-support-tickets from HuggingFace...")
    dataset = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")

    logger.info(f"Downloaded {len(dataset)} records")
    logger.info(f"Columns: {dataset.column_names}")

    # Convert to DataFrame
    df = dataset.to_pandas()

    # Save full raw dataset
    full_path = output_dir / "tickets_full.csv"
    df.to_csv(full_path, index=False)
    logger.info(f"Saved full dataset to {full_path}")

    # Print language distribution
    if 'language' in df.columns:
        logger.info(f"\nLanguage distribution:\n{df['language'].value_counts()}")

        # Filter to English only
        df_en = df[df['language'].str.lower() == 'en'].copy()
        logger.info(f"\nFiltered to English: {len(df_en)} records (from {len(df)} total)")
    else:
        logger.warning("No 'language' column found. Using all records.")
        df_en = df.copy()

    # Save English-only dataset
    en_path = output_dir / "tickets_english.csv"
    df_en.to_csv(en_path, index=False)
    logger.info(f"Saved English dataset to {en_path}")

    # Print basic statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total records (all languages): {len(df)}")
    print(f"English records: {len(df_en)}")
    print(f"Columns: {list(df_en.columns)}")
    print(f"\nColumn dtypes:\n{df_en.dtypes}")
    print(f"\nMissing values:\n{df_en.isnull().sum()}")
    print(f"\nDuplicate rows: {df_en.duplicated().sum()}")

    if 'queue' in df_en.columns:
        print(f"\nQueue (Category) distribution:\n{df_en['queue'].value_counts()}")
    if 'type' in df_en.columns:
        print(f"\nType distribution:\n{df_en['type'].value_counts()}")
    if 'priority' in df_en.columns:
        print(f"\nPriority distribution:\n{df_en['priority'].value_counts()}")

    print("=" * 60)
    return df_en


if __name__ == "__main__":
    download_dataset()
