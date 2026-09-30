"""
Exploratory Data Analysis (EDA) for Customer Support Tickets.

Generates visualizations and statistics for:
- Record counts and column info
- Missing values and duplicates
- Text length distribution
- Category (queue) distribution
- Priority distribution
- Most common terms
- Class imbalance analysis

Saves plots to docs/images/
"""

import os
import sys
import logging
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_data():
    """Load the English customer support tickets."""
    data_path = PROJECT_ROOT / "data" / "raw" / "tickets_english.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}. Run scripts/download_data.py first.")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records from {data_path}")
    return df


def basic_stats(df):
    """Print basic dataset statistics."""
    print("\n" + "=" * 70)
    print("BASIC DATASET STATISTICS")
    print("=" * 70)
    print(f"Number of records: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nMissing value percentages:\n{(df.isnull().sum() / len(df) * 100).round(2)}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    # Text column analysis
    for col in ['subject', 'body']:
        if col in df.columns:
            non_null = df[col].dropna()
            lengths = non_null.str.len()
            word_counts = non_null.str.split().str.len()
            print(f"\n{col.upper()} statistics:")
            print(f"  Non-null: {len(non_null)}")
            print(f"  Char length — mean: {lengths.mean():.1f}, median: {lengths.median():.1f}, "
                  f"min: {lengths.min()}, max: {lengths.max()}")
            print(f"  Word count — mean: {word_counts.mean():.1f}, median: {word_counts.median():.1f}, "
                  f"min: {word_counts.min()}, max: {word_counts.max()}")


def analyze_categories(df, images_dir):
    """Analyze and visualize category distribution."""
    # Try 'queue' first, then 'type'
    cat_col = 'queue' if 'queue' in df.columns else 'type'
    if cat_col not in df.columns:
        logger.warning("No category column found")
        return

    print(f"\n{'=' * 70}")
    print(f"CATEGORY DISTRIBUTION (column: {cat_col})")
    print(f"{'=' * 70}")
    cat_counts = df[cat_col].value_counts()
    print(cat_counts)
    print(f"\nNumber of categories: {len(cat_counts)}")
    print(f"Most common: {cat_counts.index[0]} ({cat_counts.iloc[0]})")
    print(f"Least common: {cat_counts.index[-1]} ({cat_counts.iloc[-1]})")
    print(f"Imbalance ratio (max/min): {cat_counts.iloc[0] / cat_counts.iloc[-1]:.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(cat_counts)), cat_counts.values, color=sns.color_palette("viridis", len(cat_counts)))
    ax.set_xticks(range(len(cat_counts)))
    ax.set_xticklabels(cat_counts.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Count')
    ax.set_title(f'Category Distribution ({cat_col})')
    for bar, count in zip(bars, cat_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(count), ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(images_dir / 'category_distribution.png', dpi=150)
    plt.close()
    logger.info("Saved category_distribution.png")


def analyze_priority(df, images_dir):
    """Analyze and visualize priority distribution."""
    if 'priority' not in df.columns:
        logger.warning("No priority column found")
        return

    print(f"\n{'=' * 70}")
    print("PRIORITY DISTRIBUTION")
    print(f"{'=' * 70}")
    prio_counts = df['priority'].value_counts()
    print(prio_counts)
    print(f"\nNumber of priority levels: {len(prio_counts)}")
    print(f"Imbalance ratio (max/min): {prio_counts.iloc[0] / prio_counts.iloc[-1]:.2f}")

    # Plot
    colors = {'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c',
              'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c',
              '1': '#2ecc71', '2': '#f39c12', '3': '#e74c3c'}
    bar_colors = [colors.get(str(p), '#3498db') for p in prio_counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(prio_counts.index.astype(str), prio_counts.values, color=bar_colors)
    ax.set_ylabel('Count')
    ax.set_title('Priority Distribution')
    for bar, count in zip(bars, prio_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(count), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(images_dir / 'priority_distribution.png', dpi=150)
    plt.close()
    logger.info("Saved priority_distribution.png")


def analyze_text_length(df, images_dir):
    """Analyze and visualize text length distribution."""
    # Combine subject + body for analysis
    df_text = df.copy()
    df_text['full_text'] = df_text['subject'].fillna('') + ' ' + df_text['body'].fillna('')
    df_text['text_length'] = df_text['full_text'].str.len()
    df_text['word_count'] = df_text['full_text'].str.split().str.len()

    print(f"\n{'=' * 70}")
    print("TEXT LENGTH ANALYSIS (subject + body combined)")
    print(f"{'=' * 70}")
    print(f"Character length — mean: {df_text['text_length'].mean():.1f}, "
          f"median: {df_text['text_length'].median():.1f}, "
          f"std: {df_text['text_length'].std():.1f}")
    print(f"Word count — mean: {df_text['word_count'].mean():.1f}, "
          f"median: {df_text['word_count'].median():.1f}, "
          f"std: {df_text['word_count'].std():.1f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df_text['text_length'], bins=50, color='#3498db', edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('Character Length')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Text Length Distribution (Characters)')
    axes[0].axvline(df_text['text_length'].median(), color='red', linestyle='--', label=f"Median: {df_text['text_length'].median():.0f}")
    axes[0].legend()

    axes[1].hist(df_text['word_count'], bins=50, color='#2ecc71', edgecolor='white', alpha=0.8)
    axes[1].set_xlabel('Word Count')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Text Length Distribution (Words)')
    axes[1].axvline(df_text['word_count'].median(), color='red', linestyle='--', label=f"Median: {df_text['word_count'].median():.0f}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(images_dir / 'text_length_distribution.png', dpi=150)
    plt.close()
    logger.info("Saved text_length_distribution.png")


def analyze_top_words(df, images_dir):
    """Analyze most common words in ticket text."""
    import re

    print(f"\n{'=' * 70}")
    print("TOP WORDS ANALYSIS")
    print(f"{'=' * 70}")

    # Common English stopwords (manually defined to avoid NLTK dependency here)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                 'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
                 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                 'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
                 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
                 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
                 'because', 'but', 'and', 'or', 'if', 'while', 'about', 'up', 'it',
                 'its', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'you', 'your',
                 'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their', 'what',
                 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
                 'get', 'got', 'also', 'hi', 'hello', 'dear', 'please', 'thank', 'thanks'}

    full_text = (df['subject'].fillna('') + ' ' + df['body'].fillna('')).str.lower()
    all_words = []
    for text in full_text:
        words = re.findall(r'\b[a-z]+\b', text)
        all_words.extend([w for w in words if w not in stopwords and len(w) > 2])

    word_counts = Counter(all_words).most_common(30)
    print("Top 30 words (excluding stopwords):")
    for word, count in word_counts:
        print(f"  {word}: {count}")

    # Plot top 20
    words, counts = zip(*word_counts[:20])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(words)), counts, color=sns.color_palette("viridis", len(words)))
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    ax.set_title('Top 20 Most Common Words (excl. stopwords)')
    plt.tight_layout()
    plt.savefig(images_dir / 'top_words.png', dpi=150)
    plt.close()
    logger.info("Saved top_words.png")


def create_combined_plot(df, images_dir):
    """Create a combined class distribution plot."""
    cat_col = 'queue' if 'queue' in df.columns else 'type'

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Category distribution
    if cat_col in df.columns:
        cat_counts = df[cat_col].value_counts()
        axes[0].bar(range(len(cat_counts)), cat_counts.values,
                    color=sns.color_palette("viridis", len(cat_counts)))
        axes[0].set_xticks(range(len(cat_counts)))
        axes[0].set_xticklabels(cat_counts.index, rotation=45, ha='right', fontsize=8)
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Category Distribution ({cat_col})')

    # Priority distribution
    if 'priority' in df.columns:
        prio_counts = df['priority'].value_counts()
        colors = {'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c',
                  'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c',
                  '1': '#2ecc71', '2': '#f39c12', '3': '#e74c3c'}
        bar_colors = [colors.get(str(p), '#3498db') for p in prio_counts.index]
        axes[1].bar(prio_counts.index.astype(str), prio_counts.values, color=bar_colors)
        axes[1].set_ylabel('Count')
        axes[1].set_title('Priority Distribution')

    plt.tight_layout()
    plt.savefig(images_dir / 'class_distribution_combined.png', dpi=150)
    plt.close()
    logger.info("Saved class_distribution_combined.png")


def run_eda():
    """Run the complete EDA pipeline."""
    images_dir = PROJECT_ROOT / "docs" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    basic_stats(df)
    analyze_categories(df, images_dir)
    analyze_priority(df, images_dir)
    analyze_text_length(df, images_dir)
    analyze_top_words(df, images_dir)
    create_combined_plot(df, images_dir)

    print(f"\n{'=' * 70}")
    print("EDA COMPLETE")
    print(f"{'=' * 70}")
    print(f"Plots saved to: {images_dir}")


if __name__ == "__main__":
    run_eda()
