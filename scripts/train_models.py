"""
Master end-to-end training pipeline script.
Runs:
1. Preprocessing & stratified splits
2. Category model training & selection
3. Urgency model training & selection
4. Advanced NLP comparison
5. In-depth error analysis
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_models_master")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.preprocess import preprocess_data
from ml.training.train_category import train_category_models
from ml.training.train_urgency import train_urgency_models
from ml.evaluation.error_analysis import run_error_analysis


def main():
    logger.info("=== STEP 1: Running Preprocessing & Data Splitting ===")
    preprocess_data()

    logger.info("\n=== STEP 2: Training Category Classification Models ===")
    train_category_models()

    logger.info("\n=== STEP 3: Training Urgency Classification Models ===")
    train_urgency_models()

    logger.info("\n=== STEP 4: Running In-depth Error Analysis ===")
    run_error_analysis()

    logger.info("\n[SUCCESS] Entire ML pipeline executed successfully! All models and reports saved.")


if __name__ == "__main__":
    main()
