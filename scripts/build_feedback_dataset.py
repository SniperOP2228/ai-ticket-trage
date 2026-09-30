"""
Extract and Validate Human Review Feedback Dataset.

Reads human corrections from the database, validates labels against
the recognized category and urgency taxonomy, and writes clean feedback
samples to data/processed/human_feedback.csv for retraining.
"""

import sys
import logging
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import settings
from app.models.ticket import Ticket, HumanReview, Prediction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VALID_CATEGORIES = {
    "Technical Support",
    "Product Support",
    "Customer Service",
    "IT Support",
    "Billing and Payments",
    "Returns and Exchanges",
    "Service Outages and Maintenance",
    "Sales and Pre-Sales",
    "Human Resources",
    "General Inquiry",
}

VALID_URGENCIES = {"High", "Medium", "Low"}


def build_feedback_dataset():
    """Extract, validate, and persist reviewed tickets for retraining."""
    try:
        from app.database import SessionLocal
        db = SessionLocal()
        reviews = db.query(HumanReview).all()
    except Exception as e:
        logger.warning(f"Primary database connection unavailable ({e}). Using local fallback SQLite database...")
        # Check both root and backend SQLite locations
        sqlite_paths = ["sqlite:///./ticket_triage.db", "sqlite:///backend/ticket_triage.db"]
        reviews = []
        db = None
        for p in sqlite_paths:
            try:
                eng = create_engine(p, connect_args={"check_same_thread": False})
                Sess = sessionmaker(bind=eng)
                test_db = Sess()
                reviews = test_db.query(HumanReview).all()
                db = test_db
                break
            except Exception:
                continue
        if db is None:
            reviews = []
    logger.info(f"Found {len(reviews)} raw human review records in database.")

    if not reviews:
        logger.info("No human review records found. Creating empty feedback schema...")
        out_dir = PROJECT_ROOT / "data" / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        empty_df = pd.DataFrame(columns=[
            'ticket_id', 'cleaned_text', 'category', 'urgency',
            'original_category', 'original_urgency', 'reviewer', 'notes', 'created_at'
        ])
        empty_df.to_csv(out_dir / 'human_feedback.csv', index=False)
        if db:
            db.close()
        return empty_df

    validated_records = []
    skipped_count = 0

    for r in reviews:
        ticket = db.query(Ticket).filter(Ticket.id == r.ticket_id).first()
        if not ticket or not ticket.cleaned_text:
            skipped_count += 1
            continue

        corrected_cat = r.corrected_category.strip()
        corrected_urg = r.corrected_urgency.strip()

        # Validate against recognized taxonomy
        if corrected_cat not in VALID_CATEGORIES or corrected_urg not in VALID_URGENCIES:
            logger.warning(f"Skipping record {r.id}: invalid category '{corrected_cat}' or urgency '{corrected_urg}'")
            skipped_count += 1
            continue

        validated_records.append({
            'ticket_id': r.ticket_id,
            'cleaned_text': ticket.cleaned_text,
            'category': corrected_cat,
            'urgency': corrected_urg,
            'original_category': r.original_category,
            'original_urgency': r.original_urgency,
            'reviewer': r.reviewer,
            'notes': r.notes or '',
            'created_at': r.created_at.isoformat() if r.created_at else ''
        })

        if db:
            db.close()

    feedback_df = pd.DataFrame(validated_records)
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'human_feedback.csv'
    feedback_df.to_csv(out_path, index=False)

    logger.info(f"Successfully validated {len(feedback_df)} human review feedback samples ({skipped_count} skipped).")
    logger.info(f"Saved feedback dataset to {out_path}")
    return feedback_df


if __name__ == "__main__":
    build_feedback_dataset()
