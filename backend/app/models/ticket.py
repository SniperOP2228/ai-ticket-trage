"""SQLAlchemy models for the ticket triage database."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, Boolean, Enum
from app.database import Base


class Ticket(Base):
    """Stores customer support tickets."""
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    text = Column(Text, nullable=False)
    cleaned_text = Column(Text, nullable=True)
    status = Column(String(50), default="pending")  # pending, routed, review_required, reviewed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Prediction(Base):
    """Stores ML predictions for each ticket."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ticket_id = Column(Integer, nullable=False, index=True)
    category = Column(String(100), nullable=False)
    category_confidence = Column(Float, nullable=False)
    urgency = Column(String(50), nullable=False)
    urgency_confidence = Column(Float, nullable=False)
    queue = Column(String(100), nullable=False)
    model_version = Column(String(50), default="1.0.0")
    requires_human_review = Column(Boolean, default=False)
    important_features = Column(Text, nullable=True)  # JSON string of important features
    created_at = Column(DateTime, default=datetime.utcnow)


class HumanReview(Base):
    """Stores human corrections for low-confidence predictions."""
    __tablename__ = "human_reviews"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ticket_id = Column(Integer, nullable=False, index=True)
    original_category = Column(String(100), nullable=False)
    corrected_category = Column(String(100), nullable=False)
    original_urgency = Column(String(50), nullable=False)
    corrected_urgency = Column(String(50), nullable=False)
    original_confidence = Column(Float, nullable=True)
    reviewer = Column(String(100), default="reviewer")
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    """Tracks model versions deployed."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    algorithm = Column(String(100), nullable=True)
    features = Column(String(200), nullable=True)
    accuracy = Column(Float, nullable=True)
    f1_macro = Column(Float, nullable=True)
    f1_weighted = Column(Float, nullable=True)
    training_date = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
