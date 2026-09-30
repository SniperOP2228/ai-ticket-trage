"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ============================================================
# Prediction Schemas
# ============================================================

class PredictRequest(BaseModel):
    """Request body for ticket prediction."""
    text: str = Field(..., min_length=3, max_length=5000, description="Customer support ticket text")

    model_config = {"json_schema_extra": {
        "examples": [{"text": "My payment was deducted but my order was cancelled. Please resolve this immediately."}]
    }}


class PredictionResponse(BaseModel):
    """Response for ticket prediction."""
    ticket_id: int
    category: str
    category_confidence: float
    urgency: str
    urgency_confidence: float
    queue: str
    requires_human_review: bool
    important_features: List[str]
    model_version: str
    status: str  # "auto_routed" or "human_review_required"


# ============================================================
# Ticket Schemas
# ============================================================

class TicketResponse(BaseModel):
    """Response for a single ticket."""
    id: int
    text: str
    status: str
    category: Optional[str] = None
    category_confidence: Optional[float] = None
    urgency: Optional[str] = None
    urgency_confidence: Optional[float] = None
    queue: Optional[str] = None
    requires_human_review: Optional[bool] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class TicketListResponse(BaseModel):
    """Response for ticket list."""
    tickets: List[TicketResponse]
    total: int
    page: int
    page_size: int


# ============================================================
# Review Schemas
# ============================================================

class ReviewRequest(BaseModel):
    """Request body for submitting a human review or manual override."""
    corrected_category: str
    corrected_urgency: str
    reviewer: str = "reviewer"
    notes: Optional[str] = None
    is_override: Optional[bool] = False


class ReviewResponse(BaseModel):
    """Response for a submitted review or manual override."""
    id: int
    ticket_id: int
    original_category: str
    corrected_category: str
    original_urgency: str
    corrected_urgency: str
    reviewer: str
    is_override: bool = False
    notes: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ReviewTicketResponse(BaseModel):
    """A ticket that needs human review, with its prediction."""
    ticket_id: int
    text: str
    predicted_category: str
    category_confidence: float
    predicted_urgency: str
    urgency_confidence: float
    created_at: datetime


# ============================================================
# Analytics Schemas
# ============================================================

class AnalyticsResponse(BaseModel):
    """Response for analytics endpoint."""
    total_tickets: int
    auto_routed: int
    human_review_required: int
    reviewed: int
    avg_category_confidence: float
    avg_urgency_confidence: float
    high_urgency_count: int
    category_distribution: dict
    urgency_distribution: dict
    human_correction_rate: Optional[float] = None
    confidence_distribution: dict
    recent_predictions: int


# ============================================================
# Model Info Schemas
# ============================================================

class ModelInfoResponse(BaseModel):
    """Response for model information."""
    category_model: dict
    urgency_model: dict
    confidence_threshold: float
    model_version: str
