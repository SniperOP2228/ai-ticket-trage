"""Analytics API endpoints for dashboard visualization."""

import json
from collections import Counter
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.models.ticket import Ticket, Prediction, HumanReview
from app.schemas.ticket import AnalyticsResponse, ModelInfoResponse
from app.ml.predictor import predictor
from app.config import settings

router = APIRouter(tags=["Analytics & Model Info"])


@router.get("/analytics", response_model=AnalyticsResponse)
def get_analytics(db: Session = Depends(get_db)):
    """Computes aggregated metrics for the dashboard."""
    total_tickets = db.query(Ticket).count()
    auto_routed = db.query(Ticket).filter(Ticket.status == "routed").count()
    human_review_required = db.query(Ticket).filter(Ticket.status == "review_required").count()
    reviewed = db.query(Ticket).filter(Ticket.status == "reviewed").count()

    # Category distribution
    cat_rows = db.query(Prediction.category, func.count(Prediction.id)).group_by(Prediction.category).all()
    category_distribution = {cat: count for cat, count in cat_rows}

    # Urgency distribution
    urg_rows = db.query(Prediction.urgency, func.count(Prediction.id)).group_by(Prediction.urgency).all()
    urgency_distribution = {urg: count for urg, count in urg_rows}

    # High urgency count
    high_urgency_count = urgency_distribution.get("High", 0) + urgency_distribution.get("high", 0)

    # Average confidences
    avg_cat_conf = db.query(func.avg(Prediction.category_confidence)).scalar() or 0.0
    avg_urg_conf = db.query(func.avg(Prediction.urgency_confidence)).scalar() or 0.0

    # Human correction rate
    total_reviews = db.query(HumanReview).count()
    human_correction_rate = None
    if total_reviews > 0:
        corrections = db.query(HumanReview).filter(
            (HumanReview.original_category != HumanReview.corrected_category) |
            (HumanReview.original_urgency != HumanReview.corrected_urgency)
        ).count()
        human_correction_rate = round(corrections / total_reviews, 4)

    # Confidence distribution bins
    preds = db.query(Prediction.category_confidence).all()
    conf_bins = {"0.5-0.6": 0, "0.6-0.7": 0, "0.7-0.8": 0, "0.8-0.9": 0, "0.9-1.0": 0}
    for (c,) in preds:
        if c < 0.6:
            conf_bins["0.5-0.6"] += 1
        elif c < 0.7:
            conf_bins["0.6-0.7"] += 1
        elif c < 0.8:
            conf_bins["0.7-0.8"] += 1
        elif c < 0.9:
            conf_bins["0.8-0.9"] += 1
        else:
            conf_bins["0.9-1.0"] += 1

    return AnalyticsResponse(
        total_tickets=total_tickets,
        auto_routed=auto_routed,
        human_review_required=human_review_required,
        reviewed=reviewed,
        avg_category_confidence=round(float(avg_cat_conf), 4),
        avg_urgency_confidence=round(float(avg_urg_conf), 4),
        high_urgency_count=high_urgency_count,
        category_distribution=category_distribution,
        urgency_distribution=urgency_distribution,
        human_correction_rate=human_correction_rate,
        confidence_distribution=conf_bins,
        recent_predictions=min(total_tickets, 100)
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    """Returns active model metadata, parameters, and configured thresholds."""
    info = predictor.get_model_info()
    return ModelInfoResponse(
        category_model=info.get("category_model", {}),
        urgency_model=info.get("urgency_model", {}),
        confidence_threshold=settings.confidence_threshold,
        model_version=settings.model_version
    )
