"""Prediction endpoints for customer support tickets."""

import json
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.ticket import Ticket, Prediction
from app.schemas.ticket import PredictRequest, PredictionResponse
from app.ml.predictor import predictor
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
def predict_ticket(
    payload: PredictRequest,
    db: Session = Depends(get_db)
):
    """
    Accepts customer support ticket text, classifies category and urgency,
    generates confidence scores and explainability features, routes the ticket,
    and persists ticket and prediction to the database.
    """
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Ticket text cannot be empty or blank")

    try:
        # Run prediction via ML predictor service
        res = predictor.predict(text, confidence_threshold=settings.confidence_threshold)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}. Ensure ML models are trained and available."
        )

    # Persist Ticket in DB
    ticket_status = "review_required" if res["requires_human_review"] else "routed"
    db_ticket = Ticket(
        text=text,
        cleaned_text=res["cleaned_text"],
        status=ticket_status
    )
    db.add(db_ticket)
    db.commit()
    db.refresh(db_ticket)

    # Persist Prediction in DB
    db_prediction = Prediction(
        ticket_id=db_ticket.id,
        category=res["category"],
        category_confidence=res["category_confidence"],
        urgency=res["urgency"],
        urgency_confidence=res["urgency_confidence"],
        queue=res["queue"],
        model_version=res["model_version"],
        requires_human_review=res["requires_human_review"],
        important_features=json.dumps(res["important_features"])
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)

    logger.info(
        f"Ticket {db_ticket.id} processed | Category: {res['category']} ({res['category_confidence']:.2f}) | "
        f"Urgency: {res['urgency']} ({res['urgency_confidence']:.2f}) | Route: {res['status']}"
    )

    return PredictionResponse(
        ticket_id=db_ticket.id,
        category=res["category"],
        category_confidence=res["category_confidence"],
        urgency=res["urgency"],
        urgency_confidence=res["urgency_confidence"],
        queue=res["queue"],
        requires_human_review=res["requires_human_review"],
        important_features=res["important_features"],
        model_version=res["model_version"],
        status=res["status"]
    )
