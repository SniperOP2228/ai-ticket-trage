"""Human-in-the-Loop review API endpoints.

Allows support agents / reviewers to inspect tickets that were flagged
as low-confidence and submit category/urgency corrections.
Corrections are recorded in human_reviews and the ticket status is updated to 'reviewed'.
"""

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db
from app.models.ticket import Ticket, Prediction, HumanReview
from app.schemas.ticket import ReviewRequest, ReviewResponse, ReviewTicketResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/review", tags=["Human Review"])


@router.get("/queue", response_model=List[ReviewTicketResponse])
def get_review_queue(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """Retrieve tickets that require human review (low confidence predictions)."""
    tickets = db.query(Ticket).filter(
        Ticket.status == "review_required"
    ).order_by(desc(Ticket.created_at)).limit(limit).all()

    queue_items = []
    for t in tickets:
        pred = db.query(Prediction).filter(
            Prediction.ticket_id == t.id
        ).order_by(desc(Prediction.created_at)).first()

        if pred:
            queue_items.append(ReviewTicketResponse(
                ticket_id=t.id,
                text=t.text,
                predicted_category=pred.category,
                category_confidence=pred.category_confidence,
                predicted_urgency=pred.urgency,
                urgency_confidence=pred.urgency_confidence,
                created_at=t.created_at
            ))

    return queue_items


@router.post("/{ticket_id}", response_model=ReviewResponse, status_code=status.HTTP_201_CREATED)
def submit_review(
    ticket_id: int,
    payload: ReviewRequest,
    db: Session = Depends(get_db)
):
    """
    Submits a human correction for a low-confidence ticket.
    Saves the original vs corrected values, reviewer info, timestamp,
    and updates ticket status to 'reviewed'.
    """
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    pred = db.query(Prediction).filter(
        Prediction.ticket_id == ticket.id
    ).order_by(desc(Prediction.created_at)).first()

    if not pred:
        raise HTTPException(status_code=400, detail="Cannot review a ticket without prediction record")

    # Record human review
    review = HumanReview(
        ticket_id=ticket.id,
        original_category=pred.category,
        corrected_category=payload.corrected_category,
        original_urgency=pred.urgency,
        corrected_urgency=payload.corrected_urgency,
        original_confidence=pred.category_confidence,
        reviewer=payload.reviewer,
        notes=payload.notes
    )
    db.add(review)

    # Update ticket status to reviewed
    ticket.status = "reviewed"
    db.commit()
    db.refresh(review)

    logger.info(
        f"Human review submitted for Ticket {ticket.id} by {payload.reviewer} | "
        f"Category: {pred.category} -> {payload.corrected_category} | "
        f"Urgency: {pred.urgency} -> {payload.corrected_urgency}"
    )

    return ReviewResponse(
        id=review.id,
        ticket_id=review.ticket_id,
        original_category=review.original_category,
        corrected_category=review.corrected_category,
        original_urgency=review.original_urgency,
        corrected_urgency=review.corrected_urgency,
        reviewer=review.reviewer,
        created_at=review.created_at
    )


@router.get("/history", response_model=List[ReviewResponse])
def get_review_history(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """Retrieve history of submitted human reviews."""
    reviews = db.query(HumanReview).order_by(desc(HumanReview.created_at)).limit(limit).all()
    return reviews
