"""API routes for ticket listing, retrieval, and status updates."""

import json
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db
from app.models.ticket import Ticket, Prediction
from app.schemas.ticket import TicketResponse, TicketListResponse

router = APIRouter(prefix="/tickets", tags=["Tickets"])


@router.get("", response_model=TicketListResponse)
def get_tickets(
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status: pending, routed, review_required, reviewed"),
    category_filter: Optional[str] = Query(None, alias="category", description="Filter by category"),
    urgency_filter: Optional[str] = Query(None, alias="urgency", description="Filter by urgency"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List tickets with optional filters and pagination, including latest prediction data."""
    query = db.query(Ticket).order_by(desc(Ticket.created_at))

    if status_filter:
        query = query.filter(Ticket.status == status_filter)

    total = query.count()
    tickets = query.offset((page - 1) * page_size).limit(page_size).all()

    # Join predictions for each ticket
    results = []
    for t in tickets:
        pred = db.query(Prediction).filter(Prediction.ticket_id == t.id).order_by(desc(Prediction.created_at)).first()
        results.append(TicketResponse(
            id=t.id,
            text=t.text,
            status=t.status,
            category=pred.category if pred else None,
            category_confidence=pred.category_confidence if pred else None,
            urgency=pred.urgency if pred else None,
            urgency_confidence=pred.urgency_confidence if pred else None,
            queue=pred.queue if pred else None,
            requires_human_review=pred.requires_human_review if pred else None,
            created_at=t.created_at
        ))

    return TicketListResponse(
        tickets=results,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{ticket_id}", response_model=TicketResponse)
def get_ticket(ticket_id: int, db: Session = Depends(get_db)):
    """Fetch details of a single ticket with its prediction."""
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    pred = db.query(Prediction).filter(Prediction.ticket_id == ticket.id).order_by(desc(Prediction.created_at)).first()

    return TicketResponse(
        id=ticket.id,
        text=ticket.text,
        status=ticket.status,
        category=pred.category if pred else None,
        category_confidence=pred.category_confidence if pred else None,
        urgency=pred.urgency if pred else None,
        urgency_confidence=pred.urgency_confidence if pred else None,
        queue=pred.queue if pred else None,
        requires_human_review=pred.requires_human_review if pred else None,
        created_at=ticket.created_at
    )
