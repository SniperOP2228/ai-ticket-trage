"""Service layer handling support queue assignment and human review gating."""

from typing import Dict, Tuple
from app.config import settings


class RoutingService:
    """Evaluates prediction confidence scores and determines target queue and review status."""

    QUEUE_MAP = {
        "Technical Support": "technical_support",
        "Product Support": "product_support",
        "Customer Service": "customer_service",
        "IT Support": "it_support",
        "Billing and Payments": "billing_support",
        "Returns and Exchanges": "returns_support",
        "Service Outages and Maintenance": "outage_support",
        "Sales and Pre-Sales": "sales_support",
        "Human Resources": "hr_support",
        "General Inquiry": "general_support",
    }

    @classmethod
    def determine_route(cls, category: str, category_confidence: float, urgency_confidence: float) -> Tuple[str, bool, str]:
        """
        Calculates destination queue and routing decision.
        Returns: (queue_name, requires_human_review, routing_status)
        """
        queue = cls.QUEUE_MAP.get(category, category.lower().replace(" ", "_"))
        min_conf = min(category_confidence, urgency_confidence)
        requires_review = min_conf < settings.confidence_threshold
        status = "human_review_required" if requires_review else "auto_routed"
        return queue, requires_review, status


routing_service = RoutingService()
