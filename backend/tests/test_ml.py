"""Automated tests for ML preprocessing, model loading, confidence scoring, and explainability."""

import pytest
from app.ml.predictor import predictor


def test_text_cleaning_pipeline():
    """Test text cleaning handles HTML, URLs, emails, and punctuation."""
    dirty_text = "Hello! Check http://test.com and email me at test@example.com <p>URGENT issue</p> #12345."
    cleaned = predictor.clean_text(dirty_text)

    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "<p>" not in cleaned
    assert "</p>" not in cleaned
    assert "urgent" in cleaned
    assert "issue" in cleaned


def test_model_loading_and_attributes():
    """Verify that models and vectorizers load properly with non-empty vocabularies."""
    if not predictor.is_loaded:
        predictor.load_models()

    assert predictor.category_model is not None
    assert predictor.category_vectorizer is not None
    assert predictor.urgency_model is not None
    assert predictor.urgency_vectorizer is not None
    assert len(predictor.category_vectorizer.vocabulary_) > 0
    assert len(predictor.urgency_vectorizer.vocabulary_) > 0


def test_prediction_output_schema():
    """Test predictor returns correct schema and keys."""
    text = "Database connection failed during customer checkout"
    res = predictor.predict(text, confidence_threshold=0.80)

    assert "category" in res
    assert "category_confidence" in res
    assert "urgency" in res
    assert "urgency_confidence" in res
    assert "queue" in res
    assert "requires_human_review" in res
    assert "important_features" in res
    assert "model_version" in res
    assert "status" in res

    assert isinstance(res["requires_human_review"], bool)
    assert res["status"] in ["auto_routed", "human_review_required"]


def test_confidence_threshold_logic():
    """Verify threshold accurately flags human review."""
    text = "General question about service"

    # With very high threshold, it should require human review
    res_high = predictor.predict(text, confidence_threshold=0.999)
    assert res_high["requires_human_review"] is True
    assert res_high["status"] == "human_review_required"

    # With 0.0 threshold, it should auto-route
    res_low = predictor.predict(text, confidence_threshold=0.0)
    assert res_low["requires_human_review"] is False
    assert res_low["status"] == "auto_routed"


def test_explainability_features():
    """Verify that explainability returns relevant tokens without crashing."""
    text = "Payment charged twice but no refund received on my credit card"
    features = predictor.get_important_features(text, "Billing and Payments", top_n=5)

    assert isinstance(features, list)
    assert len(features) > 0


def test_queue_mapping():
    """Verify queue mapping returns standardized snake_case queue identifiers."""
    assert predictor._map_queue("Technical Support") == "technical_support"
    assert predictor._map_queue("Billing and Payments") == "billing_support"
    assert predictor._map_queue("Customer Service") == "customer_service"
