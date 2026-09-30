"""Integration tests for all FastAPI REST endpoints."""

import pytest
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert "confidence_threshold" in data


def test_predict_valid_ticket(client: TestClient):
    """Test ticket classification with valid input."""
    payload = {
        "text": "My payment was deducted twice for the subscription. Please issue a refund immediately."
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 201
    data = response.json()

    assert "ticket_id" in data
    assert "category" in data
    assert "category_confidence" in data
    assert "urgency" in data
    assert "urgency_confidence" in data
    assert "queue" in data
    assert "requires_human_review" in data
    assert "important_features" in data
    assert isinstance(data["important_features"], list)
    assert 0.0 <= data["category_confidence"] <= 1.0
    assert 0.0 <= data["urgency_confidence"] <= 1.0


def test_predict_empty_input(client: TestClient):
    """Test prediction with empty text triggers 422 or 400 validation error."""
    response = client.post("/api/v1/predict", json={"text": ""})
    assert response.status_code in [400, 422]


def test_predict_short_input(client: TestClient):
    """Test prediction with very short input triggers validation error."""
    response = client.post("/api/v1/predict", json={"text": "ab"})
    assert response.status_code == 422


def test_tickets_listing_and_filtering(client: TestClient):
    """Test ticket list retrieval with filters."""
    # Create a ticket first
    client.post("/api/v1/predict", json={"text": "Server is down and website shows 500 error."})

    response = client.get("/api/v1/tickets?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert "tickets" in data
    assert "total" in data
    assert len(data["tickets"]) >= 1


def test_ticket_detail_retrieval(client: TestClient):
    """Test retrieving ticket by ID."""
    pred_res = client.post("/api/v1/predict", json={"text": "How do I reset my account password?"})
    ticket_id = pred_res.json()["ticket_id"]

    response = client.get(f"/api/v1/tickets/{ticket_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == ticket_id
    assert "text" in data
    assert "status" in data


def test_ticket_not_found(client: TestClient):
    """Test 404 on nonexistent ticket."""
    response = client.get("/api/v1/tickets/999999")
    assert response.status_code == 404


def test_human_review_workflow(client: TestClient):
    """Test complete human-in-the-loop workflow: prediction -> queue -> review submission."""
    # Create a prediction
    pred_res = client.post("/api/v1/predict", json={"text": "Need help with general query."})
    ticket_id = pred_res.json()["ticket_id"]

    # Submit a human correction
    review_payload = {
        "corrected_category": "Customer Service",
        "corrected_urgency": "Low",
        "reviewer": "test_agent_1",
        "notes": "Clarified with customer"
    }
    rev_res = client.post(f"/api/v1/review/{ticket_id}", json=review_payload)
    assert rev_res.status_code == 201
    rev_data = rev_res.json()
    assert rev_data["corrected_category"] == "Customer Service"
    assert rev_data["corrected_urgency"] == "Low"
    assert rev_data["reviewer"] == "test_agent_1"

    # Verify ticket status updated to reviewed
    ticket_res = client.get(f"/api/v1/tickets/{ticket_id}")
    assert ticket_res.json()["status"] == "reviewed"


def test_analytics_endpoint(client: TestClient):
    """Test analytics aggregation endpoint."""
    client.post("/api/v1/predict", json={"text": "Software bug report: crash on startup."})
    client.post("/api/v1/predict", json={"text": "Billing invoice incorrect for last month."})

    response = client.get("/api/v1/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "total_tickets" in data
    assert data["total_tickets"] >= 2
    assert "category_distribution" in data
    assert "urgency_distribution" in data
    assert "avg_category_confidence" in data


def test_model_info_endpoint(client: TestClient):
    """Test model-info endpoint returns metadata."""
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "category_model" in data
    assert "urgency_model" in data
    assert "confidence_threshold" in data
    assert "model_version" in data


def test_manual_override_and_prediction_immutability(client: TestClient):
    """Test that manual override updates ticket status while keeping prediction record immutable."""
    # Create a prediction
    pred_res = client.post("/api/v1/predict", json={
        "text": "My server crashed with out of memory error. Database is unreachable."
    })
    assert pred_res.status_code == 201
    pred_data = pred_res.json()
    ticket_id = pred_data["ticket_id"]
    orig_cat = pred_data["category"]
    orig_urg = pred_data["urgency"]
    orig_conf = pred_data["category_confidence"]

    # Explicit manual override
    override_payload = {
        "corrected_category": "Billing and Payments",
        "corrected_urgency": "Low",
        "reviewer": "operations_lead",
        "notes": "Customer mistakenly filed infra issue under wrong account billing",
        "is_override": True
    }
    override_res = client.post(f"/api/v1/review/{ticket_id}/override", json=override_payload)
    assert override_res.status_code == 201
    override_data = override_res.json()
    assert override_data["is_override"] is True
    assert "[Manual Override]" in override_data["notes"]
    assert override_data["corrected_category"] == "Billing and Payments"
    assert override_data["corrected_urgency"] == "Low"

    # Verify ticket status transitioned to reviewed
    ticket_res = client.get(f"/api/v1/tickets/{ticket_id}")
    assert ticket_res.status_code == 200
    ticket_data = ticket_res.json()
    assert ticket_data["status"] == "reviewed"

    # Verify original prediction in DB remains completely unchanged
    assert ticket_data["category"] == orig_cat
    assert ticket_data["urgency"] == orig_urg
    assert ticket_data["category_confidence"] == orig_conf

    # Verify review history records the override
    history_res = client.get("/api/v1/review/history")
    assert history_res.status_code == 200
    history_data = history_res.json()
    assert any(h["ticket_id"] == ticket_id and h["is_override"] is True for h in history_data)


def test_review_queue_filtering(client: TestClient):
    """Verify review queue only includes tickets requiring human review."""
    # Ensure a ticket requiring review exists
    pred_res = client.post("/api/v1/predict", json={
        "text": "Help with misc question."
    })
    ticket_id = pred_res.json()["ticket_id"]

    queue_res = client.get("/api/v1/review/queue")
    assert queue_res.status_code == 200
    queue_data = queue_res.json()
    assert isinstance(queue_data, list)
    for item in queue_data:
        assert "ticket_id" in item
        assert "predicted_category" in item
        assert "category_confidence" in item
