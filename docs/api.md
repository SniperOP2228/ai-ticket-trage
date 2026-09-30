# REST API Specification

## 1. Overview
The AutoTriage service exposes a fully typed, OpenAPI-compliant REST API powered by FastAPI.
- **Base URL**: `http://localhost:8000/api/v1`
- **Interactive Swagger Documentation**: `http://localhost:8000/docs`
- **ReDoc Specification**: `http://localhost:8000/redoc`

---

## 2. API Endpoints

### 2.1 Predict & Route Ticket
**Endpoint**: `POST /api/v1/predict`  
**Description**: Analyzes ticket text, predicts category and urgency, calculates confidence scores, derives explainable features, and executes automated routing.

#### Request Body
```json
{
  "text": "My payment was deducted twice for order #49281 but order status still shows cancelled. Need immediate refund!"
}
```

#### Response (HTTP 201 Created)
```json
{
  "ticket_id": 42,
  "category": "Billing and Payments",
  "category_confidence": 0.9421,
  "urgency": "High",
  "urgency_confidence": 0.8850,
  "queue": "billing_support",
  "requires_human_review": false,
  "important_features": [
    "refund",
    "payment",
    "deducted",
    "order",
    "cancelled"
  ],
  "model_version": "1.0.0",
  "status": "auto_routed"
}
```

---

### 2.2 List Tickets
**Endpoint**: `GET /api/v1/tickets`  
**Description**: Retrieves paginated tickets with optional filtering.

#### Query Parameters
- `status` (string, optional): Filter by `routed`, `review_required`, `reviewed`.
- `page` (integer, default: 1): Page number.
- `page_size` (integer, default: 20, max: 100): Records per page.

#### Response (HTTP 200 OK)
```json
{
  "tickets": [
    {
      "id": 42,
      "text": "My payment was deducted twice...",
      "status": "routed",
      "category": "Billing and Payments",
      "category_confidence": 0.9421,
      "urgency": "High",
      "urgency_confidence": 0.8850,
      "queue": "billing_support",
      "requires_human_review": false,
      "created_at": "2026-09-25T01:05:00.000Z"
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20
}
```

---

### 2.3 Submit Human Review Correction
**Endpoint**: `POST /api/v1/review/{ticket_id}`  
**Description**: Submits human reviewer corrections for an uncertain prediction. Updates ticket status to `reviewed` and stores the override for model retraining.

#### Request Body
```json
{
  "corrected_category": "Customer Service",
  "corrected_urgency": "Medium",
  "reviewer": "agent_sarah",
  "notes": "User was inquiring on shipping delay rather than hardware failure."
}
```

#### Response (HTTP 201 Created)
```json
{
  "id": 1,
  "ticket_id": 43,
  "original_category": "Technical Support",
  "corrected_category": "Customer Service",
  "original_urgency": "High",
  "corrected_urgency": "Medium",
  "reviewer": "agent_sarah",
  "created_at": "2026-09-25T01:10:00.000Z"
}
```

---

### 2.4 Operational Analytics
**Endpoint**: `GET /api/v1/analytics`  
**Description**: Provides real-time aggregated metrics for the dashboard.

#### Response (HTTP 200 OK)
```json
{
  "total_tickets": 150,
  "auto_routed": 122,
  "human_review_required": 28,
  "reviewed": 15,
  "avg_category_confidence": 0.8245,
  "avg_urgency_confidence": 0.7912,
  "high_urgency_count": 45,
  "category_distribution": {
    "Technical Support": 48,
    "Billing and Payments": 32,
    "Customer Service": 25
  },
  "urgency_distribution": {
    "High": 45,
    "Medium": 70,
    "Low": 35
  },
  "human_correction_rate": 0.2000,
  "confidence_distribution": {
    "0.5-0.6": 5,
    "0.6-0.7": 8,
    "0.7-0.8": 15,
    "0.8-0.9": 62,
    "0.9-1.0": 60
  },
  "recent_predictions": 100
}
```

---

### 2.5 Health Check
**Endpoint**: `GET /api/v1/health`  
**Description**: System health check for Docker container liveness probes.

#### Response (HTTP 200 OK)
```json
{
  "status": "healthy",
  "models_loaded": true,
  "model_version": "1.0.0",
  "confidence_threshold": 0.80
}
```
