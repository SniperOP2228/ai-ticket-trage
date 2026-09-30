# System Architecture & Technical Design

## 1. Executive Architecture Overview

```
                                  [ Customer / Support Agent ]
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │     React + Vite + Tailwind      │
                              │     Dashboard (Port 5173 / 80)   │
                              └────────────────┬─────────────────┘
                                               │ HTTP / REST
                                               ▼
                              ┌──────────────────────────────────┐
                              │      FastAPI Gateway (Port 8000) │
                              │   Pydantic Validation & Routing  │
                              └────────────────┬─────────────────┘
                                               │
             ┌─────────────────────────────────┴─────────────────────────────────┐
             │                                                                   │
             ▼                                                                   ▼
┌───────────────────────────┐                                       ┌───────────────────────────┐
│     Inference Service     │                                       │   PostgreSQL / SQLite     │
│  - Text Normalization     │                                       │  - tickets                │
│  - Dual ML Classifiers    │                                       │  - predictions            │
│  - Calibrated Confidence  │                                       │  - human_reviews          │
│  - Explainability Engine  │                                       │  - model_versions         │
└────────────┬──────────────┘                                       └───────────────────────────┘
             │
             ├── If min(Category Conf, Urgency Conf) >= 0.80 ──> [ Auto-Routed to Queue ]
             │                                                          (billing, tech, outage...)
             │
             └── If Confidence < 0.80 ─────────────────────────> [ Human Review Queue ]
                                                                        │
                                                                 (Correction feedback)
                                                                        │
                                                                        ▼
                                                             [ Closed-Loop Database ]
```

## 2. Core Architectural Pillars

### A. Modular Service Decomposition
1. **API Gateway (`backend/app/main.py`)**:
   - Built on FastAPI with asynchronous lifespan management.
   - Automatically initializes ML models into memory once at startup, achieving single-digit millisecond inference latency.
   - CORS middleware configured for secure frontend cross-origin requests.

2. **Data Model & Storage Layer (`backend/app/models/ticket.py`, `backend/app/database.py`)**:
   - SQLAlchemy 2.0 ORM with connection pooling.
   - Native PostgreSQL engine for production deployments.
   - Automatic SQLite fallback for local test suites and offline verification without external infrastructure dependencies.

3. **Inference & Explainability Engine (`backend/app/ml/predictor.py`)**:
   - Encapsulated singleton pattern.
   - Preprocessing parity between training and inference pipelines.
   - Dual-head prediction: Category Classifier (10 classes) and Urgency Classifier (3 classes).
   - Dynamic linguistic token importance extraction (TF-IDF feature weights & coefficients).

4. **Human-in-the-Loop Feedback Loop (`backend/app/api/routes_review.py`)**:
   - Gated decision routing: predictions scoring below the configurable confidence threshold (`CONFIDENCE_THRESHOLD=0.80`) are redirected to the Human Review Queue.
   - Human corrections capture both category and urgency overrides with reviewer identity and timestamp, building a high-value dataset for active learning and model retraining.

5. **Operational Analytics Telemetry (`backend/app/api/routes_analytics.py`)**:
   - Aggregates throughput, confidence distributions, queue volumes, and human correction rate:
     $$\text{Human Correction Rate} = \frac{\text{Human Overrides}}{\text{Total Reviewed Tickets}}$$

6. **Frontend Dashboard (`frontend/src/`)**:
   - High-performance SPA built with React 18 and Vite.
   - Styled with Tailwind CSS and visual telemetry powered by Recharts.
   - Real-time ticket submission, interactive confidence meters, and filterable audit logs.
