# AI Customer Support Ticket Triage & Intelligent Routing System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-quality, end-to-end Machine Learning and NLP system that classifies customer support tickets into 10 operational departments, estimates ticket urgency across 3 tiers, computes **calibrated confidence scores** via cross-validated Platt scaling, dynamically routes high-confidence inquiries, flags uncertain predictions for **Human-in-the-Loop review**, and provides a closed-loop retraining pipeline with real-time operational analytics through a React dashboard.

---

## 1. Problem Statement & Architecture

In customer service organizations, high ticket volumes create triage bottlenecks, routing delays, and delayed responses to critical incidents. Rule-based keyword matching breaks down on colloquial descriptions, multi-intent tickets, and spelling variations. Conversely, un-gated black-box AI risks silently misrouting critical issues.

AutoTriage AI addresses this using a hybrid intelligence architecture:

```
                            [ Inbound Customer Support Ticket ]
                                            │
                                            ▼
                           ┌──────────────────────────────────┐
                           │      FastAPI REST Gateway        │
                           │   Pydantic Schema Validation     │
                           └────────────────┬─────────────────┘
                                            │
                                            ▼
                           ┌──────────────────────────────────┐
                           │     NLP Normalization Engine     │
                           │  - Email/URL/HTML stripping      │
                           │  - Domain keyword preservation   │
                           └────────────────┬─────────────────┘
                                            │
                   ┌────────────────────────┴────────────────────────┐
                   ▼                                                 ▼
     ┌────────────────────────────┐                    ┌────────────────────────────┐
     │ Category Model (10 Classes)│                    │ Urgency Model (3 Tiers)    │
     │ Calibrated Random Forest   │                    │ Calibrated Random Forest   │
     │ (5-Fold CV Sigmoid Scaling)│                    │ (5-Fold CV Sigmoid Scaling)│
     └─────────────┬──────────────┘                    └─────────────┬──────────────┘
                   │                                                 │
                   └────────────────────────┬────────────────────────┘
                                            │
                                            ▼
                           ┌──────────────────────────────────┐
                           │   Confidence & Routing Engine    │
                           │   min(Category, Urgency) >= 0.80 │
                           └────────┬───────────────────┬─────┘
                                    │                   │
                     YES (>= 0.80)  │                   │  NO (< 0.80)
                                    ▼                   ▼
                     ┌───────────────────────┐ ┌─────────────────────────┐
                     │ Automatically Routed  │ │   Human Review Queue    │
                     │  - billing_support    │ │  - Low-confidence queue │
                     │  - technical_support  │ │  - Manual override opt  │
                     │  - outage_support ... │ │  - Immutable predictions│
                     └───────────────────────┘ └───────────┬─────────────┘
                                                           │
                                                           ▼
                                               [ Continuous Retraining ]
                                               scripts/retrain_from_feedback.py
```

---

## 2. Tech Stack

- **Backend**: Python 3.11, FastAPI, Pydantic v2, Uvicorn, SQLAlchemy 2.0
- **Machine Learning**: Scikit-Learn (CalibratedClassifierCV, RandomForest, LogisticRegression, LinearSVC), NumPy, Pandas, Joblib (compressed serialization)
- **NLP**: Sublinear N-gram TF-IDF Vectorization, Hugging Face `all-MiniLM-L6-v2` Sentence Transformers benchmark
- **Database**: PostgreSQL with automatic SQLite fallback for local development
- **Frontend**: React 18, Vite, Tailwind CSS, Recharts, Lucide Icons
- **Testing**: Pytest, FastAPI TestClient (20 automated unit & integration tests)
- **Deployment**: Docker, Multi-stage Dockerfiles, Docker Compose, CORS security middleware

---

## 3. Dataset Provenance

- **Source**: [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) on Hugging Face Hub (DOI: `10.57967/hf/6184`).
- **License**: Creative Commons Attribution-NonCommercial 4.0 International (`CC-BY-NC-4.0`).
- **Total Records**: 23,747 cleaned English tickets partitioned into stratified 70/15/15 splits (16,622 Train, 3,562 Validation, 3,562 Test).
- **Categories (10 classes)**: `Technical Support`, `Product Support`, `Customer Service`, `IT Support`, `Billing and Payments`, `Returns and Exchanges`, `Service Outages and Maintenance`, `Sales and Pre-Sales`, `Human Resources`, `General Inquiry`.
- **Urgency (3 classes)**: `High`, `Medium`, `Low`.

---

## 4. Model Experimentation & Measured Performance

All candidate models were tuned and compared on the **Validation split** before final test evaluation:

### 4.1 Category Model Comparison (Validation Set)

| Candidate Model | Features | Accuracy | Macro F1 | Weighted F1 | Train Latency | Inference Latency |
|---|---|---|---|---|---|---|
| **Logistic Regression** | TF-IDF (10k) | 44.13% | 0.4307 | 0.4433 | 3.12s | 3.0 ms |
| **Multinomial Naive Bayes** | TF-IDF (10k) | 41.58% | 0.3251 | 0.3891 | 0.05s | 3.0 ms |
| **Linear SVM** | TF-IDF (10k) | 54.01% | 0.5274 | 0.5318 | 8.87s | 14.5 ms |
| **Random Forest (Selected)** | TF-IDF (10k) | **54.66%** | **0.5672** | **0.5512** | 5.55s | 89.1 ms |
| **Logistic Regression** | Sentence Embeddings (384d) | 29.62% | 0.2805 | 0.3059 | 35.06s | 2.0 ms |

### 4.2 Urgency Model Comparison (Validation Set)

| Candidate Model | Features | Accuracy | Macro F1 | Weighted F1 | Train Latency |
|---|---|---|---|---|---|
| **Logistic Regression** | TF-IDF (8k) | 53.93% | 0.5303 | 0.5427 | 1.09s |
| **Multinomial Naive Bayes** | TF-IDF (8k) | 49.58% | 0.4345 | 0.4749 | 0.04s |
| **Linear SVM** | TF-IDF (8k) | 56.71% | 0.5151 | 0.5503 | 3.82s |
| **Random Forest (Selected)** | TF-IDF (8k) | **68.95%** | **0.6761** | **0.6871** | 15.78s |
| **Logistic Regression** | Sentence Embeddings (384d) | 43.77% | 0.4305 | 0.4410 | 2.50s |

### 4.3 Final Calibrated Deployed Models (Held-Out Test Set)

Evaluated **only once** on the untouched test set ($N = 3,562$ tickets):

- **Calibrated Category Model**:
  - Test Accuracy: **55.39%**
  - Test Macro F1: **0.5820**
  - Test Weighted F1: **0.5503**
- **Calibrated Urgency Model**:
  - Test Accuracy: **65.24%**
  - Test Macro F1: **0.6309**
  - Test Weighted F1: **0.6475**

---

## 5. Probability Calibration & Reliability Metrics

In automated decision systems, raw classifier confidence scores (`predict_proba()`) from uncalibrated tree ensembles frequently distort probabilities toward extremes. Calibration ensures that a prediction assigned 80% confidence is empirically correct approximately 80% of the time.

### Methodology
- **Platt Scaling (Sigmoid)**: Applied `CalibratedClassifierCV(method='sigmoid', cv=5)` fitted strictly on training data using 5-fold cross-validation.
- **Why Sigmoid over Isotonic?**: The dataset has 10 classes with severe class imbalance (e.g., `General Inquiry` contains only 238 training examples). Isotonic regression is non-parametric and overfits sparse classes, creating piecewise flat probability distributions. Platt sigmoid calibration provides smooth, well-regularized parametric scaling without overfitting.
- **Model Compression**: Models are serialized using `joblib.dump(..., compress=3)`, keeping artifact sizes at 75.9MB and 62.8MB, safely within GitHub's 100MB limit.

### Measured Calibration Results on Held-Out Test Set

| Metric | Category Model | Urgency Model |
|---|---|---|
| **Brier Score** | **0.5885** | **0.4679** |
| **Expected Calibration Error (ECE)** | **9.84%** (`0.0984`) | **6.21%** (`0.0621`) |
| **Maximum Calibration Error (MCE)** | **29.77%** (`0.2977`) | **12.08%** (`0.1208`) |
| **Reliability Diagram** | [`docs/images/category_calibration.png`](docs/images/category_calibration.png) | [`docs/images/urgency_calibration.png`](docs/images/urgency_calibration.png) |

---

## 6. Confidence Threshold Analysis (Human-in-the-Loop Gating)

To evaluate the operational trade-off between automation coverage and routing error rate, thresholds were swept across the validation set before selecting the operating threshold:

### 6.1 Validation Threshold Sweep

| Threshold | Auto-Routed | Needs Review | Coverage (%) | Review Rate (%) | Auto-Route Category Accuracy (%) | Auto-Route Joint Accuracy (%) | Auto-Route Error Rate (%) |
|---|---|---|---|---|---|---|---|
| **0.50** | 790 | 2,772 | 22.2% | 77.8% | 86.6% | 74.9% | 13.4% |
| **0.60** | 272 | 3,290 | 7.6% | 92.4% | 92.6% | 84.6% | 7.4% |
| **0.70** | 95 | 3,467 | 2.7% | 97.3% | 92.6% | 88.4% | 7.4% |
| **0.80** | 38 | 3,524 | 1.1% | 98.9% | 89.5% | 86.8% | 10.5% |
| **0.90** | 4 | 3,558 | 0.1% | 99.9% | 100.0% | 100.0% | 0.0% |

### 6.2 Held-Out Test Benchmark at Selected Threshold (0.80)

- **Total Test Tickets**: 3,562
- **Automatically Routed**: 48 tickets (**1.3%** coverage)
- **Sent to Human Review**: 3,514 tickets (**98.7%** review rate)
- **Auto-Route Category Accuracy**: **100.0%** (48 / 48)
- **Auto-Route Joint Accuracy (Category + Urgency)**: **100.0%** (48 / 48)
- **Auto-Route Error Rate**: **0.0%**

*Operational Insight*: At a conservative 0.80 threshold, auto-routed tickets achieved 100% accuracy on test data, preventing misrouted customer escalations. Organizations prioritizing higher automated throughput can configure `CONFIDENCE_THRESHOLD=0.50`, achieving **21.6% coverage** with **89.8% category accuracy**.

---

## 7. NLP Comparison: TF-IDF vs. Sentence Transformers

| Evaluation Dimension | N-Gram TF-IDF + Random Forest | `all-MiniLM-L6-v2` + Logistic Regression |
|---|---|---|
| **Category Macro F1** | **0.5820** | 0.2805 |
| **Urgency Macro F1** | **0.6309** | 0.4305 |
| **Feature Extraction Time** | **1.5 seconds** | 243.0 seconds (~4 minutes) |
| **Inference Compute** | Minimal CPU footprint | Requires dense PyTorch runtime |
| **Domain Specificity** | Directly isolates technical terms (`payout`, `crash`, `malwarebytes`) | Dilutes rare domain technical tokens in dense latent space |

---

## 8. Human Review, Manual Override & Active Learning

### Review vs. Manual Override Distinction
- **Standard Review (`/review/{ticket_id}`)**: Handles tickets with confidence below 0.80 (`Ticket.status == "review_required"`).
- **Manual Override (`/review/{ticket_id}/override`)**: Allows support leads to manually override already auto-routed tickets (`Ticket.status == "routed"`).
- **Prediction Immutability**: In all review and override operations, the original `Prediction` record in the database is **never modified or deleted**, ensuring an audit trail for model monitoring and drift analysis.

### Retraining Feedback Pipeline
- **Dataset Builder (`scripts/build_feedback_dataset.py`)**: Validates reviewed tickets against canonical schemas and exports verified samples to `data/processed/human_feedback.csv`.
- **Candidate Trainer (`scripts/retrain_from_feedback.py`)**: Combines base training data with verified feedback, trains candidate models with 5-fold cross-validated calibration, benchmarks performance against the baseline on test data, and saves to `ml/models/candidates/`. Models are only promoted to production if the `--promote` flag is provided.

---

## 9. Quickstart: Running the Application

### Option A: Docker Compose (Recommended)
```bash
# 1. Clone repository
git clone https://github.com/your-username/ai-ticket-triage.git
cd ai-ticket-triage

# 2. Configure environment
cp .env.example .env

# 3. Launch full stack
docker compose up --build -d
```
- **Dashboard UI**: `http://localhost:5173`
- **FastAPI Interactive Docs**: `http://localhost:8000/docs`

### Option B: Local Development
```bash
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows (or: source .venv/bin/activate on Linux/Mac)

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Run automated tests (20 passed)
pytest backend/tests/ -v

# 4. Start backend
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Start frontend (in a separate terminal)
cd ../frontend
npm install
npm run dev
```

---

## 10. Automated Testing Suite

The project includes 20 comprehensive unit and integration tests covering API endpoints, model calibration, prediction immutability, and queue routing:

```bash
pytest backend/tests/ -v
```
```
backend/tests/test_api.py::test_health_endpoint PASSED                   [  5%]
backend/tests/test_api.py::test_predict_valid_ticket PASSED              [ 10%]
backend/tests/test_api.py::test_predict_empty_input PASSED               [ 15%]
backend/tests/test_api.py::test_predict_short_input PASSED               [ 20%]
backend/tests/test_api.py::test_tickets_listing_and_filtering PASSED     [ 25%]
backend/tests/test_api.py::test_ticket_detail_retrieval PASSED           [ 30%]
backend/tests/test_api.py::test_ticket_not_found PASSED                  [ 35%]
backend/tests/test_api.py::test_human_review_workflow PASSED             [ 40%]
backend/tests/test_api.py::test_analytics_endpoint PASSED                [ 45%]
backend/tests/test_api.py::test_model_info_endpoint PASSED               [ 50%]
backend/tests/test_api.py::test_manual_override_and_prediction_immutability PASSED [ 55%]
backend/tests/test_api.py::test_review_queue_filtering PASSED            [ 60%]
backend/tests/test_ml.py::test_text_cleaning_pipeline PASSED             [ 65%]
backend/tests/test_ml.py::test_model_loading_and_attributes PASSED       [ 70%]
backend/tests/test_ml.py::test_prediction_output_schema PASSED           [ 75%]
backend/tests/test_ml.py::test_confidence_threshold_logic PASSED         [ 80%]
backend/tests/test_ml.py::test_explainability_features PASSED            [ 85%]
backend/tests/test_ml.py::test_queue_mapping PASSED                      [ 90%]
backend/tests/test_ml.py::test_probability_calibration_properties PASSED [ 95%]
backend/tests/test_ml.py::test_empty_or_whitespace_prediction_handling PASSED [100%]
======================== 20 passed in 66.11s ========================
```

---

## 11. System Limitations

1. **Class Imbalance**: The dataset contains significant category skew (over 4,800 `Technical Support` tickets versus only ~240 `General Inquiry` tickets in train). While balanced class weights and Platt scaling mitigate bias, rare categories exhibit wider confidence intervals.
2. **Lexical Representation**: N-gram TF-IDF relies on explicit keyword occurrence. It may struggle with highly metaphorical phrasing or nuanced sarcasm that lacks domain vocabulary.
3. **Threshold vs. Coverage Trade-off**: At a strict 0.80 calibrated threshold, auto-routing coverage is 1.3%, meaning most tickets are queued for human review. In production, this threshold should be dynamically calibrated to balance team capacity against error tolerance.
4. **Authentication & RBAC**: The current release provides public REST endpoints suitable for internal VPC or staging environments. Enterprise production deployment requires integrating OAuth2/JWT or SSO role-based access control.

---

## 12. Project Structure

```
ai-ticket-triage/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, lifespan loader & CORS
│   │   ├── config.py            # Pydantic settings & environment configuration
│   │   ├── database.py          # SQLAlchemy PostgreSQL/SQLite engine
│   │   ├── api/                 # REST routes (predict, tickets, review, analytics)
│   │   ├── models/              # SQLAlchemy ORM models (Ticket, Prediction, HumanReview)
│   │   ├── schemas/             # Pydantic validation schemas
│   │   └── ml/                  # Singleton TicketPredictor & XAI engine
│   ├── tests/                   # 20 Pytest integration & unit tests
│   ├── requirements.txt         # Production dependencies
│   └── Dockerfile               # Backend container definition
├── frontend/
│   ├── src/
│   │   ├── pages/               # Dashboard, Predict, Tickets, Review, Analytics, ModelInfo
│   │   ├── components/          # Navbar, StatCard, Recharts widgets
│   │   ├── services/api.js      # Axios API client
│   │   └── App.jsx
│   ├── package.json
│   └── Dockerfile               # Multi-stage production Nginx build
├── data/
│   ├── raw/                     # Original dataset
│   └── processed/               # Stratified train/val/test splits & human feedback
├── ml/
│   ├── preprocessing/           # EDA and text normalization scripts
│   ├── training/                # Category, Urgency, and Embedding trainers (with CV calibration)
│   ├── evaluation/              # Calibration & threshold analysis scripts and logs
│   └── models/                  # Calibrated .joblib artifacts (< 100MB) & metadata
├── scripts/
│   ├── build_feedback_dataset.py # Active learning feedback validator
│   └── retrain_from_feedback.py # Candidate model retraining & promotion
├── docs/                        # Architectural, ML, and calibration documentation
├── docker-compose.yml           # Multi-container orchestration
└── README.md
```

---

## 13. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Dataset is distributed under `CC-BY-NC-4.0`.
