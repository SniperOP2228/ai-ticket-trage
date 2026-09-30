# Deployment & Production Setup Guide

## 1. Quick Start with Docker Compose
The recommended approach for local development and cloud production deployment is Docker Compose.

### Prerequisites
- Docker Engine 24+ & Docker Compose v2+
- 4GB available RAM

### Launch Commands
```bash
# Clone the repository
git clone https://github.com/your-username/ai-ticket-triage.git
cd ai-ticket-triage

# Copy environment variables
cp .env.example .env

# Build and start all three services (PostgreSQL, FastAPI Backend, React Frontend)
docker compose up --build -d
```

### Accessing Running Services
- **React Dashboard**: `http://localhost:5173` (or port `80` if using production nginx)
- **FastAPI REST API**: `http://localhost:8000`
- **Swagger Documentation**: `http://localhost:8000/docs`
- **PostgreSQL Database**: `localhost:5432`

---

## 2. Local Bare-Metal Setup (Without Docker)

### Step 1: Environment Setup
```bash
cd ai-ticket-triage

# Create Python 3.11 virtual environment
uv venv .venv --python 3.11
# Or: python -m venv .venv

# Activate environment
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install backend dependencies
uv pip install -r backend/requirements.txt
```

### Step 2: Download Data & Run ML Training (If Retraining)
```bash
# 1. Download HuggingFace dataset
python scripts/download_data.py

# 2. Run Preprocessing & Splitting
python ml/preprocessing/preprocess.py

# 3. Train Category Models
python ml/training/train_category.py

# 4. Train Urgency Models
python ml/training/train_urgency.py
```

### Step 3: Launch FastAPI Backend
```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 4: Launch React Frontend
```bash
cd ../frontend
npm install
npm run dev
```

---

## 3. Environment Configuration Reference (`.env`)

| Variable | Default Value | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:password@localhost:5432/ticket_triage` | Connection string for database |
| `API_HOST` | `0.0.0.0` | API bind address |
| `API_PORT` | `8000` | API listening port |
| `CONFIDENCE_THRESHOLD` | `0.80` | Cutoff for Human-in-the-Loop review routing |
| `MODEL_VERSION` | `1.0.0` | Active model deployment version |
| `VITE_API_URL` | `http://localhost:8000` | Backend API URL for frontend client |

---

## 4. Continuous Integration (GitHub Actions)
The repository includes `.github/workflows/ci.yml`:
1. Executes on every `push` and `pull_request` to `main`.
2. Sets up Python 3.11 and runs full automated test suite (`pytest backend/tests/`).
3. Sets up Node.js 20, installs dependencies, and runs `npm run build` to verify frontend production compilation.
