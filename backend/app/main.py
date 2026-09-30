"""Main FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import engine, Base
from app.models.ticket import Ticket, Prediction, HumanReview, ModelVersion
from app.ml.predictor import predictor
from app.config import settings

# Import API routers
from app.api.routes_predict import router as predict_router
from app.api.routes_tickets import router as tickets_router
from app.api.routes_review import router as review_router
from app.api.routes_analytics import router as analytics_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ticket_triage_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("Starting up AI Ticket Triage API...")

    # Create database tables if they do not exist
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}")

    # Load ML models into memory
    try:
        predictor.load_models()
        logger.info("ML models loaded and ready for inference.")
    except Exception as e:
        logger.warning(f"Could not load ML models on startup: {e}. Models will load on first request.")

    yield
    logger.info("Shutting down AI Ticket Triage API.")


app = FastAPI(
    title="AI Customer Support Ticket Triage & Intelligent Routing System",
    description="""
    Production-grade AI/ML service for customer support ticket triage:
    - Multi-class Category Classification (10 support categories)
    - Ticket Urgency / Priority Classification (High, Medium, Low)
    - Calibrated Confidence Scoring & Dynamic Routing
    - Human-in-the-Loop review for low-confidence tickets
    - Explainable AI (Top contributing linguistic features)
    - Full Analytics & Real-time Metrics
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 Routers
app.include_router(predict_router, prefix="/api/v1")
app.include_router(tickets_router, prefix="/api/v1")
app.include_router(review_router, prefix="/api/v1")
app.include_router(analytics_router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
@app.get("/api/v1/health", tags=["Health"])
def health_check():
    """Health check endpoint for Docker and monitoring."""
    return {
        "status": "healthy",
        "models_loaded": predictor.is_loaded,
        "model_version": settings.model_version,
        "confidence_threshold": settings.confidence_threshold
    }


@app.get("/", tags=["Root"])
def root():
    """Root endpoint welcoming users and pointing to docs."""
    return {
        "message": "AI Customer Support Ticket Triage & Intelligent Routing System API",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health": "/api/v1/health",
        "version": settings.model_version
    }
