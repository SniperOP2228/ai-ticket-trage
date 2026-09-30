"""Service layer integrating ML model inference with business logic."""

from typing import Dict
from app.ml.predictor import predictor
from app.config import settings


class MLService:
    """Wrapper service for model predictions and explainability."""

    @staticmethod
    def predict_ticket(text: str) -> Dict:
        """Runs end-to-end inference using the singleton predictor."""
        return predictor.predict(text, confidence_threshold=settings.confidence_threshold)

    @staticmethod
    def get_model_metadata() -> Dict:
        """Retrieves active model registry info."""
        return predictor.get_model_info()


ml_service = MLService()
