"""
ML Predictor Service — loads trained models and makes predictions.

Loads models once at startup and reuses them for all requests.
Provides:
- Category prediction with confidence
- Urgency prediction with confidence
- Important feature extraction (explainability)
- Confidence-based routing decision
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import joblib

logger = logging.getLogger(__name__)

# Model directory — when running from backend/, models are at ../ml/models/
# When running in Docker, models are mounted at /app/ml/models/
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent.parent.parent.parent / "ml" / "models"))


class TicketPredictor:
    """
    Singleton predictor that loads models once and serves predictions.
    
    Uses TF-IDF + classifier for both category and urgency.
    Models are loaded lazily on first prediction or explicitly via load_models().
    """

    def __init__(self):
        self.category_model = None
        self.category_vectorizer = None
        self.urgency_model = None
        self.urgency_vectorizer = None
        self.category_metadata = None
        self.urgency_metadata = None
        self.is_loaded = False

    def load_models(self):
        """Load all trained models from disk."""
        try:
            logger.info(f"Loading models from {MODEL_DIR}")

            self.category_model = joblib.load(MODEL_DIR / "category_model.joblib")
            self.category_vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
            self.urgency_model = joblib.load(MODEL_DIR / "urgency_model.joblib")
            self.urgency_vectorizer = joblib.load(MODEL_DIR / "urgency_tfidf_vectorizer.joblib")

            # Load metadata
            cat_meta_path = MODEL_DIR / "category_model_metadata.json"
            if cat_meta_path.exists():
                with open(cat_meta_path) as f:
                    self.category_metadata = json.load(f)

            urg_meta_path = MODEL_DIR / "urgency_model_metadata.json"
            if urg_meta_path.exists():
                with open(urg_meta_path) as f:
                    self.urgency_metadata = json.load(f)

            self.is_loaded = True
            logger.info("All models loaded successfully")

        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise RuntimeError(f"Model file not found: {e}. Run training scripts first.")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean input text using the same preprocessing as training."""
        if not text or not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^a-z0-9\s\.\,\!\?\-]', ' ', text)
        text = re.sub(r'\b\d+\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_important_features(self, text: str, category: str, top_n: int = 5) -> List[str]:
        """
        Extract important features contributing to the prediction.
        
        For TF-IDF models, shows which terms had the highest TF-IDF weights
        among the features used by the model for the predicted class.
        
        Note: These are correlated features, not causal explanations.
        """
        try:
            text_vec = self.category_vectorizer.transform([text])
            feature_names = self.category_vectorizer.get_feature_names_out()

            # Get non-zero features for this text
            non_zero = text_vec.nonzero()[1]
            if len(non_zero) == 0:
                return []

            # Get model coefficients for the predicted class
            if hasattr(self.category_model, 'coef_'):
                classes = list(self.category_model.classes_)
                if category in classes:
                    class_idx = classes.index(category)
                    coefs = self.category_model.coef_[class_idx]

                    # Score = TF-IDF weight * model coefficient
                    scores = {}
                    for idx in non_zero:
                        feature = feature_names[idx]
                        tfidf_weight = text_vec[0, idx]
                        model_weight = coefs[idx]
                        scores[feature] = float(tfidf_weight * model_weight)

                    # Return top features by positive contribution
                    sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    return [f for f, s in sorted_features[:top_n] if s > 0]

            # Fallback: return top TF-IDF terms
            tfidf_scores = [(feature_names[idx], float(text_vec[0, idx])) for idx in non_zero]
            tfidf_scores.sort(key=lambda x: x[1], reverse=True)
            return [f for f, s in tfidf_scores[:top_n]]

        except Exception as e:
            logger.warning(f"Could not extract features: {e}")
            return []

    def predict(self, text: str, confidence_threshold: float = 0.80) -> Dict:
        """
        Make a complete prediction for a customer support ticket.
        
        Returns category, urgency, confidences, routing decision, and explanations.
        """
        if not self.is_loaded:
            self.load_models()

        # Clean text
        cleaned = self.clean_text(text)
        if not cleaned:
            raise ValueError("Text is empty after cleaning")

        # Category prediction
        X_cat = self.category_vectorizer.transform([cleaned])
        category = self.category_model.predict(X_cat)[0]

        if hasattr(self.category_model, 'predict_proba'):
            cat_probs = self.category_model.predict_proba(X_cat)[0]
            cat_confidence = float(max(cat_probs))
        else:
            cat_confidence = 0.5  # Default if no probability available

        # Urgency prediction
        X_urg = self.urgency_vectorizer.transform([cleaned])
        urgency = self.urgency_model.predict(X_urg)[0]

        if hasattr(self.urgency_model, 'predict_proba'):
            urg_probs = self.urgency_model.predict_proba(X_urg)[0]
            urg_confidence = float(max(urg_probs))
        else:
            urg_confidence = 0.5

        # Routing decision
        min_confidence = min(cat_confidence, urg_confidence)
        requires_review = min_confidence < confidence_threshold

        # Queue mapping
        queue = self._map_queue(category)

        # Explainability
        important_features = self.get_important_features(cleaned, category)

        # Model version
        version = "1.0.0"
        if self.category_metadata:
            version = self.category_metadata.get('model_version', '1.0.0')

        return {
            "category": category,
            "category_confidence": round(cat_confidence, 4),
            "urgency": urgency,
            "urgency_confidence": round(urg_confidence, 4),
            "queue": queue,
            "requires_human_review": requires_review,
            "important_features": important_features,
            "model_version": version,
            "cleaned_text": cleaned,
            "status": "human_review_required" if requires_review else "auto_routed",
        }

    def _map_queue(self, category: str) -> str:
        """Map category to support queue name."""
        queue_map = {
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
        return queue_map.get(category, category.lower().replace(" ", "_"))

    def get_model_info(self) -> Dict:
        """Return model metadata for the model information page."""
        return {
            "category_model": self.category_metadata or {},
            "urgency_model": self.urgency_metadata or {},
            "is_loaded": self.is_loaded,
            "model_dir": str(MODEL_DIR),
        }


# Global singleton instance
predictor = TicketPredictor()
