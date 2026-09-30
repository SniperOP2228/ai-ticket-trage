"""
ML Predictor Service — loads trained models and makes predictions.

Loads models once at startup and reuses them for all requests.
Provides:
- Calibrated category prediction with probability scores
- Calibrated urgency prediction with probability scores
- Feature association extraction (Explainability)
- Confidence-based routing decisions
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import joblib
from scipy import sparse

logger = logging.getLogger(__name__)

# Model directory — when running from backend/, models are at ../ml/models/
# When running in Docker, models are mounted at /app/ml/models/
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent.parent.parent.parent / "ml" / "models"))


class TicketPredictor:
    """
    Singleton predictor that loads calibrated models once and serves predictions.
    
    Uses TF-IDF + CalibratedClassifierCV for both category and urgency.
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

    @staticmethod
    def _ensure_tfidf_transformer_state(vectorizer):
        """Restore TF-IDF internals required by older scikit-learn runtimes."""
        tfidf = getattr(vectorizer, "_tfidf", None)
        if tfidf is None:
            return
        idf = tfidf.__dict__.get("idf_")
        if idf is None or hasattr(tfidf, "_idf_diag"):
            return
        tfidf._idf_diag = sparse.spdiags(
            idf,
            diags=0,
            m=len(idf),
            n=len(idf),
        )

    def load_models(self):
        """Load all trained calibrated models from disk."""
        try:
            logger.info(f"Loading models from {MODEL_DIR}")

            self.category_model = joblib.load(MODEL_DIR / "category_model.joblib")
            self.category_vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
            self.urgency_model = joblib.load(MODEL_DIR / "urgency_model.joblib")
            self.urgency_vectorizer = joblib.load(MODEL_DIR / "urgency_tfidf_vectorizer.joblib")
            self._ensure_tfidf_transformer_state(self.category_vectorizer)
            self._ensure_tfidf_transformer_state(self.urgency_vectorizer)

            # Load metadata
            cat_meta_path = MODEL_DIR / "category_model_metadata.json"
            if cat_meta_path.exists():
                with open(cat_meta_path, encoding='utf-8') as f:
                    self.category_metadata = json.load(f)

            urg_meta_path = MODEL_DIR / "urgency_model_metadata.json"
            if urg_meta_path.exists():
                with open(urg_meta_path, encoding='utf-8') as f:
                    self.urgency_metadata = json.load(f)

            self.is_loaded = True
            logger.info("All calibrated models loaded successfully")

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
        Extract important features associated with the prediction.
        
        Methodology:
        Computes the product of the term's document TF-IDF weight and the global
        tree ensemble feature importances across the calibration folds:
        score(w) = tfidf(w) * mean(tree_importance(w))
        
        Note: These represent correlated model features, not causal claims.
        """
        try:
            text_vec = self.category_vectorizer.transform([text])
            feature_names = self.category_vectorizer.get_feature_names_out()

            # Get non-zero features for this text
            non_zero = text_vec.nonzero()[1]
            if len(non_zero) == 0:
                return []

            # Check if model has calibrated fold estimators with feature_importances_
            tree_importances = None
            if hasattr(self.category_model, 'calibrated_classifiers_'):
                fold_importances = []
                for clf in self.category_model.calibrated_classifiers_:
                    est = getattr(clf, 'estimator', None)
                    if est and hasattr(est, 'feature_importances_'):
                        fold_importances.append(est.feature_importances_)
                if fold_importances:
                    tree_importances = np.mean(fold_importances, axis=0)
            elif hasattr(self.category_model, 'feature_importances_'):
                tree_importances = self.category_model.feature_importances_

            if tree_importances is not None:
                scores = {}
                for idx in non_zero:
                    feature = feature_names[idx]
                    tfidf_w = text_vec[0, idx]
                    tree_w = tree_importances[idx]
                    scores[feature] = float(tfidf_w * tree_w)

                sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                return [f for f, s in sorted_features[:top_n] if s > 0]

            # Linear model fallback if coef_ is available
            if hasattr(self.category_model, 'coef_'):
                classes = list(self.category_model.classes_)
                if category in classes:
                    class_idx = classes.index(category)
                    coefs = self.category_model.coef_[class_idx]
                    scores = {feature_names[idx]: float(text_vec[0, idx] * coefs[idx]) for idx in non_zero}
                    sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    return [f for f, s in sorted_features[:top_n] if s > 0]

            # TF-IDF frequency fallback
            tfidf_scores = [(feature_names[idx], float(text_vec[0, idx])) for idx in non_zero]
            tfidf_scores.sort(key=lambda x: x[1], reverse=True)
            return [f for f, s in tfidf_scores[:top_n]]

        except Exception as e:
            logger.warning(f"Could not extract features: {e}")
            return []

    def predict(self, text: str, confidence_threshold: float = 0.80) -> Dict:
        """
        Make a calibrated prediction for a customer support ticket.
        
        Returns category, urgency, calibrated confidences, routing decision, and feature associations.
        """
        if not self.is_loaded:
            self.load_models()

        cleaned = self.clean_text(text)
        if not cleaned:
            raise ValueError("Text is empty after cleaning")

        # Category prediction via calibrated classifier
        X_cat = self.category_vectorizer.transform([cleaned])
        category = str(self.category_model.predict(X_cat)[0])

        if hasattr(self.category_model, 'predict_proba'):
            cat_probs = self.category_model.predict_proba(X_cat)[0]
            cat_confidence = float(np.max(cat_probs))
        else:
            cat_confidence = 0.5

        # Urgency prediction via calibrated classifier
        X_urg = self.urgency_vectorizer.transform([cleaned])
        urgency = str(self.urgency_model.predict(X_urg)[0])

        if hasattr(self.urgency_model, 'predict_proba'):
            urg_probs = self.urgency_model.predict_proba(X_urg)[0]
            urg_confidence = float(np.max(urg_probs))
        else:
            urg_confidence = 0.5

        # Routing decision based on joint confidence
        min_confidence = min(cat_confidence, urg_confidence)
        requires_review = bool(min_confidence < confidence_threshold)

        queue = self._map_queue(category)
        important_features = self.get_important_features(cleaned, category)

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
        """Return model metadata for the model information page and API."""
        return {
            "category_model": self.category_metadata or {},
            "urgency_model": self.urgency_metadata or {},
            "is_loaded": self.is_loaded,
            "model_dir": str(MODEL_DIR),
        }


# Global singleton instance
predictor = TicketPredictor()
