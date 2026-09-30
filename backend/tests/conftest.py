"""Pytest configuration and fixtures for backend testing."""

import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure backend root is on sys.path
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

# Set test environment
os.environ["DATABASE_URL"] = "sqlite:///./test_tickets.db"
os.environ["CONFIDENCE_THRESHOLD"] = "0.80"

from app.database import Base, get_db
from app.main import app
from app.ml.predictor import predictor

# Use in-memory SQLite or test db for tests
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///./test_tickets.db"
test_engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def setup_models():
    """Ensure models are loaded for testing."""
    if not predictor.is_loaded:
        predictor.load_models()


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database for each test function."""
    Base.metadata.create_all(bind=test_engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def client(db_session):
    """FastAPI TestClient with overridden get_db dependency."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
