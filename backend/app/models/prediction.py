"""
SQLAlchemy models for Prediction results.
"""
from sqlalchemy import Column, Integer
from app.database import Base

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
