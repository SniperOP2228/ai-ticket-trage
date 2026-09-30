"""
SQLAlchemy models for human reviews of predictions.
"""
from sqlalchemy import Column, Integer
from app.database import Base

class Review(Base):
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True)
