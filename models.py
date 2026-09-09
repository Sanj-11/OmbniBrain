from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    datasets = relationship("Dataset", back_populates="owner", cascade="all, delete-orphan")

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String, nullable=False)
    stored_filename = Column(String, nullable=False, unique=True)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Float, nullable=False)
    rows = Column(Integer, nullable=False)
    columns = Column(Integer, nullable=False)
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    owner = relationship("User", back_populates="datasets")