# -*- coding: utf-8 -*-
import os
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "detection.db")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DetectionRecord(Base):
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(100), nullable=False)
    score = Column(Float, nullable=False)
    screenshot_path = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.now)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def extract_camera_id(video_name: str) -> str:
    """
    Extract camera ID from video file name.
    Example: 'camera_01.avi' -> 'camera_01'
    """
    if video_name.endswith('.avi'):
        return video_name[:-4]
    elif video_name.endswith('.mp4'):
        return video_name[:-4]
    return video_name


def save_fighting_result(
    video_name: str,
    score: float,
    screenshot_path: Optional[str] = None
) -> Optional[DetectionRecord]:
    """
    Save fighting detection result to database.
    Only called when fighting is detected.

    Args:
        video_name: Video file name (e.g. 'camera_01.avi')
        score: Confidence score
        screenshot_path: Screenshot save path

    Returns:
        DetectionRecord: The saved record object
    """
    db = SessionLocal()
    try:
        camera_id = extract_camera_id(video_name)

        record = DetectionRecord(
            camera_id=camera_id,
            score=score,
            screenshot_path=screenshot_path
        )

        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    finally:
        db.close()


def get_all_records() -> list:
    """Get all fighting detection records."""
    db = SessionLocal()
    try:
        return db.query(DetectionRecord).order_by(DetectionRecord.created_at.desc()).all()
    finally:
        db.close()


def get_records_by_camera(camera_id: str) -> list:
    """Get fighting records by camera ID."""
    db = SessionLocal()
    try:
        return db.query(DetectionRecord).filter(
            DetectionRecord.camera_id == camera_id
        ).order_by(DetectionRecord.created_at.desc()).all()
    finally:
        db.close()
