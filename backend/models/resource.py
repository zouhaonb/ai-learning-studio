"""AI智学 - 学习资源和学习记录数据模型"""

import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, Integer, Float, JSON, ForeignKey, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from db.database import Base


class Resource(Base):
    """学习资源表 - 6种类型: document/mindmap/quiz/reading/multimedia/code_case"""
    __tablename__ = "resources"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(30), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="")
    topic: Mapped[str] = mapped_column(String(100), nullable=False)
    difficulty: Mapped[int] = mapped_column(Integer, default=3)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_for_user: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    learning_records = relationship("LearningRecord", back_populates="resource")


class LearningRecord(Base):
    """学习记录表"""
    __tablename__ = "learning_records"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    resource_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("resources.id"), nullable=True
    )
    action_type: Mapped[str] = mapped_column(String(30), nullable=False)
    topic: Mapped[str] = mapped_column(String(100), default="")
    score: Mapped[float] = mapped_column(Float, nullable=True)
    duration_seconds: Mapped[int] = mapped_column(Integer, default=0)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="learning_records")
    resource = relationship("Resource", back_populates="learning_records")


class LearningPath(Base):
    """学习路径表"""
    __tablename__ = "learning_paths"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    path_name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    steps: Mapped[list] = mapped_column(JSON, default=list)
    current_step: Mapped[int] = mapped_column(Integer, default=1)
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    progress_percent: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
