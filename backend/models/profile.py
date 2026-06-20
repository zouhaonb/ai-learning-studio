"""AI智学 - 学生画像数据模型

8维度动态学生画像，由画像构建智能体通过自然语言对话自主构建。
"""

import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, Float, Integer, JSON, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from db.database import Base


class StudentProfile(Base):
    """学生画像表 - 8维度动态画像"""
    __tablename__ = "student_profiles"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), unique=True, nullable=False
    )

    # 维度1: 知识基础 - 各知识点掌握程度 (0-100)
    # 格式: {"搜索算法": 75, "机器学习": 60, "深度学习": 40}
    knowledge_level: Mapped[dict] = mapped_column(JSON, default=dict)

    # 维度2: 认知风格 (visual/auditory/kinesthetic/reading/balanced)
    cognitive_style: Mapped[str] = mapped_column(String(20), default="balanced")

    # 维度3: 学习目标
    # 格式: {"short_term": "...", "mid_term": "...", "long_term": "..."}
    learning_goals: Mapped[dict] = mapped_column(JSON, default=dict)

    # 维度4: 易错点偏好
    # 格式: [{"topic": "梯度下降", "error_type": "概念混淆", "frequency": 3}]
    error_patterns: Mapped[list] = mapped_column(JSON, default=list)

    # 维度5: 学习节奏
    # 格式: {"preferred_duration_min": 30, "intensity": "moderate"}
    learning_pace: Mapped[dict] = mapped_column(JSON, default=dict)

    # 维度6: 兴趣方向
    # 格式: ["自然语言处理", "计算机视觉", "强化学习"]
    interests: Mapped[list] = mapped_column(JSON, default=list)

    # 维度7: 元认知水平 (0-1)
    metacognition_level: Mapped[float] = mapped_column(Float, default=0.5)

    # 维度8: 学习动机
    # 格式: {"intrinsic": 0.7, "extrinsic": 0.5, "overall": 0.6}
    motivation: Mapped[dict] = mapped_column(JSON, default=dict)

    # 画像完成度 (0-8)
    profile_completeness: Mapped[int] = mapped_column(Integer, default=0)

    # 对话历史摘要
    conversation_summary: Mapped[str] = mapped_column(String(2000), default="")

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    user = relationship("User", back_populates="profile")

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "knowledge_level": self.knowledge_level,
            "cognitive_style": self.cognitive_style,
            "learning_goals": self.learning_goals,
            "error_patterns": self.error_patterns,
            "learning_pace": self.learning_pace,
            "interests": self.interests,
            "metacognition_level": self.metacognition_level,
            "motivation": self.motivation,
            "profile_completeness": self.profile_completeness,
        }
