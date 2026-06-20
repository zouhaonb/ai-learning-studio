"""AI智学 - 画像API (带持久化)"""

import json
from pathlib import Path
from fastapi import APIRouter

router = APIRouter()

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
PROFILE_FILE = DATA_DIR / "profiles.json"

_DEFAULT_PROFILE = {
    "user_id": "default_user",
    "knowledge_level": {"搜索算法": 0, "机器学习": 0, "深度学习": 0, "NLP": 0, "计算机视觉": 0, "知识表示": 0, "神经网络": 0, "优化算法": 0},
    "cognitive_style": "balanced",
    "learning_goals": {"short_term": "", "mid_term": "", "long_term": ""},
    "error_patterns": [], "learning_pace": {"preferred_duration_min": 30, "intensity": "moderate"},
    "interests": [], "metacognition_level": 0.5,
    "motivation": {"intrinsic": 0.7, "extrinsic": 0.5},
    "profile_completeness": 0, "recent_activities": [],
}


def _load_profiles() -> dict:
    if PROFILE_FILE.exists():
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"default_user": _DEFAULT_PROFILE.copy()}


def _save_profiles(data: dict):
    with open(PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


_PROFILES = _load_profiles()

_PROFILES = {
    "default_user": {
        "user_id": "default_user",
        "knowledge_level": {
            "搜索算法": 75, "机器学习": 60, "深度学习": 45,
            "NLP": 30, "计算机视觉": 35, "知识表示": 55,
            "神经网络": 50, "优化算法": 40,
        },
        "cognitive_style": "visual",
        "learning_goals": {
            "short_term": "掌握CNN和RNN基础",
            "mid_term": "理解Transformer架构",
            "long_term": "深入大语言模型",
        },
        "error_patterns": [
            {"topic": "反向传播", "error_type": "公式推导错误", "frequency": 3},
            {"topic": "SVM", "error_type": "对偶问题理解不清", "frequency": 2},
        ],
        "learning_pace": {"preferred_duration_min": 30, "intensity": "moderate"},
        "interests": ["深度学习", "自然语言处理", "大语言模型"],
        "metacognition_level": 0.6,
        "motivation": {"intrinsic": 0.8, "extrinsic": 0.5},
        "profile_completeness": 6,
        "recent_activities": [
            {"topic": "卷积神经网络", "type": "document", "time": "2小时前", "status": "completed"},
            {"topic": "反向传播算法", "type": "quiz", "time": "3小时前", "status": "in_progress"},
            {"topic": "Transformer架构", "type": "reading", "time": "昨天", "status": "pending"},
            {"topic": "梯度下降优化", "type": "code", "time": "昨天", "status": "completed"},
        ],
    }
}


@router.get("/{user_id}")
async def get_profile(user_id: str):
    profile = _PROFILES.get(user_id)
    if not profile:
        return {"user_id": user_id, "knowledge_level": {}, "profile_completeness": 0, "recent_activities": []}
    return profile


@router.post("/{user_id}/update")
async def update_profile(user_id: str, update_data: dict):
    if user_id not in _PROFILES:
        _PROFILES[user_id] = {"user_id": user_id, "knowledge_level": {}, "profile_completeness": 0}
    _PROFILES[user_id].update(update_data)
    _save_profiles(_PROFILES)
    return {"status": "updated", "user_id": user_id}
