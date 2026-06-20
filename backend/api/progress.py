"""AI智学 - 学习进度API (带持久化)"""

import json
from pathlib import Path
from fastapi import APIRouter
from pydantic import BaseModel
from loguru import logger

router = APIRouter()

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
PROGRESS_FILE = DATA_DIR / "progress.json"


class QuizResult(BaseModel):
    topic_id: str
    score: float
    total_questions: int
    correct_answers: int
    wrong_topics: list[str] = []


def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_progress(data: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


_PROGRESS: dict[str, dict] = _load_progress()


def _get_user_progress(user_id: str) -> dict:
    if user_id not in _PROGRESS:
        _PROGRESS[user_id] = {}
    return _PROGRESS[user_id]


@router.get("/path/{user_id}")
async def get_learning_path(user_id: str):
    from core.knowledge_graph import get_knowledge_graph
    kg = get_knowledge_graph()
    all_topics = kg.get_all_topics()
    progress = _get_user_progress(user_id)

    nodes = []
    for t in all_topics:
        tp = progress.get(t["id"], {})
        score = tp.get("best_score", 0)
        attempts = tp.get("attempts", 0)
        status = "mastered" if attempts > 0 and score >= 80 else "in_progress" if attempts > 0 else "not_started"
        detail = kg.get_topic_detail(t["id"])
        nodes.append({
            "id": t["id"], "name": t["name"], "level": t["level"], "difficulty": t["difficulty"],
            "status": status, "score": score, "attempts": attempts,
            "prerequisites": detail["prerequisites"] if detail else [],
            "description": detail["description"] if detail else "",
            "keywords": detail["keywords"] if detail else [],
        })

    total = len(nodes)
    mastered = sum(1 for n in nodes if n["status"] == "mastered")
    return {"nodes": nodes, "stats": {
        "total": total, "mastered": mastered,
        "in_progress": sum(1 for n in nodes if n["status"] == "in_progress"),
        "not_started": total - mastered - sum(1 for n in nodes if n["status"] == "in_progress"),
        "avg_score": round(sum(n["score"] for n in nodes) / total, 1) if total else 0,
        "progress_percent": round(mastered / total * 100, 1) if total else 0,
    }}


@router.post("/quiz-result")
async def submit_quiz_result(result: QuizResult, user_id: str = "default_user"):
    progress = _get_user_progress(user_id)
    if result.topic_id not in progress:
        progress[result.topic_id] = {"attempts": 0, "best_score": 0, "scores": [], "wrong_topics_history": []}
    tp = progress[result.topic_id]
    tp["attempts"] += 1
    tp["scores"].append(result.score)
    tp["best_score"] = max(tp["best_score"], result.score)
    tp["latest_score"] = result.score
    if result.wrong_topics:
        tp["wrong_topics_history"].extend(result.wrong_topics)
    _save_progress(progress)

    from api.profile import _PROFILES
    if user_id in _PROFILES:
        _PROFILES[user_id]["knowledge_level"][result.topic_id] = tp["best_score"]
        _PROFILES[user_id]["recent_activities"] = [{"topic": result.topic_id, "type": "quiz", "time": "刚刚", "status": "completed", "score": result.score}] + _PROFILES[user_id].get("recent_activities", [])[:9]

    status = "mastered" if tp["best_score"] >= 80 else "in_progress"
    return {"status": status, "score": result.score, "best_score": tp["best_score"], "attempts": tp["attempts"],
            "message": f"{'恭喜通过！' if result.score >= 80 else '继续努力！'}得分: {result.score}%"}


@router.post("/study/{topic_id}")
async def mark_studied(topic_id: str, user_id: str = "default_user"):
    progress = _get_user_progress(user_id)
    if topic_id not in progress:
        progress[topic_id] = {"attempts": 0, "best_score": 0, "scores": [], "wrong_topics_history": []}
    progress[topic_id]["studied"] = True
    return {"status": "ok"}


@router.get("/diagnosis/{user_id}")
async def get_diagnosis(user_id: str):
    progress = _get_user_progress(user_id)
    return {k: v.get("best_score", 0) for k, v in progress.items() if v.get("attempts", 0) > 0}
