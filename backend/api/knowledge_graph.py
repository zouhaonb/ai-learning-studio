"""AI智学 - 知识图谱API"""

from fastapi import APIRouter
from core.knowledge_graph import get_knowledge_graph

router = APIRouter()


@router.get("/graph")
async def get_graph_data():
    """获取知识图谱可视化数据"""
    return get_knowledge_graph().get_graph_data()


@router.get("/topics")
async def list_topics():
    return {"topics": get_knowledge_graph().get_all_topics()}


@router.get("/topics/{topic_id}")
async def get_topic_detail(topic_id: str):
    detail = get_knowledge_graph().get_topic_detail(topic_id)
    return detail if detail else {"error": "Topic not found"}


@router.get("/search")
async def search_topics(keyword: str):
    return {"results": get_knowledge_graph().search_topics(keyword)}


@router.get("/learning-path")
async def get_learning_path(target: str, mastered: str = ""):
    kg = get_knowledge_graph()
    mastered_list = [m.strip() for m in mastered.split(",") if m.strip()] if mastered else []
    path = kg.get_learning_path(target, mastered_list)
    return {"target": target, "path": path, "total_steps": len(path)}
