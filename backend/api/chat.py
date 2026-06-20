"""AI智学 - 对话API (连接多智能体 + 自动更新画像)"""

from fastapi import APIRouter
from pydantic import BaseModel
from loguru import logger

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    agent: str
    delegated_to: str | None = None
    resources: list[dict] = []


TOPIC_KEYWORDS = {
    "搜索算法": ["搜索", "bfs", "dfs", "a*", "minimax", "alpha-beta"],
    "机器学习": ["机器学习", "线性回归", "逻辑回归", "决策树", "svm", "朴素贝叶斯", "k-means", "pca"],
    "深度学习": ["深度学习", "cnn", "卷积", "resnet", "vgg", "alexnet"],
    "神经网络": ["神经网络", "感知机", "反向传播", "梯度下降", "激活函数", "adam", "sgd"],
    "NLP": ["nlp", "自然语言", "word2vec", "词嵌入", "文本分类", "bert", "gpt", "transformer", "注意力"],
    "计算机视觉": ["计算机视觉", "图像分类", "目标检测", "yolo", "rcnn", "gan", "vae"],
    "知识表示": ["知识表示", "逻辑", "命题", "谓词", "贝叶斯网络"],
    "优化算法": ["优化", "梯度下降", "学习率", "收敛"],
}


def _extract_topics(text: str) -> list[str]:
    text_lower = text.lower()
    return [t for t, kws in TOPIC_KEYWORDS.items() if any(kw in text_lower for kw in kws)]


def _update_profile(user_id: str, user_msg: str, agent_reply: str, delegated_to: str | None):
    from api.profile import _PROFILES
    if user_id not in _PROFILES:
        _PROFILES[user_id] = {
            "user_id": user_id, "knowledge_level": {}, "cognitive_style": "balanced",
            "learning_goals": {"short_term": "", "mid_term": "", "long_term": ""},
            "error_patterns": [], "learning_pace": {"preferred_duration_min": 30, "intensity": "moderate"},
            "interests": [], "metacognition_level": 0.5,
            "motivation": {"intrinsic": 0.7, "extrinsic": 0.5},
            "profile_completeness": 0, "recent_activities": [],
        }
    p = _PROFILES[user_id]
    topics = _extract_topics(user_msg + " " + agent_reply)

    for t in topics:
        p["knowledge_level"][t] = min(100, p["knowledge_level"].get(t, 0) + (8 if delegated_to else 5))

    atype = "quiz" if any(w in user_msg for w in ["练习", "题目", "测试"]) else \
            "code" if any(w in user_msg for w in ["代码", "实现", "编程"]) else "document"
    p["recent_activities"] = [{"topic": topics[0] if topics else user_msg[:15], "type": atype, "time": "刚刚", "status": "completed"}] + p.get("recent_activities", [])[:9]

    for t in topics:
        if t not in p["interests"]:
            p["interests"].append(t)
    p["interests"] = p["interests"][:8]

    filled = sum(1 for k in ["knowledge_level","cognitive_style","learning_goals","error_patterns","learning_pace","interests","metacognition_level","motivation"] if p.get(k))
    p["profile_completeness"] = min(8, filled)

    from api.profile import _save_profiles, _PROFILES
    _save_profiles(_PROFILES)


@router.post("/send", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    from agents.team import get_team
    team = get_team()
    try:
        result = team.chat(request.message, request.user_id)
        _update_profile(request.user_id, request.message, result["reply"], result["delegated_to"])
        return ChatResponse(reply=result["reply"], agent=result["agent"],
                            delegated_to=result["delegated_to"], resources=result["resources"])
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(reply=f"抱歉，处理请求时出错: {str(e)}", agent="System")


@router.get("/agents")
async def list_agents():
    from agents.team import get_team
    return {"agents": get_team().get_agent_info()}
