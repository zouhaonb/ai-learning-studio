"""AI智学 - 学习内容API (直接调用LLM，快速生成)"""

from fastapi import APIRouter
from pydantic import BaseModel
from loguru import logger

router = APIRouter()


class ContentRequest(BaseModel):
    topic_id: str
    topic_name: str
    description: str = ""
    keywords: list[str] = []


class QuizRequest(BaseModel):
    topic_id: str
    topic_name: str
    num_questions: int = 5


@router.post("/generate-content")
async def generate_content(req: ContentRequest):
    from core.llm_client import get_llm_client
    llm = get_llm_client()
    prompt = f"""你是AI课程讲师，请详细讲解"{req.topic_name}"。
要求：概念定义、关键公式/算法、具体例子、常见误区、与相关知识点联系。
关键词：{', '.join(req.keywords)}
{f'简介：{req.description}' if req.description else ''}
Markdown格式，结构清晰。"""
    try:
        content = llm.chat([{"role": "user", "content": prompt}])
        return {"content": content, "topic_id": req.topic_id}
    except Exception as e:
        logger.error(f"Content error: {e}")
        return {"content": f"生成失败: {e}", "topic_id": req.topic_id}


@router.post("/generate-quiz")
async def generate_quiz(req: QuizRequest):
    from core.llm_client import get_llm_client
    llm = get_llm_client()
    prompt = f"""为"{req.topic_name}"生成{req.num_questions}道选择题。
严格JSON输出，不要其他内容：
{{"questions":[{{"question":"题目","options":["A.选项1","B.选项2","C.选项3","D.选项4"],"answer":"A","explanation":"详细解析"}}]}}
每题4选项1正确答案，覆盖核心概念，难度适中。"""
    try:
        import json, re
        raw = llm.chat([{"role": "user", "content": prompt}])
        match = re.search(r'\{[\s\S]*"questions"[\s\S]*\}', raw)
        if match:
            data = json.loads(match.group())
            return {"questions": data["questions"], "topic_id": req.topic_id}
        return {"questions": [], "topic_id": req.topic_id, "error": "解析失败"}
    except Exception as e:
        logger.error(f"Quiz error: {e}")
        return {"questions": [], "topic_id": req.topic_id, "error": str(e)}
