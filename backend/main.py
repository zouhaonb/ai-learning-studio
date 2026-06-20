"""AI智学 - FastAPI应用主入口"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from core.config import get_settings
from db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("AI智学系统启动中...")
    await init_db()
    logger.info("数据库初始化完成")
    yield
    logger.info("AI智学系统关闭")


def create_app() -> FastAPI:
    """创建FastAPI应用实例"""
    settings = get_settings()

    app = FastAPI(
        title="AI智学 - 个性化学习多智能体系统",
        description="基于讯飞星火大模型的个性化资源生成与学习多智能体系统",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_url, "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    from api.chat import router as chat_router
    from api.profile import router as profile_router
    from api.resources import router as resources_router
    from api.knowledge_graph import router as knowledge_graph_router
    from api.progress import router as progress_router
    from api.learn import router as learn_router

    app.include_router(chat_router, prefix="/api/chat", tags=["对话"])
    app.include_router(profile_router, prefix="/api/profile", tags=["画像"])
    app.include_router(resources_router, prefix="/api/resources", tags=["资源"])
    app.include_router(knowledge_graph_router, prefix="/api/knowledge-graph", tags=["知识图谱"])
    app.include_router(progress_router, prefix="/api/progress", tags=["学习进度"])
    app.include_router(learn_router, prefix="/api/learn", tags=["学习内容"])

    @app.get("/")
    async def root():
        return {"name": "AI智学", "version": "1.0.0", "description": "基于大模型的个性化资源生成与学习多智能体系统"}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("main:app", host=settings.app_host, port=settings.app_port, reload=settings.debug)
