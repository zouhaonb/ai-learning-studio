"""AI智学 - 数据库配置

SQLAlchemy异步数据库引擎配置。
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from core.config import get_settings


class Base(DeclarativeBase):
    """所有ORM模型的基类"""
    pass


def get_engine():
    """创建异步数据库引擎"""
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        echo=settings.debug,
        future=True,
    )


def get_session_factory():
    """创建异步会话工厂"""
    engine = get_engine()
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def get_db() -> AsyncSession:
    """FastAPI依赖注入：获取数据库会话"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """初始化数据库：创建所有表"""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
