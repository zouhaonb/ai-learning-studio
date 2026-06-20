"""AI智学 - 配置管理模块

使用 pydantic-settings 管理所有配置项，支持从 .env 文件和环境变量加载。
"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # === 讯飞星火大模型 ===
    iflytek_app_id: str = ""
    iflytek_api_key: str = ""
    iflytek_api_secret: str = ""
    iflytek_api_password: str = ""

    # === 模型配置 ===
    spark_model_primary: str = "4.0Ultra"
    spark_model_lite: str = "lite"
    spark_base_url: str = "https://spark-api-open.xf-yun.com/v1"

    # === 数据库 ===
    database_url: str = "sqlite+aiosqlite:///./data/ai_learning.db"

    # === ChromaDB ===
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection: str = "ai_course_knowledge"

    # === 应用配置 ===
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True
    secret_key: str = "change-this-to-a-random-secret-key"

    # === 前端 ===
    frontend_url: str = "http://localhost:3000"

    @property
    def spark_api_url(self) -> str:
        """星火OpenAI兼容接口完整URL"""
        return f"{self.spark_base_url}/chat/completions"

    @property
    def data_dir(self) -> Path:
        """数据存储目录"""
        path = Path("./data")
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache()
def get_settings() -> Settings:
    """获取全局配置单例"""
    return Settings()
