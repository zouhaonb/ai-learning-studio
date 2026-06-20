"""AI智学 - 讯飞星火LLM客户端 (官方SDK版)

使用 spark-ai-python 官方SDK，通过WebSocket直连星火大模型。
支持流式/非流式对话，无需本地代理。
"""

from typing import AsyncIterator
from loguru import logger
from sparkai.llm.llm import ChatSparkLLM
from sparkai.core.messages import ChatMessage
from core.config import get_settings


class SparkLLMClient:
    """讯飞星火大模型客户端 (基于官方SDK)"""

    def __init__(self):
        settings = get_settings()
        self._api_key = settings.iflytek_api_key
        self._api_secret = settings.iflytek_api_secret
        self._app_id = settings.iflytek_app_id
        self._primary_model = settings.spark_model_primary
        self._lite_model = settings.spark_model_lite
        logger.info(f"SparkLLMClient initialized: primary={self._primary_model}")

    def _create_client(self, model: str, streaming: bool = False) -> ChatSparkLLM:
        return ChatSparkLLM(
            spark_api_key=self._api_key,
            spark_api_secret=self._api_secret,
            spark_app_id=self._app_id,
            domain=model,
            streaming=streaming,
            request_timeout=120,
        )

    def chat(self, messages: list[dict], model: str | None = None) -> str:
        """同步对话"""
        target_model = model or self._primary_model
        client = self._create_client(target_model, streaming=False)
        spark_messages = [
            ChatMessage(role=m["role"], content=m["content"])
            for m in messages
        ]
        response = client.invoke(spark_messages)
        logger.debug(f"Spark response ({target_model}): {response.content[:100]}...")
        return response.content

    def chat_stream(self, messages: list[dict], model: str | None = None):
        """流式对话 (生成器)"""
        target_model = model or self._primary_model
        client = self._create_client(target_model, streaming=True)
        spark_messages = [
            ChatMessage(role=m["role"], content=m["content"])
            for m in messages
        ]
        for chunk in client.stream(spark_messages):
            if chunk.content:
                yield chunk.content

    def get_model_info(self) -> dict:
        return {
            "primary_model": self._primary_model,
            "lite_model": self._lite_model,
            "sdk": "spark-ai-python",
        }


# 全局单例
_client: SparkLLMClient | None = None


def get_llm_client() -> SparkLLMClient:
    global _client
    if _client is None:
        _client = SparkLLMClient()
    return _client
