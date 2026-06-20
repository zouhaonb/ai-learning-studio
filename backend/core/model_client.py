"""AI智学 - AutoGen模型客户端

使用 spark-ai-python SDK 的内置代理将星火WebSocket转为HTTP接口，
AutoGen通过OpenAI兼容接口调用。
"""

import subprocess
import time
import socket
from loguru import logger
from autogen_ext.models.openai import OpenAIChatCompletionClient
from core.config import get_settings

# SDK代理进程
_proxy_process = None
_PROXY_PORT = 8009


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def ensure_proxy_running():
    """确保SDK代理服务在运行"""
    global _proxy_process
    if _is_port_in_use(_PROXY_PORT):
        logger.info(f"Spark proxy already running on port {_PROXY_PORT}")
        return

    settings = get_settings()
    _proxy_process = subprocess.Popen(
        [
            "python", "-m", "sparkai.openai_api_server",
            "--host", "127.0.0.1",
            "--port", str(_PROXY_PORT),
            "--appid", settings.iflytek_app_id,
            "--api_secret", settings.iflytek_api_secret,
            "--spark_api_key", settings.iflytek_api_key,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(2)
    if _is_port_in_use(_PROXY_PORT):
        logger.info(f"Spark proxy started on port {_PROXY_PORT}")
    else:
        logger.warning("Spark proxy failed to start, using direct SDK calls")


def get_spark_model_client(
    model: str | None = None,
    temperature: float = 0.7,
) -> OpenAIChatCompletionClient:
    """创建AutoGen模型客户端 (通过SDK代理连接星火)"""
    settings = get_settings()
    target_model = model or settings.spark_model_primary

    return OpenAIChatCompletionClient(
        model=target_model,
        base_url=f"http://127.0.0.1:{_PROXY_PORT}/v1",
        api_key="spark",  # SDK代理不需要真实key
        temperature=temperature,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
        },
    )


def get_spark_lite_client(temperature: float = 0.3) -> OpenAIChatCompletionClient:
    """轻量级客户端"""
    settings = get_settings()
    return get_spark_model_client(model=settings.spark_model_lite, temperature=temperature)
