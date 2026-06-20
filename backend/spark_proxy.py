"""AI智学 - 讯飞星火本地代理服务器

将讯飞WebSocket API转换为OpenAI兼容的HTTP API。
使用APPID/APIKey/APISecret直接认证，无需APIPassword。

使用方式: python spark_proxy.py
启动后: http://localhost:8008/v1/chat/completions
"""

import asyncio
import json
import hashlib
import hmac
import base64
from datetime import datetime
from urllib.parse import urlencode, urlparse
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import websockets


def load_env():
    env_path = Path(__file__).parent / ".env"
    config = {}
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip()
    return config


ENV = load_env()
APP_ID = ENV.get("IFLYTEK_APP_ID", "")
API_KEY = ENV.get("IFLYTEK_API_KEY", "")
API_SECRET = ENV.get("IFLYTEK_API_SECRET", "")

MODEL_URLS = {
    "4.0Ultra": "wss://spark-api.xf-yun.com/v4.0/chat",
    "generalv3.5": "wss://spark-api.xf-yun.com/v3.5/chat",
    "generalv3": "wss://spark-api.xf-yun.com/v3.1/chat",
    "lite": "wss://spark-api.xf-yun.com/v1.1/chat",
}

MODEL_ALIASES = {
    "spark-4.0-ultra": "4.0Ultra",
    "spark-max": "generalv3.5",
    "spark-pro": "generalv3",
    "spark-lite": "lite",
}


def create_auth_url(ws_url: str) -> str:
    """生成讯飞WebSocket认证URL (HMAC-SHA256签名)"""
    parsed = urlparse(ws_url)
    host = parsed.hostname
    path = parsed.path
    date = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')

    # Step 1: 构建签名原始串
    signature_origin = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"

    # Step 2: HMAC-SHA256签名，key为API_SECRET
    sign = hmac.new(API_SECRET.encode("utf-8"), signature_origin.encode("utf-8"), hashlib.sha256).digest()
    sign_base64 = base64.b64encode(sign).decode("utf-8")

    # Step 3: 构建authorization原始串
    authorization_origin = (
        f'api_key="{API_KEY}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{sign_base64}"'
    )

    # Step 4: 对authorization做base64编码
    authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")

    # Step 5: 拼接最终URL
    return f"{ws_url}?authorization={authorization}&date={date}&host={host}"


def build_payload(messages: list, model: str, max_tokens: int) -> dict:
    system_msg = ""
    chat_msgs = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            chat_msgs.append({"role": m["role"], "content": m["content"]})
    if system_msg:
        chat_msgs.insert(0, {"role": "system", "content": system_msg})
    return {
        "header": {"app_id": APP_ID, "uid": "ai_learning_user"},
        "parameter": {"chat": {"domain": model, "max_tokens": max_tokens, "temperature": 0.7}},
        "payload": {"message": {"text": chat_msgs}},
    }


async def call_spark(messages: list, model: str, max_tokens: int) -> str:
    model_domain = MODEL_ALIASES.get(model, model)
    ws_url = MODEL_URLS.get(model_domain, MODEL_URLS["4.0Ultra"])
    auth_url = create_auth_url(ws_url)
    payload = build_payload(messages, model_domain, max_tokens)
    result = ""
    async with websockets.connect(auth_url, max_size=10*1024*1024) as ws:
        await ws.send(json.dumps(payload))
        while True:
            resp = json.loads(await ws.recv())
            if resp["header"]["code"] != 0:
                raise Exception(f"Spark error: {resp['header'].get('message', '')}")
            text = resp["payload"]["choices"]["text"]
            result += text[0]["content"] if text else ""
            if resp["header"]["status"] == 2:
                break
    return result


async def call_spark_stream(messages: list, model: str, max_tokens: int):
    model_domain = MODEL_ALIASES.get(model, model)
    ws_url = MODEL_URLS.get(model_domain, MODEL_URLS["4.0Ultra"])
    auth_url = create_auth_url(ws_url)
    payload = build_payload(messages, model_domain, max_tokens)
    async with websockets.connect(auth_url, max_size=10*1024*1024) as ws:
        await ws.send(json.dumps(payload))
        while True:
            resp = json.loads(await ws.recv())
            if resp["header"]["code"] != 0:
                yield f"data: {json.dumps({'error': resp['header'].get('message', '')})}\n\n"
                break
            text = resp["payload"]["choices"]["text"]
            if text:
                chunk = {"choices": [{"delta": {"content": text[0]["content"]}, "index": 0}]}
                yield f"data: {json.dumps(chunk)}\n\n"
            if resp["header"]["status"] == 2:
                yield "data: [DONE]\n\n"
                break


app = FastAPI(title="Spark Proxy")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://localhost:8000"], allow_methods=["*"], allow_headers=["*"])


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "4.0Ultra")
    max_tokens = body.get("max_tokens", 4096)
    stream = body.get("stream", False)
    if stream:
        return StreamingResponse(call_spark_stream(messages, model, max_tokens), media_type="text/event-stream")
    result = await call_spark(messages, model, max_tokens)
    return {
        "id": "chatcmpl-spark", "object": "chat.completion", "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": result}, "finish_reason": "stop"}],
    }


@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": m, "object": "model", "owned_by": "iflytek"} for m in MODEL_URLS]}


@app.get("/health")
async def health():
    return {"status": "ok", "app_id": APP_ID[:4] + "****"}


if __name__ == "__main__":
    print(f"Spark Proxy: http://localhost:8008/v1/chat/completions")
    print(f"APPID: {APP_ID[:4]}****")
    uvicorn.run(app, host="127.0.0.1", port=8008)
