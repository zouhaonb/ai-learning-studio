"""AI智学 - 知识库索引脚本

将Markdown知识文档拆分为语义段落，索引到ChromaDB向量数据库。
使用方式: cd backend && python -m knowledge_base.index
"""

import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.rag_engine import get_rag_engine
from loguru import logger


def split_into_chunks(text: str) -> list[dict]:
    """按章节拆分为语义段落 (500-2000字符)"""
    chunks = []
    sections = re.split(r'\n(?=## )', text)

    for section in sections:
        section = section.strip()
        if not section or len(section) < 50:
            continue

        title_match = re.match(r'^#{1,3}\s+(.+)', section)
        title = title_match.group(1) if title_match else "未命名"
        topic = classify_topic(title, section)

        if len(section) > 2000:
            subsections = re.split(r'\n(?=### )', section)
            for sub in subsections:
                sub = sub.strip()
                if len(sub) < 50:
                    continue
                sub_title = re.match(r'^#{1,3}\s+(.+)', sub)
                chunks.append({
                    "content": sub,
                    "topic": topic,
                    "title": sub_title.group(1) if sub_title else title,
                    "difficulty": estimate_difficulty(sub),
                })
        else:
            chunks.append({
                "content": section, "topic": topic, "title": title,
                "difficulty": estimate_difficulty(section),
            })
    return chunks


def classify_topic(title: str, content: str) -> str:
    title_lower = title.lower()
    content_lower = content[:500].lower()
    keywords_map = {
        "ai_overview": ["概述", "历史", "定义", "图灵", "流派"],
        "search": ["搜索", "bfs", "dfs", "a*", "minimax", "alpha-beta"],
        "knowledge": ["知识表示", "逻辑", "命题", "谓词", "贝叶斯网络"],
        "ml_basics": ["线性回归", "逻辑回归", "决策树", "svm", "朴素贝叶斯", "k-means", "pca"],
        "neural_network": ["神经元", "感知机", "反向传播", "梯度下降", "激活函数", "adam"],
        "deep_learning": ["卷积", "cnn", "rnn", "lstm", "transformer", "注意力", "resnet"],
        "nlp": ["词嵌入", "word2vec", "文本分类", "bert", "gpt", "语言模型"],
        "cv": ["图像分类", "目标检测", "yolo", "rcnn", "gan", "生成模型"],
    }
    for topic, kws in keywords_map.items():
        for kw in kws:
            if kw in title_lower or kw in content_lower:
                return topic
    return "general"


def estimate_difficulty(content: str) -> int:
    signals = {
        5: ["transformer", "注意力机制", "gan", "大语言模型"],
        4: ["svm", "反向传播", "cnn", "rnn", "lstm", "核函数"],
        3: ["a*", "决策树", "梯度下降", "pca", "k-means"],
        2: ["线性回归", "逻辑回归", "感知机", "bfs", "dfs"],
        1: ["概述", "历史", "定义", "图灵测试"],
    }
    cl = content.lower()
    for level in [5, 4, 3, 2, 1]:
        for s in signals[level]:
            if s in cl:
                return level
    return 3


def index_knowledge_base():
    docs_dir = Path(__file__).parent / "documents"
    all_chunks = []

    for fpath in sorted(docs_dir.glob("*")):
        if fpath.suffix not in (".md", ".txt"):
            continue
        logger.info(f"Reading: {fpath.name}")
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = split_into_chunks(content)
        logger.info(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    logger.info(f"Total: {len(all_chunks)} chunks")

    topic_counts = {}
    for c in all_chunks:
        topic_counts[c["topic"]] = topic_counts.get(c["topic"], 0) + 1
    for t, n in sorted(topic_counts.items()):
        logger.info(f"  {t}: {n} chunks")

    rag = get_rag_engine()
    rag.add_documents(
        documents=[c["content"] for c in all_chunks],
        metadatas=[{"topic": c["topic"], "title": c["title"], "difficulty": c["difficulty"]} for c in all_chunks],
        ids=[str(uuid.uuid4()) for _ in all_chunks],
    )
    logger.info(f"Indexed! Total docs: {rag.get_document_count()}")

    for q in ["卷积神经网络", "反向传播公式", "Transformer注意力", "决策树特征选择", "GAN损失函数", "BERT预训练"]:
        results = rag.search(q, top_k=2)
        logger.info(f"Q: {q}")
        for r in results:
            logger.info(f"  [{r['score']:.3f}] {r['metadata'].get('title', '')}")


if __name__ == "__main__":
    index_knowledge_base()
