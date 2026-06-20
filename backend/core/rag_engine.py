"""AI智学 - RAG检索增强生成引擎

基于TF-IDF的轻量级检索引擎，不依赖外部模型下载。
支持中文分词、语义检索、主题过滤。
"""

import re
import math
from collections import Counter
from pathlib import Path
from loguru import logger
from core.config import get_settings


def _tokenize(text: str) -> list[str]:
    """中文分词 (基于字符n-gram + 英文单词)"""
    # 英文单词
    words = re.findall(r'[a-zA-Z]+(?:\.[a-zA-Z]+)*', text.lower())
    # 中文字符bigram
    chinese = re.findall(r'[一-鿿]+', text)
    for seg in chinese:
        for i in range(len(seg)):
            words.append(seg[i])  # unigram
        for i in range(len(seg) - 1):
            words.append(seg[i:i+2])  # bigram
    return words


class RAGEngine:
    """轻量级RAG检索引擎 (TF-IDF + 倒排索引)"""

    def __init__(self):
        self._documents: list[dict] = []  # [{id, content, metadata}]
        self._idf: dict[str, float] = {}
        self._tfidf: list[dict[str, float]] = []
        self._load_persisted()

    def _load_persisted(self):
        """从磁盘加载已索引的文档"""
        persist_path = Path(get_settings().chroma_persist_dir) / "documents.json"
        if persist_path.exists():
            import json
            with open(persist_path, "r", encoding="utf-8") as f:
                self._documents = json.load(f)
            self._rebuild_index()
            logger.info(f"Loaded {len(self._documents)} documents from disk")

    def _persist(self):
        """持久化文档到磁盘"""
        persist_path = Path(get_settings().chroma_persist_dir) / "documents.json"
        persist_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(persist_path, "w", encoding="utf-8") as f:
            json.dump(self._documents, f, ensure_ascii=False)

    def add_documents(self, documents: list[str], metadatas: list[dict] | None = None, ids: list[str] | None = None):
        """添加文档"""
        import uuid
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        if metadatas is None:
            metadatas = [{}] * len(documents)

        for i, doc in enumerate(documents):
            self._documents.append({"id": ids[i], "content": doc, "metadata": metadatas[i]})

        self._rebuild_index()
        self._persist()
        logger.info(f"Added {len(documents)} documents, total: {len(self._documents)}")

    def _rebuild_index(self):
        """重建TF-IDF索引"""
        n = len(self._documents)
        if n == 0:
            return

        # 计算TF和DF
        df: dict[str, int] = Counter()
        doc_tokens = []
        for doc in self._documents:
            tokens = _tokenize(doc["content"])
            doc_tokens.append(Counter(tokens))
            for term in set(tokens):
                df[term] += 1

        # 计算IDF
        self._idf = {term: math.log(n / (1 + count)) for term, count in df.items()}

        # 计算每篇文档的TF-IDF向量
        self._tfidf = []
        for tokens in doc_tokens:
            total = sum(tokens.values()) or 1
            vec = {}
            for term, count in tokens.items():
                tf = count / total
                vec[term] = tf * self._idf.get(term, 0)
            self._tfidf.append(vec)

    def search(self, query: str, top_k: int = 5, where: dict | None = None) -> list[dict]:
        """语义检索"""
        if not self._documents:
            return []

        query_tokens = _tokenize(query)
        query_tf = Counter(query_tokens)
        query_total = sum(query_tf.values()) or 1

        # 构建查询向量
        query_vec = {}
        for term, count in query_tf.items():
            tf = count / query_total
            query_vec[term] = tf * self._idf.get(term, 0)

        # 计算余弦相似度
        query_norm = math.sqrt(sum(v * v for v in query_vec.values())) or 1
        scores = []
        for i, doc_vec in enumerate(self._tfidf):
            # 主题过滤
            if where:
                match = True
                for key, value in where.items():
                    if self._documents[i]["metadata"].get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            # 余弦相似度
            dot = sum(query_vec.get(t, 0) * doc_vec.get(t, 0) for t in query_vec)
            doc_norm = math.sqrt(sum(v * v for v in doc_vec.values())) or 1
            similarity = dot / (query_norm * doc_norm)
            scores.append((similarity, i))

        # 排序取top_k
        scores.sort(reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            if score > 0.01:
                results.append({
                    "content": self._documents[idx]["content"],
                    "metadata": self._documents[idx]["metadata"],
                    "score": round(score, 4),
                })
        return results

    def delete_by_topic(self, topic: str):
        self._documents = [d for d in self._documents if d["metadata"].get("topic") != topic]
        self._rebuild_index()
        self._persist()

    def get_document_count(self) -> int:
        return len(self._documents)


# 全局单例
_rag_engine: RAGEngine | None = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
