"""AI智学 - 资源API (连接真实数据)"""

from fastapi import APIRouter

router = APIRouter()

_RESOURCES = [
    {"id": "1", "title": "卷积神经网络(CNN)完整讲解", "type": "document", "topic": "深度学习", "difficulty": 4,
     "description": "从卷积操作到经典架构(LeNet到ResNet)的系统讲解。", "agent": "ContentGenerator", "duration": "15分钟",
     "topic_ids": ["cnn"]},
    {"id": "2", "title": "反向传播算法练习题集", "type": "quiz", "topic": "神经网络", "difficulty": 3,
     "description": "选择题、计算题和编程题，覆盖链式法则、权重更新。", "agent": "QuizGenerator", "duration": "10道题",
     "topic_ids": ["backpropagation"]},
    {"id": "3", "title": "Transformer自注意力图解", "type": "multimedia", "topic": "深度学习", "difficulty": 5,
     "description": "可视化动画讲解Q/K/V计算和多头注意力。", "agent": "MultiModalGenerator", "duration": "8分钟",
     "topic_ids": ["transformer"]},
    {"id": "4", "title": "Python实现手写数字识别", "type": "code", "topic": "深度学习", "difficulty": 3,
     "description": "PyTorch从零实现CNN，MNIST 99%准确率。", "agent": "ContentGenerator", "duration": "30分钟",
     "topic_ids": ["cnn", "image_classification"]},
    {"id": "5", "title": "机器学习经典论文导读", "type": "reading", "topic": "机器学习", "difficulty": 4,
     "description": "SVM、随机森林、XGBoost等经典论文核心思想。", "agent": "ContentGenerator", "duration": "20分钟",
     "topic_ids": ["ml_overview", "svm", "decision_tree"]},
    {"id": "6", "title": "线性回归与正规方程详解", "type": "document", "topic": "机器学习", "difficulty": 2,
     "description": "从几何直觉到矩阵推导，完整理解线性回归。", "agent": "ContentGenerator", "duration": "12分钟",
     "topic_ids": ["linear_regression"]},
    {"id": "7", "title": "梯度下降优化算法对比", "type": "quiz", "topic": "神经网络", "difficulty": 4,
     "description": "SGD/Momentum/Adam原理对比和超参数选择。", "agent": "QuizGenerator", "duration": "8道题",
     "topic_ids": ["optimization", "backpropagation"]},
    {"id": "8", "title": "知识图谱可视化技术博客", "type": "reading", "topic": "知识表示", "difficulty": 3,
     "description": "D3.js力导向图实现详解，附完整代码。", "agent": "ContentGenerator", "duration": "15分钟",
     "topic_ids": ["knowledge_overview"]},
    {"id": "9", "title": "搜索算法练习题", "type": "quiz", "topic": "搜索算法", "difficulty": 2,
     "description": "BFS、DFS、A*等搜索算法练习题。", "agent": "QuizGenerator", "duration": "10道题",
     "topic_ids": ["uninformed_search", "informed_search"]},
    {"id": "10", "title": "对抗搜索实战(Minimax)", "type": "code", "topic": "搜索算法", "difficulty": 4,
     "description": "Minimax算法实现井字棋AI。", "agent": "ContentGenerator", "duration": "25分钟",
     "topic_ids": ["adversarial_search"]},
    {"id": "11", "title": "概率推理与贝叶斯网络", "type": "document", "topic": "概率推理", "difficulty": 4,
     "description": "贝叶斯网络基础和概率推理方法。", "agent": "ContentGenerator", "duration": "18分钟",
     "topic_ids": ["probabilistic"]},
    {"id": "12", "title": "逻辑推理入门指南", "type": "document", "topic": "逻辑推理", "difficulty": 2,
     "description": "命题逻辑和谓词逻辑基础。", "agent": "ContentGenerator", "duration": "12分钟",
     "topic_ids": ["logic"]},
    {"id": "13", "title": "无监督学习：K-Means与PCA", "type": "document", "topic": "机器学习", "difficulty": 3,
     "description": "聚类和降维算法详解。", "agent": "ContentGenerator", "duration": "15分钟",
     "topic_ids": ["unsupervised"]},
    {"id": "14", "title": "词嵌入技术对比", "type": "multimedia", "topic": "自然语言处理", "difficulty": 3,
     "description": "Word2Vec、GloVe、FastText可视化对比。", "agent": "MultiModalGenerator", "duration": "10分钟",
     "topic_ids": ["word_embedding"]},
    {"id": "15", "title": "文本分类实战", "type": "code", "topic": "自然语言处理", "difficulty": 3,
     "description": "使用LSTM实现文本分类。", "agent": "ContentGenerator", "duration": "20分钟",
     "topic_ids": ["text_classification"]},
    {"id": "16", "title": "RNN/LSTM序列建模", "type": "reading", "topic": "深度学习", "difficulty": 4,
     "description": "循环神经网络和长短期记忆网络详解。", "agent": "ContentGenerator", "duration": "18分钟",
     "topic_ids": ["rnn_lstm"]},
    {"id": "17", "title": "目标检测从R-CNN到YOLO", "type": "document", "topic": "计算机视觉", "difficulty": 5,
     "description": "目标检测算法演进详解。", "agent": "ContentGenerator", "duration": "22分钟",
     "topic_ids": ["object_detection"]},
    {"id": "18", "title": "生成模型(GAN/VAE/Diffusion)", "type": "multimedia", "topic": "深度学习", "difficulty": 5,
     "description": "生成对抗网络、变分自编码器和扩散模型。", "agent": "MultiModalGenerator", "duration": "15分钟",
     "topic_ids": ["generation"]},
    {"id": "19", "title": "逻辑回归与Sigmoid详解", "type": "document", "topic": "机器学习", "difficulty": 2,
     "description": "逻辑回归原理和实现。", "agent": "ContentGenerator", "duration": "10分钟",
     "topic_ids": ["logistic_regression"]},
    {"id": "20", "title": "决策树与随机森林", "type": "code", "topic": "机器学习", "difficulty": 3,
     "description": "决策树和随机森林算法实现。", "agent": "ContentGenerator", "duration": "20分钟",
     "topic_ids": ["decision_tree"]},
    {"id": "21", "title": "朴素贝叶斯分类器", "type": "quiz", "topic": "机器学习", "difficulty": 2,
     "description": "朴素贝叶斯算法练习题。", "agent": "QuizGenerator", "duration": "8道题",
     "topic_ids": ["naive_bayes"]},
    {"id": "22", "title": "SVM支持向量机详解", "type": "document", "topic": "机器学习", "difficulty": 4,
     "description": "支持向量机原理和核函数。", "agent": "ContentGenerator", "duration": "16分钟",
     "topic_ids": ["svm"]},
    {"id": "23", "title": "神经元与感知机基础", "type": "document", "topic": "神经网络", "difficulty": 1,
     "description": "神经网络基础概念。", "agent": "ContentGenerator", "duration": "8分钟",
     "topic_ids": ["neuron"]},
    {"id": "24", "title": "大语言模型(LLM)前沿", "type": "reading", "topic": "深度学习", "difficulty": 5,
     "description": "GPT、LLaMA等大语言模型技术解析。", "agent": "ContentGenerator", "duration": "25分钟",
     "topic_ids": ["llm"]},
]


@router.get("/list")
async def list_resources(user_id: str = "default_user", resource_type: str | None = None):
    filtered = _RESOURCES
    if resource_type and resource_type != "all":
        filtered = [r for r in _RESOURCES if r["type"] == resource_type]
    return {"resources": filtered, "total": len(filtered)}


@router.get("/by-topic/{topic_id}")
async def get_resources_by_topic(topic_id: str):
    """Get resources related to a specific knowledge graph topic"""
    matching = [r for r in _RESOURCES if topic_id in r.get("topic_ids", [])]
    return {"resources": matching, "total": len(matching), "topic_id": topic_id}


@router.get("/recommendations/{user_id}")
async def get_recommendations(user_id: str):
    """Get resource recommendations for weak topics"""
    from api.progress import _get_user_progress
    progress = _get_user_progress(user_id)

    # Find weak topics (score < 50, attempted at least once)
    weak_topic_ids = [
        tid for tid, data in progress.items()
        if data.get("attempts", 0) > 0 and data.get("best_score", 0) < 50
    ]

    # Find resources matching weak topics
    recommendations = []
    seen_ids = set()
    for topic_id in weak_topic_ids:
        matching = [r for r in _RESOURCES if topic_id in r.get("topic_ids", [])]
        for r in matching:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                recommendations.append({**r, "weak_topic_id": topic_id})

    return {
        "recommendations": recommendations,
        "weak_topics": weak_topic_ids,
        "total": len(recommendations)
    }


@router.get("/by-learning-path")
async def get_resources_by_learning_path(user_id: str = "default_user"):
    """Get resources organized by learning path level"""
    from core.knowledge_graph import get_knowledge_graph
    kg = get_knowledge_graph()
    all_topics = kg.get_all_topics()

    # Group resources by level
    level_resources = {}
    seen_ids = set()
    for topic in all_topics:
        level = topic["level"]
        matching = [r for r in _RESOURCES if topic["id"] in r.get("topic_ids", [])]
        if matching:
            if level not in level_resources:
                level_resources[level] = []
            for r in matching:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    entry = {**r, "topic_id": topic["id"], "topic_name": topic["name"]}
                    level_resources[level].append(entry)

    return {"levels": level_resources, "total": sum(len(v) for v in level_resources.values())}


@router.get("/{resource_id}")
async def get_resource(resource_id: str):
    for r in _RESOURCES:
        if r["id"] == resource_id:
            return r
    return {"error": "Resource not found"}


@router.post("/generate")
async def generate_resource(request: dict):
    return {"status": "generating", "task_id": "pending"}
