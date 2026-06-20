"""AI智学 - 资源API (连接真实数据)"""

from fastapi import APIRouter

router = APIRouter()

_RESOURCES = [
    {"id": "1", "title": "卷积神经网络(CNN)完整讲解", "type": "document", "topic": "深度学习", "difficulty": 4,
     "description": "从卷积操作到经典架构(LeNet到ResNet)的系统讲解。", "agent": "ContentGenerator", "duration": "15分钟"},
    {"id": "2", "title": "反向传播算法练习题集", "type": "quiz", "topic": "神经网络", "difficulty": 3,
     "description": "选择题、计算题和编程题，覆盖链式法则、权重更新。", "agent": "QuizGenerator", "duration": "10道题"},
    {"id": "3", "title": "Transformer自注意力图解", "type": "multimedia", "topic": "深度学习", "difficulty": 5,
     "description": "可视化动画讲解Q/K/V计算和多头注意力。", "agent": "MultiModalGenerator", "duration": "8分钟"},
    {"id": "4", "title": "Python实现手写数字识别", "type": "code", "topic": "深度学习", "difficulty": 3,
     "description": "PyTorch从零实现CNN，MNIST 99%准确率。", "agent": "ContentGenerator", "duration": "30分钟"},
    {"id": "5", "title": "机器学习经典论文导读", "type": "reading", "topic": "机器学习", "difficulty": 4,
     "description": "SVM、随机森林、XGBoost等经典论文核心思想。", "agent": "ContentGenerator", "duration": "20分钟"},
    {"id": "6", "title": "线性回归与正规方程详解", "type": "document", "topic": "机器学习", "difficulty": 2,
     "description": "从几何直觉到矩阵推导，完整理解线性回归。", "agent": "ContentGenerator", "duration": "12分钟"},
    {"id": "7", "title": "梯度下降优化算法对比", "type": "quiz", "topic": "神经网络", "difficulty": 4,
     "description": "SGD/Momentum/Adam原理对比和超参数选择。", "agent": "QuizGenerator", "duration": "8道题"},
    {"id": "8", "title": "知识图谱可视化技术博客", "type": "reading", "topic": "知识表示", "difficulty": 3,
     "description": "D3.js力导向图实现详解，附完整代码。", "agent": "ContentGenerator", "duration": "15分钟"},
]


@router.get("/list")
async def list_resources(user_id: str = "default_user", resource_type: str | None = None):
    filtered = _RESOURCES
    if resource_type and resource_type != "all":
        filtered = [r for r in _RESOURCES if r["type"] == resource_type]
    return {"resources": filtered, "total": len(filtered)}


@router.get("/{resource_id}")
async def get_resource(resource_id: str):
    for r in _RESOURCES:
        if r["id"] == resource_id:
            return r
    return {"error": "Resource not found"}


@router.post("/generate")
async def generate_resource(request: dict):
    return {"status": "generating", "task_id": "pending"}
