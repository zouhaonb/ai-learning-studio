# 🎓 AI智学 — 个性化学习多智能体系统

> 第十五届中国软件杯 A组参赛作品
> 基于科大讯飞星火大模型 + AutoGen多智能体框架

## 系统简介

AI智学通过8个专业智能体的协作，为学生提供个性化的学习体验。

### 核心功能

| 功能 | 描述 | 智能体 |
|------|------|--------|
| 对话式画像构建 | 8维度动态学生画像 | ProfileBuilder |
| 多智能体资源生成 | 6种类型学习资源 | ContentGenerator/QuizGenerator/MultiModalGenerator |
| 学习路径规划 | 知识图谱驱动 | PathPlanner |
| 智能辅导 | 多模态答疑 | TutorAgent |
| 效果评估 | 多维评估+动态调整 | Evaluator |

### 技术栈

- **前端**: Next.js 14 + React 18 + TypeScript + Shadcn/ui
- **后端**: Python 3.11 + FastAPI
- **智能体**: AutoGen 0.7.x
- **大模型**: 科大讯飞星火 4.0 Ultra
- **向量数据库**: ChromaDB
- **知识图谱**: NetworkX + D3.js
- **数据库**: SQLite + SQLAlchemy

## 快速开始

```bash
# 后端
cd backend && pip install -r requirements.txt
cp .env.example .env  # 填入讯飞API Key
python main.py

# 前端
cd frontend && npm install && npm run dev
```

## 讯飞API集成

| 服务 | 用途 |
|------|------|
| 星火4.0 Ultra | 核心推理、内容生成 |
| 星火Lite | 意图分类 |
| 讯飞文生图 | 思维导图/示意图 |
| 讯飞TTS | 语音讲解 |
| 讯飞ASR | 语音输入 |
