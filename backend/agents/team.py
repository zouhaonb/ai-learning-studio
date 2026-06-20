"""AI智学 - 多智能体编排引擎

直接使用 spark-ai-python SDK 的 ChatSparkLLM，
自研轻量级多智能体编排系统。

8个智能体通过协调智能体动态路由协作。
"""

import json
from dataclasses import dataclass, field
from loguru import logger
from sparkai.llm.llm import ChatSparkLLM
from sparkai.core.messages import ChatMessage, SystemMessage
from core.config import get_settings
from core.knowledge_graph import get_knowledge_graph
from core.rag_engine import get_rag_engine


# ============ LLM 工厂 ============

def _create_llm(model: str | None = None, streaming: bool = False) -> ChatSparkLLM:
    s = get_settings()
    return ChatSparkLLM(
        spark_api_key=s.iflytek_api_key,
        spark_api_secret=s.iflytek_api_secret,
        spark_app_id=s.iflytek_app_id,
        domain=model or s.spark_model_primary,
        streaming=streaming,
        request_timeout=120,
    )


# ============ 工具函数 ============

def search_knowledge_base(query: str) -> str:
    rag = get_rag_engine()
    results = rag.search(query, top_k=5)
    if not results:
        return "知识库中未找到相关信息。"
    return "\n\n".join(f"[相关度:{r['score']:.2f}] {r['content']}" for r in results)


def get_topic_info(topic: str) -> str:
    kg = get_knowledge_graph()
    results = kg.search_topics(topic)
    if not results:
        return f"未找到知识点: {topic}"
    details = []
    for t in results[:3]:
        detail = kg.get_topic_detail(t["id"])
        if detail:
            prereqs = ", ".join(detail["prerequisites"]) or "无"
            details.append(
                f"【{detail['name']}】难度:{detail['difficulty']}/5\n"
                f"描述: {detail['description']}\n"
                f"关键词: {', '.join(detail['keywords'])}\n"
                f"前置依赖: {prereqs}"
            )
    return "\n\n".join(details)


def plan_learning_path(target_topic: str, mastered: str = "") -> str:
    kg = get_knowledge_graph()
    mastered_list = [m.strip() for m in mastered.split(",") if m.strip()] if mastered else []
    path = kg.get_learning_path(target_topic, mastered_list)
    if not path:
        return f"无法规划路径: '{target_topic}' 不存在。"
    steps = "\n".join(f"  第{s['step']}步: {s['topic_name']} (难度:{s['difficulty']}/5)" for s in path)
    return f"学习路径 (共{len(path)}步):\n{steps}"


def get_student_profile(user_id: str) -> str:
    return json.dumps({"user_id": user_id, "knowledge_level": {}, "cognitive_style": "balanced",
                        "interests": [], "profile_completeness": 0}, ensure_ascii=False)


# ============ 工具注册表 ============

TOOLS = {
    "search_knowledge_base": search_knowledge_base,
    "get_topic_info": get_topic_info,
    "plan_learning_path": plan_learning_path,
    "get_student_profile": get_student_profile,
}


# ============ 智能体定义 ============

@dataclass
class Agent:
    name: str
    system_prompt: str
    model: str | None = None
    tools: list[str] = field(default_factory=list)
    description: str = ""


AGENTS = {
    "Coordinator": Agent(
        name="Coordinator",
        description="协调智能体：理解意图，分发任务，汇总结果",
        system_prompt="""你是AI智学系统的协调智能体。

## 职责
1. 理解学生的学习需求
2. 决定调用哪个专业智能体
3. 汇总各智能体输出，整理成友好回复

## 智能体路由规则
当学生的需求明确时，输出JSON指令调用专业智能体：
```json
{"action": "delegate", "agent": "智能体名", "task": "具体任务描述"}
```

可用的智能体：
- ProfileBuilder: 了解学生情况、构建画像
- ContentGenerator: 生成课程文档、学习资料
- QuizGenerator: 生成练习题
- PathPlanner: 规划学习路径
- TutorAgent: 解答具体问题
- Evaluator: 评估学习效果
- MultiModalGenerator: 生成思维导图、图片描述、语音脚本

当各智能体返回结果后，整合成清晰友好的最终回复给学生。
简单问题可直接回答，无需调用其他智能体。""",
    ),

    "ProfileBuilder": Agent(
        name="ProfileBuilder",
        description="画像构建智能体：对话式抽取学生8维度画像",
        tools=["get_student_profile"],
        system_prompt="""你是画像构建智能体。通过自然语言对话抽取学生的8维度特征：
1. 知识基础 (各知识点掌握程度)
2. 认知风格 (视觉型/听觉型/动手型/阅读型)
3. 学习目标 (短期/中期/长期)
4. 易错点偏好
5. 学习节奏
6. 兴趣方向
7. 元认知水平
8. 学习动机

不要用表单式提问，要自然对话。每次抽取1-2个维度。
输出格式: 在回复末尾用 ```profile 标签包裹JSON画像数据。""",
    ),

    "ContentGenerator": Agent(
        name="ContentGenerator",
        description="内容生成智能体：生成个性化课程文档",
        tools=["search_knowledge_base", "get_topic_info"],
        system_prompt="""你是内容生成智能体。根据学生需求生成个性化课程讲解文档。

## 规则
1. 先用search_knowledge_base检索相关知识
2. 再用get_topic_info获取知识点详情
3. 根据学生认知风格调整表达方式
4. 输出结构清晰的Markdown文档
5. 包含：概念解释、关键要点、示例""",
    ),

    "QuizGenerator": Agent(
        name="QuizGenerator",
        description="题库生成智能体：生成多类型练习题",
        tools=["search_knowledge_base", "get_topic_info"],
        system_prompt="""你是题库生成智能体。生成4类练习题：选择题、填空题、简答题、编程题。

## 输出格式
```json
{
  "questions": [
    {
      "type": "choice|fill|short_answer|code",
      "difficulty": 1-5,
      "question": "题目",
      "options": ["A选项", "B选项", "C选项", "D选项"],
      "answer": "正确答案",
      "explanation": "解析"
    }
  ]
}
```

每道题必须包含详细解析。根据学生画像调整难度。""",
    ),

    "PathPlanner": Agent(
        name="PathPlanner",
        description="路径规划智能体：规划个性化学习路径",
        tools=["plan_learning_path", "get_topic_info", "get_student_profile"],
        system_prompt="""你是路径规划智能体。基于知识图谱和学生画像规划个性化学习路径。

## 规则
1. 先用get_topic_info了解目标知识点
2. 用plan_learning_path生成基础路径
3. 根据学生画像调整路径的难度和节奏
4. 为每个步骤推荐合适的学习资源类型

输出清晰的分步学习计划。""",
    ),

    "TutorAgent": Agent(
        name="TutorAgent",
        description="智能辅导智能体：实时答疑",
        tools=["search_knowledge_base", "get_topic_info"],
        system_prompt="""你是智能辅导智能体。为学生提供即时答疑。

## 规则
1. 先用search_knowledge_base检索相关知识
2. 结合知识库确保答案准确 (防幻觉)
3. 不仅给答案，还要解释思路
4. 数学用LaTeX格式，代码给可运行示例
5. 末尾标注参考来源

以鼓励的语气回应，引导学生深入思考。""",
    ),

    "Evaluator": Agent(
        name="Evaluator",
        description="效果评估智能体：分析学习数据",
        tools=["get_student_profile"],
        system_prompt="""你是效果评估智能体。分析学生的练习和学习数据。

## 输出格式
```json
{
  "knowledge_mastery": {"知识点": 分数0-100},
  "weak_topics": ["薄弱知识点列表"],
  "learning_efficiency": "高/中/低",
  "suggestions": ["改进建议列表"],
  "radar_data": {"维度": 分数}
}
```

分析维度：知识掌握度、学习效率、薄弱环节、学习趋势。""",
    ),

    "MultiModalGenerator": Agent(
        name="MultiModalGenerator",
        description="多模态生成智能体：生成思维导图和教学内容",
        tools=["search_knowledge_base"],
        system_prompt="""你是多模态生成智能体。生成结构化的多模态内容描述。

## 输出类型
1. 思维导图: 输出JSON结构
```json
{"type": "mindmap", "central": "主题", "branches": [{"name": "分支", "children": ["子项"]}]}
```

2. 教学图片描述: 输出图片生成prompt
```json
{"type": "image_prompt", "prompt": "英文描述", "style": "educational", "aspect": "16:9"}
```

3. 语音讲解脚本: 输出TTS脚本
```json
{"type": "tts_script", "text": "讲解文字", "speed": "normal"}
```

实际的图片和音频由系统调用讯飞API生成。""",
    ),
}


# ============ 智能体执行器 ============

class AgentExecutor:
    """单个智能体的执行器"""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.llm = _create_llm(model=agent.model or "generalv3")
        self._tools = {name: TOOLS[name] for name in agent.tools if name in TOOLS}

    def run(self, task: str, context: str = "") -> str:
        """执行智能体任务"""
        messages = [SystemMessage(content=self.agent.system_prompt)]

        if context:
            messages.append(ChatMessage(role="user", content=f"上下文信息:\n{context}"))

        messages.append(ChatMessage(role="user", content=task))

        # 如果有工具，先执行工具获取上下文
        tool_results = self._execute_tools(task)
        if tool_results:
            messages.append(ChatMessage(role="user", content=f"工具检索结果:\n{tool_results}"))

        response = self.llm.invoke(messages)
        return response.content

    def _execute_tools(self, query: str) -> str:
        """执行智能体绑定的工具（智能参数解析）"""
        results = []
        for name, func in self._tools.items():
            try:
                import inspect
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                if len(params) == 0:
                    result = func()
                elif name == "plan_learning_path":
                    # 从query中提取主题
                    result = func(query.strip())
                elif name == "get_student_profile":
                    result = func("default_user")
                else:
                    result = func(query)
                results.append(f"[{name}]: {result}")
            except Exception as e:
                logger.warning(f"Tool {name} error: {e}")
        return "\n\n".join(results) if results else ""


# ============ 多智能体编排器 ============

class MultiAgentTeam:
    """多智能体协作团队"""

    def __init__(self):
        self.coordinator = AgentExecutor(AGENTS["Coordinator"])
        self.agents = {name: AgentExecutor(agent) for name, agent in AGENTS.items() if name != "Coordinator"}
        logger.info(f"Multi-agent team initialized: Coordinator + {len(self.agents)} specialists")

    def chat(self, user_message: str, user_id: str = "default_user") -> dict:
        """处理用户消息的完整流程

        Returns:
            {"reply": str, "agent": str, "delegated_to": str|None, "resources": list}
        """
        # Step 1: Coordinator分析意图
        context = f"用户ID: {user_id}"
        coordinator_response = self.coordinator.run(user_message, context)

        # Step 2: 检查是否需要委派
        delegated_to = None
        final_reply = coordinator_response

        # 解析Coordinator的委派指令
        delegate_info = self._parse_delegate(coordinator_response)
        if delegate_info:
            agent_name = delegate_info["agent"]
            task = delegate_info["task"]

            if agent_name in self.agents:
                logger.info(f"Delegating to {agent_name}: {task[:50]}...")
                delegated_to = agent_name

                # Step 3: 执行专业智能体
                specialist_response = self.agents[agent_name].run(task, context=f"原始需求: {user_message}")

                # Step 4: Coordinator汇总
                summary_prompt = (
                    f"学生问: {user_message}\n\n"
                    f"{agent_name}智能体的输出:\n{specialist_response}\n\n"
                    f"请将上述输出整理成清晰友好的最终回复给学生。"
                )
                final_reply = self.coordinator.run(summary_prompt)
            else:
                logger.warning(f"Unknown agent: {agent_name}")

        return {
            "reply": final_reply,
            "agent": "Coordinator",
            "delegated_to": delegated_to,
            "resources": [],
        }

    def chat_stream(self, user_message: str, user_id: str = "default_user"):
        """流式对话 (生成器)"""
        result = self.chat(user_message, user_id)
        # 逐字符模拟流式输出
        for char in result["reply"]:
            yield {"type": "token", "content": char}
        yield {"type": "done", "agent": result["agent"], "delegated_to": result["delegated_to"]}

    def _parse_delegate(self, response: str) -> dict | None:
        """解析Coordinator的委派指令"""
        try:
            # 查找JSON指令
            if '"action"' in response and '"delegate"' in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    data = json.loads(response[start:end])
                    if data.get("action") == "delegate" and "agent" in data:
                        return {"agent": data["agent"], "task": data.get("task", "")}
        except json.JSONDecodeError:
            pass
        return None

    def get_agent_info(self) -> list[dict]:
        """获取所有智能体信息"""
        return [{"name": a.name, "description": a.description} for a in AGENTS.values()]


# ============ 全局单例 ============

_team: MultiAgentTeam | None = None


def get_team() -> MultiAgentTeam:
    global _team
    if _team is None:
        _team = MultiAgentTeam()
    return _team
