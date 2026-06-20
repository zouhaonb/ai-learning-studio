# 人工智能导论 - 竞赛知识库

> 综合 Stanford CS229/CS231n/CS224n、MIT 6.034、Andrew Ng 机器学习课程及国内《人工智能导论》教材整理

---

## 第一部分：人工智能概述与历史

### 1.1 人工智能的定义

人工智能（Artificial Intelligence, AI）是计算机科学的一个分支，旨在创建能够模拟人类智能行为的系统，包括学习、推理、问题求解、感知和语言理解。

**Stuart Russell & Peter Norvig 的四种定义分类：**

| 类别 | 类人思维 | 理性思维 | 类人行为 | 理性行为 |
|------|----------|----------|----------|----------|
| 思考 | 像人一样思考 | 像理性主体一样思考 | | |
| 行为 | | | 像人一样行动 | 像理性主体一样行动 |
| 代表 | 认知建模 | 逻辑推理 | 图灵测试 | 理性主体 |
| 代表人物 | Newell & Simon | 逻辑学家 | Turing | Russell & Norvig |

### 1.2 图灵测试（Turing Test）

- **提出者**：Alan Turing，1950年论文《Computing Machinery and Intelligence》
- **定义**：如果人类评估者通过自然语言对话无法可靠地区分机器与人类，则该机器通过图灵测试
- **原名**："模仿游戏"（The Imitation Game）
- **三个参与者**：人类评估者（C）、人类（B）、机器（A）
- **评价标准**：机器能否在对话中骗过评估者
- **局限性**：
  - 仅测试行为表现，不关注内部机制
  - 依赖语言能力，忽略了感知和运动智能
  - 可能通过欺骗而非真正智能通过测试

### 1.3 AI 发展历史（重要里程碑）

| 年份 | 事件 |
|------|------|
| 1943 | McCulloch & Pitts 提出第一个人工神经元模型（M-P模型） |
| 1950 | Turing 发表《Computing Machinery and Intelligence》 |
| 1956 | **Dartmouth 会议**——AI 作为学科正式诞生（John McCarthy 命名） |
| 1957 | Frank Rosenblatt 发明感知机（Perceptron） |
| 1966 | ELIZA 聊天程序（Joseph Weizenbaum） |
| 1969 | Minsky & Papert 指出感知机局限性，第一次AI寒冬开始 |
| 1974-1980 | 第一次AI寒冬 |
| 1980s | 专家系统兴起（如 MYCIN 医疗诊断） |
| 1986 | Rumelhart, Hinton, Williams 提出反向传播算法（BP算法） |
| 1987-1993 | 第二次AI寒冬 |
| 1997 | IBM Deep Blue 击败国际象棋世界冠军 Kasparov |
| 2006 | Hinton 提出深度信念网络，开启深度学习复兴 |
| 2012 | AlexNet 在 ImageNet 竞赛中大幅领先，深度学习爆发 |
| 2016 | AlphaGo 击败围棋世界冠军李世石 |
| 2017 | Vaswani 等提出 Transformer 架构（"Attention Is All You Need"） |
| 2020 | GPT-3 发布，大语言模型时代开启 |
| 2022 | ChatGPT 发布，生成式AI进入主流 |

### 1.4 AI 的主要流派

| 流派 | 核心思想 | 代表方法 |
|------|----------|----------|
| 符号主义（Symbolism） | 智能基于符号推理 | 专家系统、逻辑编程 |
| 连接主义（Connectionism） | 智能源于神经网络连接 | 神经网络、深度学习 |
| 行为主义（Actionism） | 智能通过与环境交互产生 | 强化学习、进化算法 |

---

## 第二部分：搜索算法

### 2.1 搜索问题的基本概念

- **状态空间**：所有可能状态的集合
- **初始状态**（Initial State）
- **动作**（Actions）：给定状态s下可用的行动集合
- **转移模型**（Transition Model）：执行动作后的结果状态
- **目标测试**（Goal Test）：判断是否达到目标状态
- **路径代价**（Path Cost）：从初始状态到当前状态的总代价
- **启发函数** h(n)：从节点n到目标的估计代价

### 2.2 无信息搜索（Uninformed Search）

#### 2.2.1 广度优先搜索（BFS）

**定义**：逐层扩展节点，先访问所有深度为d的节点，再访问深度为d+1的节点。

**伪代码：**
```
function BFS(problem):
    node ← Node(problem.initial)
    if problem.is_goal(node.state): return node
    frontier ← Queue()  // FIFO队列
    frontier.add(node)
    explored ← Set()
    while not frontier.is_empty():
        node ← frontier.pop()
        explored.add(node.state)
        for each action in problem.actions(node.state):
            child ← child_node(problem, node, action)
            if child.state not in explored and child.state not in frontier:
                if problem.is_goal(child.state): return child
                frontier.add(child)
    return failure
```

**性质：**
- **完备性**：是（若有有限分支因子b和有限深度d）
- **最优性**：是（当所有边代价相同时）
- **时间复杂度**：O(b^d)
- **空间复杂度**：O(b^d)（需存储所有已生成节点）

#### 2.2.2 深度优先搜索（DFS）

**定义**：优先扩展最深层节点，回溯到最近的未扩展节点。

**伪代码：**
```
function DFS(problem):
    node ← Node(problem.initial)
    frontier ← Stack()  // LIFO栈
    frontier.add(node)
    explored ← Set()
    while not frontier.is_empty():
        node ← frontier.pop()
        if problem.is_goal(node.state): return node
        explored.add(node.state)
        for each action in problem.actions(node.state):
            child ← child_node(problem, node, action)
            if child.state not in explored:
                frontier.add(child)
    return failure
```

**性质：**
- **完备性**：否（可能陷入无限分支）
- **最优性**：否
- **时间复杂度**：O(b^m)（m为最大搜索深度）
- **空间复杂度**：O(bm)（仅需存储当前路径）

#### 2.2.3 BFS vs DFS 对比

| 特性 | BFS | DFS |
|------|-----|-----|
| 数据结构 | 队列 | 栈 |
| 完备性 | 是 | 否 |
| 最优性 | 是（等代价） | 否 |
| 时间 | O(b^d) | O(b^m) |
| 空间 | O(b^d) | O(bm) |

### 2.3 有信息搜索（Informed Search）

#### 2.3.1 A* 搜索算法

**核心思想**：结合实际代价和启发式估计，选择最有希望的节点扩展。

**评价函数：**
```
f(n) = g(n) + h(n)
```
- g(n)：从初始状态到节点n的**实际代价**
- h(n)：从节点n到目标状态的**启发式估计代价**
- f(n)：经过节点n到达目标的**总估计代价**

**伪代码：**
```
function A_Star(problem, h):
    node ← Node(problem.initial)
    node.f ← h(node)
    frontier ← PriorityQueue()  // 按f(n)排序
    frontier.add(node)
    explored ← Set()
    while not frontier.is_empty():
        node ← frontier.pop()  // 取f值最小的节点
        if problem.is_goal(node.state): return node
        explored.add(node.state)
        for each action in problem.actions(node.state):
            child ← child_node(problem, node, action)
            child.f ← child.g + h(child)
            if child.state not in explored and child.state not in frontier:
                frontier.add(child)
            else if child.state in frontier and child.f < frontier[child].f:
                frontier.replace(child)  // 更新为更小的f值
    return failure
```

**关键概念——可采纳启发函数（Admissible Heuristic）：**
- 对所有节点n，h(n) <= h*(n)，其中h*(n)是n到目标的真实最小代价
- 即h(n)**从不高估**实际代价
- **定理**：若h是可采纳的，则A*搜索是最优的

**一致性（Consistency/Monotonicity）：**
- 对所有节点n和后继节点n'：h(n) <= c(n,n') + h(n')
- 三角不等式：直接估计不超过经过中间节点的估计
- 一致性蕴含可采纳性

**常见启发函数：**
- 曼哈顿距离：h(n) = |x₁-x₂| + |y₁-y₂|
- 欧几里得距离：h(n) = √((x₁-x₂)² + (y₁-y₂)²)

### 2.4 博弈搜索

#### 2.4.1 Minimax 算法

**适用场景**：两人零和博弈（一方获利等于另一方损失）

**核心思想**：
- MAX 玩家试图最大化评估值
- MIN 玩家试图最小化评估值
- 双方都按最优策略行动

**伪代码：**
```
function Minimax(state):
    return Max_Value(state)

function Max_Value(state):
    if is_terminal(state): return utility(state)
    v ← -∞
    for each action in actions(state):
        v ← max(v, Min_Value(result(state, action)))
    return v

function Min_Value(state):
    if is_terminal(state): return utility(state)
    v ← +∞
    for each action in actions(state):
        v ← min(v, Max_Value(result(state, action)))
    return v
```

**性质：**
- 在完备信息的两人零和博弈中，Minimax 产生最优策略
- 时间复杂度：O(b^m)
- 空间复杂度：O(bm)

#### 2.4.2 Alpha-Beta 剪枝

**核心思想**：在 Minimax 基础上，通过维护两个边界值 α 和 β 来剪去不影响最终决策的分支。

- **α**：MAX 玩家目前已保证的最好值（下界）
- **β**：MIN 玩家目前已保证的最好值（上界）
- 当 α ≥ β 时，剪去当前分支

**伪代码：**
```
function Alpha_Beta(state):
    return Max_Value(state, -∞, +∞)

function Max_Value(state, α, β):
    if is_terminal(state): return utility(state)
    v ← -∞
    for each action in actions(state):
        v ← max(v, Min_Value(result(state, action), α, β))
        if v ≥ β: return v  // β剪枝
        α ← max(α, v)
    return v

function Min_Value(state, α, β):
    if is_terminal(state): return utility(state)
    v ← +∞
    for each action in actions(state):
        v ← min(v, Max_Value(result(state, action), α, β))
        if v ≤ α: return v  // α剪枝
        β ← min(β, v)
    return v
```

**优化效果：**
- 最优情况下时间复杂度降为 O(b^(m/2))
- 节点排序良好时等效于将搜索树宽度开方

### 2.5 搜索算法常见考题与易错点

**常见考题：**
1. 给定搜索树，手动执行 BFS/DFS/A* 搜索过程
2. 判断启发函数是否可采纳/一致
3. 给定博弈树，执行 Minimax 和 Alpha-Beta 剪枝

**易错点：**
- A* 搜索的最优性要求：必须使用可采纳启发函数
- BFS 的最优性仅在等代价边成立
- Alpha-Beta 剪枝中 α 和 β 的更新时机
- DFS 不保证找到最短路径

---

## 第三部分：知识表示与推理

### 3.1 命题逻辑（Propositional Logic）

#### 3.1.1 基本概念

- **命题**：具有确定真值的陈述句（真 T 或假 F）
- **原子命题**：不可再分的简单命题
- **复合命题**：由联结词组合的命题

#### 3.1.2 逻辑联结词

| 联结词 | 符号 | 名称 | 含义 |
|--------|------|------|------|
| ¬P | 否定 | 非 | P为假时 ¬P 为真 |
| P∧Q | 合取 | 与 | 两者都为真时为真 |
| P∨Q | 析取 | 或 | 至少一个为真时为真 |
| P→Q | 蕴含 | 如果...那么 | P为真且Q为假时为假 |
| P↔Q | 等价 | 当且仅当 | 真值相同时为真 |

**蕴含的真值表（重点）：**

| P | Q | P→Q |
|---|---|-----|
| T | T | T |
| T | F | F |
| F | T | T |
| F | F | T |

**关键**：前件为假时，蕴含式恒为真（空真/vacuously true）

#### 3.1.3 重要的逻辑等价式

- **德摩根律**：¬(P∧Q) ≡ ¬P∨¬Q；¬(P∨Q) ≡ ¬P∧¬Q
- **双重否定**：¬(¬P) ≡ P
- **逆否律**：(P→Q) ≡ (¬Q→¬P)
- **分配律**：P∧(Q∨R) ≡ (P∧Q)∨(P∧R)
- **排中律**：P∨¬P ≡ T
- **矛盾律**：P∧¬P ≡ F

#### 3.1.4 推理规则

| 规则 | 形式 |
|------|------|
| 假言推理（Modus Ponens） | P, P→Q ⊢ Q |
| 拒取式（Modus Tollens） | ¬Q, P→Q ⊢ ¬P |
| 假言三段论 | P→Q, Q→R ⊢ P→R |
| 析取三段论 | P∨Q, ¬P ⊢ Q |

#### 3.1.5 范式

- **合取范式（CNF）**：子句的合取，每个子句是文字的析取
  - 例：(P∨¬Q)∧(¬P∨R)∧(Q∨R)
- **析取范式（DNF）**：文字合取的析取
  - 例：(P∧Q)∨(¬P∧R)

### 3.2 谓词逻辑（Predicate Logic / First-Order Logic）

#### 3.2.1 基本概念

- **个体**：讨论的对象（常量 a, b, c 或变量 x, y, z）
- **谓词**：描述个体性质或关系的符号 P(x), R(x,y)
- **量词**：
  - **全称量词** ∀x P(x)：对所有x，P(x)成立
  - **存在量词** ∃x P(x)：存在某个x使P(x)成立

#### 3.2.2 合式公式（WFF）

- 原子公式是WFF
- 若A、B是WFF，则 ¬A、A∧B、A∨B、A→B、A↔B 是WFF
- 若A是WFF，x是变量，则 ∀xA 和 ∃xA 是WFF

#### 3.2.3 量词的否定

- ¬(∀x P(x)) ≡ ∃x ¬P(x)
- ¬(∃x P(x)) ≡ ∀x ¬P(x)

#### 3.2.4 转换为子句形（Clausal Form）

步骤：
1. 消去蕴含和等价（用 ¬ 和 ∨ 表示）
2. 将否定符号内移到谓词前（德摩根律 + 量词否定）
3. 变量标准化（不同量词的变量名不同）
4. 存在量词 Skolem 化（用Skolem函数替换存在量词变量）
5. 全称量词前束化
6. 化为合取范式（CNF）
7. 去掉合取号，得到子句集

#### 3.2.5 归结推理（Resolution）

**核心思想**：通过合一消去互补文字

**归结规则：**
```
子句1: C₁ ∨ L₁
子句2: C₂ ∨ L₂  （L₂ 是 L₁ 的互补文字）
归结式: C₁ ∨ C₂
```

**合一（Unification）**：找到替换 θ 使得两个表达式相同
- MGU（最一般合一子）是代价最小的合一替换

**归结反驳证明过程：**
1. 将前提条件转化为子句集
2. 将待证结论的否定加入子句集
3. 反复应用归结规则
4. 若导出空子句（矛盾），则原结论成立

### 3.3 贝叶斯网络（Bayesian Network）

#### 3.3.1 概率论基础

**条件概率：**
```
P(A|B) = P(A∩B) / P(B)
```

**贝叶斯定理（核心公式）：**
```
P(A|B) = P(B|A) × P(A) / P(B)
```
- P(A)：先验概率
- P(B|A)：似然
- P(A|B)：后验概率
- P(B)：证据（归一化常数）

**全概率公式：**
```
P(B) = Σᵢ P(B|Aᵢ) × P(Aᵢ)
```

#### 3.3.2 贝叶斯网络的定义

贝叶斯网络 = **有向无环图（DAG）** + **条件概率表（CPT）**

- 节点表示随机变量
- 有向边表示因果/依赖关系
- 每个节点X有P(X | Parents(X))的条件概率表

**联合概率分布（链式法则）：**
```
P(X₁, X₂, ..., Xₙ) = ∏ᵢ P(Xᵢ | Parents(Xᵢ))
```

#### 3.3.3 条件独立性（D-Separation）

**D-Separation 判定规则**：节点X和Y在给定证据集Z下条件独立，当且仅当X和Y之间的**所有路径**都被Z阻塞。

一条路径被阻塞的条件：
- 路径上存在节点W ∈ Z，且W是链式或分叉结构（链：A→W→B 或共因：A←W→B）
- 或路径上存在节点W ∉ Z，且W是汇聚结构（共果：A→W←B）且W及其后代都不在Z中

三种基本结构：

| 结构 | 形式 | 独立条件 |
|------|------|----------|
| 链式 | A→B→C | 给定B时，A与C独立 |
| 分叉（共因） | A←B→C | 给定B时，A与C独立 |
| 汇聚（共果） | A→B←C | 不给定B时，A与C独立 |

#### 3.3.4 推理类型

| 推理类型 | 说明 | 示例 |
|----------|------|------|
| 因果推理（预测） | 原因→结果 | P(疾病\|症状) |
| 诊断推理（溯因） | 结果→原因 | P(症状\|疾病) |
| 跨因推理 | 原因间相互影响 | A和B共同导致C，已知C时A和B不再独立 |

### 3.4 知识表示常见考题与易错点

**常见考题：**
1. 命题逻辑真值表计算
2. 将自然语言转化为谓词逻辑公式
3. 归结推理证明
4. 贝叶斯网络中利用D-Separation判定条件独立性
5. 利用贝叶斯定理计算后验概率

**易错点：**
- 蕴含 P→Q 在P为假时恒为真
- 全称量词和存在量词的顺序影响语义：∀x∃y ≠ ∃y∀x
- Skolem化时，存在量词变量依赖于其左侧的所有全称量词变量
- D-Separation 中汇聚结构（共果）的独立条件与链式/分叉结构相反
- 贝叶斯网络中"解释消除"效应：已知共同效果后，原因变得不独立

---

## 第四部分：机器学习基础

### 4.1 机器学习基本概念

**定义（Tom Mitchell）**：一个计算机程序被称为从经验E中学习关于某任务T和性能度量P，如果它在T上的性能P随着经验E的增加而提高。

**学习类型分类：**
- **监督学习**：有标签数据（回归、分类）
- **无监督学习**：无标签数据（聚类、降维）
- **强化学习**：通过与环境交互获得奖励信号

**关键概念：**
- **偏差-方差权衡（Bias-Variance Tradeoff）**：
  - 高偏差 = 欠拟合（模型太简单）
  - 高方差 = 过拟合（模型太复杂）
- **正则化**：L1正则化（Lasso，稀疏性）、L2正则化（Ridge，权重衰减）
- **交叉验证**：k折交叉验证评估模型泛化能力

### 4.2 线性回归（Linear Regression）

#### 4.2.1 模型假设

```
h_θ(x) = θᵀx = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```

#### 4.2.2 代价函数（均方误差 MSE）

```
J(θ) = (1/2m) Σᵢ₌₁ᵐ (h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
```

#### 4.2.3 梯度下降

```
θⱼ := θⱼ - α × ∂J(θ)/∂θⱼ
    = θⱼ - α × (1/m) Σᵢ₌₁ᵐ (h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾) × xⱼ⁽ⁱ⁾
```

**梯度下降变体：**
| 类型 | 描述 | 特点 |
|------|------|------|
| 批量梯度下降（BGD） | 使用全部训练数据计算梯度 | 稳定但慢 |
| 随机梯度下降（SGD） | 每次使用一个样本 | 快但波动大 |
| 小批量梯度下降（Mini-batch） | 使用一小批样本 | 折中方案 |

#### 4.2.4 正规方程（Normal Equation）

```
θ = (XᵀX)⁻¹Xᵀy
```

- **优点**：无需选择学习率，无需迭代
- **缺点**：需要计算矩阵逆，时间复杂度O(n³)，特征维度高时不可行
- **注意**：当XᵀX不可逆时（特征线性相关或特征数>样本数），需使用正则化或伪逆

### 4.3 逻辑回归（Logistic Regression）

#### 4.3.1 Sigmoid 函数

```
g(z) = 1 / (1 + e⁻ᶻ)
```
- 输出范围 (0, 1)，可解释为概率
- g'(z) = g(z)(1 - g(z))

#### 4.3.2 模型假设

```
h_θ(x) = g(θᵀx) = 1 / (1 + e⁻⁽θᵀˣ⁾)
```

#### 4.3.3 代价函数（交叉熵损失）

```
J(θ) = -(1/m) Σᵢ₌₁ᵐ [y⁽ⁱ⁾ log(h_θ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾) log(1 - h_θ(x⁽ⁱ⁾))]
```

**推导思路**：通过最大似然估计（MLE）得到：
```
对数似然 = Σᵢ [y⁽ⁱ⁾ log h + (1-y⁽ⁱ⁾) log(1-h)]
代价 = -对数似然 / m
```

#### 4.3.4 梯度下降更新

形式与线性回归相同：
```
θⱼ := θⱼ - α × (1/m) Σᵢ₌₁ᵐ (h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾) × xⱼ⁽ⁱ⁾
```

### 4.4 决策树（Decision Tree）

#### 4.4.1 核心思想

通过递归地选择最优特征进行分裂，构建树形分类结构。

#### 4.4.2 分裂准则

**信息增益（Information Gain）—— ID3 算法：**

```
H(D) = -Σₖ₌₁ᴷ pₖ log₂(pₖ)    // 信息熵

H(D|A) = Σᵢ₌₁ⁿ (|Dᵢ|/|D|) × H(Dᵢ)    // 条件熵

IG(D,A) = H(D) - H(D|A)    // 信息增益
```

**信息增益比（Gain Ratio）—— C4.5 算法：**
```
GR(D,A) = IG(D,A) / H_A(D)
```
其中H_A(D)是特征A的固有值（Intrinsic Value），用于惩罚取值较多的特征。

**基尼指数（Gini Index）—— CART 算法：**
```
Gini(D) = 1 - Σₖ₌₁ᴷ pₖ²

Gini(D,A) = (|D₁|/|D|)Gini(D₁) + (|D₂|/|D|)Gini(D₂)
```
选择使基尼指数最小的特征进行分裂。

#### 4.4.3 剪枝策略

| 方法 | 描述 |
|------|------|
| 预剪枝 | 在构建过程中提前停止（限制深度、最小样本数等） |
| 后剪枝 | 先完整构建，再自底向上删除不显著的分支 |

#### 4.4.4 常见算法对比

| 算法 | 分裂准则 | 特点 |
|------|----------|------|
| ID3 | 信息增益 | 仅处理离散特征，易过拟合 |
| C4.5 | 信息增益比 | 支持连续特征，支持剪枝 |
| CART | 基尼指数 | 二叉树，支持回归和分类 |

### 4.5 支持向量机（SVM）

#### 4.5.1 线性可分 SVM

**目标**：找到将两类数据分开且间隔（margin）最大的超平面。

**决策超平面**：w·x + b = 0

**优化问题（硬间隔）：**
```
min (1/2)||w||²
s.t. yᵢ(w·xᵢ + b) ≥ 1, ∀i
```

**函数间隔**：γ̂ᵢ = yᵢ(w·xᵢ + b)
**几何间隔**：γᵢ = yᵢ(w·xᵢ + b) / ||w||
**支持向量**：满足 yᵢ(w·xᵢ + b) = 1 的训练样本（位于间隔边界上）

#### 4.5.2 软间隔 SVM

引入松弛变量 ξᵢ ≥ 0 允许部分误分类：

```
min (1/2)||w||² + C Σᵢ ξᵢ
s.t. yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
```

- C > 0 为惩罚参数：C越大对误分类惩罚越重
- C→∞ 退化为硬间隔

#### 4.5.3 Hinge Loss

```
L = max(0, 1 - yᵢ(w·xᵢ + b))
```

#### 4.5.4 核技巧（Kernel Trick）

将数据映射到高维空间 φ(x) 使其线性可分，但避免显式计算高维映射：

```
K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ)
```

**常用核函数：**

| 核函数 | 公式 | 特点 |
|--------|------|------|
| 线性核 | K(x,y) = x·y | 最简单 |
| 多项式核 | K(x,y) = (x·y + c)ᵈ | 参数d控制复杂度 |
| RBF/高斯核 | K(x,y) = exp(-γ\|\|x-y\|\|²) | 最常用，隐式映射到无穷维 |
| Sigmoid核 | K(x,y) = tanh(αx·y + c) | 类似神经网络 |

#### 4.5.5 对偶问题

```
max Σᵢ αᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
s.t. 0 ≤ αᵢ ≤ C, Σᵢ αᵢyᵢ = 0
```

**决策函数**：
```
f(x) = sign(Σᵢ αᵢyᵢK(xᵢ, x) + b)
```

### 4.6 朴素贝叶斯（Naive Bayes）

#### 4.6.1 核心思想

基于贝叶斯定理和**特征条件独立假设**：

```
P(C|X) ∝ P(X|C) × P(C)

"朴素"假设：P(X|C) = ∏ᵢ P(xᵢ|C)

y = argmax_c P(c) ∏ᵢ P(xᵢ|c)
```

#### 4.6.2 参数估计

- **先验概率**：P(c) = |Dc| / |D|
- **条件概率**：
  - 离散特征：P(xᵢ|c) = |Dc,xi| / |Dc|（加拉普拉斯平滑）
  - 连续特征：假设高斯分布 P(xᵢ|c) = (1/√(2πσ²c)) exp(-(xᵢ-μc)²/2σ²c)

#### 4.6.3 拉普拉斯平滑

防止零概率问题：
```
P(xᵢ|c) = (|Dc,xi| + 1) / (|Dc| + Nᵢ)
```
其中Nᵢ是特征xᵢ的可能取值数。

#### 4.6.4 三种朴素贝叶斯模型

| 模型 | 适用特征 | 场景 |
|------|----------|------|
| 高斯朴素贝叶斯 | 连续特征 | 假设特征服从正态分布 |
| 多项式朴素贝叶斯 | 离散计数 | 文本分类（词频） |
| 伯努利朴素贝叶斯 | 二值特征 | 文档特征（词是否出现） |

### 4.7 K-Means 聚类

#### 4.7.1 算法流程

```
输入：数据集D，聚类数K
1. 随机初始化K个聚类中心 μ₁, μ₂, ..., μₖ
2. 重复直到收敛：
   a. 分配步骤：将每个样本分配到最近的聚类中心
      c⁽ⁱ⁾ = argmin_k ||x⁽ⁱ⁾ - μₖ||²
   b. 更新步骤：重新计算聚类中心
      μₖ = (1/|Cₖ|) Σ_{x∈Cₖ} x
3. 输出：聚类结果
```

#### 4.7.2 目标函数

最小化类内平方和（SSE / inertia）：
```
J = Σₖ₌₁ᴷ Σ_{x∈Cₖ} ||x - μₖ||²
```

#### 4.7.3 选择K值

- **肘部法则（Elbow Method）**：绘制SSE随K变化的曲线，选择"拐点"
- **轮廓系数（Silhouette Score）**：衡量簇内紧密度和簇间分离度

#### 4.7.4 优缺点

| 优点 | 缺点 |
|------|------|
| 简单高效 | 需要预先指定K |
| 大数据集可扩展 | 对初始中心敏感 |
| | 仅发现球形簇 |
| | 对异常值敏感 |
| | 可能陷入局部最优 |

### 4.8 PCA 主成分分析

#### 4.8.1 核心思想

找到数据方差最大的正交方向（主成分），将数据投影到低维空间。

#### 4.8.2 算法步骤

```
输入：数据集X (n×d)，目标维度k
1. 数据标准化：减均值，可选除标准差
2. 计算协方差矩阵：Σ = (1/n) XᵀX
3. 特征值分解：Σu = λu
4. 按特征值从大到小排序，取前k个特征向量
5. 投影：Z = X × Uₖ (Uₖ为前k个特征向量组成的矩阵)
输出：降维后的数据Z (n×k)
```

#### 4.8.3 方差解释比

```
解释方差比 = λᵢ / Σⱼ λⱼ
```
选择累计解释方差比 >= 95% 的主成分数。

#### 4.8.4 PCA 的性质

- **无监督**降维方法
- 最大化数据方差 ≈ 最小化重构误差
- 主成分之间正交（不相关）
- 对数据尺度敏感，需要先标准化
- 线性降维方法，不能处理非线性结构（核PCA可解决）

### 4.9 机器学习常见考题与易错点

**常见考题：**
1. 手动计算线性回归的梯度下降更新步骤
2. 逻辑回归损失函数的推导
3. 决策树构建过程（选择分裂特征、计算信息增益）
4. SVM支持向量的概念和软间隔参数C的影响
5. K-Means迭代过程
6. PCA降维计算步骤

**易错点：**
- 线性回归正规方程中 (XᵀX) 不可逆的处理
- 逻辑回归虽然名字中有"回归"，但用于分类
- 朴素贝叶斯的"条件独立"假设在实际中常不成立但效果仍然好
- SVM的核函数将低维映射到高维空间，但实际计算中不需要显式映射
- K-Means不保证全局最优解
- PCA降维后丢失了部分信息，不能完全代表原始数据

---

## 第五部分：神经网络

### 5.1 感知机（Perceptron）

#### 5.1.1 单层感知机

```
输出：y = f(Σᵢ wᵢxᵢ + b)
其中 f(z) = { 1 if z ≥ 0; 0 otherwise }（阶跃函数）
```

**感知机学习规则：**
```
wᵢ := wᵢ + η(y - ŷ)xᵢ
b := b + η(y - ŷ)
```
其中 η 为学习率，y 为真实标签，ŷ 为预测输出。

**局限性**：只能解决线性可分问题（Minsky & Papert, 1969）。经典反例：XOR问题不可解。

#### 5.1.2 多层感知机（MLP）

通过引入隐藏层解决非线性问题：
- 输入层 → 隐藏层（一层或多层）→ 输出层
- 每层的神经元与下一层全连接
- 万能近似定理：单隐层网络（足够宽）可以近似任意连续函数

### 5.2 激活函数

| 函数 | 公式 | 导数 | 特点 |
|------|------|------|------|
| Sigmoid | σ(z)=1/(1+e⁻ᶻ) | σ(z)(1-σ(z)) | 输出(0,1)，梯度消失 |
| Tanh | tanh(z) | 1-tanh²(z) | 输出(-1,1)，零中心 |
| ReLU | max(0,z) | 0 if z<0; 1 if z≥0 | 简单高效，Dead ReLU问题 |
| Leaky ReLU | max(αz,z), α≈0.01 | α if z<0; 1 if z≥0 | 解决Dead ReLU |
| ELU | z if z≥0; α(eᶻ-1) if z<0 | 1 if z≥0; ELU+α if z<0 | 零中心，平滑 |
| Softmax | eᶻⁱ/Σⱼ eᶻʲ | -- | 多分类输出层 |

**激活函数选择建议：**
- 隐藏层：ReLU（首选）/ Leaky ReLU
- 二分类输出层：Sigmoid
- 多分类输出层：Softmax
- 回归输出层：无激活/线性激活

### 5.3 反向传播算法（Backpropagation）

#### 5.3.1 核心思想

利用链式法则，从输出层到输入层逐层计算损失函数对每个参数的梯度。

#### 5.3.2 前向传播

对于第l层：
```
z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = f(z⁽ˡ⁾)
```

#### 5.3.3 反向传播推导

**输出层误差：**
```
δ⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ = (a⁽ˡ⁾ - y) ⊙ f'(z⁽ˡ⁾)  // 对于MSE + Sigmoid
```

**隐藏层误差（递推）：**
```
δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ δ⁽ˡ⁺¹⁾ ⊙ f'(z⁽ˡ⁾)
```

**参数梯度：**
```
∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ (a⁽ˡ⁻¹⁾)ᵀ
∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾
```

#### 5.3.4 参数更新

```
W⁽ˡ⁾ := W⁽ˡ⁾ - α × ∂L/∂W⁽ˡ⁾
b⁽ˡ⁾ := b⁽ˡ⁾ - α × ∂L/∂b⁽ˡ⁾
```

### 5.4 梯度下降优化

| 优化器 | 核心思想 | 特点 |
|--------|----------|------|
| SGD | w := w - α∇L | 基础，可能震荡 |
| Momentum | 引入动量项，指数加权平均梯度 | 加速收敛，减少震荡 |
| AdaGrad | 自适应学习率，频繁更新的参数学习率减小 | 稀疏数据好，学习率单调递减 |
| RMSProp | 使用梯度平方的指数移动平均修正AdaGrad | 解决学习率递减过快 |
| Adam | 结合Momentum和RMSProp | 当前最常用，自适应学习率 |

**Adam 优化器更新公式：**
```
mₜ = β₁mₜ₋₁ + (1-β₁)gₜ         // 一阶矩估计（动量）
vₜ = β₂vₜ₋₁ + (1-β₂)gₜ²        // 二阶矩估计（自适应学习率）
m̂ₜ = mₜ / (1-β₁ᵗ)              // 偏差修正
v̂ₜ = vₜ / (1-β₂ᵗ)              // 偏差修正
wₜ₊₁ = wₜ - α × m̂ₜ / (√v̂ₜ + ε)
```

### 5.5 正则化技术

| 技术 | 描述 |
|------|------|
| L1 正则化 | 添加 Σ\|wᵢ\| 到损失函数，产生稀疏权重 |
| L2 正则化 | 添加 λΣwᵢ² 到损失函数（权重衰减） |
| Dropout | 训练时随机将一定比例神经元置零（通常p=0.5） |
| Early Stopping | 验证集损失不再下降时停止训练 |
| Batch Normalization | 对每层输入做归一化，加速训练 |

### 5.6 神经网络常见考题与易错点

**常见考题：**
1. 给定网络结构和权重，手动计算前向传播和反向传播
2. 不同激活函数的优缺点比较
3. 梯度消失/爆炸的原因和解决方案
4. Dropout的工作原理

**易错点：**
- 反向传播中链式法则的正确应用
- 梯度消失主要发生在Sigmoid/Tanh激活函数中（导数最大值为0.25/1）
- Dropout仅在训练时使用，测试时需要缩放或使用Inverted Dropout
- Batch Normalization 的训练和推理行为不同

---

## 第六部分：深度学习

### 6.1 卷积神经网络（CNN）

#### 6.1.1 核心组件

**卷积层（Convolutional Layer）：**
- 局部连接（Local Connectivity）：每个神经元只连接输入的一个局部区域
- 权值共享（Weight Sharing）：同一个卷积核在整个输入上滑动
- 参数量：K × K × C_in × C_out + C_out（K为核大小，C为通道数）

**输出尺寸计算公式：**
```
Output = (W - K + 2P) / S + 1
```
- W：输入尺寸
- K：卷积核尺寸
- P：填充（Padding）
- S：步长（Stride）

**池化层（Pooling Layer）：**
- 最大池化（Max Pooling）：取窗口最大值
- 平均池化（Average Pooling）：取窗口平均值
- 无学习参数
- 输出尺寸用相同公式计算

#### 6.1.2 经典 CNN 架构

| 架构 | 年份 | 关键创新 |
|------|------|----------|
| LeNet-5 | 1998 | 首个成功的CNN（手写数字识别） |
| AlexNet | 2012 | ReLU、Dropout、GPU训练，ImageNet突破 |
| VGGNet | 2014 | 统一3×3卷积核，增加深度（16-19层） |
| GoogLeNet/Inception | 2014 | Inception模块（多尺度卷积并行），1×1卷积降维 |
| ResNet | 2015 | **残差连接（Skip Connection）**，解决深层网络退化问题 |

**ResNet 残差块：**
```
输出 = F(x) + x    // 恒等映射（identity mapping）
```
- F(x) = W₂σ(W₁x + b₁) + b₂（两层卷积）
- 解决梯度消失和网络退化问题
- 允许训练极深的网络（152层+）

#### 6.1.3 其他重要技术

- **Batch Normalization**：对每个mini-batch归一化，加速训练
- **Dropout**：防止过拟合
- **数据增强**：随机裁剪、翻转、颜色抖动等
- **迁移学习**：使用预训练模型在新任务上微调（Fine-tuning）

### 6.2 循环神经网络（RNN）

#### 6.2.1 基本 RNN

**前向传播：**
```
hₜ = tanh(Wₓₕxₜ + Wₕₕhₜ₋₁ + bₕ)
yₜ = Wₕᵧhₜ + bᵧ
```

**问题：梯度消失/爆炸**
- 反向传播通过时间（BPTT）时，梯度需连乘多个时间步的权重矩阵
- 如果权重矩阵特征值 < 1，梯度指数级缩小（消失）
- 如果权重矩阵特征值 > 1，梯度指数级增大（爆炸）
- 解决梯度爆炸：梯度裁剪（Gradient Clipping）

#### 6.2.2 LSTM（长短期记忆网络）

**核心思想**：通过门控机制和细胞状态实现长距离依赖。

**三个门和细胞状态的完整方程：**

```
遗忘门：fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)

输入门：iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)
候选值：C̃ₜ = tanh(WC · [hₜ₋₁, xₜ] + bC)

细胞状态更新：Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ

输出门：oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)
隐藏状态：hₜ = oₜ * tanh(Cₜ)
```

**LSTM 为何能解决梯度消失：**
- 细胞状态Cₜ的更新是**加法**操作（不是连乘）
- 当遗忘门 fₜ ≈ 1 时，梯度可以几乎无损地通过细胞状态传递
- 细胞状态像一条"传送带"，信息可以长距离传递

#### 6.2.3 GRU（门控循环单元）

简化版 LSTM，合并遗忘门和输入门：
```
更新门：zₜ = σ(Wz · [hₜ₋₁, xₜ])
重置门：rₜ = σ(Wr · [hₜ₋₁, xₜ])
候选隐藏状态：h̃ₜ = tanh(W · [rₜ * hₜ₋₁, xₜ])
隐藏状态：hₜ = (1-zₜ) * hₜ₋₁ + zₜ * h̃ₜ
```

### 6.3 Transformer 架构

#### 6.3.1 自注意力机制（Self-Attention）

**核心公式——缩放点积注意力（Scaled Dot-Product Attention）：**
```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
```
- Q（Query）：查询矩阵
- K（Key）：键矩阵
- V（Value）：值矩阵
- dₖ：键的维度（缩放因子防止内积过大导致softmax梯度消失）

**计算步骤：**
1. 计算注意力分数：S = QKᵀ
2. 缩放：S = S / √dₖ
3. 归一化：A = softmax(S)
4. 加权求和：Output = AV

#### 6.3.2 多头注意力（Multi-Head Attention）

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) Wᴼ

其中 headᵢ = Attention(QWᵢᵠ, KWᵢᴷ, VWᵢⱽ)
```
- h 个注意力头并行计算
- 每个头关注不同的表示子空间
- Wᵢᵠ, Wᵢᴷ, Wᵢⱽ ∈ ℝ^(d_model × dₖ) 为投影矩阵
- dₖ = dᵥ = d_model / h

#### 6.3.3 位置编码（Positional Encoding）

由于自注意力是排列不变的，需要注入位置信息：
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

#### 6.3.4 Transformer 整体架构

**编码器（Encoder）：**
```
输入 → 位置编码 → [多头自注意力 → Add&LayerNorm → 前馈网络 → Add&LayerNorm] × N
```

**解码器（Decoder）：**
```
输出 → 位置编码 → [带掩码多头自注意力 → Add&LayerNorm → 编码器-解码器注意力 → Add&LayerNorm → 前馈网络 → Add&LayerNorm] × N → 线性层 → Softmax
```

**关键组件：**
- **Add & Layer Normalization**：残差连接 + 层归一化
- **前馈网络（FFN）**：两层全连接 + ReLU/GELU
  ```
  FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
  ```
- **掩码自注意力**：解码器中防止看到未来位置

#### 6.3.5 Transformer vs RNN

| 特性 | RNN/LSTM | Transformer |
|------|----------|-------------|
| 并行性 | 时间步串行 | 完全并行 |
| 长距离依赖 | 受限于梯度传播 | 直接建模 |
| 计算复杂度 | O(n) 每步 | O(n²d) 自注意力 |
| 位置信息 | 天然有序 | 需位置编码 |
| 内存 | 与序列长度无关 | 需存储完整注意力矩阵 |

### 6.4 深度学习常见考题与易错点

**常见考题：**
1. CNN输出尺寸计算
2. LSTM各门的作用和计算过程
3. Transformer自注意力机制的工作原理
4. ResNet残差连接为什么有效

**易错点：**
- 卷积层参数量的正确计算（需考虑输入通道数）
- 池化层没有学习参数
- LSTM中遗忘门控制的是细胞状态，不是隐藏状态
- Transformer的缩放因子√dₖ的作用（防止点积过大）
- 位置编码不是学习得到的，而是固定的正弦/余弦函数

---

## 第七部分：自然语言处理（NLP）

### 7.1 词嵌入（Word Embeddings）

#### 7.1.1 词的表示方法

| 方法 | 描述 | 维度 | 特点 |
|------|------|------|------|
| One-Hot | 每个词一个维度为1的向量 | 词表大小 | 稀疏，无语义信息 |
| 词袋模型（BoW） | 统计词频 | 词表大小 | 忽略词序 |
| TF-IDF | 词频-逆文档频率 | 词表大小 | 衡量词的重要性 |
| 分布式表示 | 低维稠密向量 | 100-300 | 捕获语义关系 |

#### 7.1.2 Word2Vec

**两种架构：**

**CBOW（连续词袋模型）：**
- 输入：上下文词
- 目标：预测中心词
- 目标函数：
  ```
  L = (1/T) Σₜ log P(wₜ | wₜ₋c, ..., wₜ₋₁, wₜ₊₁, ..., wₜ₊c)
  ```

**Skip-Gram：**
- 输入：中心词
- 目标：预测上下文词
- 目标函数：
  ```
  L = (1/T) Σₜ Σ_{-c≤j≤c, j≠0} log P(wₜ₊ⱼ | wₜ)
  ```
- 条件概率（Softmax）：
  ```
  P(wₒ|wᵢ) = exp(v'ₒᵀvᵢ) / Σw exp(v'wᵀvᵢ)
  ```

**负采样（Negative Sampling）：**
- 避免计算全词表Softmax的高昂代价
- 将多分类问题转化为多个二分类问题
- 目标函数：
  ```
  log σ(v'ₒᵀvᵢ) + Σᵢ₌₁ᴷ E_{wᵢ~Pn(w)} [log σ(-v'ᵢᵀvᵢ)]
  ```

**经典语义关系：**
```
king - man + woman ≈ queen
Paris - France + Italy ≈ Rome
```

#### 7.1.3 GloVe（Global Vectors）

- 结合全局统计（共现矩阵）和局部上下文窗口
- 目标函数：
  ```
  J = Σᵢ,ⱼ f(Xᵢⱼ)(vᵢᵀvⱼ + bᵢ + bⱼ - log Xᵢⱼ)²
  ```
- f(x) 为权重函数，限制高频词的影响

### 7.2 文本分类

#### 7.2.1 传统方法

- 词袋模型 + 朴素贝叶斯
- TF-IDF + SVM
- N-gram 特征 + 逻辑回归

#### 7.2.2 深度学习方法

| 方法 | 架构 | 特点 |
|------|------|------|
| TextCNN | CNN + 多种卷积核 | 捕获局部N-gram特征，速度快 |
| BiLSTM | 双向LSTM | 捕获双向上下文依赖 |
| BiLSTM+Attention | 双向LSTM + 注意力 | 关注重要词 |
| BERT | Transformer Encoder | 预训练+微调，效果最优 |
| GPT | Transformer Decoder | 自回归语言模型 |

### 7.3 预训练语言模型

#### 7.3.1 BERT（Bidirectional Encoder Representations from Transformers）

- **架构**：Transformer Encoder
- **预训练任务**：
  - **MLM（Masked Language Model）**：随机遮盖15%的词，预测被遮盖的词
  - **NSP（Next Sentence Prediction）**：预测两句是否相邻
- **微调**：在预训练模型上加任务特定的输出层

#### 7.3.2 GPT（Generative Pre-trained Transformer）

- **架构**：Transformer Decoder
- **预训练**：自回归语言模型（预测下一个词）
- **生成**：从左到右逐词生成

### 7.4 NLP 常见考题与易错点

**易错点：**
- Word2Vec的Skip-Gram和CBOW的关系：Skip-Gram适合小数据集和低频词，CBOW适合大数据集
- BERT是双向的（看上下文），GPT是单向的（只看左侧）
- 词嵌入能捕获语义相似性，但不能处理一词多义（需要上下文相关表示如ELMo/BERT）

---

## 第八部分：计算机视觉（CV）

### 8.1 图像分类

#### 8.1.1 基本流程

图像 → 预处理 → 特征提取 → 分类器 → 类别标签

#### 8.1.2 传统方法

- **手工特征**：SIFT、HOG、LBP
- **分类器**：SVM、随机森林

#### 8.1.3 深度学习方法

使用 CNN（见第6.1节）进行端到端学习。

**经典网络架构**：LeNet → AlexNet → VGG → GoogLeNet → ResNet

### 8.2 目标检测

#### 8.2.1 基本任务

在图像中定位并分类多个目标物体，输出边界框（bounding box）和类别。

#### 8.2.2 两阶段检测器（Two-Stage）

**R-CNN 系列：**

| 方法 | 流程 | 特点 |
|------|------|------|
| R-CNN | 候选区域 → CNN特征 → SVM分类 | 慢，需分别训练 |
| Fast R-CNN | 整图CNN → ROI Pooling → 分类+回归 | 共享卷积计算 |
| Faster R-CNN | RPN生成候选区域 → ROI Pooling → 分类+回归 | 端到端训练 |

**Faster R-CNN 核心——区域提议网络（RPN）：**
- 在特征图上滑动窗口生成候选区域
- 使用锚框（Anchor Boxes）预测不同尺度和比例的候选框

#### 8.2.3 单阶段检测器（One-Stage）

**YOLO（You Only Look Once）：**
- 将检测视为回归问题
- 一次前向传播同时预测所有边界框和类别
- 将图像分为 S×S 网格，每个网格预测 B 个边界框
- 损失函数 = 定位损失 + 置信度损失 + 分类损失
- 速度快，适合实时检测

**SSD（Single Shot MultiBox Detector）：**
- 多尺度特征图检测
- 不同层检测不同大小的目标

#### 8.2.4 评估指标

**IoU（交并比）：**
```
IoU = Area(Bbox_pred ∩ Bbox_gt) / Area(Bbox_pred ∪ Bbox_gt)
```

**mAP（平均精度均值）：**
- Precision-Recall 曲线下面积
- 在不同 IoU 阈值下的 AP 求均值

### 8.3 生成对抗网络（GAN）

#### 8.3.1 基本架构

两个网络对抗训练：
- **生成器 G（Generator）**：从随机噪声 z 生成假数据 G(z)
- **判别器 D（Discriminator）**：区分真实数据和生成数据

#### 8.3.2 目标函数（原始 GAN）

```
min_G max_D V(D,G) = E_{x~pdata}[log D(x)] + E_{z~pz}[log(1 - D(G(z)))]
```

**训练过程：**
1. 固定 G，训练 D 最大化判别准确率
2. 固定 D，训练 G 最小化 log(1 - D(G(z)))
3. 交替优化

**理论最优：** 当 G 的生成分布 = 真实数据分布时，D(x) = 1/2

#### 8.3.3 GAN 变体

| 变体 | 改进点 |
|------|--------|
| DCGAN | 使用 CNN 架构 |
| WGAN | Wasserstein 距离，训练更稳定 |
| LSGAN | 最小二乘损失替代对数损失 |
| Conditional GAN (cGAN) | 条件生成（如根据标签生成） |
| CycleGAN | 无配对的图像风格转换 |
| StyleGAN | 高质量人脸生成，可控风格 |

#### 8.3.4 GAN 的应用

- 图像生成（人脸、风景）
- 图像超分辨率
- 图像修复（Inpainting）
- 风格迁移
- 数据增强
- 文本到图像生成

### 8.4 计算机视觉常见考题与易错点

**常见考题：**
1. CNN 各层特征图尺寸计算
2. 目标检测中 IoU 的计算
3. GAN 的训练不稳定性问题
4. Faster R-CNN 与 YOLO 的区别

**易错点：**
- GAN 训练中生成器和判别器的交替优化过程
- mAP 不是单一指标，而是不同类别和阈值的综合
- 卷积层输出尺寸公式中不要忘记加 padding
- ResNet 的残差连接不是简单的跳跃连接，而是学习残差函数 F(x)

---

## 第九部分：重要公式速查表

### 损失函数

| 名称 | 公式 | 适用场景 |
|------|------|----------|
| 均方误差 MSE | (1/n)Σ(yᵢ - ŷᵢ)² | 回归 |
| 交叉熵 | -Σyᵢlog(ŷᵢ) | 分类 |
| Hinge Loss | max(0, 1-y·f(x)) | SVM |
| 二元交叉熵 | -[y·log(p)+(1-y)·log(1-p)] | 二分类 |
| KL散度 | Σpᵢlog(pᵢ/qᵢ) | 分布匹配 |

### 正则化

| 方法 | 公式 | 效果 |
|------|------|------|
| L1 | λΣ\|wᵢ\| | 稀疏性（特征选择） |
| L2 | λΣwᵢ² | 权重衰减（防过拟合） |
| Elastic Net | α\|w\|₁ + (1-α)\|w\|₂² | L1+L2组合 |

### 信息论

| 名称 | 公式 |
|------|------|
| 信息熵 | H(X) = -Σp(x)log₂p(x) |
| 条件熵 | H(Y\|X) = Σp(x)H(Y\|X=x) |
| 信息增益 | IG = H(Y) - H(Y\|X) |
| 互信息 | I(X;Y) = H(X) - H(X\|Y) |
| 基尼指数 | Gini = 1 - Σpᵢ² |

---

## 第十部分：竞赛高频考点与陷阱总结

### 10.1 高频考点 TOP 20

1. 图灵测试的定义和局限性
2. BFS/DFS/A* 的时间空间复杂度对比
3. A* 可采纳启发函数与一致性的关系
4. Minimax + Alpha-Beta 剪枝手动模拟
5. 命题逻辑/谓词逻辑的推理规则
6. 贝叶斯定理的计算应用
7. 贝叶斯网络 D-Separation 条件独立性判定
8. 线性回归/逻辑回归的损失函数和梯度下降
9. 决策树分裂准则（信息增益、基尼指数）
10. SVM 的软间隔和核函数
11. 朴素贝叶斯的条件独立假设和拉普拉斯平滑
12. K-Means 算法迭代过程
13. PCA 降维步骤
14. 反向传播算法的链式法则
15. CNN 卷积层输出尺寸计算
16. LSTM 三门机制和细胞状态更新
17. Transformer 自注意力机制
18. Word2Vec 的 Skip-Gram 和 CBOW
19. GAN 的对抗训练过程
20. 目标检测 IoU 和 mAP 的计算

### 10.2 常见概念混淆

| 混淆点 | 正确理解 |
|--------|----------|
| 深度学习 vs 机器学习 | 深度学习是机器学习的子集 |
| 监督学习 vs 无监督学习 | 区别在于是否有标签 |
| 过拟合 vs 欠拟合 | 过拟合=方差大，欠拟合=偏差大 |
| L1 vs L2 正则化 | L1产生稀疏解，L2权重衰减 |
| SGD vs 批量梯度下降 | SGD每次用一个样本 |
| CNN vs 全连接网络 | CNN利用局部连接和权值共享 |
| RNN vs LSTM | LSTM有门控机制，解决梯度消失 |
| BERT vs GPT | BERT双向（编码器），GPT单向（解码器） |
| 生成模型 vs 判别模型 | 生成模型建模P(X,Y)，判别模型建模P(Y|X) |
| 偏差 vs 方差 | 偏差衡量预测值与真实值偏离，方差衡量预测值的离散程度 |

---

## 参考资源

- Stanford CS229 - Machine Learning: [cs229.stanford.edu](https://cs229.stanford.edu/)
- Stanford CS231n - CNN for Visual Recognition: [cs231n.stanford.edu](https://cs231n.stanford.edu/)
- Stanford CS224n - NLP with Deep Learning: [web.stanford.edu/class/cs224n/](http://web.stanford.edu/class/cs224n/)
- MIT 6.034 - Artificial Intelligence: [ocw.mit.edu/courses/6-034-artificial-intelligence-fall-2010/](https://ocw.mit.edu/courses/6-034-artificial-intelligence-fall-2010/)
- Andrew Ng Machine Learning Course: [Coursera](https://www.coursera.org/learn/machine-learning)
- Russell & Norvig, *Artificial Intelligence: A Modern Approach* (4th Edition)
- Goodfellow, Bengio & Courville, *Deep Learning* (deeplearningbook.org)
- 王万良《人工智能导论》
- 蔡自兴《人工智能基础教程》
- 李德毅《人工智能导论》
- Vaswani et al., "Attention Is All You Need" (2017)
- He et al., "Deep Residual Learning for Image Recognition" (2015)
- Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013)
- Goodfellow et al., "Generative Adversarial Networks" (2014)
