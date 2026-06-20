"""AI智学 - 知识图谱引擎

基于NetworkX构建人工智能导论课程知识图谱，
支持知识点关系查询、路径规划、依赖分析。
"""

import json
from pathlib import Path
from loguru import logger
import networkx as nx


class KnowledgeGraphEngine:
    """课程知识图谱引擎"""

    def __init__(self):
        self._graph = nx.DiGraph()
        self._knowledge_data: dict = {}
        self._load_graph()
        logger.info(f"Knowledge Graph loaded: {self._graph.number_of_nodes()} nodes, {self._graph.number_of_edges()} edges")

    def _load_graph(self) -> None:
        graph_path = Path(__file__).parent.parent / "knowledge_base" / "knowledge_graph.json"
        if graph_path.exists():
            with open(graph_path, "r", encoding="utf-8") as f:
                self._knowledge_data = json.load(f)
        else:
            self._knowledge_data = self._get_default_knowledge_graph()
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            with open(graph_path, "w", encoding="utf-8") as f:
                json.dump(self._knowledge_data, f, ensure_ascii=False, indent=2)

        for node in self._knowledge_data.get("nodes", []):
            self._graph.add_node(node["id"], name=node["name"], level=node.get("level", 0),
                                 difficulty=node.get("difficulty", 3), description=node.get("description", ""),
                                 keywords=node.get("keywords", []))
        for edge in self._knowledge_data.get("edges", []):
            self._graph.add_edge(edge["from"], edge["to"], relation=edge.get("relation", "prerequisite"))

    def get_all_topics(self) -> list[dict]:
        return [{"id": nid, "name": d["name"], "level": d["level"], "difficulty": d["difficulty"]}
                for nid, d in self._graph.nodes(data=True)]

    def get_topic_detail(self, topic_id: str) -> dict | None:
        if topic_id not in self._graph:
            return None
        d = self._graph.nodes[topic_id]
        return {"id": topic_id, "name": d["name"], "level": d["level"], "difficulty": d["difficulty"],
                "description": d["description"], "keywords": d["keywords"],
                "prerequisites": list(self._graph.predecessors(topic_id)),
                "next_topics": list(self._graph.successors(topic_id))}

    def get_learning_path(self, target_topic: str, mastered_topics: list[str] | None = None) -> list[dict]:
        """规划学习路径 - 拓扑排序，排除已掌握"""
        if target_topic not in self._graph:
            return []
        mastered = set(mastered_topics or [])
        predecessors = set()
        queue = [target_topic]
        visited = set()
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for pred in self._graph.predecessors(node):
                predecessors.add(pred)
                queue.append(pred)

        sub_graph = self._graph.subgraph(predecessors | {target_topic})
        try:
            ordered = list(nx.topological_sort(sub_graph))
        except nx.NetworkXError:
            ordered = list(predecessors | {target_topic})

        path, step = [], 1
        for node_id in ordered:
            if node_id in mastered:
                continue
            d = self._graph.nodes[node_id]
            path.append({"step": step, "topic_id": node_id, "topic_name": d["name"],
                         "difficulty": d["difficulty"], "status": "pending"})
            step += 1
        return path

    def search_topics(self, keyword: str) -> list[dict]:
        kw = keyword.lower()
        return [{"id": nid, "name": d["name"], "difficulty": d["difficulty"], "description": d["description"]}
                for nid, d in self._graph.nodes(data=True)
                if kw in d["name"].lower() or kw in d.get("description", "").lower()
                or any(kw in k.lower() for k in d.get("keywords", []))]

    def get_graph_data(self) -> dict:
        """获取D3.js可视化数据"""
        nodes = [{"id": nid, "name": d["name"], "level": d["level"], "difficulty": d["difficulty"], "group": d["level"]}
                 for nid, d in self._graph.nodes(data=True)]
        links = [{"source": u, "target": v, "relation": d.get("relation", "")}
                 for u, v, d in self._graph.edges(data=True)]
        return {"nodes": nodes, "links": links}

    @staticmethod
    def _get_default_knowledge_graph() -> dict:
        """AI导论课程默认知识图谱 (30个核心知识点)"""
        return {
            "nodes": [
                {"id": "ai_overview", "name": "AI概述与发展历史", "level": 0, "difficulty": 1,
                 "description": "人工智能(AI)是计算机科学的核心分支，旨在创建能模拟人类智能行为的系统。本节涵盖Russell和Norvig提出的四种AI定义框架（像人一样思考/行动、理性地思考/行动），图灵测试的原理与局限性，以及AI从1956年达特茅斯会议至今的三次发展浪潮和两次寒冬。重点理解符号主义、连接主义、行为主义三大流派的核心思想差异。",
                 "keywords": ["AI", "图灵测试", "达特茅斯会议", "符号主义", "连接主义"]},

                {"id": "search_overview", "name": "搜索算法概述", "level": 1, "difficulty": 2,
                 "description": "搜索是AI解决问题的基础方法，将问题抽象为状态空间中的路径寻找。本节介绍搜索问题的形式化定义：状态空间、初始状态、动作集合、转移模型、目标测试和路径代价。理解搜索树与状态空间图的区别，掌握无信息搜索和有信息搜索的分类，以及完备性、最优性、时间复杂度、空间复杂度四个评价指标。",
                 "keywords": ["状态空间", "搜索树", "完备性", "最优性"]},

                {"id": "uninformed_search", "name": "无信息搜索", "level": 2, "difficulty": 2,
                 "description": "无信息搜索（盲搜索）不利用任何领域知识，仅按系统规则扩展节点。广度优先搜索(BFS)逐层扩展，保证最浅路径最优但空间开销大；深度优先搜索(DFS)沿分支深入，空间效率高但不保证最优；统一代价搜索(UCS)按路径代价排序，保证最小代价最优；迭代加深搜索(IDS)结合BFS和DFS优点，是树搜索的最优无信息算法。需要掌握每种算法的伪代码、时间空间复杂度、完备性和最优性条件。",
                 "keywords": ["BFS", "DFS", "UCS", "迭代加深"]},

                {"id": "informed_search", "name": "有信息搜索", "level": 2, "difficulty": 3,
                 "description": "有信息搜索利用启发函数h(n)估计当前节点到目标的距离，大幅提高搜索效率。A*算法是最经典的有信息搜索，评估函数f(n)=g(n)+h(n)。关键概念：可纳性（启发值不高估实际代价）、一致性（三角不等式条件）、一致性蕴含可纳性的证明。启发函数设计方法如松弛法，以及A*最优性的反证法证明都是考试重点。",
                 "keywords": ["A*", "启发函数", "可纳性", "一致性"]},

                {"id": "adversarial_search", "name": "对抗搜索", "level": 2, "difficulty": 3,
                 "description": "对抗搜索用于双人零和博弈场景（如国际象棋、围棋）。Minimax算法通过递归评估博弈树，假设双方都采取最优策略。Alpha-Beta剪枝通过维护上下界剪除不可能影响最终决策的分支，将时间复杂度从O(b^m)优化到最优O(b^(m/2))。需要掌握Minimax伪代码、博弈树手动追踪、以及Alpha-Beta剪枝的详细过程。蒙特卡洛树搜索(MCTS)是现代围棋AI的核心技术。",
                 "keywords": ["Minimax", "Alpha-Beta", "博弈树", "MCTS"]},

                {"id": "knowledge_overview", "name": "知识表示与推理", "level": 1, "difficulty": 2,
                 "description": "知识表示是AI的核心问题之一，研究如何将人类知识编码为计算机可处理的形式。主要方法包括逻辑表示（命题逻辑、谓词逻辑）、产生式规则、语义网络、框架、本体论等。推理是从已有知识推导新结论的过程，分为演绎推理（从一般到特殊）、归纳推理（从特殊到一般）和溯因推理（从结果推原因）。",
                 "keywords": ["知识表示", "推理", "逻辑", "本体论"]},

                {"id": "logic", "name": "逻辑推理", "level": 2, "difficulty": 3,
                 "description": "命题逻辑用命题变量和联结词（与、或、非、蕴含、等价）表示知识，通过真值表判断可满足性。谓词逻辑引入量词（全称量词、存在量词）和谓词函数，表达能力更强。关键推理方法：归结推理（将公式转为CNF后应用归结规则）、反演法（否定结论导出矛盾来证明）、合一算法（寻找变量替换使原子公式一致）。Skolem化和前束范式转换是重要步骤。",
                 "keywords": ["命题逻辑", "谓词逻辑", "归结推理", "合一算法", "Skolem化"]},

                {"id": "probabilistic", "name": "概率推理", "level": 2, "difficulty": 4,
                 "description": "概率推理处理不完整和不确定信息。贝叶斯网络用有向无环图表示变量间的条件依赖关系，每个节点附带条件概率表(CPT)。核心概念：贝叶斯定理P(A|B)=P(B|A)P(A)/P(B)、联合概率分解、条件独立性。D-Separation是判断条件独立的图方法，包含链式、分叉、对撞三种基本结构。变量消除法是精确推理的主要算法。在医疗诊断、垃圾邮件过滤等场景有广泛应用。",
                 "keywords": ["贝叶斯网络", "条件概率", "D-Separation", "变量消除"]},

                {"id": "ml_overview", "name": "机器学习概述", "level": 1, "difficulty": 2,
                 "description": "机器学习使计算机能从数据中自动学习规律而无需显式编程。三大范式：监督学习（有标签数据，分类和回归）、无监督学习（无标签数据，聚类和降维）、强化学习（与环境交互获取奖励）。关键概念：过拟合与欠拟合、偏差-方差权衡、交叉验证评估方法、训练集/验证集/测试集划分。理解特征工程、模型选择和超参数调优的基本流程。",
                 "keywords": ["监督学习", "无监督学习", "过拟合", "交叉验证", "偏差方差"]},

                {"id": "linear_regression", "name": "线性回归", "level": 2, "difficulty": 2,
                 "description": "线性回归是最基础的监督学习算法，通过线性函数y=w^T*x+b拟合连续目标值。损失函数使用均方误差MSE，可通过正规方程w=(X^TX)^(-1)X^Ty直接求解，或通过梯度下降迭代优化。正则化防止过拟合：L2正则化(Ridge)使权重趋向小值；L1正则化(Lasso)产生稀疏解实现特征选择。R²决定系数衡量拟合优度。这是理解后续所有算法的数学基础。",
                 "keywords": ["线性回归", "MSE", "正规方程", "正则化", "Ridge", "Lasso"]},

                {"id": "logistic_regression", "name": "逻辑回归", "level": 2, "difficulty": 2,
                 "description": "逻辑回归是经典的二分类算法。核心是Sigmoid函数σ(z)=1/(1+e^(-z))，将线性输出映射到(0,1)概率区间。损失函数使用交叉熵（从最大似然估计MLE推导），梯度形式与线性回归统一。多分类扩展使用Softmax函数。决策边界是线性的。理解Sigmoid的导数σ'=σ(1-σ)、交叉熵推导过程、以及逻辑回归与线性回归的联系和区别。",
                 "keywords": ["逻辑回归", "Sigmoid", "交叉熵", "Softmax", "决策边界"]},

                {"id": "decision_tree", "name": "决策树", "level": 2, "difficulty": 2,
                 "description": "决策树通过递归分裂特征空间进行分类或回归，具有良好的可解释性。三种经典算法：ID3使用信息增益（基于信息熵H=-Σp*log₂p），C4.5使用增益率修正偏差，CART使用基尼指数Gini=1-Σp²。剪枝策略防止过拟合：预剪枝（限制深度）和后剪枝（代价复杂度剪枝CCP）。回归树使用方差作为分裂标准，叶节点输出均值。",
                 "keywords": ["决策树", "信息增益", "基尼指数", "ID3", "C4.5", "CART", "剪枝"]},

                {"id": "svm", "name": "支持向量机", "level": 2, "difficulty": 4,
                 "description": "SVM基于统计学习理论，核心思想是找到最大间隔超平面分离数据。软间隔引入松弛变量和惩罚参数C。通过拉格朗日对偶转化为对偶问题，最优解仅依赖支持向量。核技巧处理非线性：线性核、多项式核、RBF核、Sigmoid核。Hinge Loss是SVM的损失函数。KKT条件是求解对偶问题的必要条件。掌握对偶推导过程和核函数选择是考试重点。",
                 "keywords": ["SVM", "最大间隔", "核函数", "对偶问题", "KKT条件", "Hinge Loss"]},

                {"id": "naive_bayes", "name": "朴素贝叶斯", "level": 2, "difficulty": 2,
                 "description": "朴素贝叶斯基于贝叶斯定理和特征条件独立假设P(y|X)∝P(y)∏P(xi|y)。三种变体：高斯（连续特征）、多项式（离散计数，常用于文本分类）、伯努利（二值特征）。拉普拉斯平滑解决零概率问题。训练和预测速度极快O(nd)，是文本分类的常用基线模型。理解从贝叶斯定理到分类决策的完整推导过程。",
                 "keywords": ["朴素贝叶斯", "贝叶斯定理", "拉普拉斯平滑", "条件独立"]},

                {"id": "unsupervised", "name": "无监督学习", "level": 2, "difficulty": 3,
                 "description": "无监督学习从无标签数据中发现隐藏结构。K-Means：随机初始化K个质心→分配→更新→迭代收敛。K-Means++改进初始化。选K方法：肘部法则、轮廓系数。PCA降维：中心化→协方差矩阵→特征值分解→前k个特征向量投影。PCA最大化投影方差，等价于最小化重构误差。理解K-Means目标函数和PCA的完整数学推导。",
                 "keywords": ["K-Means", "PCA", "聚类", "降维", "特征值分解"]},

                {"id": "neuron", "name": "神经元与感知机", "level": 2, "difficulty": 2,
                 "description": "M-P神经元模型（1943）：输入加权求和后通过激活函数产生输出。感知机（Rosenblatt, 1958）能学习线性可分问题，但无法解决异或(XOR)问题（Minsky, 1969）。多层感知机(MLP)通过引入隐藏层解决非线性。常用激活函数：Sigmoid（梯度消失）、Tanh（零中心）、ReLU（缓解梯度消失）、GELU、Swish。万能近似定理证明单隐层MLP可逼近任意连续函数。",
                 "keywords": ["感知机", "M-P模型", "激活函数", "ReLU", "MLP", "万能近似定理"]},

                {"id": "backpropagation", "name": "反向传播", "level": 2, "difficulty": 4,
                 "description": "反向传播(BP)算法是训练神经网络的核心。前向传播逐层计算z=W*a+b, a=g(z)。反向传播利用链式法则从输出层向输入层计算梯度：输出层误差δ_L=ŷ-y，隐藏层误差δ_l=(W_{l+1})^T*δ_{l+1}⊙g'(z_l)。Batch Normalization通过对mini-batch标准化加速训练。Dropout随机丢弃神经元防止过拟合。掌握完整推导过程和三种输出层梯度的统一形式是考试核心。",
                 "keywords": ["反向传播", "链式法则", "BatchNorm", "Dropout", "梯度"]},

                {"id": "optimization", "name": "优化算法", "level": 2, "difficulty": 3,
                 "description": "优化算法决定神经网络参数的更新方式。SGD用mini-batch估计梯度。Momentum引入动量项加速收敛。Adam结合一阶矩和二阶矩自适应调整学习率，包含偏差校正，是最常用的优化器。学习率调度：StepLR、CosineAnnealing、Warmup。梯度消失（Sigmoid深层梯度指数衰减）和梯度爆炸的原因及解决方案（ReLU、残差连接、梯度裁剪、BN）。",
                 "keywords": ["SGD", "Adam", "学习率", "梯度消失", "梯度爆炸", "Momentum"]},

                {"id": "cnn", "name": "卷积神经网络(CNN)", "level": 1, "difficulty": 4,
                 "description": "CNN是处理网格结构数据的核心模型，通过局部感受野和参数共享减少参数量。卷积层提取特征，输出尺寸公式out=(in-k+2p)/s+1。池化层降低尺寸增强平移不变性。经典架构：LeNet-5→AlexNet(ReLU+Dropout)→VGG(3×3小卷积核)→GoogLeNet(Inception模块)→ResNet(残差连接y=F(x)+x)→DenseNet。残差连接有效原因：梯度直通、集成学习视角、损失曲面平滑。",
                 "keywords": ["CNN", "卷积", "池化", "ResNet", "残差连接", "VGG"]},

                {"id": "rnn_lstm", "name": "循环神经网络(RNN/LSTM)", "level": 1, "difficulty": 4,
                 "description": "RNN处理序列数据，通过隐状态h_t=tanh(W_hh·h_{t-1}+W_xh·x_t)传递时序信息。Vanilla RNN存在梯度消失问题（梯度连乘导致长距离依赖信息丢失）。LSTM通过门控机制解决：遗忘门f_t、输入门i_t、输出门o_t，细胞状态C_t提供梯度直通路径。GRU是LSTM简化版（合并遗忘门和输入门）。Bi-RNN同时正向反向编码。Seq2Seq+Attention是机器翻译经典架构。",
                 "keywords": ["RNN", "LSTM", "GRU", "梯度消失", "门控机制", "Seq2Seq"]},

                {"id": "transformer", "name": "Transformer", "level": 1, "difficulty": 5,
                 "description": "Transformer（Vaswani等, 2017）彻底改变了深度学习。核心是自注意力机制：Attention(Q,K,V)=softmax(QK^T/√d_k)V。除以√d_k防止softmax梯度极小。多头注意力并行捕获不同子空间模式。位置编码用正弦/余弦函数注入位置信息。Encoder-Decoder架构：编码器双向注意力，解码器因果注意力+交叉注意力。BERT(自编码)vs GPT(自回归)是两种预训练范式。",
                 "keywords": ["Transformer", "自注意力", "多头注意力", "位置编码", "BERT", "GPT"]},

                {"id": "llm", "name": "大语言模型(LLM)", "level": 1, "difficulty": 5,
                 "description": "大语言模型是当前AI最前沿方向。GPT系列：GPT-1(1.17亿)→GPT-2(15亿,零样本)→GPT-3(1750亿,In-context Learning)→GPT-4(多模态)。BERT通过MLM和NSP双向预训练。Prompt Engineering：Zero-shot、Few-shot、Chain-of-Thought。RLHF是ChatGPT核心训练方法：SFT→奖励模型→PPO优化。RAG结合外部知识库解决幻觉问题。理解预训练-微调-对齐全流程。",
                 "keywords": ["LLM", "GPT", "BERT", "RLHF", "Prompt", "RAG", "In-context Learning"]},

                {"id": "word_embedding", "name": "词嵌入", "level": 1, "difficulty": 3,
                 "description": "词嵌入将离散词汇映射到低维稠密向量空间，使语义相似的词距离相近。Word2Vec有两种架构：CBOW（上下文预测中心词）和Skip-gram（中心词预测上下文），负采样提高效率。GloVe结合全局共现矩阵和局部上下文窗口。FastText引入字符n-gram，能为未登录词生成向量。评估：词类比任务（king-man+woman≈queen）、词相似度任务。",
                 "keywords": ["Word2Vec", "GloVe", "FastText", "词嵌入", "CBOW", "Skip-gram"]},

                {"id": "text_classification", "name": "文本分类", "level": 1, "difficulty": 3,
                 "description": "文本分类将文本分配到预定义类别。传统方法：BoW+TF-IDF+SVM。深度学习：TextCNN用不同宽度卷积核捕获n-gram特征；LSTM/BiLSTM序列建模；BERT微调取[CLS]隐状态分类。评估指标：混淆矩阵→精确率Precision=TP/(TP+FP)→召回率Recall=TP/(TP+FN)→F1=2PR/(P+R)。多分类：Macro-F1、Micro-F1、Weighted-F1。",
                 "keywords": ["文本分类", "TextCNN", "BERT微调", "F1", "精确率", "召回率"]},

                {"id": "image_classification", "name": "图像分类", "level": 1, "difficulty": 3,
                 "description": "图像分类是计算机视觉基础任务。传统方法：HOG（方向梯度直方图）、SIFT（尺度不变特征变换，128维描述子）。CNN端到端分类是主流。迁移学习：冻结卷积层训练分类头或微调。数据增强：几何变换、Mixup、CutMix。ImageNet竞赛推动深度学习革命：AlexNet(15.3%)→VGG(7.3%)→ResNet(3.57%，超越人类5.1%)。",
                 "keywords": ["图像分类", "HOG", "SIFT", "迁移学习", "数据增强", "ImageNet"]},

                {"id": "object_detection", "name": "目标检测", "level": 1, "difficulty": 4,
                 "description": "目标检测同时定位和识别多个目标。两阶段：R-CNN→Fast R-CNN→Faster R-CNN(RPN+Anchor机制)。边框编码t_x=(x-x_a)/w_a。一阶段：YOLO直接预测，SSD多尺度检测。评估：IoU=Area(A∩B)/Area(A∪B)、NMS去冗余、mAP各类别AP均值。COCO使用mAP@[0.5:0.95]。掌握RPN原理和Anchor机制是核心考点。",
                 "keywords": ["目标检测", "R-CNN", "YOLO", "RPN", "Anchor", "IoU", "mAP"]},

                {"id": "generation", "name": "生成模型", "level": 1, "difficulty": 5,
                 "description": "生成模型学习数据分布并生成新样本。GAN通过G和D对抗训练：min_G max_D V=E[logD(x)]+E[log(1-D(G(z)))]。VAE基于变分推断优化ELBO，重参数化技巧z=μ+σε使采样可微。扩散模型DDPM前向逐步加噪，反向学习去噪，训练L=E[||ε-ε_θ||²]。对比：GAN质量高但不稳定，VAE稳定但模糊，扩散质量最高但慢。掌握GAN损失推导和VAE的ELBO推导是考试重点。",
                 "keywords": ["GAN", "VAE", "扩散模型", "DDPM", "ELBO", "重参数化"]},
            ],
            "edges": [
                {"from": "ai_overview", "to": "search_overview", "relation": "prerequisite"},
                {"from": "ai_overview", "to": "knowledge_overview", "relation": "prerequisite"},
                {"from": "ai_overview", "to": "ml_overview", "relation": "prerequisite"},
                {"from": "search_overview", "to": "uninformed_search", "relation": "prerequisite"},
                {"from": "uninformed_search", "to": "informed_search", "relation": "prerequisite"},
                {"from": "search_overview", "to": "adversarial_search", "relation": "prerequisite"},
                {"from": "knowledge_overview", "to": "logic", "relation": "prerequisite"},
                {"from": "logic", "to": "probabilistic", "relation": "prerequisite"},
                {"from": "ml_overview", "to": "linear_regression", "relation": "prerequisite"},
                {"from": "linear_regression", "to": "logistic_regression", "relation": "prerequisite"},
                {"from": "ml_overview", "to": "decision_tree", "relation": "prerequisite"},
                {"from": "ml_overview", "to": "naive_bayes", "relation": "prerequisite"},
                {"from": "linear_regression", "to": "svm", "relation": "prerequisite"},
                {"from": "ml_overview", "to": "unsupervised", "relation": "prerequisite"},
                {"from": "linear_regression", "to": "neuron", "relation": "prerequisite"},
                {"from": "logistic_regression", "to": "neuron", "relation": "prerequisite"},
                {"from": "neuron", "to": "backpropagation", "relation": "prerequisite"},
                {"from": "backpropagation", "to": "optimization", "relation": "prerequisite"},
                {"from": "backpropagation", "to": "cnn", "relation": "prerequisite"},
                {"from": "backpropagation", "to": "rnn_lstm", "relation": "prerequisite"},
                {"from": "rnn_lstm", "to": "transformer", "relation": "prerequisite"},
                {"from": "transformer", "to": "llm", "relation": "prerequisite"},
                {"from": "ml_overview", "to": "word_embedding", "relation": "prerequisite"},
                {"from": "word_embedding", "to": "text_classification", "relation": "prerequisite"},
                {"from": "word_embedding", "to": "rnn_lstm", "relation": "prerequisite"},
                {"from": "cnn", "to": "image_classification", "relation": "prerequisite"},
                {"from": "image_classification", "to": "object_detection", "relation": "prerequisite"},
                {"from": "cnn", "to": "generation", "relation": "prerequisite"},
                {"from": "probabilistic", "to": "generation", "relation": "prerequisite"},
            ],
        }


_kg_engine: KnowledgeGraphEngine | None = None


def get_knowledge_graph() -> KnowledgeGraphEngine:
    global _kg_engine
    if _kg_engine is None:
        _kg_engine = KnowledgeGraphEngine()
    return _kg_engine
