# NLP与计算机视觉完整知识库

## 一、词嵌入（Word Embeddings）

### 1.1 One-Hot编码的缺点
- 维度灾难：词汇表V通常数万维，极度稀疏
- 无法表达语义相似性：任意两个不同词正交，余弦相似度恒为0
- 离散无结构：无法进行语义推理（如 king - man + woman ≈ queen）
- 泛化能力差：每个词都是孤立符号

### 1.2 TF-IDF
TF(t,d) = f(t,d) / Σf(t',d)，IDF(t) = log(N/df(t))，TF-IDF = TF × IDF
能降低常见词权重，提升区分力词权重，但仍无法捕获词序和深层语义。

### 1.3 Word2Vec
CBOW：给上下文预测中心词，v_context = (1/2m)Σv_{w_{t+j}}，P(w_t|ctx) = softmax(u^T·v)
Skip-gram：给中心词预测上下文，P(o|c) = softmax(u_o^T·v_c)
负采样：J = logσ(u_o^T·v_c) + ΣE[logσ(-u_k^T·v_c)]，噪声分布P_n(w)∝f(w)^{3/4}

### 1.4 GloVe
共现矩阵X，加权最小二乘：J = Σf(X_ij)(w_i^T·w_j + b_i + b_j - logX_ij)^2
权重函数f(x) = (x/x_max)^α，x_max=100, α=0.75

### 1.5 FastText
词向量 = 所有字符n-gram向量之和：v_w = Σz_g
优势：可为OOV生成向量，在形态丰富语言中表现优异

### 1.6 评估方法
类比任务：argmax cos(d, b*-a*+a)，相似度任务：cos(u,v)与人工标注相关性

---

## 二、文本分类（Text Classification）

### 2.1 传统方法：BoW/TF-IDF + SVM
词袋模型忽略词序，SVM在高维稀疏特征上表现优异。
L = (1/2)||w||^2 + C·Σmax(0, 1-y_i(w^T·x_i+b))

### 2.2 TextCNN (Kim 2014)
输入n×k矩阵(n句长,k嵌入维度)，卷积c_i = f(w·x_{i:i+h-1}+b)
最大池化ĉ = max{c_1,...,c_{n-h+1}}，多尺度卷积核(3,4,5)拼接
输出y = softmax(W·z+b)，损失为交叉熵

### 2.3 LSTM文本分类
遗忘门f_t=σ(W_f·[h_{t-1},x_t])，输入门i_t=σ(W_i·[h_{t-1},x_t])
候选C̃_t=tanh(W_C·[h_{t-1},x_t])，记忆C_t=f_t*C_{t-1}+i_t*C̃_t
输出门o_t=σ(W_o·[h_{t-1},x_t])，h_t=o_t*tanh(C_t)
BiLSTM双向编码拼接，取h_T或注意力加权后分类

### 2.4 BERT微调
取[CLS]的h_{[CLS]}，P(y|x)=softmax(W·h_{[CLS]}+b)
学习率2e-5~5e-5，线性warmup+衰减，微调2~4 epoch

### 2.5 评估指标
Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，F1=2PR/(P+R)
Macro-F1：各类别F1算术平均；Micro-F1：汇总TP/FP/FN后计算

---

## 三、图像分类（Image Classification）

### 3.1 传统特征
HOG：cell内梯度方向直方图(9bin)→block归一化→拼接
SIFT：DoG尺度空间→关键点定位→方向分配→128维描述子(4×4×8)

### 3.2 CNN架构演进
LeNet-5(1998)→AlexNet(2012,15.3%)→VGG(2014,7.3%)→GoogLeNet(2014,6.7%)
→ResNet(2015,3.57%,残差连接y=F(x)+x，首次超越人类5.1%)

### 3.3 迁移学习
特征提取：冻结卷积层，只训练分类头
微调：冻结浅层(边缘纹理)，微调深层(任务特定)
渐进式解冻：逐步解冻更深层

### 3.4 数据增强
几何变换：裁剪/翻转/旋转/仿射
颜色变换：亮度/对比度/饱和度抖动
Mixup：x̃=λx_i+(1-λ)x_j, λ~Beta(α,α)
CutMix：随机区域替换，标签按面积混合

---

## 四、目标检测（Object Detection）

### 4.1 两阶段：R-CNN系列
R-CNN(2014)：Selective Search→CNN→SVM（47s/张）
Fast R-CNN(2015)：整图CNN一次+RoI Pooling+多任务损失
Faster R-CNN(2015)：RPN替代SS，Anchor(3尺度×3比例=9个/位置)
边框编码：t_x=(x-x_a)/w_a, t_w=log(w/w_a)
RPN损失：L_cls(前景/背景交叉熵) + λ·L_reg(Smooth L1)

### 4.2 一阶段：YOLO系列
YOLOv1：7×7网格，S×S×(B×5+C)输出
YOLOv3：多尺度预测(13/26/52)，Darknet-53
YOLOv8：Anchor-free，C2f模块，解耦头
SSD：多分辨率特征图检测，速度接近YOLO，精度接近Faster R-CNN

### 4.3 评估指标
IoU = Area(A∩B)/Area(A∪B)
NMS：按置信度排序，删除高IoU重叠框
mAP = (1/C)ΣAP_c，COCO用mAP@[0.5:0.95]

---

## 五、生成模型（Generative Models）

### 5.1 GAN (Goodfellow 2014)
min_G max_D V = E[logD(x)] + E[log(1-D(G(z)))]
最优判别器D* = p_data/(p_data+p_g)
模式崩溃：WGAN用Wasserstein距离替代JS散度

### 5.2 VAE (Kingma 2014)
ELBO = E[logp(x|z)] - KL(q(z|x)||p(z))
KL解析解 = -(1/2)Σ(1+logσ²-μ²-σ²)
重参数化：z = μ + σ·ε, ε~N(0,I)

### 5.3 扩散模型 DDPM (Ho 2020)
前向：x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
反向：p_θ(x_{t-1}|x_t) = N(μ_θ, Σ_θ)
训练：L = E[||ε - ε_θ(x_t, t)||²]
采样：x_T~N(0,I)，逐步去噪

### 5.4 对比
GAN：质量高但不稳定，模式覆盖差
VAE：稳定但模糊，潜在空间连续可插值
扩散：质量最高但速度慢，DALL-E 2/Stable Diffusion

---

## 六、大语言模型（Large Language Models）

### 6.1 预训练范式
自回归(GPT)：P(x)=ΠP(x_t|x_{<t})，因果掩码，适合生成
自编码(BERT)：MLM+NSP，双向注意力，适合理解
编码器-解码器(T5)：统一理解和生成

### 6.2 GPT系列
GPT-1(1.17亿)→GPT-2(15亿)→GPT-3(1750亿,In-context Learning)→GPT-4(多模态)

### 6.3 BERT预训练
MLM：15%token处理(80%MASK+10%随机+10%不变)
NSP：50%正例+50%负例二分类
L = L_MLM + L_NSP，RoBERTa去NSP效果更好

### 6.4 Prompt Engineering
Zero-shot/Few-shot/Chain-of-Thought/Self-Consistency/ReAct

### 6.5 RLHF
SFT→奖励模型(L_RM=-E[logσ(r(y_w)-r(y_l))])→PPO优化
DPO：直接从偏好数据优化，无需独立奖励模型

### 6.6 应用
文本生成、代码生成、RAG知识问答、摘要翻译、推理规划、多模态、Agent系统
