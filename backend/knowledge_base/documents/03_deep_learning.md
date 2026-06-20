# 神经网络基础
# Neural Network Foundations

## 参考来源
- Stanford CS231n: Convolutional Neural Networks for Visual Recognition
- d2l.ai: Dive into Deep Learning (动手学深度学习)
- PyTorch 官方文档
- Rosenblatt (1958), Minsky & Papert (1969), Hornik et al. (1989)

---

## 1. McCulloch-Pitts (M-P) 神经元模型

### 1.1 历史背景

1943年，Warren McCulloch 和 Walter Pitts 提出了第一个数学化的神经元模型，即 M-P 模型。该模型模仿生物神经元的工作方式：接收多个输入信号，当信号总和超过某个阈值时，神经元被激活（输出1），否则保持静默（输出0）。

### 1.2 数学定义

M-P 神经元的输入输出关系定义如下：

```
给定 n 个二值输入 x_1, x_2, ..., x_n ∈ {0, 1}
以及对应的权重 w_1, w_2, ..., w_n
阈值为 θ

输出 y 定义为:

y = f(Σ(w_i * x_i) - θ)

其中 f 是阶跃函数 (step function):

f(z) = 1,  if z >= 0
f(z) = 0,  if z < 0
```

### 1.3 模型局限性

- 权重和阈值需要人工设定，没有学习能力
- 只能表示线性可分的布尔函数
- 无法解决 XOR（异或）问题，Minsky 和 Papert 在1969年《Perceptrons》一书中严格证明了这一点

### 1.4 M-P 模型与现代神经元的对比

| 特征 | M-P 模型 | 现代神经元 |
|------|----------|-----------|
| 输入 | 二值 {0, 1} | 连续实数 |
| 权重 | 人工设定 | 通过学习自动调整 |
| 激活函数 | 阶跃函数 | Sigmoid/ReLU等连续函数 |
| 可学习性 | 不可学习 | 可通过梯度下降学习 |
| 能力 | 仅线性可分函数 | 万能近似能力 |

---

## 2. 感知机学习算法 (Perceptron Learning Algorithm)

### 2.1 模型定义

感知机（Perceptron）由 Frank Rosenblatt 于1958年提出，是第一个具有学习能力的神经网络模型。

```
输入: x = (x_1, x_2, ..., x_n)
权重: w = (w_1, w_2, ..., w_n)
偏置: b

输出: y = f(w · x + b) = sign(w · x + b)

其中:
sign(z) = +1,  if z >= 0
sign(z) = -1, if z < 0
```

### 2.2 学习规则

给定训练样本 (x_i, y_i)，其中 y_i ∈ {+1, -1}，感知机的学习规则为：

```
当预测错误时（即 y_i * (w · x_i + b) <= 0），更新权重：

w ← w + η * y_i * x_i
b ← b + η * y_i

其中 η > 0 是学习率
```

直觉解释：当把正样本误判为负类时（y_i = +1 但预测为负），增大权重使 w · x_i 增大；反之亦然。

### 2.3 收敛性定理 (Perceptron Convergence Theorem)

**定理（Novikoff, 1963）**: 若训练数据集是线性可分的，即存在超平面 w* · x + b* = 0 将正负样本完全分开，则感知机学习算法在有限步数内收敛。

**证明概要**：

假设存在分离超平面 (w*, b*) 满足：
```
对所有正样本: w* · x_i + b* >= γ > 0
对所有负样本: w* · x_i + b* <= -γ < 0

其中 γ = min_i |w* · x_i + b*| > 0 是数据集到超平面的最小间隔（margin）
```

不失一般性，设 ||w*|| = 1。

令 R = max_i ||x_i|| 表示样本的最大范数。

**第 k 次更新的分析**：

(1) 内积增长下界：第 k 次更新发生在样本 (x_k, y_k) 上：
```
w · w* >= (k-1) * η * γ

推导：每次更新使 w · w* 至少增加 η * γ
w_{t+1} · w* = w_t · w* + η * y_k * (x_k · w*)
             >= w_t · w* + η * γ
```

(2) 权重范数增长上界：
```
||w_{t+1}||^2 = ||w_t + η * y_k * x_k||^2
              = ||w_t||^2 + 2η * y_k * (w_t · x_k) + η^2 * ||x_k||^2
              <= ||w_t||^2 + η^2 * R^2

（因为 y_k * (w_t · x_k) <= 0 是更新条件）

递推得: ||w_k||^2 <= k * η^2 * R^2
```

(3) 结合(1)和(2)：
```
cos θ = (w · w*) / (||w|| * ||w*||)
      >= (k-1) * η * γ / sqrt(k * η^2 * R^2)
      = (k-1) * γ / (sqrt(k) * R)

由于 cos θ <= 1，解不等式:

k <= (R / γ)^2

即更新次数 k 有限，算法必然在有限步内收敛。
```

**推论**：收敛速度与 R^2/γ^2 成正比。数据集的间隔 γ 越大，收敛越快；样本范数 R 越大，收敛越慢。

### 2.4 感知机的局限

- 只能处理线性可分问题
- 无法学习 XOR 等非线性函数
- 解的不唯一性：依赖于初始权重和样本顺序
- 对噪声数据敏感

---

## 3. 多层感知机 (Multi-Layer Perceptron, MLP)

### 3.1 网络结构

MLP 由输入层、一个或多个隐藏层和输出层组成。每层的每个神经元与下一层的所有神经元全连接。

```
网络架构示例 (输入维度 n, 隐藏层维度 h, 输出维度 o):

输入层:   x ∈ R^n
隐藏层1:  z_1 = W_1 * x + b_1,    a_1 = σ(z_1),     W_1 ∈ R^{h×n},  b_1 ∈ R^h
隐藏层2:  z_2 = W_2 * a_1 + b_2,   a_2 = σ(z_2),     W_2 ∈ R^{h×h},  b_2 ∈ R^h
输出层:   z_3 = W_3 * a_2 + b_3,   y = f_out(z_3),    W_3 ∈ R^{o×h},  b_3 ∈ R^o

其中 σ 是隐藏层激活函数, f_out 是输出层激活函数
```

### 3.2 前向传播通用公式

对于 L 层 MLP，前向传播可递归表示为：

```
a_0 = x                          (输入)
z_l = W_l * a_{l-1} + b_l        (线性变换, l = 1, 2, ..., L)
a_l = σ(z_l)                     (非线性激活, l = 1, 2, ..., L-1)
y = f_out(z_L)                   (输出层激活)
```

### 3.3 损失函数

**回归任务**：均方误差损失
```
L_MSE = (1/N) * Σ_i ||y_i - ŷ_i||^2
```

**二分类任务**：二元交叉熵损失
```
L_BCE = -(1/N) * Σ_i [y_i * log(ŷ_i) + (1 - y_i) * log(1 - ŷ_i)]
```

**多分类任务**：交叉熵损失（配合 Softmax 输出）
```
L_CE = -(1/N) * Σ_i Σ_c y_{i,c} * log(ŷ_{i,c})

其中 ŷ = softmax(z_L)
softmax(z)_c = exp(z_c) / Σ_j exp(z_j)
```

---

## 4. 激活函数完整对比

### 4.1 Sigmoid 函数

**公式**：
```
σ(x) = 1 / (1 + exp(-x))
```

**导数**：
```
σ'(x) = σ(x) * (1 - σ(x))

性质: σ'(x) ∈ (0, 0.25], 最大值在 x=0 处取得
```

**输出范围**: (0, 1)

**优点**：
- 输出有明确的概率解释
- 平滑可微
- 历史上广泛使用，有丰富的理论基础

**缺点**：
- 梯度消失问题：当 |x| 较大时，梯度趋近于 0，导致深层网络训练困难
- 输出非零中心：σ(x) 恒大于 0，导致权重更新始终同号，使收敛变慢（zig-zag path）
- 包含指数运算，计算开销相对较大
- 不是零中心的 (not zero-centered)

### 4.2 Tanh 函数

**公式**：
```
tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        = 2σ(2x) - 1
```

**导数**：
```
tanh'(x) = 1 - tanh^2(x)

性质: tanh'(x) ∈ (0, 1], 最大值在 x=0 处为 1
```

**输出范围**: (-1, 1)

**优点**：
- 零中心输出，解决了 Sigmoid 的非零中心问题
- 梯度值范围更大（最大为1 vs Sigmoid的最大为0.25），梯度消失相对较轻

**缺点**：
- 仍然存在梯度消失问题（当 |x| 较大时梯度趋近于0）
- 仍然需要指数运算

### 4.3 ReLU 函数 (Rectified Linear Unit)

**公式**：
```
ReLU(x) = max(0, x)
```

**导数**：
```
ReLU'(x) = 1,  if x > 0
ReLU'(x) = 0,  if x < 0
x = 0 处不可微，实践中取 0 或 1
```

**输出范围**: [0, +∞)

**优点**：
- 计算极其简单高效
- 正区间梯度恒为1，有效缓解梯度消失问题
- 引入稀疏性：约50%的神经元输出为0
- 加速收敛（约6倍于 Sigmoid/Tanh，Krizhevsky et al., 2012）

**缺点**：
- "Dead ReLU" 问题：如果神经元的输入始终为负，该神经元将永远输出0，梯度也为0，无法更新
- 输出非零中心
- 在 x=0 处不可微（实践中影响不大）

**Dead ReLU 问题的直觉解释**：较大的学习率可能导致权重更新过大，使得神经元的输入 w·x+b 始终为负。一旦进入"死亡"状态，梯度为零，该神经元再也无法恢复。

### 4.4 Leaky ReLU 函数

**公式**：
```
LeakyReLU(x) = x,          if x > 0
LeakyReLU(x) = α * x,      if x <= 0

其中 α 是一个较小的正数（通常 α = 0.01）
```

**导数**：
```
LeakyReLU'(x) = 1,      if x > 0
LeakyReLU'(x) = α,      if x < 0
```

**输出范围**: (-∞, +∞)

**优点**：
- 解决了 Dead ReLU 问题：负区间有非零梯度 α
- 继承了 ReLU 计算高效的特点
- 输出范围包括负值，近似零中心

**缺点**：
- α 需要手动设定（虽然实践表明对 α 的选择不太敏感）
- 效果不一定总优于标准 ReLU

**变体**：
- **PReLU** (Parametric ReLU): α 作为可学习参数
- **RReLU** (Randomized ReLU): α 在训练时从均匀分布中随机采样，测试时取均值

### 4.5 GELU 函数 (Gaussian Error Linear Unit)

**公式**：
```
GELU(x) = x * Φ(x)

其中 Φ(x) 是标准正态分布的累积分布函数 (CDF):
Φ(x) = (1/2) * [1 + erf(x / sqrt(2))]
```

**近似公式**（实践中常用）：
```
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

或者更精确的 Sigmoid 近似:
GELU(x) ≈ x * σ(1.702 * x)
```

**导数**（精确形式）：
```
GELU'(x) = Φ(x) + x * φ(x)

其中 φ(x) 是标准正态分布的概率密度函数:
φ(x) = (1/sqrt(2π)) * exp(-x^2/2)
```

**输出范围**: 约 (-0.17, +∞)

**优点**：
- 非单调：在 x 约为 -0.17 处有最小值约 -0.17
- 在零点附近有平滑的过渡，比 ReLU 更自然
- Transformer 和 BERT、GPT 等模型的标配激活函数
- 有概率论直觉：以概率 Φ(x) 保留输入 x，类似随机正则化

**缺点**：
- 计算较复杂，但近似版本效率接近 ReLU
- 理论分析较困难

**直觉解释**：GELU 可以理解为一种"软门控"——输入 x 越大（正值），被保留的概率越高；输入越小（负值），被丢弃的概率越高。这种行为比 ReLU 的硬阈值更柔和。

### 4.6 Swish / SiLU 函数

**公式**：
```
Swish(x) = x * σ(βx)

其中 σ 是 Sigmoid 函数，β 是可学习参数或固定常数。
当 β = 1 时，称为 SiLU (Sigmoid Linear Unit):
SiLU(x) = x * σ(x) = x / (1 + exp(-x))
```

**导数**：
```
Swish'(x) = σ(βx) + βx * σ(βx) * (1 - σ(βx))
          = σ(βx) + βx * σ(βx) - βx * σ(βx)^2
          = σ(βx) * (1 + βx * (1 - σ(βx)))
          = σ(βx) + βx * σ(βx) * (1 - σ(βx))

当 β = 1 时:
SiLU'(x) = σ(x) + x * σ(x) * (1 - σ(x))
         = σ(x) * (1 + x - x * σ(x))
```

**输出范围**: 约 (-0.278, +∞)（当 β=1 时）

**优点**：
- 非单调特性，与 GELU 类似
- 在多项基准测试中优于 ReLU（Ramachandran et al., 2017）
- EfficientNet 和多项现代架构中的默认激活函数
- 平滑、处处可微

**缺点**：
- 计算成本高于 ReLU
- 需要 Sigmoid 运算

### 4.7 激活函数综合对比表

| 激活函数 | 公式 | 输出范围 | 零中心 | 单调性 | 计算效率 | 主要问题 | 典型应用 |
|---------|------|---------|--------|--------|---------|---------|---------|
| Sigmoid | 1/(1+e^-x) | (0,1) | 否 | 是 | 低 | 梯度消失，非零中心 | 二分类输出层 |
| Tanh | (e^x-e^-x)/(e^x+e^-x) | (-1,1) | 是 | 是 | 低 | 梯度消失 | RNN |
| ReLU | max(0,x) | [0,+∞) | 否 | 非负区间是 | 高 | Dead ReLU | CNN默认 |
| LeakyReLU | max(αx,x) | (-∞,+∞) | 近似 | 非负区间是 | 高 | 需设α | 替代ReLU |
| GELU | x*Φ(x) | (-0.17,+∞) | 否 | 非单调 | 中 | 计算较复杂 | Transformer |
| Swish/SiLU | x*σ(x) | (-0.278,+∞) | 否 | 非单调 | 中 | 计算较复杂 | EfficientNet |

### 4.8 PyTorch 代码示例

```python
import torch
import torch.nn.functional as F

x = torch.linspace(-5, 5, 1000)

# Sigmoid
sigmoid = torch.sigmoid(x)
# Tanh
tanh = torch.tanh(x)
# ReLU
relu = F.relu(x)
# LeakyReLU
leaky_relu = F.leaky_relu(x, negative_slope=0.01)
# GELU
gelu = F.gelu(x)
# SiLU (Swish with β=1)
silu = F.silu(x)

# 在网络中使用
class SimpleMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))     # 隐藏层使用GELU
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)              # 输出层不加激活（配合CrossEntropyLoss）
        return x
```

---

## 5. 万能近似定理 (Universal Approximation Theorem)

### 5.1 定理陈述

**定理 (Hornik, Stinchcombe & White, 1989; Cybenko, 1989)**：

设 σ: R → R 是一个非常数的、有界的、单调递增的连续函数（如 Sigmoid）。则对于任意连续函数 f: [0,1]^n → R 和任意 ε > 0，存在正整数 N、实数 v_i, b_i 和实向量 w_i，使得：

```
F(x) = Σ_{i=1}^{N} v_i * σ(w_i · x + b_i)

满足: sup_{x ∈ [0,1]^n} |F(x) - f(x)| < ε
```

即：具有单个隐藏层、有限个神经元和适当激活函数的前馈网络，可以以任意精度逼近任意连续函数。

### 5.2 推广版本

**Leshno et al. (1993)** 将条件推广：只要激活函数不是多项式函数，万能近似定理就成立。这意味着 ReLU、GELU、Swish 等非多项式激活函数都满足条件。

**深度网络版本 (Lu et al., 2017)**：宽度为 n+1（输入维度加1）的 ReLU 网络即可逼近任意连续函数，且指数级深度的网络逼近能力比多项式宽度的浅层网络更强。

### 5.3 定理的局限性

- **存在性定理，非构造性**：只保证存在这样的网络，不告诉我们如何找到它
- **不保证可训练性**：梯度下降等优化算法不一定能找到正确的参数
- **未指明网络大小**：所需的神经元数量可能是指数级的
- **实践意义**：深层网络比浅层宽网络更高效地表示同一函数，这是深度学习优于浅层模型的理论基础之一

### 5.4 深度 vs 宽度

虽然单层 MLP 是万能近似器，但实践中有以下原因支持使用深度网络：

1. **参数效率**：深度网络可以用指数级更少的参数表示同一函数
2. **层次化特征提取**：深度网络逐层提取从低级到高级的特征
3. **实际优化更好**：虽然理论上浅层网络足够，但深度网络的优化景观对梯度下降更友好

---

## 6. 常见错误与最佳实践

### 6.1 常见错误

1. **激活函数选择不当**：
   - 在输出层使用 Sigmoid + MSE 损失（应该用交叉熵损失）
   - 在深层网络中使用 Sigmoid/Tanh（梯度消失）
   - 在所有层使用相同激活函数而不考虑输出层的特殊需求

2. **权重初始化不当**：
   - 使用全零初始化（所有神经元学到相同特征，对称性问题）
   - 使用过大的随机初始化（输出爆炸）或过小的随机初始化（梯度消失）

3. **忘记偏置项**：在有 BatchNorm 的层中偏置项是冗余的，但在普通全连接层中不可或缺

### 6.2 最佳实践

1. **默认使用 ReLU** 作为隐藏层激活函数，配合 He 初始化
2. **Transformer 类模型使用 GELU**
3. **输出层**根据任务选择：二分类用 Sigmoid，多分类用 Softmax，回归用线性输出
4. **权重初始化**：Xavier（用于 Sigmoid/Tanh）或 He（用于 ReLU）
   - Xavier: W ~ N(0, 2/(n_in + n_out))
   - He: W ~ N(0, 2/n_in)

---
---

# 反向传播完整推导
# Complete Derivation of Backpropagation

## 参考来源
- Rumelhart, Hinton & Williams (1986): Learning representations by back-propagating errors
- Stanford CS231n Lecture: Backpropagation
- d2l.ai Chapter: Backpropagation
- Goodfellow et al. (2016): Deep Learning (Chapter 6)

---

## 1. 前向传播 (Forward Propagation)

### 1.1 符号约定

```
L: 网络总层数 (不含输入层)
x: 输入向量 (第0层激活值 a_0 = x)
W_l: 第 l 层的权重矩阵
b_l: 第 l 层的偏置向量
z_l: 第 l 层的线性变换输出 (pre-activation)
a_l: 第 l 层的激活输出 (post-activation)
σ: 激活函数
y: 真实标签
ŷ = a_L: 网络输出 (预测值)
L(ŷ, y): 损失函数
```

### 1.2 前向传播公式

对于第 l 层 (l = 1, 2, ..., L)：

```
线性变换:   z_l = W_l * a_{l-1} + b_l
激活输出:   a_l = σ(z_l)

其中:
- W_l ∈ R^{n_l × n_{l-1}},  n_l 是第 l 层的神经元数
- b_l ∈ R^{n_l}
- z_l ∈ R^{n_l}
- a_l ∈ R^{n_l}
```

特殊处理输入层和输出层：
```
a_0 = x                              (输入层)
对于输出层:
  回归任务:  ŷ = a_L = z_L           (无激活/线性激活)
  二分类:    ŷ = a_L = σ(z_L)        (Sigmoid激活)
  多分类:    ŷ = softmax(z_L)        (Softmax激活)
```

### 1.3 完整前向传播计算图

```
x (= a_0) → [W_1, b_1] → z_1 → [σ] → a_1
                                        ↓
a_1 → [W_2, b_2] → z_2 → [σ] → a_2
                                    ↓
                                   ...
                                    ↓
a_{L-1} → [W_L, b_L] → z_L → [f_out] → ŷ → [L(ŷ, y)] → Loss
```

---

## 2. 链式法则 (Chain Rule)

### 2.1 标量链式法则

若 y = f(g(x))，则：
```
dy/dx = f'(g(x)) * g'(x) = (dy/dg) * (dg/dx)
```

### 2.2 多变量链式法则

若 y 依赖于 z_1, z_2, ..., z_n，每个 z_i 依赖于 x：
```
∂y/∂x = Σ_i (∂y/∂z_i) * (∂z_i/∂x)
```

### 2.3 反向传播的核心思想

反向传播本质上就是链式法则在计算图上的系统应用。从输出层开始，反向计算每个参数的梯度：

```
核心操作: 对于每个中间变量 z_l，计算 δ_l = ∂L/∂z_l (误差信号)

然后通过以下关系计算参数梯度:
∂L/∂W_l = δ_l * a_{l-1}^T   (外积)
∂L/∂b_l = δ_l                (直接传递)
```

---

## 3. 输出层误差推导

### 3.1 Softmax + 交叉熵的梯度（多分类）

输出层使用 Softmax，损失函数为交叉熵：

```
Softmax:  ŷ_c = softmax(z_L)_c = exp(z_{L,c}) / Σ_j exp(z_{L,j})
Loss:     L = -Σ_c y_c * log(ŷ_c)

其中 y 是 one-hot 编码的真实标签
```

对 z_L 的梯度推导：

```
∂L/∂z_{L,c} = ∂L/∂ŷ * ∂ŷ/∂z_{L,c}

先求 ∂ŷ_c/∂z_{L,k}（Softmax 的雅可比矩阵）:
当 c = k:  ∂ŷ_c/∂z_{L,c} = ŷ_c * (1 - ŷ_c)
当 c ≠ k:  ∂ŷ_c/∂z_{L,k} = -ŷ_c * ŷ_k

然后:
∂L/∂z_{L,k} = -Σ_c (y_c / ŷ_c) * ∂ŷ_c/∂z_{L,k}
             = -Σ_c (y_c / ŷ_c) * [ŷ_c(δ_{ck} - ŷ_k)]
             = -Σ_c y_c * (δ_{ck} - ŷ_k)
             = -y_k + ŷ_k * Σ_c y_c
             = ŷ_k - y_k

结论: δ_L = ∂L/∂z_L = ŷ - y
```

**重要结论**：Softmax + 交叉熵的组合使得输出层梯度的形式极其简洁——预测值与真实值之差。这也是为什么分类任务推荐使用这个组合。

### 3.2 Sigmoid + 二元交叉熵的梯度（二分类）

```
ŷ = σ(z_L)
L = -[y * log(ŷ) + (1-y) * log(1-ŷ)]

∂L/∂z_L = ∂L/∂ŷ * ∂ŷ/∂z_L
        = [-y/ŷ + (1-y)/(1-ŷ)] * ŷ(1-ŷ)
        = -y(1-ŷ) + (1-y)ŷ
        = ŷ - y

结论: δ_L = ŷ - y (与多分类形式相同!)
```

### 3.3 线性输出 + MSE 的梯度（回归）

```
ŷ = z_L
L = (1/2)(ŷ - y)^2

∂L/∂z_L = ŷ - y

结论: δ_L = ŷ - y (同样简洁)
```

---

## 4. 隐藏层误差传播

### 4.1 误差信号的反向传播

已知第 l+1 层的误差信号 δ_{l+1} = ∂L/∂z_{l+1}，需要推导第 l 层的误差信号 δ_l。

```
由于 z_{l+1} = W_{l+1} * a_l + b_l，且 a_l = σ(z_l):

δ_l = ∂L/∂z_l
    = (∂L/∂z_{l+1}) * (∂z_{l+1}/∂a_l) * (∂a_l/∂z_l)
    = W_{l+1}^T * δ_{l+1} ⊙ σ'(z_l)

其中:
- W_{l+1}^T * δ_{l+1}: 误差从第 l+1 层"反向传播"到第 l 层
- ⊙: 逐元素乘法 (Hadamard product)
- σ'(z_l): 激活函数的导数
```

展开写成标量形式：

```
δ_{l,i} = σ'(z_{l,i}) * Σ_j W_{l+1,j,i} * δ_{l+1,j}
```

### 4.2 参数梯度计算

有了误差信号 δ_l，参数梯度可以直接计算：

```
∂L/∂W_{l,i,j} = δ_{l,i} * a_{l-1,j}

矩阵形式:
∂L/∂W_l = δ_l * a_{l-1}^T    (外积, 维度: n_l × n_{l-1})
∂L/∂b_l = δ_l                 (维度: n_l × 1)

对于 batch_size = N 的 mini-batch:
∂L/∂W_l = (1/N) * Δ_l * A_{l-1}^T
∂L/∂b_l = (1/N) * Σ_i δ_l^{(i)}   (对 batch 维求和)
```

### 4.3 完整反向传播算法

```
输入: 训练样本 (x, y)
输出: 损失 L 对所有参数的梯度

步骤1 - 前向传播:
  a_0 = x
  for l = 1 to L:
    z_l = W_l * a_{l-1} + b_l
    a_l = σ(z_l)
  计算 Loss = L(a_L, y)

步骤2 - 反向传播:
  δ_L = ∂L/∂z_L          (根据输出层和损失函数计算)
  for l = L-1 down to 1:
    δ_l = (W_{l+1}^T * δ_{l+1}) ⊙ σ'(z_l)

步骤3 - 参数梯度:
  for l = 1 to L:
    ∂L/∂W_l = δ_l * a_{l-1}^T
    ∂L/∂b_l = δ_l
```

---

## 5. Batch Normalization (批归一化)

### 5.1 公式 (Ioffe & Szegedy, 2015)

BatchNorm 在每一层的激活值上进行归一化，通常作用于线性变换之后、激活函数之前：

```
输入: mini-batch B = {x_1, x_2, ..., x_m}

(1) 计算 batch 均值:     μ_B = (1/m) * Σ_{i=1}^{m} x_i
(2) 计算 batch 方差:     σ_B^2 = (1/m) * Σ_{i=1}^{m} (x_i - μ_B)^2
(3) 归一化:              x̂_i = (x_i - μ_B) / sqrt(σ_B^2 + ε)
(4) 缩放和平移:          y_i = γ * x̂_i + β

其中:
- ε > 0 是防止除零的小常数（通常 ε = 1e-5）
- γ (scale) 和 β (shift) 是可学习参数
- 当 γ = sqrt(σ_B^2) 且 β = μ_B 时，BatchNorm 等价于恒等变换
```

### 5.2 训练 vs 推理

```
训练时:
  使用当前 mini-batch 的统计量 μ_B, σ_B^2
  同时更新全局统计量（指数移动平均）:
    μ_running = (1 - α) * μ_running + α * μ_B
    σ_running^2 = (1 - α) * σ_running^2 + α * σ_B^2
  其中 α 是动量参数（通常 α = 0.1）

推理时:
  使用训练期间积累的全局统计量 μ_running, σ_running^2
  不再依赖 batch 内的统计量
```

### 5.3 BatchNorm 为什么有效？

**解释1：减少内部协变量偏移 (Internal Covariate Shift)**
- 原始论文的解释：训练过程中，每层输入的分布会随着前面层参数的更新而变化，导致训练困难
- BatchNorm 稳定了每层的输入分布

**解释2：平滑损失曲面 (Santurkar et al., 2018)**
- 实验表明 BatchNorm 并没有真正减少 ICS
- 真正作用是使损失函数更加平滑（Lipschitz 连续性更好）
- 梯度的 Lipschitz 常数更小，允许使用更大学习率

**解释3：允许更大学习率**
- 没有 BatchNorm 时，过大的学习率导致梯度爆炸
- BatchNorm 将每层的激活值归一化到合理范围，允许使用更大学习率

**解释4：隐式正则化**
- 每个样本的归一化依赖于同一 mini-batch 中的其他样本
- 引入了随机性，类似于 Dropout 的正则化效果
- 实验表明使用 BatchNorm 后可以减弱甚至去掉 Dropout

### 5.4 BatchNorm 的反向传播

```
∂L/∂γ = Σ_i (∂L/∂y_i) * x̂_i
∂L/∂β = Σ_i (∂L/∂y_i)

∂L/∂x̂_i = (∂L/∂y_i) * γ

∂L/∂σ_B^2 = Σ_i (∂L/∂x̂_i) * (x_i - μ_B) * (-1/2) * (σ_B^2 + ε)^{-3/2}

∂L/∂μ_B = Σ_i (∂L/∂x̂_i) * (-1/sqrt(σ_B^2 + ε))
         + ∂L/∂σ_B^2 * (1/m) * Σ_i (-2)(x_i - μ_B) * (1/m)

∂L/∂x_i = (∂L/∂x̂_i) / sqrt(σ_B^2 + ε)
         + ∂L/∂σ_B^2 * (2/m) * (x_i - μ_B)
         + ∂L/∂μ_B * (1/m)
```

### 5.5 BatchNorm 的常见变体

| 变体 | 归一化维度 | 适用场景 |
|------|-----------|---------|
| BatchNorm | batch 维度 | CNN，大 batch |
| LayerNorm | 特征维度 | Transformer，RNN，小 batch |
| InstanceNorm | 单个样本单个通道 | 风格迁移 |
| GroupNorm | 通道分组 | 小 batch，检测任务 |

---

## 6. Dropout 原理

### 6.1 算法描述 (Srivastava et al., 2014)

```
训练时 (Dropout rate = p, 通常 p = 0.5):
  for each layer l:
    r_l ~ Bernoulli(1 - p)     (生成掩码，每个元素独立以概率 1-p 保留)
    ã_l = r_l ⊙ a_l            (逐元素乘以掩码)
    a_l_out = ã_l / (1 - p)     (inverted dropout: 缩放以保持期望值不变)

推理时:
  不做任何 Dropout，直接使用 a_l（因为训练时已经缩放过）
```

### 6.2 为什么 Dropout 有效？

**解释1：模型平均 (Model Averaging)**
- Dropout 在每次训练时随机采样一个子网络
- 共有 2^n 种可能的子网络（n 为神经元数）
- 推理时相当于对所有子网络的预测取平均
- 类似于集成学习（Ensemble），降低方差

**解释2：打破共适应 (Co-adaptation)**
- 神经元不能依赖特定的其他神经元的存在
- 每个神经元必须学习更鲁棒的特征
- 类似于生态系统中的"强制多样化"

**解释3：等效于 L2 正则化**
- 对于线性回归，Dropout 近似等价于 L2 正则化
- 正则化强度与 Dropout rate 和学习率相关

### 6.3 Dropout 代码示例

```python
import torch
import torch.nn as nn

class ModelWithDropout(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)       # 训练时随机丢弃，推理时直通
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Dropout 自动处理训练/推理模式切换
model = ModelWithDropout(784, 256, 10)

model.train()   # 训练模式: Dropout 生效
output = model(x_train)

model.eval()    # 推理模式: Dropout 关闭
output = model(x_test)
```

### 6.4 Dropout 使用注意事项

1. **只在训练时使用**：PyTorch 的 nn.Dropout 自动处理
2. **典型 Dropout rate**：隐藏层 0.5，输入层 0.1-0.2
3. **CNN 中使用 Dropout2d**：丢弃整个通道而非单个像素
4. **与 BatchNorm 的关系**：两者都有正则化效果，使用 BatchNorm 时可以降低或去掉 Dropout
5. **最后一层通常不加 Dropout**

---
---

# 优化算法
# Optimization Algorithms

## 参考来源
- Ruder (2016): An overview of gradient descent optimization algorithms
- Kingma & Ba (2015): Adam: A Method for Stochastic Optimization
- Loshchilov & Hutter (2019): Decoupled Weight Decay Regularization
- Smith (2017): Cyclical Learning Rates
- PyTorch 官方文档

---

## 1. 随机梯度下降 (SGD)

### 1.1 基本 SGD

```
参数更新公式:
θ_{t+1} = θ_t - η * g_t

其中:
- θ: 模型参数
- η: 学习率 (learning rate)
- g_t = ∇L(θ_t; x_i, y_i): 在单个样本 (或 mini-batch) 上计算的梯度
```

**Mini-batch SGD** 使用一个 mini-batch 的平均梯度：
```
g_t = (1/|B|) * Σ_{i ∈ B} ∇L(θ_t; x_i, y_i)
```

**收敛性分析**（凸优化设定下）：
```
对于凸函数，SGD 的收敛速度为 O(1/sqrt(T))
即 L(θ_T) - L(θ*) <= O(R / (η * sqrt(T)))

其中 R 是参数空间直径，T 是迭代次数
```

### 1.2 SGD with Momentum

**动机**：SGD 在梯度方向变化较大的维度上收敛缓慢（之字形振荡）。

```
动量累积:
v_t = μ * v_{t-1} + g_t

参数更新:
θ_{t+1} = θ_t - η * v_t

其中:
- μ: 动量系数 (通常 μ = 0.9)
- v_t: 速度向量 (velocity)
- v_0 = 0
```

展开可得指数移动平均：
```
v_t = g_t + μ * g_{t-1} + μ^2 * g_{t-2} + μ^3 * g_{t-3} + ...
    = Σ_{k=0}^{t} μ^k * g_{t-k}

等效步长: η / (1 - μ)    (当 μ = 0.9 时，等效步长为原始学习率的 10 倍)
```

**直觉解释**：动量就像小球在损失曲面上滚动，不仅受当前坡度影响，还受到之前累积速度的影响。在梯度方向一致的维度上加速，在梯度方向变化的维度上减速。

### 1.3 Nesterov Accelerated Gradient (NAG)

```
先做一步"前瞻":
θ̃_t = θ_t - η * μ * v_{t-1}

在前瞻位置计算梯度:
g_t = ∇L(θ̃_t)

更新动量:
v_t = μ * v_{t-1} + g_t

参数更新:
θ_{t+1} = θ_t - η * v_t
```

**NAG 与标准 Momentum 的区别**：NAG 先根据动量方向"预看"一步，然后在预看位置计算梯度。这提供了更好的理论收敛率，并在实践中对变化的损失曲面更敏感。

**等价形式**（实践中更常用）：
```
v_t = μ * v_{t-1} + ∇L(θ_t - η * μ * v_{t-1})
θ_{t+1} = θ_t - η * v_t
```

---

## 2. 自适应学习率方法

### 2.1 AdaGrad (Adaptive Gradient)

**核心思想**：对每个参数使用不同的学习率，频繁更新的参数使用较小学习率，不频繁更新的参数使用较大学习率。

```
累积梯度平方:
r_t = r_{t-1} + g_t ⊙ g_t

参数更新:
θ_{t+1} = θ_t - (η / (sqrt(r_t) + ε)) ⊙ g_t

其中:
- r_0 = 0
- ε = 1e-8 (防止除零)
- / 和 ⊙ 是逐元素操作
```

**优点**：自动调整学习率，适合稀疏数据

**缺点**：累积梯度平方 r_t 单调递增，导致学习率单调递减，后期学习率可能变得过小，训练过早停止

### 2.2 RMSProp (Root Mean Square Propagation)

**核心思想**：用指数移动平均替代 AdaGrad 的累积，使学习率不会单调递减。

```
梯度平方的指数移动平均:
r_t = ρ * r_{t-1} + (1 - ρ) * g_t ⊙ g_t

参数更新:
θ_{t+1} = θ_t - (η / (sqrt(r_t) + ε)) ⊙ g_t

其中:
- ρ: 衰减率 (通常 ρ = 0.99)
- r_0 = 0
- ε = 1e-8
```

**与 AdaGrad 的区别**：RMSProp 用指数移动平均代替简单累加，使得旧的梯度平方会逐渐衰减，避免了学习率单调递减的问题。

### 2.3 Adam (Adaptive Moment Estimation)

**核心思想**：同时维护梯度的一阶矩（均值）和二阶矩（未中心化方差）的估计，结合了 Momentum 和 RMSProp 的优点。

```
一阶矩估计 (动量项):
m_t = β_1 * m_{t-1} + (1 - β_1) * g_t

二阶矩估计 (自适应学习率项):
v_t = β_2 * v_{t-1} + (1 - β_2) * g_t ⊙ g_t

偏差校正 (因为 m_0 = 0, v_0 = 0 导致初始估计偏向 0):
m̂_t = m_t / (1 - β_1^t)
v̂_t = v_t / (1 - β_2^t)

参数更新:
θ_{t+1} = θ_t - η * m̂_t / (sqrt(v̂_t) + ε)

默认超参数:
- β_1 = 0.9       (一阶矩衰减率)
- β_2 = 0.999     (二阶矩衰减率)
- η = 0.001       (学习率)
- ε = 1e-8        (数值稳定性)
```

**偏差校正的推导**：
```
在第 t 步:
m_t = (1-β_1) * Σ_{i=1}^{t} β_1^{t-i} * g_i

取期望 (假设 g_i 独立同分布):
E[m_t] = E[g_t] * (1-β_1) * Σ_{i=1}^{t} β_1^{t-i}
       = E[g_t] * (1 - β_1^t)

因此:
E[m̂_t] = E[m_t] / (1-β_1^t) = E[g_t]  (无偏估计)
```

**收敛性**：Adam 在凸优化设定下可以证明收敛，但 Reddi et al. (2018) 指出原始 Adam 的收敛性证明存在缺陷，提出了 AMSGrad 修正。

### 2.4 AdamW (Adam with Decoupled Weight Decay)

**动机**：L2 正则化在 Adam 中并不等价于权重衰减（这与 SGD 不同）。

```
标准 Adam + L2 正则化:
g_t = ∇L(θ_t) + λ * θ_t         (L2 正则化梯度)
然后用 Adam 更新规则处理 g_t

AdamW (解耦权重衰减):
m_t = β_1 * m_{t-1} + (1 - β_1) * ∇L(θ_t)
v_t = β_2 * v_{t-1} + (1 - β_2) * (∇L(θ_t))^2
m̂_t = m_t / (1 - β_1^t)
v̂_t = v_t / (1 - β_2^t)
θ_{t+1} = θ_t - η * (m̂_t / (sqrt(v̂_t) + ε) + λ * θ_t)

关键区别: 权重衰减项 λ * θ_t 直接作用于参数，而不是通过梯度
```

**为什么 AdamW 优于 Adam + L2？**

在 Adam + L2 中，L2 正则化项 λ * θ_t 被自适应学习率缩放，导致正则化强度因参数而异，不一致。AdamW 将权重衰减与梯度更新解耦，使正则化效果更一致。

---

## 3. 优化器对比表

| 优化器 | 核心机制 | 学习率 | 适用场景 | 超参数 |
|--------|---------|--------|---------|--------|
| SGD | 基本梯度下降 | 固定 | 简单问题，已知好解 | η |
| SGD+Momentum | 累积动量 | 固定 | CV 任务首选 | η, μ |
| NAG | 先前瞻再求梯度 | 固定 | 理论更优 | η, μ |
| AdaGrad | 累积梯度平方 | 自适应（递减） | 稀疏数据 | η, ε |
| RMSProp | 指数移动平均 | 自适应 | RNN | η, ρ, ε |
| Adam | 一阶+二阶矩 | 自适应 | 通用默认 | η, β1, β2, ε |
| AdamW | Adam + 解耦权重衰减 | 自适应 | Transformer, 大模型 | η, β1, β2, ε, λ |

---

## 4. 学习率调度 (Learning Rate Scheduling)

### 4.1 StepLR (阶梯衰减)

```
η_t = η_0 * γ^(floor(t / step_size))

其中:
- η_0: 初始学习率
- γ: 衰减因子 (通常 γ = 0.1)
- step_size: 每隔多少个 epoch 衰减一次

示例: η_0=0.1, step_size=30, γ=0.1
  epoch 0-29:  η = 0.1
  epoch 30-59: η = 0.01
  epoch 60-89: η = 0.001
```

### 4.2 CosineAnnealingLR (余弦退火)

```
η_t = η_min + (1/2) * (η_max - η_min) * (1 + cos(π * t / T_max))

其中:
- η_max: 最大（初始）学习率
- η_min: 最小学习率 (通常为 0)
- T_max: 半周期长度

直觉: 学习率按余弦曲线从 η_max 平滑下降到 η_min
```

**PyTorch 实现**:
```python
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

model = torch.nn.Linear(10, 2)
optimizer = SGD(model.parameters(), lr=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

for epoch in range(100):
    train(...)
    optimizer.step()
    scheduler.step()
    print(f"Epoch {epoch}, LR: {scheduler.get_last_lr()[0]:.6f}")
```

### 4.3 Warmup (预热)

```
Warmup 阶段 (t < T_warmup):
η_t = η_max * t / T_warmup    (线性增加)

主阶段 (t >= T_warmup):
使用 CosineAnnealing 或其他策略

典型 Warmup 步数: 5-10 个 epoch，或总步数的 5-10%
```

**为什么需要 Warmup？**
- 训练初期，参数离最优解较远，梯度可能很大且不稳定
- 大学习率 + 不稳定的梯度 = 灾难性的参数更新
- Warmup 让模型在学习率较小时先"热身"，等梯度稳定后再使用大学习率
- 对 Adam 等自适应优化器，Warmup 还帮助二阶矩估计 v_t 获得有意义的初始值

### 4.4 OneCycleLR

```
阶段1 (上升): η 从 η_min 线性上升到 η_max
阶段2 (下降): η 从 η_max 按余弦下降到 η_min
同时: 动量从 μ_max 线性下降到 μ_min 再上升

提出者: Leslie Smith (2017)
论文: Cyclical Learning Rates for Training Neural Networks
效果: 通常能显著加速训练，被称为"1cycle policy"
```

---

## 5. 梯度消失与梯度爆炸

### 5.1 梯度消失 (Vanishing Gradients)

**原因分析**：

反向传播中，梯度需要经过多次连乘：
```
δ_l = (∏_{k=l+1}^{L} W_k^T * diag(σ'(z_k))) * δ_L
```

若每层的梯度因子 ||W_k^T * σ'(z_k)|| < 1，则连乘后梯度指数级衰减。

**具体场景**：
- Sigmoid 函数: σ'(x) ∈ (0, 0.25]，连续 L 层后梯度最多缩小到 0.25^L
  - L=10 时: 0.25^10 ≈ 10^{-6}
  - L=20 时: 0.25^20 ≈ 10^{-12}
- 权重矩阵的谱范数 < 1 时也会导致梯度衰减

### 5.2 梯度爆炸 (Exploding Gradients)

**原因分析**：

与梯度消失相反，若每层的梯度因子 > 1：
```
||δ_l|| >= (∏_{k=l+1}^{L} ρ_k) * ||δ_L||

当 ρ_k > 1 时，梯度指数级增长
```

**具体场景**：
- RNN 中最常见（同一权重矩阵反复相乘）
- 学习率过大 + 权重初始化不当

### 5.3 解决方案

| 方案 | 解决问题 | 原理 |
|------|---------|------|
| **ReLU 激活函数** | 梯度消失 | 正区间梯度恒为1，不衰减 |
| **残差连接 (ResNet)** | 梯度消失 | 跳跃连接提供梯度直通路径 |
| **BatchNorm** | 两者都缓解 | 归一化激活值，稳定梯度分布 |
| **LSTM/GRU 门控** | RNN 梯度消失 | 门控机制控制信息流 |
| **梯度裁剪** | 梯度爆炸 | 限制梯度范数上限 |
| **合适初始化** | 两者都缓解 | Xavier/He 初始化保持梯度方差 |
| **LSTM forget gate bias** | 梯度消失 | 初始化 forget gate bias 为 1 |

**梯度裁剪**：
```
方法1 - 按范数裁剪:
  if ||g|| > clip_norm:
    g = clip_norm * g / ||g||

方法2 - 按值裁剪:
  g = clamp(g, -clip_value, clip_value)

典型值: clip_norm = 1.0 或 5.0
```

### 5.4 PyTorch 梯度裁剪代码

```python
import torch.nn.utils as nn_utils

# 训练循环中
optimizer.zero_grad()
loss.backward()

# 按范数裁剪 (推荐)
nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 或按值裁剪
nn_utils.clip_grad_value_(model.parameters(), clip_value=0.5)

optimizer.step()
```

---
---

# 卷积神经网络 (CNN)
# Convolutional Neural Networks

## 参考来源
- Stanford CS231n: Convolutional Neural Networks for Visual Recognition
- d2l.ai Chapter: Convolutional Neural Networks
- LeCun et al. (1998), Krizhevsky et al. (2012), Simonyan & Zisserman (2014), Szegedy et al. (2015), He et al. (2016), Huang et al. (2017)

---

## 1. 卷积操作数学定义

### 1.1 离散卷积 (Discrete Convolution)

对于二维输入 X 和卷积核 K，卷积操作定义为：

```
(X * K)(i, j) = Σ_m Σ_n X(i+m, j+n) * K(m, n)

注意：深度学习中的"卷积"实际上是互相关 (cross-correlation)，即没有翻转核。
严格数学卷积需要翻转核: (X * K)(i,j) = Σ_m Σ_n X(i-m, j-n) * K(m, n)
但在深度学习中，由于核是学习得到的，翻转与否不影响表达能力。
```

### 1.2 多通道卷积

```
输入: X ∈ R^{C_in × H × W}    (C_in 个输入通道)
卷积核: K ∈ R^{C_out × C_in × K_h × K_w}
输出: Y ∈ R^{C_out × H_out × W_out}

Y_{c_out, i, j} = Σ_{c_in=1}^{C_in} Σ_{m=0}^{K_h-1} Σ_{n=0}^{K_w-1}
    X_{c_in, i·s+m, j·s+n} * K_{c_out, c_in, m, n} + b_{c_out}

其中 s 是步长 (stride)
```

### 1.3 输出尺寸公式

```
H_out = floor((H_in + 2*P - K_h) / S + 1)
W_out = floor((W_in + 2*P - K_w) / S + 1)

其中:
- H_in, W_in: 输入高、宽
- K_h, K_w: 卷积核高、宽
- P: 填充 (padding)
- S: 步长 (stride)

特殊情况:
- "same" padding: P = (K-1)/2 (当 K 为奇数)，输出尺寸 = 输入尺寸 / S
- "valid" padding: P = 0
```

### 1.4 参数量计算

```
单个卷积层参数量:
Params = C_out × (C_in × K_h × K_w + 1)
       = C_out × C_in × K_h × K_w + C_out  (卷积核权重 + 偏置)

示例: Conv2d(3, 64, kernel_size=3, padding=1)
Params = 64 × (3 × 3 × 3 + 1) = 64 × 28 = 1,792

全连接层参数量 (对比):
如果输入为 224×224×3，输出为 4096:
Params = 224 × 224 × 3 × 4096 = 616,562,688  (约6亿!)

这正是卷积层的优势: 参数共享 + 局部连接大幅减少参数量
```

### 1.5 计算量 (FLOPs)

```
单个卷积层 FLOPs (乘加操作):
FLOPs = 2 × C_out × C_in × K_h × K_w × H_out × W_out

(因子2表示一次乘法和一次加法)
```

---

## 2. 卷积的关键性质

### 2.1 局部连接 (Local Connectivity)

每个输出神经元只与输入的一个局部区域相连（感受野），而非全部输入。

### 2.2 参数共享 (Weight Sharing)

同一个卷积核在整个输入上滑动，所有位置共享相同的参数。这大幅减少参数量，并使网络具有平移不变性。

### 2.3 平移等变性 (Translation Equivariance)

```
如果输入平移，输出也相应平移:
f(shift(x)) = shift(f(x))

这意味着卷积网络对物体在图像中的位置不敏感。
```

---

## 3. 经典架构完整对比

### 3.1 LeNet-5 (LeCun et al., 1998)

```
架构 (输入: 1×32×32):
  Conv1:    6 个 5×5 核 → 6×28×28
  AvgPool1: 2×2, stride=2 → 6×14×14
  Conv2:    16 个 5×5 核 → 16×10×10
  AvgPool2: 2×2, stride=2 → 16×5×5
  FC1:      120
  FC2:      84
  FC3:      10 (输出)

参数量: ~60K
激活函数: Tanh (原始使用 Sigmoid)
特点: 最早的 CNN 之一，用于手写数字识别 (MNIST)
```

### 3.2 AlexNet (Krizhevsky et al., 2012)

```
架构 (输入: 3×227×227):
  Conv1: 96 个 11×11 核, stride=4 → 96×55×55    (LRN + MaxPool)
  Conv2: 256 个 5×5 核, pad=2 → 256×27×27       (LRN + MaxPool)
  Conv3: 384 个 3×3 核, pad=1 → 384×13×13
  Conv4: 384 个 3×3 核, pad=1 → 384×13×13
  Conv5: 256 个 3×3 核, pad=1 → 256×13×13        (MaxPool)
  FC1: 4096 (Dropout)
  FC2: 4096 (Dropout)
  FC3: 1000

参数量: ~60M
Top-5 错误率: 16.4% (ImageNet, 大幅优于传统方法)
关键创新:
  - ReLU 激活函数 (替代 Sigmoid/Tanh)
  - Dropout 正则化
  - 数据增强 (random crop, horizontal flip, color jitter)
  - GPU 训练 (双 GTX 580 模型并行)
  - Local Response Normalization (LRN，后来被 BatchNorm 替代)
```

### 3.3 VGGNet (Simonyan & Zisserman, 2014)

```
核心思想: 用多个 3×3 小卷积核替代大卷积核

为什么 3×3 核更好？
两个 3×3 核的感受野 = 一个 5×5 核
三个 3×3 核的感受野 = 一个 7×7 核

参数量对比:
- 一个 7×7 核, C 通道: 7×7×C×C = 49C²
- 三个 3×3 核, C 通道: 3×(3×3×C×C) = 27C²  (节省45%参数!)
- 更重要的: 三个非线性激活 vs 一个，表达能力更强

VGG-16 架构:
  Block1: 2× [Conv3-64] + MaxPool  → 64×112×112
  Block2: 2× [Conv3-128] + MaxPool → 128×56×56
  Block3: 3× [Conv3-256] + MaxPool → 256×28×28
  Block4: 3× [Conv3-512] + MaxPool → 512×14×14
  Block5: 3× [Conv3-512] + MaxPool → 512×7×7
  FC1: 4096, FC2: 4096, FC3: 1000

参数量: ~138M
Top-5 错误率: 7.3% (ImageNet)
特点: 结构规整统一，易于理解和实现
```

### 3.4 GoogLeNet / Inception (Szegedy et al., 2015)

```
核心创新: Inception 模块 (多尺度特征提取)

Inception 模块结构:
输入 → 并行分支:
  Branch 1: 1×1 Conv
  Branch 2: 1×1 Conv → 3×3 Conv
  Branch 3: 1×1 Conv → 5×5 Conv
  Branch 4: 3×3 MaxPool → 1×1 Conv
→ 沿通道维度拼接 (Concat)

关键设计:
  - 1×1 卷积用于降维 (减少通道数)，大幅减少计算量
  - 多尺度并行提取特征 (1×1, 3×3, 5×5)
  - 全局平均池化替代全连接层

GoogLeNet = 22 层，9 个 Inception 模块
参数量: ~6.8M (远少于 VGG!)
Top-5 错误率: 6.7% (ImageNet)

后续版本: Inception v2/v3 (加入 BN), Inception v4 (结合残差连接)
```

### 3.5 ResNet (He et al., 2016)

```
核心创新: 残差连接 (Skip Connection / Shortcut Connection)

残差块 (Residual Block):
  输入 x
  → Conv → BN → ReLU → Conv → BN  (= F(x), 残差映射)
  → + x                            (跳跃连接)
  → ReLU
  → 输出

公式: y = F(x, {W_i}) + x

当输入输出维度不同时，使用投影快捷连接:
  y = F(x, {W_i}) + W_s * x
  其中 W_s 是 1×1 卷积 (用于匹配维度)

ResNet-50 架构:
  Conv1:    7×7, 64, stride=2  + MaxPool stride=2
  Conv2_x: 3 个 Bottleneck 块 (64→256 通道)
  Conv3_x: 4 个 Bottleneck 块 (128→512 通道)
  Conv4_x: 6 个 Bottleneck 块 (256→1024 通道)
  Conv5_x: 3 个 Bottleneck 块 (512→2048 通道)
  Global AvgPool + FC 1000

Bottleneck 结构 (ResNet-50/101/152):
  1×1 Conv (降维到 1/4 通道) → 3×3 Conv → 1×1 Conv (升回通道)

参数量: ResNet-50 约 25.6M
Top-5 错误率: 3.6% (ImageNet)

变体: ResNet-18/34/50/101/152/200+
```

### 3.6 DenseNet (Huang et al., 2017)

```
核心创新: 密集连接 (Dense Connectivity)

Dense Block:
  每一层接收前面所有层的特征图作为输入
  x_l = H_l([x_0, x_1, ..., x_{l-1}])

  其中 [·] 表示沿通道维度拼接 (concatenation)
  不同于 ResNet 的相加 (addition)

增长率 (Growth Rate) k:
  每层输出 k 个特征图
  第 l 层的输入通道数 = k_0 + k * (l-1)
  其中 k_0 是初始通道数

Transition Layer (在 Dense Block 之间):
  1×1 Conv (压缩通道) → 2×2 AvgPool (降采样)
  压缩因子 θ (通常 θ = 0.5)

DenseNet-121 架构:
  Conv1: 7×7, 64
  Dense Block 1: 6 层, growth rate=32
  Transition 1
  Dense Block 2: 12 层
  Transition 2
  Dense Block 3: 24 层
  Transition 3
  Dense Block 4: 16 层
  Global AvgPool + FC

参数量: DenseNet-121 约 8M
Top-5 错误率: ~3.5% (ImageNet)

优势: 参数效率高，特征复用，梯度流更好
劣势: 内存占用大（需要存储所有中间特征图用于拼接）
```

---

## 4. 残差连接为什么有效

### 4.1 梯度流分析

```
没有残差连接:
  x_{l+1} = F(x_l)
  x_L = F_L(F_{L-1}(...F_1(x)...))
  ∂L/∂x_l = ∂L/∂x_L × Π_{k=l}^{L-1} ∂F_k/∂x_k

  若 ||∂F_k/∂x_k|| < 1，梯度指数级衰减

有残差连接:
  x_{l+1} = F(x_l) + x_l
  x_L = x_l + Σ_{k=l}^{L-1} F(x_k)   (展开后)
  ∂L/∂x_l = ∂L/∂x_L × (1 + ∂(Σ F_k)/∂x_l)

  关键: 始终有 "1" 的梯度直通路径，梯度不会消失!
```

### 4.2 集成学习视角 (Veit et al., 2016)

ResNet 可以解释为大量不同深度路径的隐式集成：
```
ResNet-1031 可以展开为 2^N 条不同路径的集成
(每层可以选择走残差分支还是跳过)

实验证明:
- 删除单条路径对性能影响很小
- 主要贡献来自中等长度的路径
- 很少有路径经过所有层
```

### 4.3 损失曲面平滑性

残差连接使损失曲面更平滑，更容易优化。Li et al. (2018) 的可视化实验表明，没有残差连接的深层网络的损失曲面充满"山脊"和"断崖"，而 ResNet 的损失曲面更加平缓。

---

## 5. 感受野计算

### 5.1 定义

感受野 (Receptive Field) 是输出特征图上一个像素对应输入图像的区域大小。

### 5.2 递推公式

```
设第 l 层的感受野大小为 r_l，卷积核大小为 k_l，步长为 s_l:

r_l = r_{l-1} + (k_l - 1) × Π_{i=1}^{l-1} s_i

初始条件: r_0 = 1 (输入层)

示例:
  Layer 1: Conv 3×3, stride=1  → r_1 = 1 + (3-1)×1 = 3
  Layer 2: Conv 3×3, stride=1  → r_2 = 3 + (3-1)×1 = 5
  Layer 3: Conv 3×3, stride=2  → r_3 = 5 + (3-1)×1 = 7
  Layer 4: Conv 3×3, stride=1  → r_4 = 7 + (3-1)×2 = 11

(注意: stride 的累积效应)
```

### 5.3 有效感受野 (Effective Receptive Field)

实际影响输出的区域远小于理论感受野。Luo et al. (2016) 证明有效感受野近似服从高斯分布，中心区域的贡献远大于边缘。

---

## 6. 常见错误与最佳实践

### 6.1 常见错误

1. **输出尺寸计算错误**：忘记 padding、stride、dilation 的影响
2. **通道维度混淆**：PyTorch 使用 NCHW，TensorFlow 默认 NHWC
3. **误用 MaxPool 降维**：大量使用 2×2 MaxPool 导致信息丢失过快
4. **忘记 BN 在 eval 模式**：推理时应使用 model.eval() 以使用全局统计量
5. **ResNet 中忘记投影快捷连接**：当通道数变化时必须使用 1×1 卷积

### 6.2 最佳实践

1. 使用 3×3 卷积核 (VGG 的教训)
2. 每次降采样后增加通道数 (通道数翻倍，空间减半)
3. 使用 BatchNorm + ReLU 的标准组合
4. 使用 Global Average Pooling 替代最后的全连接层
5. 数据增强：RandomCrop, HorizontalFlip, ColorJitter, CutOut, MixUp

### 6.3 PyTorch CNN 代码示例

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """ResNet 基础残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)   # 残差连接
        out = torch.relu(out)
        return out
```

---
---

# 循环神经网络 (RNN / LSTM / GRU)
# Recurrent Neural Networks

## 参考来源
- Stanford CS231n Lecture 10: Recurrent Neural Networks
- d2l.ai Chapter: Recurrent Neural Networks
- Hochreiter & Schmidhuber (1997): Long Short-Term Memory
- Cho et al. (2014): Learning Phrase Representations using RNN Encoder-Decoder
- Bahdanau et al. (2015): Neural Machine Translation by Jointly Learning to Align and Translate

---

## 1. Vanilla RNN

### 1.1 数学公式

```
给定输入序列 x = (x_1, x_2, ..., x_T)

隐藏状态:
  h_t = tanh(W_{xh} * x_t + W_{hh} * h_{t-1} + b_h)

输出:
  y_t = W_{hy} * h_t + b_y

初始状态:
  h_0 = 0 (或可学习参数)

其中:
- x_t ∈ R^{d_x}:  t 时刻的输入
- h_t ∈ R^{d_h}:  t 时刻的隐藏状态
- W_{xh} ∈ R^{d_h × d_x}:  输入到隐藏的权重
- W_{hh} ∈ R^{d_h × d_h}:  隐藏到隐藏的权重 (循环连接)
- W_{hy} ∈ R^{d_y × d_h}:  隐藏到输出的权重
- b_h ∈ R^{d_h}, b_y ∈ R^{d_y}: 偏置
```

### 1.2 时间展开图 (Unrolling Through Time)

```
x_1 → [RNN Cell] → h_1 → [RNN Cell] → h_2 → ... → [RNN Cell] → h_T
         ↑                  ↑                           ↑
        h_0               h_1                         h_{T-1}
         |                  |                           |
        y_1                y_2                         y_T
```

### 1.3 梯度消失证明

**定理**: 在 Vanilla RNN 中，隐藏状态对早期输入的梯度随序列长度指数衰减。

**证明**:

```
考虑 ∂h_T / ∂h_k (从时刻 T 反向传播到时刻 k):

∂h_T / ∂h_k = Π_{t=k+1}^{T} ∂h_t / ∂h_{t-1}

其中每一项:
∂h_t / ∂h_{t-1} = diag(tanh'(z_t)) × W_{hh}

因此:
||∂h_T / ∂h_k|| <= Π_{t=k+1}^{T} ||tanh'(z_t)|| × ||W_{hh}||

由于 tanh'(x) ∈ (0, 1]:
||tanh'(z_t)|| <= 1

当 ||W_{hh}|| 的谱范数 < 1 时 (或 tanh' < 1 的部分连乘):
||∂h_T / ∂h_k|| <= (max_t ||tanh'(z_t)|| × ||W_{hh}||)^{T-k}

若 γ = max ||tanh'(z_t)|| × ||W_{hh}|| < 1:
||∂h_T / ∂h_k|| <= γ^{T-k}

当 T - k 较大时，梯度指数级趋近于 0，即梯度消失。

反之，若 γ > 1，则梯度指数级爆炸。
```

**关键推论**: Vanilla RNN 在理论上能记忆长期依赖，但在实践中由于梯度消失/爆炸，无法学习超过 10-20 步的依赖关系。

---

## 2. LSTM (Long Short-Term Memory)

### 2.1 动机

Hochreiter & Schmidhuber (1997) 提出 LSTM 来解决 Vanilla RNN 的梯度消失问题。核心思想是引入一条"细胞状态" (cell state) 信息高速公路，以及三个门控机制来控制信息的流入和流出。

### 2.2 完整公式

```
给定: x_t (输入), h_{t-1} (前一时刻隐藏状态), c_{t-1} (前一时刻细胞状态)

(1) 遗忘门 (Forget Gate): 决定丢弃哪些旧信息
    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

    直觉: f_t ∈ (0,1)，输出越接近 0 表示"忘掉"，越接近 1 表示"记住"

(2) 输入门 (Input Gate): 决定存入哪些新信息
    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)

(3) 候选细胞值: 新信息的内容
    g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)

    直觉: i_t 控制"是否写入"，g_t 控制"写入什么内容"

(4) 更新细胞状态:
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t

    直觉: 旧记忆的保留部分 + 新信息的筛选部分
    这是 LSTM 的核心——加法更新使梯度可以无衰减地流过 c_t

(5) 输出门 (Output Gate): 决定输出哪些信息
    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

(6) 隐藏状态输出:
    h_t = o_t ⊙ tanh(c_t)

    直觉: 细胞状态经过 tanh 压缩到 (-1,1)，然后通过输出门选择性输出
```

### 2.3 为什么 LSTM 能解决梯度消失

```
关键在细胞状态的更新方式:
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t

反向传播时:
∂c_t / ∂c_{t-1} = f_t  (当 f_t ≈ 1 时，梯度接近恒等映射)

这与 Vanilla RNN 的连乘不同:
- Vanilla RNN: 连乘 W_{hh} × diag(tanh'(z))，梯度指数衰减
- LSTM: 连乘 f_t，通过学习 f_t ≈ 1，可以保持梯度不衰减

直觉: 遗忘门 f_t 充当了梯度的"阀门"，LSTM 可以学会将 f_t 设为接近1，
      使梯度在细胞状态路径上畅通无阻地传播。
```

### 2.4 PyTorch LSTM 代码

```python
import torch
import torch.nn as nn

# LSTM 层
lstm = nn.LSTM(
    input_size=256,     # 输入特征维度
    hidden_size=512,    # 隐藏状态维度
    num_layers=2,       # 堆叠2层
    batch_first=True,   # 输入形状为 (batch, seq, feature)
    dropout=0.3,        # 层间 Dropout
    bidirectional=False  # 单向
)

# 输入: (batch_size, seq_len, input_size)
x = torch.randn(32, 50, 256)

# 前向传播
output, (h_n, c_n) = lstm(x)
# output: (32, 50, 512) - 所有时间步的隐藏状态
# h_n: (2, 32, 512) - 最后一个时间步的隐藏状态 (每层)
# c_n: (2, 32, 512) - 最后一个时间步的细胞状态 (每层)
```

---

## 3. GRU (Gated Recurrent Unit)

### 3.1 完整公式

```
Cho et al. (2014) 提出的 LSTM 简化版本，合并了细胞状态和隐藏状态。

(1) 更新门 (Update Gate): 类似 LSTM 的遗忘门+输入门
    z_t = σ(W_z · [h_{t-1}, x_t] + b_z)

(2) 重置门 (Reset Gate): 控制前一隐藏状态对候选状态的影响
    r_t = σ(W_r · [h_{t-1}, x_t] + b_r)

(3) 候选隐藏状态:
    h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)

    直觉: 当 r_t ≈ 0 时，忽略之前的隐藏状态，相当于"重置"

(4) 最终隐藏状态:
    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

    直觉: z_t 控制新旧信息的混合比例
    - z_t ≈ 1: 采用候选状态 (新信息)
    - z_t ≈ 0: 保留旧状态 (旧信息)
```

### 3.2 LSTM vs GRU 对比

| 特征 | LSTM | GRU |
|------|------|-----|
| 门的数量 | 3 个 (遗忘、输入、输出) | 2 个 (更新、重置) |
| 状态 | 隐藏状态 h + 细胞状态 c | 仅隐藏状态 h |
| 参数量 | 更多 (4 组权重) | 更少 (3 组权重) |
| 训练速度 | 较慢 | 较快 |
| 长序列性能 | 通常更好 | 相当或略差 |
| 小数据集 | 可能过拟合 | 泛化更好 |
| 使用建议 | 长序列、大数据集 | 短序列、小数据集 |

---

## 4. 双向 RNN (Bidirectional RNN)

### 4.1 结构

```
正向: h_t→ = f(W_forward * x_t + U_forward * h_{t-1}→)
反向: h_t← = f(W_backward * x_t + U_backward * h_{t+1}←)

输出: h_t = [h_t→; h_t←]  (拼接)

或者: h_t = h_t→ + h_t←  (求和)
```

### 4.2 应用场景

- 适用于整个序列已知的任务（如文本分类、命名实体识别）
- 不适用于序列生成（因为在生成时看不到未来信息）
- 双向 LSTM (Bi-LSTM) 在 NLP 任务中广泛使用

---

## 5. Seq2Seq + Attention

### 5.1 编码器-解码器架构

```
编码器 (Encoder):
  将输入序列 (x_1, ..., x_T) 编码为上下文向量 c
  h_t = EncoderRNN(x_t, h_{t-1})
  c = h_T  (最后时刻的隐藏状态，作为上下文向量)

解码器 (Decoder):
  基于上下文向量 c 逐步生成输出序列 (y_1, ..., y_{T'})
  s_t = DecoderRNN(y_{t-1}, s_{t-1}, c)
  P(y_t | y_{<t}, x) = softmax(W_s * s_t + b_s)
```

### 5.2 注意力机制 (Bahdanau Attention)

```
问题: 将整个输入序列压缩为固定长度向量 c 是信息瓶颈

注意力机制解决方法: 在解码的每一步，动态关注输入序列的不同部分

注意力分数:
  e_{t,i} = score(s_{t-1}, h_i)

  其中 score 函数有多种选择:
  - 加法注意力: e_{t,i} = v^T * tanh(W_1 * s_{t-1} + W_2 * h_i)
  - 乘法注意力: e_{t,i} = s_{t-1}^T * W * h_i
  - 点积注意力: e_{t,i} = s_{t-1}^T * h_i

注意力权重 (归一化):
  α_{t,i} = exp(e_{t,i}) / Σ_j exp(e_{t,j})  (softmax)

上下文向量 (加权求和):
  c_t = Σ_i α_{t,i} * h_i

解码器输入变为 [y_{t-1}; c_t]，即拼接上一步输出和注意力上下文
```

### 5.3 PyTorch 代码示例

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell  # outputs 用于 attention

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, hidden)
        # encoder_outputs: (batch, seq_len, hidden)
        score = self.v(torch.tanh(
            self.W1(decoder_hidden).unsqueeze(1) + self.W2(encoder_outputs)
        ))  # (batch, seq_len, 1)
        weights = torch.softmax(score, dim=1)
        context = (weights * encoder_outputs).sum(dim=1)  # (batch, hidden)
        return context, weights.squeeze(-1)
```

### 5.4 Seq2Seq 常见错误

1. **Teacher Forcing 比例不当**：训练时 100% 使用真实标签，推理时完全用自己的预测，导致误差累积 (exposure bias)。解决：使用 scheduled sampling。
2. **忽略 Attention**: 简单的 Seq2Seq 在长序列上性能急剧下降，因为瓶颈效应。
3. **忘记 Masking**: 变长序列需要对 padding 位置进行 mask，否则注意力会在 padding 上浪费权重。
4. **解码时未处理 OOV**: 缺少 UNK token 处理或 Copy 机制。

---
---

# Transformer
# Attention Is All You Need

## 参考来源
- Vaswani et al. (2017): Attention Is All You Need
- Devlin et al. (2019): BERT: Pre-training of Deep Bidirectional Transformers
- Radford et al. (2019): Language Models are Unsupervised Multitask Learners (GPT-2)
- Ba et al. (2016): Layer Normalization
- d2l.ai Chapter: Transformer
- Stanford CS224n: Natural Language Processing with Deep Learning

---

## 1. 自注意力机制 (Self-Attention)

### 1.1 缩放点积注意力 (Scaled Dot-Product Attention)

```
输入: 查询 Q, 键 K, 值 V

Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

其中:
- Q ∈ R^{n × d_k}: 查询矩阵 (Query)
- K ∈ R^{m × d_k}: 键矩阵 (Key)
- V ∈ R^{m × d_v}: 值矩阵 (Value)
- d_k: 键/查询的维度
- n: 查询序列长度
- m: 键/值序列长度 (自注意力中 n = m)

计算步骤:
(1) 计算注意力分数:  S = Q * K^T ∈ R^{n × m}
(2) 缩放:            S = S / sqrt(d_k)
(3) 归一化:          A = softmax(S) ∈ R^{n × m}  (逐行 softmax)
(4) 加权求和:        Output = A * V ∈ R^{n × d_v}
```

### 1.2 为什么除以 sqrt(d_k)

**问题**: 当 d_k 较大时，Q 和 K 的点积值的方差也会很大。

**分析**: 假设 Q 和 K 的每个元素独立服从均值为0、方差为1的分布：
```
q · k = Σ_{i=1}^{d_k} q_i * k_i

E[q · k] = 0
Var[q · k] = Σ_{i=1}^{d_k} Var[q_i * k_i] = d_k × 1 = d_k

(因为 Var[q_i * k_i] = E[q_i^2] * E[k_i^2] - E[q_i]^2 * E[k_i]^2 = 1×1 - 0 = 1)
```

当 d_k = 64 时，点积的方差为 64，标准差约为 8。这意味着：
```
softmax([8, 0, 0, 0, ...]) ≈ [1.0, 0.0, 0.0, ...]

softmax 的输出接近 one-hot 向量，导致:
1. 梯度极小 (softmax 饱和区梯度接近 0)
2. 注意力过于集中，失去了"软"注意力的意义
```

除以 sqrt(d_k) 后：
```
Var[q · k / sqrt(d_k)] = d_k / d_k = 1

点积值的方差回到 1，softmax 输出更加均匀，梯度更有意义。
```

### 1.3 Self-Attention 的直觉理解

```
Self-Attention 可以理解为一种"信息检索":
- 每个位置生成一个"查询" (Query): "我在找什么信息?"
- 每个位置生成一个"键" (Key): "我有什么信息可以提供?"
- 每个位置生成一个"值" (Value): "如果被选中，我提供的信息是..."

Q*K^T 计算每对位置之间的"相关度"
softmax 归一化为概率分布
乘以 V 得到加权聚合的信息

自注意力使每个位置都能"看到"序列中的所有其他位置，
建立任意距离的依赖关系，克服了 RNN 的长距离依赖问题。
```

---

## 2. 多头注意力 (Multi-Head Attention)

### 2.1 公式

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O

其中每个头:
  head_i = Attention(Q * W_Q_i, K * W_K_i, V * W_V_i)

参数:
  W_Q_i ∈ R^{d_model × d_k}
  W_K_i ∈ R^{d_model × d_k}
  W_V_i ∈ R^{d_model × d_v}
  W_O ∈ R^{h × d_v × d_model}

通常设置: d_k = d_v = d_model / h

示例: d_model = 512, h = 8
  → d_k = d_v = 64
  → 每个头有独立的 QKV 投影 (512 → 64)
  → 8 个头的输出拼接: 8 × 64 = 512
  → 最后线性投影: 512 → 512
```

### 2.2 为什么需要多头

```
单头注意力只能学习一种"注意力模式"。
多头注意力允许模型同时关注不同类型的信息:

头1 可能学习"语法关系" (主语-谓语)
头2 可能学习"共指关系" (代词-先行词)
头3 可能学习"局部相邻" (相邻词)
头4 可能学习"语义相似" (同义词)
...

实践发现: 不同头确实学到了不同的注意力模式
(Voisin et al., 2019 可视化研究)
```

### 2.3 计算复杂度

```
自注意力的时间和空间复杂度: O(n^2 × d)

其中 n 是序列长度，d 是特征维度

这使得标准 Transformer 处理长序列 (n > 512) 非常昂贵。
改进方法:
- Sparse Attention: O(n × sqrt(n))
- Linear Attention: O(n × d)
- Flash Attention: 同等复杂度但显存更优
```

---

## 3. 位置编码 (Positional Encoding)

### 3.1 动机

自注意力是排列不变的 (permutation invariant)：打乱输入顺序，输出也相应打乱但计算结果等价。因此需要显式注入位置信息。

### 3.2 正弦位置编码 (Sinusoidal Positional Encoding)

```
PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

其中:
- pos: 位置索引 (0, 1, 2, ...)
- i: 维度索引 (0, 1, ..., d_model/2 - 1)
- d_model: 模型维度

直觉解释:
- 不同维度对应不同频率的正弦/余弦波
- 低维度: 高频 (变化快，编码局部位置)
- 高维度: 低频 (变化慢，编码全局位置)
- 类似于二进制编码，但连续可微

为什么正弦编码有效?
- PE(pos+k) 可以表示为 PE(pos) 的线性函数:
  因为 sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
  所以相对位置信息可以通过线性变换获得
- 不同位置的编码向量的内积仅取决于相对距离
```

### 3.3 可学习位置编码

```
另一种选择: 将位置编码作为可学习参数
PE = nn.Embedding(max_len, d_model)

优点: 可以学习任意位置模式
缺点: 不能泛化到训练中未见过的长度

GPT 系列和 BERT 使用可学习位置编码
原始 Transformer 论文使用正弦位置编码
```

### 3.4 旋转位置编码 (RoPE, Su et al., 2021)

```
现代 LLM (如 LLaMA) 广泛使用的相对位置编码:

将位置信息融入 Q 和 K 的旋转:
RoPE(x_m, m) = x_m * e^{imθ}

其中 m 是位置，θ 是频率

核心优势: 注意力分数自然编码了相对位置
  <RoPE(q_m, m), RoPE(k_n, n)> = f(q, k, m-n)
```

---

## 4. LayerNorm vs BatchNorm

### 4.1 Batch Normalization

```
沿 batch 维度归一化:
对于形状为 (N, C, ...) 的特征:
  μ_c = (1/N) * Σ_{i=1}^{N} x_{i,c,...}
  σ_c^2 = (1/N) * Σ_{i=1}^{N} (x_{i,c,...} - μ_c)^2
  x̂_{i,c,...} = (x_{i,c,...} - μ_c) / sqrt(σ_c^2 + ε)
  y_{i,c,...} = γ_c * x̂_{i,c,...} + β_c
```

### 4.2 Layer Normalization

```
沿特征维度归一化:
对于形状为 (N, C, ...) 的特征，对每个样本独立归一化:

  μ_i = (1/D) * Σ_{j=1}^{D} x_{i,j}
  σ_i^2 = (1/D) * Σ_{j=1}^{D} (x_{i,j} - μ_i)^2
  x̂_{i,j} = (x_{i,j} - μ_i) / sqrt(σ_i^2 + ε)
  y_{i,j} = γ_j * x̂_{i,j} + β_j

其中 D 是特征维度的总大小
```

### 4.3 为什么 Transformer 使用 LayerNorm

| 特征 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化维度 | batch 维度 | 特征维度 |
| 依赖 batch 大小 | 是 | 否 |
| 变长序列 | 不适用 | 适用 |
| 推理时统计量 | 需要 running mean/var | 直接计算 |
| 训练和推理一致性 | 不一致 | 一致 |
| RNN/Transformer | 问题较多 | 推荐 |
| CNN | 推荐 | 可以但不如BN |

**关键原因**:
1. 序列长度可变，BatchNorm 在序列维度上归一化不合理
2. BatchNorm 在小 batch 下统计量不稳定
3. LayerNorm 每个样本独立归一化，不依赖 batch 内其他样本

### 4.4 Pre-Norm vs Post-Norm

```
Post-Norm (原始 Transformer):
  x → Sublayer(x) → LayerNorm(x + Sublayer(x))

Pre-Norm (GPT-2, 大多数现代模型):
  x → x + Sublayer(LayerNorm(x))

Pre-Norm 优势:
- 训练更稳定 (梯度更平滑)
- 允许更大学习率
- 不需要学习率 warmup (通常)

Pre-Norm 劣势:
- 理论上每层输出的范数可能增长
- 某些情况下最终性能略差于调好的 Post-Norm
```

---

## 5. Transformer 完整架构

### 5.1 Encoder 结构

```
单个 Encoder 层:
  (1) Multi-Head Self-Attention
  (2) Add & Norm (残差连接 + LayerNorm)
  (3) Feed-Forward Network (FFN)
  (4) Add & Norm

FFN 详细结构:
  FFN(x) = max(0, x * W_1 + b_1) * W_2 + b_2
         = ReLU(x * W_1 + b_1) * W_2 + b_2

  其中 W_1 ∈ R^{d_model × d_ff}, W_2 ∈ R^{d_ff × d_model}
  通常 d_ff = 4 × d_model

  (现代变体使用 GELU 替代 ReLU, 或使用 SwiGLU)

完整 Encoder (6 层):
  输入 → Embedding + Positional Encoding
  → [Encoder Layer] × 6
  → Encoder 输出
```

### 5.2 Decoder 结构

```
单个 Decoder 层:
  (1) Masked Multi-Head Self-Attention (带因果掩码)
  (2) Add & Norm
  (3) Multi-Head Cross-Attention (查询来自 Decoder, KV 来自 Encoder)
  (4) Add & Norm
  (5) Feed-Forward Network
  (6) Add & Norm

因果掩码 (Causal Mask):
  防止位置 i 看到位置 i 之后的信息
  Mask[i][j] = -∞  if j > i
  Mask[i][j] = 0    if j <= i

  在 softmax 之前加到注意力分数上:
  softmax(S + Mask)

  效果: 位置 i 只能关注位置 1, 2, ..., i
```

### 5.3 完整 Transformer 架构

```
Encoder:
  Input → Token Embedding + Positional Encoding
  → Encoder Layer × N (N=6)
    → Multi-Head Self-Attention → Add & Norm
    → FFN → Add & Norm
  → Encoder Output (memory)

Decoder:
  Output → Token Embedding + Positional Encoding
  → Decoder Layer × N (N=6)
    → Masked Multi-Head Self-Attention → Add & Norm
    → Cross-Attention(memory) → Add & Norm
    → FFN → Add & Norm
  → Linear → Softmax → Output Probabilities

原始 Transformer 超参数:
  d_model = 512
  d_ff = 2048
  h = 8 (注意力头数)
  d_k = d_v = 64
  N = 6 (层数)
  Dropout = 0.1
```

### 5.4 PyTorch 代码示例

```python
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(5000, d_model)  # 可学习位置编码

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.embedding(src) + self.pos_encoding(
            torch.arange(src.size(1), device=src.device)
        )
        tgt_emb = self.embedding(tgt) + self.pos_encoding(
            torch.arange(tgt.size(1), device=tgt.device)
        )

        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        return self.output_proj(output)
```

---

## 6. BERT vs GPT 对比

### 6.1 架构对比

| 特征 | BERT | GPT |
|------|------|-----|
| 架构 | 仅 Encoder | 仅 Decoder |
| 注意力方向 | 双向 (全部可见) | 单向 (因果掩码) |
| 预训练目标 | MLM + NSP | 自回归语言建模 |
| 输入格式 | [CLS] text [SEP] text [SEP] | text |
| 微调方式 | 加任务头，端到端微调 | Prompt / In-context Learning |
| 适合任务 | 理解型 (分类, NER, QA) | 生成型 (文本生成, 对话) |
| 首次发布 | 2018 (Devlin et al.) | 2018 (Radford et al.) |

### 6.2 预训练目标

**BERT - Masked Language Model (MLM)**:
```
随机遮蔽输入中 15% 的 token:
  - 80% 替换为 [MASK]
  - 10% 替换为随机 token
  - 10% 保持不变

目标: 预测被遮蔽的 token
L_MLM = -Σ_{masked} log P(x_masked | x_context)

Next Sentence Prediction (NSP):
  判断两个句子是否连续
  (后来被证明不太重要，RoBERTa 去掉了 NSP)
```

**GPT - Causal Language Model (CLM)**:
```
目标: 给定前文，预测下一个 token
L_CLM = -Σ_{t=1}^{T} log P(x_t | x_1, ..., x_{t-1})

这是自回归 (autoregressive) 建模
天然适合文本生成，但无法双向利用上下文
```

### 6.3 规模演进

```
BERT 系列:
  BERT-Base:  110M 参数, 12 层, 768 维
  BERT-Large: 340M 参数, 24 层, 1024 维
  RoBERTa:    优化 BERT 训练策略
  ALBERT:     参数共享，减少参数量
  DeBERTa:    解耦注意力 + 增强掩码解码器

GPT 系列:
  GPT-1:   117M 参数, 12 层
  GPT-2:   1.5B 参数, 48 层
  GPT-3:   175B 参数, 96 层, 12288 维
  GPT-4:   未公开 (推测 MoE 架构, ~1.8T)

趋势: GPT 系列通过 scale (规模) 获得了涌现能力 (emergent abilities)
BERT 系列在特定 NLU 任务上通过微调仍然很强
```

### 6.4 常见错误与最佳实践

1. **忘记因果掩码**：训练 GPT 类模型时必须使用因果掩码，否则信息泄露
2. **位置编码外推**：训练 512 长度推理 1024 长度，正弦编码可以但效果下降；可学习编码直接失败
3. **LayerNorm 放置位置**：Pre-Norm 训练更稳定，是现代模型的标准选择
4. **注意力头冗余**：部分注意力头可能退化为恒等映射，可以剪枝
5. **训练不稳定**：使用 fp16 混合精度时，注意力分数可能溢出。建议使用 bf16 或 Flash Attention
6. **解码策略选择**：
   - Greedy: 选择概率最高的 token (确定性但不一定是全局最优)
   - Beam Search: 维护 k 个候选序列 (质量更高但多样性低)
   - Top-k Sampling: 从概率最高的 k 个 token 中采样
   - Top-p (Nucleus) Sampling: 从累积概率超过 p 的最小集合中采样 (推荐)
   - Temperature: T<1 更确定，T>1 更随机。softmax(logits/T)
