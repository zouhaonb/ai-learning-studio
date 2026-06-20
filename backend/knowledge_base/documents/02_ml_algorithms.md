# 机器学习核心算法知识库

> 参考资料：Stanford CS229 (Andrew Ng)、sklearn官方文档、《统计学习方法》(李航)、《机器学习》(周志华)
> 编写日期：2026年6月

---

## 第一章：线性回归 (Linear Regression)

### 1.1 算法概述

线性回归是最基础的监督学习算法，用于建立自变量 X 与因变量 y 之间的线性关系模型。它是理解更复杂算法（神经网络、广义线性模型）的基石。

**模型假设：**

给定 n 个样本、d 个特征，假设模型为：

    h(x) = w_0 + w_1*x_1 + w_2*x_2 + ... + w_d*x_d

写成向量形式（令 x_0 = 1 来吸收偏置项 w_0）：

    h(x) = w^T * x = x^T * w

其中 w = [w_0, w_1, ..., w_d]^T 为参数向量，x = [1, x_1, ..., x_d]^T 为特征向量。

对于 m 个训练样本，用矩阵表示：

    X = [x^(1), x^(2), ..., x^(m)]^T    (m × (d+1) 矩阵)
    y = [y^(1), y^(2), ..., y^(m)]^T    (m × 1 向量)
    预测值: y_hat = X * w

---

### 1.2 MSE损失函数推导

**均方误差（Mean Squared Error, MSE）** 是线性回归最常用的损失函数。

**直觉理解：** 我们希望预测值 h(x^(i)) 与真实值 y^(i) 尽可能接近，因此最小化它们之间差的平方和。

**单样本误差：**

    error(i) = (h(x^(i)) - y^(i))^2

**损失函数（Cost Function）：**

    J(w) = (1/2m) * sum_{i=1}^{m} (h(x^(i)) - y^(i))^2

写成矩阵形式：

    J(w) = (1/2m) * (X*w - y)^T * (X*w - y)

其中 1/2 是为了求导后消除系数，方便计算。

**为什么用MSE而不是MAE？**
1. MSE是可微的（处处可导），方便使用梯度下降
2. MSE对大误差惩罚更重（平方效应），鼓励模型不要偏离太远
3. 在高斯噪声假设下，MSE等价于极大似然估计（见下文）

---

### 1.3 正规方程推导（含矩阵微分）

**目标：** 求解 w* = argmin J(w)

**推导过程：**

    J(w) = (1/2m) * (Xw - y)^T * (Xw - y)
         = (1/2m) * (w^T X^T X w - w^T X^T y - y^T X w + y^T y)

由于 y^T X w 是标量，(y^T X w)^T = w^T X^T y，所以：

    J(w) = (1/2m) * (w^T X^T X w - 2 w^T X^T y + y^T y)

对 w 求梯度（使用矩阵微分公式）：

**关键矩阵微分公式：**
    d(Aw)/dw = A^T
    d(w^T A w)/dw = (A + A^T) * w  (当A对称时 = 2Aw)
    d(a^T w)/dw = a

因此：

    dJ/dw = (1/2m) * (2 X^T X w - 2 X^T y)
          = (1/m) * (X^T X w - X^T y)

令梯度为零（一阶必要条件）：

    X^T X w = X^T y

解得**正规方程（Normal Equation）**：

    w* = (X^T X)^{-1} * X^T * y

**注意：** X^T X 必须可逆。若不可逆（特征共线性或 d > m），可使用：
1. 正则化方法（Ridge回归）
2. 伪逆（Moore-Penrose Pseudoinverse）：w = pinv(X) * y

**正规方程的复杂度：** O(d^3 + md^2)，其中 d 为特征数。当 d > 10000 时，建议使用梯度下降。

---

### 1.4 梯度下降实现

**梯度下降的通用更新规则：**

    w := w - alpha * dJ/dw
    w := w - alpha * (1/m) * X^T * (X*w - y)

其中 alpha 为学习率（learning rate）。

**三种梯度下降变体：**

| 方法 | 每次更新使用样本数 | 更新频率 | 收敛性 | 适用场景 |
|------|-------------------|---------|--------|---------|
| 批量梯度下降(BGD) | 全部 m 个样本 | 每轮1次 | 稳定收敛 | 数据量小 |
| 随机梯度下降(SGD) | 1个样本 | 每轮m次 | 震荡但快 | 在线学习/大数据 |
| 小批量梯度下降(Mini-batch) | b个样本 | 每轮m/b次 | 平衡 | 实践中最常用 |

**SGD更新规则：**

    对每个样本 i:
        w := w - alpha * (h(x^(i)) - y^(i)) * x^(i)

**Mini-batch SGD更新规则：**

    对每个小批量 B：
        w := w - alpha * (1/|B|) * sum_{i in B} (h(x^(i)) - y^(i)) * x^(i)

**收敛判断：**
1. 损失变化小于阈值：|J(w^{t+1}) - J(w^t)| < epsilon
2. 参数变化小于阈值：||w^{t+1} - w^t|| < epsilon
3. 达到最大迭代次数

---

### 1.5 MSE的极大似然解释

假设数据满足：y^(i) = w^T * x^(i) + epsilon^(i)，其中 epsilon^(i) ~ N(0, sigma^2)

则 y^(i) | x^(i); w ~ N(w^T * x^(i), sigma^2)

似然函数：

    L(w) = prod_{i=1}^{m} p(y^(i) | x^(i); w)
         = prod_{i=1}^{m} (1/(sqrt(2*pi)*sigma)) * exp(-(y^(i) - w^T*x^(i))^2 / (2*sigma^2))

对数似然：

    l(w) = m * log(1/(sqrt(2*pi)*sigma)) - (1/(2*sigma^2)) * sum_{i=1}^{m} (y^(i) - w^T*x^(i))^2

最大化 l(w) 等价于最小化：

    sum_{i=1}^{m} (y^(i) - w^T*x^(i))^2

这恰好就是 MSE（不带常数项）！因此，在高斯噪声假设下，最小化MSE = 极大似然估计。

---

### 1.6 R^2 评估指标

**R^2（决定系数，Coefficient of Determination）** 衡量模型对数据的拟合程度。

    SS_tot = sum_{i=1}^{m} (y^(i) - y_bar)^2     (总平方和)
    SS_res = sum_{i=1}^{m} (y^(i) - h(x^(i)))^2   (残差平方和)
    R^2 = 1 - SS_res / SS_tot

**解释：**
- R^2 = 1：完美拟合
- R^2 = 0：模型等同于预测均值
- R^2 < 0：模型比预测均值还差（很差）

**调整R^2（Adjusted R^2）：**

    R^2_adj = 1 - (1 - R^2) * (m - 1) / (m - d - 1)

引入了特征数 d 的惩罚，防止特征越多R^2越高的虚高现象。

**其他回归评估指标：**

    MSE  = (1/m) * sum(y^(i) - h(x^(i)))^2
    RMSE = sqrt(MSE)
    MAE  = (1/m) * sum|y^(i) - h(x^(i))|
    MAPE = (100/m) * sum|y^(i) - h(x^(i))| / |y^(i)|

---

### 1.7 正则化：Ridge / Lasso / ElasticNet

当特征多或存在共线性时，线性回归容易**过拟合**。正则化通过在损失函数中添加惩罚项来约束模型复杂度。

#### 1.7.1 L2正则化 —— Ridge回归（岭回归）

    J(w) = (1/2m) * ||Xw - y||^2 + lambda * ||w||_2^2
         = (1/2m) * ||Xw - y||^2 + lambda * sum_{j=1}^{d} w_j^2

正规方程解：

    w* = (X^T X + lambda * I')^{-1} * X^T * y

其中 I' 是 (d+1)x(d+1) 单位矩阵，第一行第一列为0（不对偏置正则化）。

**特点：**
- 将权重压缩接近0但不会恰好为0
- 解决共线性问题（X^T X + lambda*I 必可逆）
- 闭式解存在

**L2正则化的贝叶斯解释：** 等价于对参数 w 施加零均值高斯先验 w ~ N(0, (1/(2*lambda)) * I)

#### 1.7.2 L1正则化 —— Lasso回归

    J(w) = (1/2m) * ||Xw - y||^2 + lambda * ||w||_1
         = (1/2m) * ||Xw - y||^2 + lambda * sum_{j=1}^{d} |w_j|

**特点：**
- 可以产生稀疏解（部分权重恰好为0），实现特征选择
- 没有闭式解，需要迭代优化
- 当多个特征相关时，倾向于选择其中一个

**L1正则化的贝叶斯解释：** 等价于对参数 w 施加拉普拉斯先验

**为什么L1产生稀疏解（直觉理解）？**
L1的约束区域是菱形（|w_1| + |w_2| <= C），MSE的等高线是椭圆形。椭圆与菱形的交点更容易在坐标轴上（即某个分量为0），而与L2圆形约束的交点一般不在轴上。

#### 1.7.3 ElasticNet

    J(w) = (1/2m) * ||Xw - y||^2 + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2

结合L1和L2的优点，通过 l1_ratio 参数控制比例。

#### 1.7.4 正则化方法对比

| 特性 | Ridge(L2) | Lasso(L1) | ElasticNet |
|------|-----------|-----------|------------|
| 惩罚项 | sum(w_j^2) | sum(\|w_j\|) | alpha*L1 + beta*L2 |
| 稀疏性 | 不稀疏 | 稀疏（特征选择） | 稀疏 |
| 闭式解 | 有 | 无 | 无 |
| 共线性处理 | 好 | 不稳定 | 好 |
| 适用场景 | 特征多但都有用 | 特征多且稀疏 | 特征多且有共线性 |
| sklearn类 | Ridge | Lasso | ElasticNet |

---

### 1.8 Python代码示例

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# ========== 1. 数据生成 ==========
np.random.seed(42)
m = 200  # 样本数
d = 5    # 特征数
X = np.random.randn(m, d)
true_w = np.array([3, -2, 0.5, 0, -1.5])  # 真实权重，第4个特征无用
y = X @ true_w + 2 + np.random.randn(m) * 0.5  # y = Xw + 噪声

# ========== 2. 数据划分 ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== 3. 标准化（正则化模型需要） ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 4. 正规方程实现 ==========
class LinearRegressionNormalEq:
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 添加偏置列
        self.w = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        return self

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.w

model_ne = LinearRegressionNormalEq()
model_ne.fit(X_train, y_train)
y_pred_ne = model_ne.predict(X_test)
print(f"正规方程 - R^2: {r2_score(y_test, y_pred_ne):.4f}, MSE: {mean_squared_error(y_test, y_pred_ne):.4f}")

# ========== 5. 梯度下降实现 ==========
class LinearRegressionGD:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.losses = []

    def fit(self, X, y):
        m, d = X.shape
        X_b = np.c_[np.ones((m, 1)), X]
        self.w = np.zeros(d + 1)

        for i in range(self.n_iters):
            y_pred = X_b @ self.w
            error = y_pred - y
            gradient = (1/m) * (X_b.T @ error)
            self.w -= self.lr * gradient
            self.losses.append((1/(2*m)) * np.sum(error**2))
        return self

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.w

model_gd = LinearRegressionGD(lr=0.1, n_iters=500)
model_gd.fit(X_train_scaled, y_train)
y_pred_gd = model_gd.predict(X_test_scaled)
print(f"梯度下降 - R^2: {r2_score(y_test, y_pred_gd):.4f}, MSE: {mean_squared_error(y_test, y_pred_gd):.4f}")

# ========== 6. sklearn正则化模型对比 ==========
models = {
    'LinearRegression': LinearRegression(),
    'Ridge(alpha=1.0)': Ridge(alpha=1.0),
    'Lasso(alpha=0.1)': Lasso(alpha=0.1),
    'ElasticNet(alpha=0.1, l1_ratio=0.5)': ElasticNet(alpha=0.1, l1_ratio=0.5),
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"{name}: R^2={r2_score(y_test, y_pred):.4f}, MSE={mean_squared_error(y_test, y_pred):.4f}")
    if hasattr(model, 'coef_'):
        print(f"  系数: {model.coef_.round(3)}")

# ========== 7. 正则化路径图 ==========
alphas = np.logspace(-3, 3, 100)
ridge_coefs = []
lasso_coefs = []
for a in alphas:
    ridge = Ridge(alpha=a).fit(X_train_scaled, y_train)
    lasso = Lasso(alpha=a, max_iter=10000).fit(X_train_scaled, y_train)
    ridge_coefs.append(ridge.coef_)
    lasso_coefs.append(lasso.coef_)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i in range(d):
    axes[0].plot(alphas, [c[i] for c in ridge_coefs], label=f'w_{i+1}')
    axes[1].plot(alphas, [c[i] for c in lasso_coefs], label=f'w_{i+1}')
axes[0].set_xscale('log'); axes[0].set_title('Ridge Regularization Path')
axes[1].set_xscale('log'); axes[1].set_title('Lasso Regularization Path')
for ax in axes:
    ax.set_xlabel('alpha'); ax.legend(); ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.tight_layout(); plt.savefig('regularization_path.png'); plt.show()
```

---

### 1.9 超参数选择指南

| 超参数 | 说明 | 选择方法 | 常用范围 |
|--------|------|---------|---------|
| alpha(lambda) | 正则化强度 | 交叉验证、网格搜索 | Ridge: 0.01~100, Lasso: 0.001~10 |
| l1_ratio | ElasticNet中L1比例 | 交叉验证 | 0.1~0.9 |
| 学习率(alpha) | 梯度下降步长 | 学习率衰减、Adam自适应 | 0.001~0.1 |
| 迭代次数 | 梯度下降轮数 | 监控验证集损失 | 直到收敛 |
| 批量大小 | Mini-batch SGD | 一般取2的幂 | 32, 64, 128, 256 |

**sklearn网格搜索示例：**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
grid.fit(X_train_scaled, y_train)
print(f"最佳alpha: {grid.best_params_}, 最佳R^2: {grid.best_score_:.4f}")
```

---

### 1.10 优缺点对比

**优点：**
1. 简单、可解释性强（系数直接表示特征重要性）
2. 有闭式解（正规方程）
3. 计算效率高
4. 在数据量小且线性关系成立时表现优异
5. 正则化变体可防止过拟合

**缺点：**
1. 假设线性关系，无法捕捉非线性模式
2. 对异常值敏感（MSE受平方影响）
3. 多重共线性问题（无正则化时）
4. 特征缩放影响梯度下降收敛
5. 高维数据需要正则化

---

### 1.11 常见面试/考试题

**Q1: 为什么MSE损失函数中要乘以1/2？**
A: 纯粹是为了求导时消除平方的系数2，使梯度公式更简洁。乘以常数不影响最优解的位置。

**Q2: 正规方程不可逆怎么办？**
A: (1) 删除冗余特征 (2) 使用正则化 Ridge (3) 使用伪逆 np.linalg.pinv (4) 增加数据量

**Q3: Ridge回归为什么能解决共线性问题？**
A: X^T X + lambda*I 保证正定可逆。lambda越大，对角线元素越占主导，条件数越小，数值越稳定。

**Q4: 梯度下降学习率如何选择？**
A: 学习率太大→震荡不收敛；太小→收敛太慢。常用方法：(1) 网格搜索 (2) 学习率衰减 (3) 自适应方法（Adam, RMSProp）

**Q5: Lasso为什么不常用闭式解？**
A: L1惩罚项 sum|w_j| 在 w_j=0 处不可微，因此没有解析解，需要用坐标下降法或次梯度方法迭代求解。

**Q6: 如何判断线性回归是否欠拟合或过拟合？**
A: 欠拟合：训练误差和测试误差都高。过拟合：训练误差低但测试误差高。用学习曲线（训练样本数 vs 误差）诊断。

**Q7: 对比正规方程和梯度下降的优缺点？**
A: 正规方程：不需要选择学习率，不需要迭代；但O(d^3)复杂度，d大时不可用。梯度下降：需要调参，需要迭代；但适合大规模数据，可在线学习。

---

## 第二章：逻辑回归 (Logistic Regression)

### 2.1 算法概述

逻辑回归是经典的**二分类**算法，虽然名字中有"回归"，但它是一个**分类**模型。它在线性回归的基础上引入Sigmoid函数，将输出映射到(0,1)区间，表示样本属于正类的概率。

**核心思想：**

    线性模型：z = w^T * x = w_0 + w_1*x_1 + ... + w_d*x_d
    概率输出：P(y=1|x) = sigma(z) = 1 / (1 + e^{-z})
    决策规则：y_hat = 1 if P(y=1|x) >= 0.5 else 0

---

### 2.2 Sigmoid函数推导——为什么选择它？

#### 2.2.1 从指数族分布推导

逻辑回归实际上可以由**广义线性模型（GLM）**框架严格推导出来。

假设 y|x 服从**伯努利分布**（二分类）：

    P(y=1|x) = phi
    P(y=0|x) = 1 - phi

伯努利分布属于指数族分布，可以写成：

    P(y|phi) = phi^y * (1-phi)^(1-y)
             = exp(y * log(phi/(1-phi)) + log(1-phi))

令自然参数 eta = log(phi/(1-phi))，解出：

    phi = 1 / (1 + e^{-eta})

在GLM中，令 eta = w^T * x，就得到了Sigmoid函数：

    P(y=1|x) = sigma(w^T * x) = 1 / (1 + exp(-w^T * x))

#### 2.2.2 Sigmoid函数的数学性质

    sigma(z) = 1 / (1 + e^{-z})

关键性质：
1. **值域 (0, 1)**：天然表示概率
2. **单调递增**：z越大，概率越高
3. **关于(0, 0.5)中心对称**：sigma(0) = 0.5
4. **导数优美**：sigma'(z) = sigma(z) * (1 - sigma(z)) = sigma(z) * sigma(-z)
5. **极限行为**：z -> +inf 时 sigma -> 1，z -> -inf 时 sigma -> 0
6. **饱和性**：|z|很大时梯度接近0（梯度消失问题）

**导数推导：**

    sigma'(z) = d/dz [1/(1+e^{-z})]
              = e^{-z} / (1+e^{-z})^2
              = [1/(1+e^{-z})] * [e^{-z}/(1+e^{-z})]
              = sigma(z) * (1 - sigma(z))

---

### 2.3 交叉熵损失函数推导（含MLE视角）

#### 2.3.1 从极大似然估计推导

假设训练集 {(x^(i), y^(i))}_{i=1}^m，y^(i) in {0, 1}。

**模型假设：**

    P(y=1|x) = h(x) = sigma(w^T * x)
    P(y=0|x) = 1 - h(x)

统一写成：

    P(y|x) = h(x)^y * (1 - h(x))^{1-y}

**似然函数：**

    L(w) = prod_{i=1}^{m} P(y^(i)|x^(i); w)
         = prod_{i=1}^{m} h(x^(i))^{y^(i)} * (1 - h(x^(i)))^{1-y^(i)}

**对数似然：**

    l(w) = log L(w)
         = sum_{i=1}^{m} [y^(i) * log(h(x^(i))) + (1-y^(i)) * log(1 - h(x^(i)))]

最大化对数似然等价于最小化**交叉熵损失（Cross-Entropy Loss）**：

    J(w) = -(1/m) * l(w)
         = -(1/m) * sum_{i=1}^{m} [y^(i) * log(h(x^(i))) + (1-y^(i)) * log(1 - h(x^(i)))]

#### 2.3.2 交叉熵损失的梯度推导

    J(w) = -(1/m) * sum [y^(i) * log(sigma(w^T x^(i))) + (1-y^(i)) * log(1 - sigma(w^T x^(i)))]

对 w 求梯度。利用 sigma'(z) = sigma(z)(1-sigma(z))：

    dJ/dw_j = -(1/m) * sum [y^(i) * (1/sigma) * sigma*(1-sigma) * x_j^(i)
                           - (1-y^(i)) * (1/(1-sigma)) * sigma*(1-sigma) * x_j^(i)]
            = -(1/m) * sum [y^(i) * (1-sigma) * x_j^(i) - (1-y^(i)) * sigma * x_j^(i)]
            = -(1/m) * sum [(y^(i) - sigma) * x_j^(i)]

写成向量形式，**梯度公式与线性回归完全一致**：

    dJ/dw = (1/m) * X^T * (h(X) - y)

其中 h(X) = sigma(Xw) 是预测概率向量。这个公式极其优美，是逻辑回归能高效训练的关键。

**参数更新：**

    w := w - alpha * (1/m) * X^T * (sigma(Xw) - y)

#### 2.3.3 为什么用交叉熵而不是MSE？

对于分类问题，MSE损失是非凸的（有多个局部最优），而交叉熵损失是**凸函数**，保证梯度下降能找到全局最优解。

直觉上：
- MSE在sigmoid饱和区（|z|很大时）梯度接近0，导致学习缓慢
- 交叉熵对错误预测惩罚更重：当y=1但h(x)->0时，loss->+inf

---

### 2.4 多分类扩展：Softmax回归

当分类数 K > 2 时，使用**Softmax回归**（也叫多项逻辑回归）。

**Softmax函数：**

    P(y=k|x) = exp(w_k^T * x) / sum_{j=1}^{K} exp(w_j^T * x)

其中 w_1, ..., w_K 是 K 个类别的参数向量（注意：通常只需 K-1 个独立参数向量，因为概率和为1）。

**交叉熵损失（多分类）：**

    J(W) = -(1/m) * sum_{i=1}^{m} sum_{k=1}^{K} [y_k^(i) * log(P(y=k|x^(i)))]

其中 y_k^(i) 是one-hot编码（样本i属于类别k则为1，否则为0）。

**Softmax的性质：**
1. 输出概率之和为1
2. 每个输出在(0,1)之间
3. 可微分，支持反向传播
4. 当K=2时退化为Sigmoid

**Softmax的梯度：**

    dJ/dw_k = (1/m) * sum_{i=1}^{m} (P(y=k|x^(i)) - y_k^(i)) * x^(i)

---

### 2.5 决策边界

逻辑回归的决策边界是**线性**的（在特征空间中是超平面）：

    P(y=1|x) = 0.5 时，sigma(w^T*x) = 0.5，即 w^T*x = 0

决策边界方程：w_0 + w_1*x_1 + w_2*x_2 + ... + w_d*x_d = 0

**非线性决策边界：** 通过特征工程（多项式特征）可以实现非线性边界：
- 二次特征：添加 x_1^2, x_2^2, x_1*x_2
- sklearn: PolynomialFeatures(degree=2)

---

### 2.6 Python代码示例

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from sklearn.datasets import make_classification, make_moons
import matplotlib.pyplot as plt

# ========== 1. 手写逻辑回归 ==========
class LogisticRegressionScratch:
    def __init__(self, lr=0.01, n_iters=1000, reg='l2', lambda_=0.01):
        self.lr = lr
        self.n_iters = n_iters
        self.reg = reg
        self.lambda_ = lambda_

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)  # 防止溢出
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, d = X.shape
        X_b = np.c_[np.ones((m, 1)), X]
        self.w = np.zeros(d + 1)
        self.losses = []

        for i in range(self.n_iters):
            z = X_b @ self.w
            h = self._sigmoid(z)
            # 交叉熵损失
            loss = -(1/m) * (y @ np.log(h + 1e-8) + (1-y) @ np.log(1-h + 1e-8))
            # 正则化
            if self.reg == 'l2':
                loss += (self.lambda_/(2*m)) * np.sum(self.w[1:]**2)
                gradient = (1/m) * (X_b.T @ (h - y))
                gradient[1:] += (self.lambda_/m) * self.w[1:]
            elif self.reg == 'l1':
                loss += (self.lambda_/m) * np.sum(np.abs(self.w[1:]))
                gradient = (1/m) * (X_b.T @ (h - y))
                gradient[1:] += (self.lambda_/m) * np.sign(self.w[1:])
            else:
                gradient = (1/m) * (X_b.T @ (h - y))

            self.w -= self.lr * gradient
            self.losses.append(loss)
        return self

    def predict_proba(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return self._sigmoid(X_b @ self.w)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# ========== 2. 生成二分类数据 ==========
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                           n_informative=2, random_state=42, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ========== 3. 手写模型训练 ==========
model_scratch = LogisticRegressionScratch(lr=0.1, n_iters=1000)
model_scratch.fit(X_train_s, y_train)
y_pred_s = model_scratch.predict(X_test_s)
print(f"手写逻辑回归 - Accuracy: {accuracy_score(y_test, y_pred_s):.4f}")

# ========== 4. sklearn模型 ==========
model_sk = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
model_sk.fit(X_train_s, y_train)
y_pred_sk = model_sk.predict(X_test_s)
y_proba_sk = model_sk.predict_proba(X_test_s)[:, 1]

print(f"sklearn逻辑回归 - Accuracy: {accuracy_score(y_test, y_pred_sk):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_sk):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_sk):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_pred_sk):.4f}")
print(f"  ROC AUC:   {roc_auc_score(y_test, y_proba_sk):.4f}")
print(classification_report(y_test, y_pred_sk))

# ========== 5. 决策边界可视化 ==========
def plot_decision_boundary(model, X, y, title, poly_degree=None):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    if poly_degree:
        poly = PolynomialFeatures(degree=poly_degree)
        grid = poly.fit_transform(grid)

    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k', s=20)
    plt.title(title)
    plt.xlabel('Feature 1'); plt.ylabel('Feature 2')
    plt.savefig(f'{title.replace(" ", "_")}.png'); plt.show()

plot_decision_boundary(model_sk, X_test_s, y_test, 'Logistic Regression Decision Boundary')

# ========== 6. ROC曲线 ==========
fpr, tpr, thresholds = roc_curve(y_test, y_proba_sk)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', label=f'AUC = {roc_auc_score(y_test, y_proba_sk):.3f}')
plt.plot([0, 1], [0, 1], 'r--', label='Random')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve'); plt.legend(); plt.grid(True)
plt.savefig('roc_curve.png'); plt.show()

# ========== 7. 非线性边界（多项式特征） ==========
X_moon, y_moon = make_moons(n_samples=300, noise=0.2, random_state=42)
X_moon_train, X_moon_test, y_moon_train, y_moon_test = train_test_split(
    X_moon, y_moon, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=3)
X_moon_train_p = poly.fit_transform(X_moon_train)
X_moon_test_p = poly.transform(X_moon_test)

model_poly = LogisticRegression(C=1.0, max_iter=1000)
model_poly.fit(X_moon_train_p, y_moon_train)
print(f"多项式逻辑回归(非线性) - Accuracy: {model_poly.score(X_moon_test_p, y_moon_test):.4f}")
```

---

### 2.7 超参数选择指南

| 超参数 | sklearn参数 | 说明 | 选择方法 |
|--------|------------|------|---------|
| 正则化强度 | C (注意: C=1/lambda) | C越大，正则化越弱 | 交叉验证，范围0.001~100 |
| 正则化类型 | penalty | 'l1', 'l2', 'elasticnet', 'none' | l2最常用 |
| 优化算法 | solver | 'lbfgs', 'liblinear', 'sag', 'saga', 'newton-cg' | 小数据:liblinear, 大数据:sag/saga |
| 最大迭代 | max_iter | 优化器迭代次数 | 观察收敛，100~1000 |
| 分类阈值 | threshold | predict的决策阈值，默认0.5 | 根据业务需求调整（如医疗诊断降低阈值） |
| 类别权重 | class_weight | 处理类别不平衡 | 'balanced'或手动设置 |
| 多分类策略 | multi_class | 'ovr'(一对多)或'multinomial' | 类别少用ovr, 多用multinomial |

**solver选择指南：**

| Solver | 支持L1 | 支持L2 | 支持多分类 | 适用场景 |
|--------|--------|--------|-----------|---------|
| liblinear | 是 | 是 | OVR | 小数据集 |
| lbfgs | 否 | 是 | 多分类 | 中等数据集（默认） |
| newton-cg | 否 | 是 | 多分类 | 中等数据集 |
| sag | 否 | 是 | 多分类 | 大数据集 |
| saga | 是 | 是 | 多分类 | 大数据集+L1 |

---

### 2.8 优缺点对比

**优点：**
1. 模型简单，计算高效，训练速度快
2. 输出具有概率意义（可解释性强）
3. 交叉熵损失是凸函数，保证全局最优
4. 不容易过拟合（低方差），适合高维稀疏数据
5. 可以通过正则化进一步控制复杂度
6. 在线性可分数据上表现优异

**缺点：**
1. 决策边界是线性的（需要特征工程来处理非线性）
2. 对特征空间的假设较强
3. 容易欠拟合（高偏差）
4. 对多重共线性敏感
5. 不适合处理大量多值特征（会导致维度过高）

---

### 2.9 常见面试/考试题

**Q1: 逻辑回归和线性回归的区别？**
A: (1) 线性回归解决回归问题，逻辑回归解决分类问题 (2) 线性回归输出连续值，逻辑回归输出概率(0,1) (3) 损失函数不同：MSE vs 交叉熵 (4) 逻辑回归在线性回归基础上加了Sigmoid激活函数

**Q2: 逻辑回归的决策边界是什么形状？**
A: 线性的（超平面）。因为决策边界是 w^T*x = 0，这是特征空间中的线性方程。通过多项式特征可以扩展为非线性边界。

**Q3: 为什么逻辑回归用交叉熵而不是MSE？**
A: (1) MSE对sigmoid输出是非凸的，有局部最优 (2) 交叉熵是凸的，保证全局最优 (3) MSE在sigmoid饱和区梯度消失，学习极慢 (4) 交叉熵在MLE框架下有严格的统计学推导

**Q4: 逻辑回归如何处理多分类？**
A: (1) OVR（One-vs-Rest）：训练K个二分类器，第k个判断"是/不是第k类" (2) Softmax：直接建模P(y=k|x)，用softmax函数保证概率归一化

**Q5: 逻辑回归中C参数和lambda参数的关系？**
A: C = 1/lambda。C越大，正则化越弱（对训练数据拟合更好，但可能过拟合）；C越小，正则化越强。

**Q6: 如何处理逻辑回归中的类别不平衡？**
A: (1) 设置class_weight='balanced' (2) 过采样少数类(SMOTE) (3) 欠采样多数类 (4) 调整决策阈值 (5) 使用F1/AUC而非Accuracy评估

**Q7: Sigmoid函数的缺点是什么？在深度学习中为什么被淘汰？**
A: (1) 梯度消失：|z|大时梯度接近0，深层网络训练困难 (2) 输出非零中心：导致梯度更新效率低 (3) 指数运算较慢。现代深度学习常用ReLU及其变体。

---

## 第三章：决策树 (Decision Tree)

### 3.1 算法概述

决策树是一种基于**树结构**进行决策的监督学习算法，可用于分类和回归。它通过对特征空间进行递归划分，构建一棵从根到叶的树。决策树的最大优势是**可解释性强**——模型的决策过程可以直观地用树形图表示。

**核心概念：**
- **内部节点**：对应一个特征（属性）上的测试
- **分支**：对应测试的一个输出（特征的一个取值或区间）
- **叶节点**：对应一个类别（分类树）或一个预测值（回归树）

**决策树学习算法的核心问题：**
1. 如何选择最优划分特征？（分裂准则）
2. 何时停止分裂？（停止条件）
3. 如何处理过拟合？（剪枝策略）

---

### 3.2 ID3算法——信息增益

#### 3.2.1 信息熵（Entropy）

信息熵是度量样本集合纯度（不确定性）的指标。对于样本集合 D，假设有 K 个类别，第 k 类样本的比例为 p_k：

    H(D) = -sum_{k=1}^{K} p_k * log_2(p_k)

**性质：**
- H(D) >= 0
- 当所有样本属于同一类别时，H(D) = 0（纯度最高，不确定性最低）
- 当所有类别等概率时，H(D) = log_2(K)（不确定性最大）

**示例：** 二分类问题，p_1 = p_2 = 0.5 时，H = -(0.5*log2(0.5) + 0.5*log2(0.5)) = 1 bit

#### 3.2.2 条件熵

在已知特征 A 的条件下，集合 D 的条件熵为：

    H(D|A) = sum_{v=1}^{V} (|D_v| / |D|) * H(D_v)

其中 V 是特征 A 的可能取值数，D_v 是特征 A 取第 v 个值的样本子集。

#### 3.2.3 信息增益（Information Gain）

**ID3算法使用信息增益作为特征选择准则：**

    Gain(D, A) = H(D) - H(D|A)

信息增益表示：知道特征 A 的信息后，数据集不确定性减少的程度。

**ID3算法步骤：**
1. 计算当前节点的信息熵 H(D)
2. 对每个候选特征 A，计算条件熵 H(D|A)
3. 选择信息增益最大的特征作为划分特征
4. 按该特征的取值划分子节点
5. 递归执行直到满足停止条件

**ID3的缺陷——偏好取值多的特征：**
假设有一个"样本编号"的特征，每个取值只对应一个样本，此时条件熵 H(D|A) = 0，信息增益最大。但这毫无意义——"样本编号"对预测没有泛化能力。

---

### 3.3 C4.5算法——增益率

为解决ID3偏好取值多的特征的问题，C4.5使用**增益率（Gain Ratio）**：

**特征 A 的固有值（Intrinsic Value）：**

    IV(A) = -sum_{v=1}^{V} (|D_v| / |D|) * log_2(|D_v| / |D|)

特征 A 的取值越多（V越大），IV(A) 越大。

**增益率：**

    GainRatio(D, A) = Gain(D, A) / IV(A)

**注意：** 增率率对取值少的特征有偏好。C4.5的做法是：先从候选特征中找出信息增益高于平均水平的特征，再从中选择增益率最高的。

---

### 3.4 CART算法——基尼指数

CART（Classification and Regression Tree）使用**基尼指数（Gini Index）** 作为分裂准则。

#### 3.4.1 基尼值

    Gini(D) = sum_{k=1}^{K} p_k * (1 - p_k) = 1 - sum_{k=1}^{K} p_k^2

**含义：** 从数据集 D 中随机抽取两个样本，其类别标记不一致的概率。

**性质：**
- Gini(D) ∈ [0, 1-1/K]
- Gini(D) = 0 时纯度最高（所有样本同一类别）
- Gini(D) 越小，纯度越高

#### 3.4.2 基尼指数（特征A的基尼指数）

    Gini_index(D, A) = sum_{v=1}^{V} (|D_v| / |D|) * Gini(D_v)

**CART算法选择基尼指数最小的特征进行划分。**

#### 3.4.3 三种分裂准则对比

| 准则 | 算法 | 公式 | 选择标准 | 特点 |
|------|------|------|---------|------|
| 信息增益 | ID3 | H(D) - H(D\|A) | 最大 | 偏好取值多的特征 |
| 增益率 | C4.5 | Gain / IV(A) | 最大 | 偏好取值少的特征（需修正） |
| 基尼指数 | CART | sum(\|D_v\|/\|D\|)*Gini(D_v) | 最小 | 计算简单，与熵近似 |

**信息熵 vs 基尼指数：** 当 K=2 时，信息熵 H = -p*log2(p) - (1-p)*log2(1-p) 和基尼指数 G = 2p(1-p) 在 p=0.5 附近非常接近，实际效果差异不大。基尼指数计算更简单（不需要对数运算），是CART的默认选择。

---

### 3.5 剪枝策略

决策树容易过拟合——如果不加限制，它会一直分裂直到每个叶节点只有一个样本（完美拟合训练集但泛化极差）。剪枝是解决过拟合的主要方法。

#### 3.5.1 预剪枝（Pre-pruning）

在树的生长过程中，提前终止分裂。

**常用停止条件：**
1. **最大深度**（max_depth）：树达到指定深度时停止
2. **最小分裂样本数**（min_samples_split）：节点样本数少于阈值时不分裂
3. **最小叶节点样本数**（min_samples_leaf）：分裂后叶节点样本数不够则不分裂
4. **最大叶节点数**（max_leaf_nodes）：限制叶节点总数
5. **最小信息增益**：分裂带来的增益小于阈值则停止
6. **最大特征数**（max_features）：每次分裂只考虑部分特征

**优点：** 训练速度快，节省计算资源
**缺点：** 可能过早停止（欠拟合），因为有些特征当前增益小但后续增益大

#### 3.5.2 后剪枝（Post-pruning）

先生成完整的决策树，再从下往上逐步剪枝。

**CART的代价复杂度剪枝（Cost-Complexity Pruning, CCP）：**

    C_alpha(T) = C(T) + alpha * |T_leaf|

其中：
- C(T) 是树 T 的训练误差（分类错误率或MSE）
- |T_leaf| 是叶节点数（树的复杂度）
- alpha 是复杂度参数（正则化系数）

**步骤：**
1. 生成完整的树 T_0
2. 对于一系列递增的 alpha 值，逐步剪枝得到嵌套的子树序列 T_0, T_1, ..., T_n
3. 用交叉验证选择最佳 alpha
4. 返回对应的最佳子树

**sklearn实现：**

```python
from sklearn.tree import DecisionTreeClassifier
# 使用ccp_alpha进行后剪枝
clf = DecisionTreeClassifier(ccp_alpha=0.01)
```

**预剪枝 vs 后剪枝对比：**

| 方面 | 预剪枝 | 后剪枝 |
|------|--------|--------|
| 训练时间 | 短 | 长（需先建完整树） |
| 欠拟合风险 | 较高 | 较低 |
| 过拟合控制 | 一般 | 较好 |
| 实现复杂度 | 简单 | 较复杂 |
| 常用方法 | max_depth, min_samples | ccp_alpha |

---

### 3.6 回归树

CART回归树使用**最小二乘法**选择划分特征和划分点。

**划分准则：** 对于特征 j 和划分点 s，将数据划分为 R_1(j,s) = {x|x_j <= s} 和 R_2(j,s) = {x|x_j > s}，选择使下式最小化的 (j, s)：

    min_{j,s} [min_{c1} sum_{x_i in R_1} (y_i - c_1)^2 + min_{c2} sum_{x_i in R_2} (y_i - c_2)^2]

其中 c_1 和 c_2 分别是两个区域的均值：

    c_1 = mean(y_i | x_i in R_1)
    c_2 = mean(y_i | x_i in R_2)

**输出：** 叶节点的预测值 = 该节点所有训练样本 y 值的均值。

---

### 3.7 Python代码示例

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.datasets import load_iris, make_regression
import matplotlib.pyplot as plt

# ========== 1. 分类树 ==========
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ID3风格（用信息增益，但sklearn CART只支持gini和entropy）
# 使用entropy即信息增益
clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf_entropy.fit(X_train, y_train)
print(f"信息熵准则 - Accuracy: {clf_entropy.score(X_test, y_test):.4f}")

# CART默认（基尼指数）
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf_gini.fit(X_train, y_train)
print(f"基尼指数准则 - Accuracy: {clf_gini.score(X_test, y_test):.4f}")

# ========== 2. 树的可视化 ==========
print("\n决策树文本表示：")
print(export_text(clf_gini, feature_names=list(feature_names)))

plt.figure(figsize=(20, 10))
plot_tree(clf_gini, feature_names=feature_names, class_names=list(class_names),
          filled=True, rounded=True, fontsize=10)
plt.title('CART Decision Tree (Gini)')
plt.tight_layout()
plt.savefig('decision_tree_iris.png', dpi=150)
plt.show()

# ========== 3. 特征重要性 ==========
importances = clf_gini.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.4f}")

# ========== 4. 后剪枝（CCP Alpha路径） ==========
path = clf_gini.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

train_scores = []
test_scores = []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, 'b-o', label='Train Accuracy')
plt.plot(ccp_alphas, test_scores, 'r-o', label='Test Accuracy')
plt.xlabel('ccp_alpha'); plt.ylabel('Accuracy')
plt.title('Cost-Complexity Pruning Path')
plt.legend(); plt.grid(True)
plt.savefig('ccp_alpha_path.png'); plt.show()

# ========== 5. 超参数网格搜索 ==========
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5, 6, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'ccp_alpha': [0.0, 0.01, 0.02, 0.05],
}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid,
                    cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print(f"\n最佳参数: {grid.best_params_}")
print(f"最佳交叉验证准确率: {grid.best_score_:.4f}")
print(f"测试集准确率: {grid.best_estimator_.score(X_test, y_test):.4f}")

# ========== 6. 回归树 ==========
X_reg, y_reg = make_regression(n_samples=200, n_features=1, noise=15, random_state=42)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

reg_tree = DecisionTreeRegressor(max_depth=4, random_state=42)
reg_tree.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_tree.predict(X_reg_test)
print(f"\n回归树 MSE: {mean_squared_error(y_reg_test, y_reg_pred):.2f}")

# 回归树预测可视化
X_plot = np.linspace(X_reg.min(), X_reg.max(), 500).reshape(-1, 1)
y_plot = reg_tree.predict(X_plot)
plt.figure(figsize=(10, 6))
plt.scatter(X_reg_test, y_reg_test, c='blue', alpha=0.5, label='Test Data')
plt.plot(X_plot, y_plot, 'r-', linewidth=2, label='Regression Tree Prediction')
plt.xlabel('X'); plt.ylabel('y')
plt.title('Decision Tree Regression')
plt.legend(); plt.grid(True)
plt.savefig('regression_tree.png'); plt.show()
```

---

### 3.8 超参数选择指南

| 超参数 | sklearn参数 | 说明 | 推荐范围 |
|--------|------------|------|---------|
| 分裂准则 | criterion | 'gini'或'entropy' | 差异不大，默认gini |
| 最大深度 | max_depth | 树的最大深度 | 3~15或None |
| 最小分裂样本 | min_samples_split | 节点最少样本才分裂 | 2~20 |
| 最小叶节点样本 | min_samples_leaf | 叶节点最少样本 | 1~10 |
| 最大叶节点数 | max_leaf_nodes | 叶节点总数限制 | 10~100 |
| 最大特征数 | max_features | 每次分裂考虑的特征 | sqrt(d)或log2(d) |
| 后剪枝参数 | ccp_alpha | 复杂度参数 | 0~0.1（观察路径选择） |

---

### 3.9 优缺点对比

**优点：**
1. **可解释性强**：决策过程直观，易于可视化
2. **不需要特征缩放**：不受特征量纲影响
3. **能处理混合类型特征**：同时处理数值和类别特征
4. **能捕捉非线性关系**：通过递归划分实现非线性决策边界
5. **计算效率高**：预测时间O(log n)
6. **支持多输出**：天然支持多分类

**缺点：**
1. **容易过拟合**：不加剪枝的树会过拟合训练数据
2. **不稳定**：数据微小变化可能导致完全不同的树结构
3. **贪心算法**：每次局部最优选择不保证全局最优
4. **对不平衡数据敏感**：偏向多数类
5. **不擅长线性关系**：用阶梯函数逼近线性关系，效率低

---

### 3.10 常见面试/考试题

**Q1: 信息增益、增益率、基尼指数的区别？**
A: 信息增益偏好取值多的特征（ID3），增益率通过除以固有值修正了这个问题但偏好取值少的特征（C4.5），基尼指数计算简单且与信息熵近似等价（CART）。

**Q2: 为什么决策树需要剪枝？**
A: 不加限制的树会一直分裂到每个叶节点只有一个样本，完美记忆训练数据但泛化能力极差（过拟合）。剪枝通过限制树的复杂度来控制过拟合。

**Q3: CART分类树和回归树的分裂准则有什么区别？**
A: 分类树用基尼指数或信息熵，回归树用最小二乘法（最小化MSE）。分类树叶节点输出多数类，回归树叶节点输出样本均值。

**Q4: 决策树的优缺点？各举两个。**
A: 优点：可解释性强、不需要特征缩放。缺点：容易过拟合、对数据变化不稳定（可引入随机森林解决）。

**Q5: 如何处理决策树中的连续特征？**
A: 将连续特征的取值排序后，依次取相邻值的中点作为候选划分点，选择使分裂准则最优的划分点。CART和C4.5都支持连续特征。

**Q6: 预剪枝和后剪枝哪个更好？**
A: 后剪枝通常泛化能力更好（不容易欠拟合），但训练时间更长。预剪枝简单高效但可能过早停止。实践中常结合使用。

---

## 第四章：支持向量机 (SVM)

### 4.1 算法概述

支持向量机（Support Vector Machine, SVM）是一种强大且优雅的监督学习算法，可用于分类、回归和异常检测。SVM的核心思想是：在特征空间中找到一个**最大间隔超平面**来划分不同类别的样本。

**核心概念：**
- **超平面（Hyperplane）**：d维空间中的(d-1)维子空间，是决策边界
- **间隔（Margin）**：超平面到最近样本点的距离
- **支持向量（Support Vectors）**：距离超平面最近的样本点，决定超平面的位置
- **核技巧（Kernel Trick）**：将数据映射到高维空间以处理非线性问题

---

### 4.2 硬间隔SVM（线性可分情况）

#### 4.2.1 问题设定

给定训练集 {(x^(i), y^(i))}_{i=1}^m，其中 y^(i) in {-1, +1}。

超平面定义为：w^T * x + b = 0

分类决策函数：f(x) = sign(w^T * x + b)

#### 4.2.2 间隔的计算

样本点 x^(i) 到超平面的**函数间隔**：

    gamma_hat^(i) = y^(i) * (w^T * x^(i) + b)

函数间隔的问题：等比例缩放 w 和 b 会使间隔变化，但超平面不变。

**几何间隔**（归一化）：

    gamma^(i) = y^(i) * (w^T * x^(i) + b) / ||w||

**所有样本的最小几何间隔（即间隔）：**

    gamma = min_{i} gamma^(i)

#### 4.2.3 最大间隔分类器

我们的目标是最大化间隔 gamma：

    max_{w,b} gamma
    s.t.  y^(i) * (w^T * x^(i) + b) / ||w|| >= gamma,  for all i

由于函数间隔可以任意缩放，我们令 gamma_hat = 1（即最近样本的函数间隔为1），则：

    y^(i) * (w^T * x^(i) + b) >= 1,  for all i

此时几何间隔 = 1/||w||。最大化几何间隔等价于最小化 ||w||：

    min_{w,b} (1/2) * ||w||^2
    s.t.  y^(i) * (w^T * x^(i) + b) >= 1,  for all i

这是SVM的**原始优化问题**（凸二次规划问题）。取1/2和平方是为了求导方便。

---

### 4.3 对偶问题推导

使用**拉格朗日乘子法**将约束优化问题转化为无约束问题。

**拉格朗日函数：**

    L(w, b, alpha) = (1/2) * ||w||^2 - sum_{i=1}^{m} alpha^(i) * [y^(i) * (w^T * x^(i) + b) - 1]

其中 alpha^(i) >= 0 是拉格朗日乘子。

**对 w 求偏导并令其为零：**

    dL/dw = w - sum_{i=1}^{m} alpha^(i) * y^(i) * x^(i) = 0
    =>  w = sum_{i=1}^{m} alpha^(i) * y^(i) * x^(i)    ... (★)

**对 b 求偏导并令其为零：**

    dL/db = -sum_{i=1}^{m} alpha^(i) * y^(i) = 0
    =>  sum_{i=1}^{m} alpha^(i) * y^(i) = 0            ... (★★)

将(★)(★★)代入拉格朗日函数，利用 w^T * x^(i) = sum_j alpha^(j) y^(j) (x^(j))^T x^(i)，得到**对偶问题**：

    max_alpha  sum_{i=1}^{m} alpha^(i) - (1/2) * sum_{i=1}^{m} sum_{j=1}^{m} alpha^(i) * alpha^(j) * y^(i) * y^(j) * (x^(i))^T * x^(j)
    s.t.  alpha^(i) >= 0,  for all i
          sum_{i=1}^{m} alpha^(i) * y^(i) = 0

**KKT条件（Karush-Kuhn-Tucker）：**

    alpha^(i) >= 0
    y^(i) * (w^T * x^(i) + b) - 1 >= 0
    alpha^(i) * [y^(i) * (w^T * x^(i) + b) - 1] = 0    （互补松弛条件）

**重要推论——支持向量：**
- 若 alpha^(i) = 0：样本不在边界上，不影响超平面
- 若 alpha^(i) > 0：必须有 y^(i) * (w^T * x^(i) + b) = 1，即样本恰好在间隔边界上——这些就是**支持向量**

**预测函数（仅依赖支持向量）：**

    f(x) = sign(sum_{i in SV} alpha^(i) * y^(i) * (x^(i))^T * x + b)

---

### 4.4 软间隔SVM

现实中数据往往**线性不可分**。软间隔SVM允许部分样本违反间隔约束。

**引入松弛变量 xi^(i) >= 0：**

    min_{w,b,xi} (1/2) * ||w||^2 + C * sum_{i=1}^{m} xi^(i)
    s.t.  y^(i) * (w^T * x^(i) + b) >= 1 - xi^(i),  for all i
          xi^(i) >= 0,  for all i

**C 的含义：**
- C 大：对误分类的惩罚大，间隔小，可能过拟合
- C 小：对误分类的容忍大，间隔大，可能欠拟合
- C -> inf 时退化为硬间隔

**对偶问题（软间隔）：**

    max_alpha  sum alpha^(i) - (1/2) * sum_i sum_j alpha^(i) * alpha^(j) * y^(i) * y^(j) * (x^(i))^T * x^(j)
    s.t.  0 <= alpha^(i) <= C,  for all i
          sum alpha^(i) * y^(i) = 0

相比硬间隔，多了 alpha^(i) <= C 的上界约束。

---

### 4.5 核函数详解

#### 4.5.1 核技巧的动机

对于非线性可分数据，将数据映射到高维空间 phi(x) 后可能线性可分。但直接计算 phi(x) 可能维度极高甚至无穷维。

**核函数定义：**

    K(x, z) = phi(x)^T * phi(z)

核函数可以直接在原始空间计算高维空间的内积，**无需显式映射**——这就是**核技巧（Kernel Trick）**。

在对偶问题中，所有 x^(i) 的出现形式都是内积 (x^(i))^T * x^(j)，将其替换为 K(x^(i), x^(j)) 即可。

#### 4.5.2 常用核函数

**1. 线性核（Linear Kernel）**

    K(x, z) = x^T * z

适用场景：特征数多、样本线性可分、文本分类（高维稀疏数据）

**2. 多项式核（Polynomial Kernel）**

    K(x, z) = (gamma * x^T * z + r)^d

参数：
- d：多项式阶数（常用2, 3）
- gamma：缩放系数
- r：常数偏移项

适用场景：数据具有多项式关系、低维数据

**3. 径向基核（RBF Kernel / 高斯核）**

    K(x, z) = exp(-gamma * ||x - z||^2)

其中 gamma = 1/(2*sigma^2)。gamma越大，高斯分布越窄，决策边界越复杂。

**RBF核可以映射到无穷维空间！** 这是它最强大之处。

适用场景：通用默认选择、不知道数据分布时、非线性关系

**4. Sigmoid核**

    K(x, z) = tanh(gamma * x^T * z + r)

等价于一个单隐层神经网络。

适用场景：启发式使用，实际较少用

#### 4.5.3 核函数选择指南

| 核函数 | 维度 | 计算复杂度 | 适用场景 | 超参数 |
|--------|------|-----------|---------|--------|
| 线性 | 原始维度 | 最低 | 高维稀疏数据、线性可分 | 无 |
| 多项式 | 有限高维 | 中等 | 低维多项式关系 | d, gamma, r |
| RBF | 无穷维 | 较高 | 通用默认、非线性 | gamma |
| Sigmoid | 可变 | 中等 | 神经网络类比 | gamma, r |

**经验法则：**
1. 先试线性核（尤其特征数 > 样本数时）
2. 通用场景用RBF核
3. 特征数少、样本少时试多项式核

**Mercer定理：** 一个函数K是合法核函数，当且仅当对任意样本集，核矩阵 K_ij = K(x^(i), x^(j)) 是半正定的。

---

### 4.6 SMO算法思路

**序列最小优化算法（Sequential Minimal Optimization, SMO）** 是John Platt于1998年提出的高效求解SVM对偶问题的算法。

**核心思想：** 将大优化问题分解为一系列最小的子问题——每次只优化两个变量（因为等式约束 sum alpha^(i)*y^(i) = 0 意味着至少要同时更新两个变量）。

**SMO步骤：**
1. 选择两个变量 alpha_i 和 alpha_j（启发式选择：违反KKT条件最严重的变量）
2. 固定其他所有变量，将目标函数对 (alpha_i, alpha_j) 进行解析优化
3. 更新 alpha_i 和 alpha_j（含裁剪到 [0, C] 范围）
4. 更新偏置 b
5. 重复直到所有变量满足KKT条件（在容忍度内）

**为什么每次只选2个变量？**
等式约束 sum alpha^(i)*y^(i) = 0 意味着改变一个变量必须至少改变另一个来保持等式成立。选2个是能解析求解的最小规模。

**时间复杂度：** 约 O(m^2) 到 O(m^3)，远优于一般的二次规划求解器。

---

### 4.7 Python代码示例

```python
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification, make_moons, make_circles
import matplotlib.pyplot as plt

# ========== 1. 数据生成 ==========
# 线性可分
X_linear, y_linear = make_classification(n_samples=200, n_features=2, n_redundant=0,
                                          n_informative=2, random_state=42, n_clusters_per_class=1)
# 非线性可分（月牙形）
X_moon, y_moon = make_moons(n_samples=300, noise=0.15, random_state=42)
# 非线性可分（同心圆）
X_circle, y_circle = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)

datasets = [
    (X_linear, y_linear, 'Linear Separable'),
    (X_moon, y_moon, 'Moons'),
    (X_circle, y_circle, 'Circles'),
]

# ========== 2. 不同核函数对比 ==========
kernels = ['linear', 'poly', 'rbf']
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for row, (X, y, data_name) in enumerate(datasets):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    for col, kernel in enumerate(kernels):
        svm = SVC(kernel=kernel, C=1.0, gamma='scale', degree=3)
        svm.fit(X_train_s, y_train)
        score = svm.score(X_test_s, y_test)

        # 绘制决策边界
        ax = axes[row][col]
        h = 0.02
        x_min, x_max = X_train_s[:, 0].min() - 1, X_train_s[:, 0].max() + 1
        y_min, y_max = X_train_s[:, 1].min() - 1, X_train_s[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.scatter(X_train_s[:, 0], X_train_s[:, 1], c=y_train, cmap='RdYlBu',
                   edgecolors='k', s=15, alpha=0.6)
        # 标记支持向量
        sv = svm.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=100, facecolors='none', edgecolors='black', linewidths=2)
        ax.set_title(f'{data_name} - {kernel} (Acc={score:.2f})')

plt.tight_layout()
plt.savefig('svm_kernels_comparison.png', dpi=150)
plt.show()

# ========== 3. C参数和gamma参数的影响 ==========
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

C_values = [0.01, 1, 100]
gamma_values = [0.1, 1, 10]
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for i, C in enumerate(C_values):
    for j, gamma in enumerate(gamma_values):
        svm = SVC(kernel='rbf', C=C, gamma=gamma)
        svm.fit(X_train_s, y_train)
        score = svm.score(X_test_s, y_test)

        ax = axes[i][j]
        h = 0.02
        x_min, x_max = X_train_s[:, 0].min() - 1, X_train_s[:, 0].max() + 1
        y_min, y_max = X_train_s[:, 1].min() - 1, X_train_s[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.scatter(X_train_s[:, 0], X_train_s[:, 1], c=y_train, cmap='RdYlBu', edgecolors='k', s=15)
        ax.set_title(f'C={C}, gamma={gamma}, Acc={score:.2f}')

plt.suptitle('Effect of C and gamma on RBF SVM', fontsize=14)
plt.tight_layout()
plt.savefig('svm_hyperparameters.png', dpi=150)
plt.show()

# ========== 4. 超参数网格搜索 ==========
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
    'kernel': ['rbf', 'linear', 'poly'],
}
grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_s, y_train)
print(f"最佳参数: {grid.best_params_}")
print(f"最佳准确率: {grid.best_score_:.4f}")
print(f"测试集准确率: {grid.best_estimator_.score(X_test_s, y_test):.4f}")
print(f"支持向量数: {grid.best_estimator_.n_support_}")

# ========== 5. 支持向量回归(SVR) ==========
from sklearn.svm import SVR
np.random.seed(42)
X_svr = np.sort(5 * np.random.rand(100, 1), axis=0)
y_svr = np.sin(X_svr).ravel() + 0.1 * np.random.randn(100)

for kernel in ['linear', 'rbf', 'poly']:
    svr = SVR(kernel=kernel, C=100, gamma='scale', epsilon=0.1)
    svr.fit(X_svr, y_svr)
    X_plot = np.linspace(0, 5, 200).reshape(-1, 1)
    y_plot = svr.predict(X_plot)
    print(f"SVR ({kernel}) - MSE: {np.mean((y_svr - svr.predict(X_svr))**2):.4f}")
```

---

### 4.8 超参数选择指南

| 超参数 | sklearn参数 | 说明 | 常用范围 |
|--------|------------|------|---------|
| 正则化参数 | C | 控制间隔与误分类的权衡 | 0.01~1000 (对数尺度) |
| 核函数 | kernel | 'linear','rbf','poly','sigmoid' | 先试linear再试rbf |
| RBF核gamma | gamma | 控制单个样本的影响范围 | 'scale','auto', 0.001~100 |
| 多项式度数 | degree | 多项式核的阶数 | 2, 3, 4 |
| 核函数系数 | coef0 | poly和sigmoid核的偏移 | 0, 1 |

**gamma的选择技巧：**
- gamma大：每个样本影响范围小，决策边界复杂，可能过拟合
- gamma小：每个样本影响范围大，决策边界平滑，可能欠拟合
- 推荐先用 'scale'（1/(n_features * X.var())）

---

### 4.9 优缺点对比

**优点：**
1. **最大间隔原则**：有良好的泛化理论保证
2. **核技巧**：能高效处理非线性问题
3. **高维有效**：在高维空间中表现优异，甚至维度>样本数时
4. **内存效率**：决策只依赖支持向量（稀疏解）
5. **理论基础扎实**：有VC维理论和统计学习理论支撑

**缺点：**
1. **大数据集慢**：训练时间复杂度 O(m^2 ~ m^3)
2. **对参数敏感**：C和gamma的选择对性能影响大
3. **不直接提供概率**：需要通过Platt缩放（SVM predict_proba较慢）
4. **对特征缩放敏感**：需要标准化
5. **核函数选择困难**：没有通用的选择方法
6. **不适合多分类**：需要OVR或OVO策略

---

### 4.10 常见面试/考试题

**Q1: SVM的核心思想是什么？**
A: 找到一个最大间隔超平面来划分数据。间隔越大，泛化能力越强。支持向量是距离超平面最近的样本，决定了超平面的位置。

**Q2: 什么是核技巧？为什么有效？**
A: 核技巧是用核函数K(x,z)代替高维映射后的内积phi(x)^T*phi(z)，避免显式计算高维映射。有效是因为SVM对偶问题中数据只以内积形式出现。

**Q3: RBF核的gamma参数有什么影响？**
A: gamma = 1/(2*sigma^2)。gamma大→sigma小→每个样本影响范围小→决策边界复杂（过拟合）。gamma小→sigma大→影响范围大→边界平滑（欠拟合）。

**Q4: 硬间隔和软间隔的区别？**
A: 硬间隔要求所有样本正确分类且在间隔外（线性可分假设）。软间隔引入松弛变量允许部分样本违反约束，通过C参数控制惩罚强度。

**Q5: SVM对偶问题中KKT条件的意义？**
A: 互补松弛条件 alpha^(i) * [y^(i)*(w^T*x^(i)+b) - 1] = 0 意味着：要么alpha=0（非支持向量），要么函数间隔=1（支持向量在间隔边界上）。这解释了SVM解的稀疏性。

**Q6: 为什么SVM的解是稀疏的？**
A: 由KKT条件，只有支持向量（alpha>0）参与决策，大部分样本的alpha=0。预测时只需计算与支持向量的核函数值，计算效率高。

---

## 第五章：朴素贝叶斯 (Naive Bayes)

### 5.1 算法概述

朴素贝叶斯是一种基于**贝叶斯定理**和**特征条件独立假设**的分类算法。虽然"朴素"的条件独立假设在现实中几乎不成立，但朴素贝叶斯在文本分类、垃圾邮件过滤等任务中表现出色，且训练和预测速度极快。

---

### 5.2 贝叶斯定理推导

#### 5.2.1 从条件概率出发

对于事件 A 和 B（P(B) > 0）：

    P(A|B) = P(A ∩ B) / P(B)

同理：

    P(B|A) = P(A ∩ B) / P(A)

因此 P(A ∩ B) = P(A|B) * P(B) = P(B|A) * P(A)

#### 5.2.2 贝叶斯定理

    P(A|B) = P(B|A) * P(A) / P(B)

在分类问题中，A 是类别 y，B 是观测到的特征 x：

    P(y|x) = P(x|y) * P(y) / P(x)

**各项的含义：**
- **后验概率 P(y|x)**：观测到特征 x 后，属于类别 y 的概率（我们要求的）
- **似然 P(x|y)**：类别 y 的条件下，出现特征 x 的概率
- **先验概率 P(y)**：类别 y 在总体中出现的概率
- **证据 P(x)**：特征 x 出现的概率（对所有类别相同，可忽略）

#### 5.2.3 分类决策规则

    y_hat = argmax_y P(y|x) = argmax_y P(x|y) * P(y)

---

### 5.3 条件独立假设

直接计算 P(x|y) = P(x_1, x_2, ..., x_d | y) 需要估计指数级的参数（每个特征组合都需要），这在高维数据中不可行。

**朴素贝叶斯的核心假设——特征条件独立：**

    P(x_1, x_2, ..., x_d | y) = prod_{j=1}^{d} P(x_j | y)

即给定类别 y 的条件下，各特征之间相互独立。

**这个假设的"朴素"之处：**
- 现实中特征通常不独立（如"出现单词free"和"出现单词money"在垃圾邮件中正相关）
- 但这个假设大大简化了计算：从估计 P(x_1,...,x_d|y) 需要 O(V^d) 个参数降到 O(d*V) 个参数
- 实践中即使假设不成立，分类精度往往仍然很好（因为分类只需要 argmax，不需要精确概率）

**最终决策函数：**

    y_hat = argmax_y P(y) * prod_{j=1}^{d} P(x_j | y)

取对数防止下溢（连乘变连加）：

    y_hat = argmax_y [log P(y) + sum_{j=1}^{d} log P(x_j | y)]

---

### 5.4 拉普拉斯平滑（Laplace Smoothing）

**问题：** 如果训练集中某个特征值在某个类别下从未出现，则 P(x_j = v | y) = 0，导致整个乘积为0（零概率问题）。

**解决方法——拉普拉斯平滑（加一平滑）：**

    P(x_j = v | y) = (count(x_j = v, y) + alpha) / (count(y) + alpha * V_j)

其中：
- count(x_j = v, y)：特征 j 取值为 v 且类别为 y 的样本数
- count(y)：类别为 y 的样本总数
- V_j：特征 j 的可能取值数
- alpha：平滑参数（alpha=1 为拉普拉斯平滑，0 < alpha < 1 为Lidstone平滑）

**先验概率的平滑：**

    P(y) = (count(y) + alpha) / (m + alpha * K)

其中 K 为类别数。

**alpha 的影响：**
- alpha = 0：无平滑，可能出现零概率
- alpha = 1：标准拉普拉斯平滑
- alpha < 1：较弱平滑
- alpha > 1：较强平滑（更强的先验假设）

---

### 5.5 三种朴素贝叶斯变体

#### 5.5.1 高斯朴素贝叶斯（Gaussian Naive Bayes）

**假设：** 每个特征在每个类别下服从**正态分布**：

    P(x_j | y) = (1 / sqrt(2*pi*sigma_{y,j}^2)) * exp(-(x_j - mu_{y,j})^2 / (2*sigma_{y,j}^2))

参数估计：
    mu_{y,j} = (1/|C_y|) * sum_{i in C_y} x_j^(i)       （类y中特征j的均值）
    sigma_{y,j}^2 = (1/|C_y|) * sum_{i in C_y} (x_j^(i) - mu_{y,j})^2  （方差）

**适用场景：** 连续特征（身高、温度、收入等），是最常用的朴素贝叶斯变体。

**sklearn类：** `GaussianNB`

#### 5.5.2 多项式朴素贝叶斯（Multinomial Naive Bayes）

**假设：** 特征服从**多项式分布**，特征值为非负整数（如词频）。

    P(x_j | y) = (count(x_j, y) + alpha) / (sum_{k=1}^{d} count(x_k, y) + alpha * V)

在文本分类中，x_j 表示词 j 在文档中出现的次数。

**适用场景：** 文本分类（词频/TF-IDF特征）、离散计数数据。

**sklearn类：** `MultinomialNB`

#### 5.5.3 伯努利朴素贝叶斯（Bernoulli Naive Bayes）

**假设：** 特征服从**伯努利分布**（0/1二值特征，即特征是否出现）。

    P(x_j | y) = P(x_j=1|y)^{x_j} * (1 - P(x_j=1|y))^{1-x_j}

参数估计：
    P(x_j=1|y) = (count(x_j=1, y) + alpha) / (count(y) + 2*alpha)

**与多项式NB的区别：**
- 多项式NB使用词频（出现几次），伯努利NB只看是否出现（0或1）
- 伯努利NB额外考虑特征不出现的证据（P(x_j=0|y)），在短文本中效果更好

**适用场景：** 短文本分类、二值特征、文档是否包含某关键词。

**sklearn类：** `BernoulliNB`

#### 5.5.4 三种变体对比

| 特性 | 高斯NB | 多项式NB | 伯努利NB |
|------|--------|---------|---------|
| 特征类型 | 连续 | 非负整数(计数) | 二值(0/1) |
| 分布假设 | 正态分布 | 多项式分布 | 伯努利分布 |
| 典型应用 | 通用分类 | 文本分类(词频) | 短文本(词是否出现) |
| 参数估计 | 均值和方差 | 频率+平滑 | 概率+平滑 |
| sklearn类 | GaussianNB | MultinomialNB | BernoulliNB |

---

### 5.6 Python代码示例

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris, fetch_20newsgroups
import matplotlib.pyplot as plt

# ========== 1. 手写高斯朴素贝叶斯 ==========
class GaussianNBNaive:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.params = {}  # {class: {'mean': [...], 'var': [...], 'prior': float}}
        for c in self.classes:
            X_c = X[y == c]
            self.params[c] = {
                'mean': X_c.mean(axis=0),
                'var': X_c.var(axis=0) + 1e-9,  # 加小值防零方差
                'prior': len(X_c) / len(X)
            }
        return self

    def _log_likelihood(self, x, mean, var):
        # log P(x|y) = sum_j log N(x_j; mean_j, var_j)
        return -0.5 * np.sum(np.log(2 * np.pi * var) + (x - mean)**2 / var)

    def predict(self, X):
        predictions = []
        for x in X:
            log_posteriors = []
            for c in self.classes:
                p = self.params[c]
                log_post = np.log(p['prior']) + self._log_likelihood(x, p['mean'], p['var'])
                log_posteriors.append(log_post)
            predictions.append(self.classes[np.argmax(log_posteriors)])
        return np.array(predictions)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

# ========== 2. 鸢尾花数据集测试 ==========
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 手写版
model_scratch = GaussianNBNaive().fit(X_train, y_train)
print(f"手写高斯NB - Accuracy: {model_scratch.score(X_test, y_test):.4f}")

# sklearn版
model_sk = GaussianNB().fit(X_train, y_train)
print(f"sklearn高斯NB - Accuracy: {model_sk.score(X_test, y_test):.4f}")
print(f"各类别先验概率: {model_sk.class_prior_}")
print(f"各类别各特征均值:\n{model_sk.theta_}")
print(f"各类别各特征方差:\n{model_sk.var_}")

# ========== 3. 文本分类：垃圾邮件检测 ==========
# 模拟邮件数据
emails = [
    "free money win cash prize now",
    "click here to claim your reward",
    "free gift card limited offer",
    "winner winner you won lottery",
    "buy cheap medications online now",
    "meeting at 3pm tomorrow",
    "please review the attached report",
    "project deadline extended to friday",
    "lunch plans for today?",
    "quarterly earnings report attached",
    "congratulations you won a free iphone",
    "urgent business proposal confidential",
    "team standup meeting tomorrow morning",
    "please update the documentation",
    "special discount offer act now",
]
labels = [1,1,1,1,1, 0,0,0,0,0, 1,1, 0,0, 1]  # 1=spam, 0=ham

# 词频特征（多项式NB）
vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(emails)

X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(
    X_text, labels, test_size=0.3, random_state=42)

# 多项式NB
mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_text_train, y_text_train)
print(f"\n多项式NB(词频) - Accuracy: {mnb.score(X_text_test, y_text_test):.4f}")

# 伯努利NB
bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_text_train, y_text_train)
print(f"伯努利NB(二值) - Accuracy: {bnb.score(X_text_test, y_text_test):.4f}")

# TF-IDF特征 + 高斯NB
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(emails).toarray()
X_tfidf_train, X_tfidf_test, _, _ = train_test_split(X_tfidf, labels, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_tfidf_train, y_text_train)
print(f"高斯NB(TF-IDF) - Accuracy: {gnb.score(X_tfidf_test, y_text_test):.4f}")

# ========== 4. 拉普拉斯平滑的影响 ==========
alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
scores = []
for a in alphas:
    clf = MultinomialNB(alpha=a)
    cv_score = cross_val_score(clf, X_text, labels, cv=3, scoring='accuracy').mean()
    scores.append(cv_score)

plt.figure(figsize=(8, 5))
plt.plot(alphas, scores, 'bo-')
plt.xscale('log')
plt.xlabel('Alpha (Laplace Smoothing)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Effect of Laplace Smoothing on MultinomialNB')
plt.grid(True)
plt.savefig('naive_bayes_smoothing.png'); plt.show()

# ========== 5. 20 Newsgroups大型文本分类 ==========
from sklearn.pipeline import Pipeline

categories = ['sci.space', 'comp.graphics', 'rec.sport.baseball', 'talk.politics.guns']
train_data = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
test_data = fetch_20newsgroups(subset='test', categories=categories, random_state=42)

# Pipeline: TF-IDF + 多项式NB
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
    ('clf', MultinomialNB(alpha=0.1)),
])
pipeline.fit(train_data.data, train_data.target)
y_pred = pipeline.predict(test_data.data)
print(f"\n20 Newsgroups分类 - Accuracy: {accuracy_score(test_data.target, y_pred):.4f}")
print(classification_report(test_data.target, y_pred, target_names=categories))
```

---

### 5.7 超参数选择指南

| 超参数 | sklearn参数 | 说明 | 推荐范围 |
|--------|------------|------|---------|
| 平滑参数 | alpha | 拉普拉斯平滑强度 | 0.01~10，常用1.0 |
| 先验概率 | class_prior | 各类别的先验概率 | None(自动)或手动指定 |
| 是否拟合先验 | fit_prior | 是否从数据估计先验 | True(默认) |

---

### 5.8 优缺点对比

**优点：**
1. **训练和预测极快**：时间复杂度O(n*d)，适合大规模数据
2. **适合高维数据**：参数数量线性增长
3. **对小样本有效**：即使训练数据少也能给出合理结果
4. **天然支持增量学习**：partial_fit支持在线更新
5. **不需调参**：基本不需要超参数调整
6. **对缺失数据鲁棒**：可以忽略缺失特征

**缺点：**
1. **条件独立假设过强**：特征相关性强时效果差
2. **概率估计不准确**：输出的概率值往往过于极端（接近0或1）
3. **对零频率敏感**：需要拉普拉斯平滑
4. **不适合特征间强相关的情况**：如"身高"和"体重"

---

### 5.9 常见面试/考试题

**Q1: 朴素贝叶斯的"朴素"体现在哪里？**
A: 体现在条件独立假设——给定类别y的条件下，各特征之间相互独立。即P(x_1,...,x_d|y) = prod P(x_j|y)。这个假设在现实中很少成立，但实践中效果仍然不错。

**Q2: 为什么朴素贝叶斯在特征不独立时仍能工作？**
A: (1) 分类只需要argmax，不需要精确概率，独立假设只影响概率估计的准确性 (2) 特征间的依赖关系可能在不同类别中以类似方式抵消 (3) 对数空间中的连加操作对个别特征的概率偏差有一定容忍度。

**Q3: 拉普拉斯平滑解决什么问题？**
A: 解决零概率问题。当某个特征值在某类中从未出现时，似然为0，导致整个后验为0。拉普拉斯平滑给每个计数加alpha（通常为1），保证所有概率非零。

**Q4: 三种朴素贝叶斯变体（高斯、多项式、伯努利）分别适用什么场景？**
A: 高斯NB用于连续特征，多项式NB用于计数/词频特征（文本分类），伯努利NB用于二值特征（文档是否包含某词）。文本分类中，长文档用多项式NB，短文本用伯努利NB。

**Q5: 朴素贝叶斯与逻辑回归的区别？**
A: (1) NB是生成模型（建模P(x|y)和P(y)），LR是判别模型（直接建模P(y|x)） (2) NB有强独立假设，LR无此假设 (3) NB训练更快，LR通常精度更高 (4) NB适合小样本，LR适合大样本

---

## 第六章：K-Means 聚类与 PCA 主成分分析

### 6.1 K-Means 聚类

#### 6.1.1 算法概述

K-Means是最经典、最广泛使用的**无监督聚类**算法。它的目标是将 n 个样本划分为 K 个簇（cluster），使得簇内样本尽量相似，簇间样本尽量不同。

#### 6.1.2 目标函数

K-Means最小化的是**簇内平方和（Within-Cluster Sum of Squares, WCSS）**，也叫**惯性（Inertia）**：

    J(C, mu) = sum_{k=1}^{K} sum_{x_i in C_k} ||x_i - mu_k||^2

其中：
- K：簇的数量
- C_k：第 k 个簇的样本集合
- mu_k：第 k 个簇的质心（centroid），mu_k = (1/|C_k|) * sum_{x_i in C_k} x_i

这是一个NP-hard问题（在一般情况下），K-Means通过贪心迭代找到局部最优解。

#### 6.1.3 算法流程（Lloyd's Algorithm）

**输入：** 数据集 X，簇数 K，最大迭代次数 T

**初始化：** 随机选择 K 个样本作为初始质心 mu_1, ..., mu_K

**重复以下步骤直到收敛：**

**步骤1 —— 分配（Assignment Step）：**
将每个样本分配到最近的质心所在的簇：

    C_k = {x_i : ||x_i - mu_k|| <= ||x_i - mu_j||, for all j != k}

**步骤2 —— 更新（Update Step）：**
重新计算每个簇的质心：

    mu_k = (1/|C_k|) * sum_{x_i in C_k} x_i

**收敛条件：** 簇分配不再变化，或质心不再变化，或达到最大迭代次数。

**时间复杂度：** O(n * K * d * T)，其中 n 是样本数，d 是维度，T 是迭代次数。

#### 6.1.4 收敛性证明

**定理：** Lloyd's算法的每次迭代（分配+更新）都不会增加目标函数 J 的值。

**证明：**

分配步骤：固定质心 mu，对每个样本 x_i，将其分配到最近的质心所在簇，这是对 J 关于 C 的最优解。

    J_new(C, mu_old) <= J(C_old, mu_old)

更新步骤：固定簇分配 C，新的质心 mu_k = mean(C_k) 是 J 对 mu_k 的最优解（求导令梯度为零）：

    dJ/dmu_k = -2 * sum_{x_i in C_k} (x_i - mu_k) = 0
    =>  mu_k = (1/|C_k|) * sum_{x_i in C_k} x_i

    J_new(C_new, mu_new) <= J_new(C_new, mu_old)

因此每次迭代 J 单调递减，而 J >= 0，由单调有界序列收敛定理，算法收敛。

**注意：** 只保证收敛到**局部最优**，不保证全局最优。因此需要多次随机初始化，选择J最小的结果。

#### 6.1.5 K-Means++ 初始化

随机初始化可能导致收敛到很差的局部最优。K-Means++通过智能初始化改善这个问题。

**算法步骤：**
1. 从数据集中**随机选择一个样本**作为第一个质心 mu_1
2. 对每个样本 x_i，计算它到最近已选质心的距离：D(x_i) = min_{j} ||x_i - mu_j||^2
3. 以概率 **P(x_i) = D(x_i) / sum_j D(x_j)** 选择下一个质心
4. 重复步骤2-3直到选满K个质心
5. 然后运行标准K-Means

**核心思想：** 距离现有质心越远的点越可能被选为新质心，使得初始质心尽量分散。

**理论保证：** K-Means++的期望近似比为 O(log K)，即 E[J_KM++] <= 8(ln K + 2) * J_optimal

**sklearn默认使用K-Means++**（init='k-means++'）。

#### 6.1.6 选择K的方法

**1. 肘部法则（Elbow Method）**

对不同的 K 值（如 K=1,2,...,10），计算WCSS（J值）。绘制 K vs J 的曲线：
- J 随 K 增大而递减
- 曲线出现"肘部"（拐点）的位置即为合适的 K
- 肘部之后J下降变缓，说明增加K的收益减小

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K'); plt.ylabel('WCSS (Inertia)')
plt.title('Elbow Method')
plt.grid(True); plt.show()
```

**2. 轮廓系数（Silhouette Coefficient）**

对每个样本 i，定义：

    a(i) = (1/|C_k|-1) * sum_{j in C_k, j != i} ||x_i - x_j||   （同簇内平均距离）
    b(i) = min_{j != k} (1/|C_j|) * sum_{x_j in C_j} ||x_i - x_j||  （最近簇平均距离）
    s(i) = (b(i) - a(i)) / max(a(i), b(i))

- s(i) 接近 1：聚类效果好（样本与同簇近，与异簇远）
- s(i) 接近 0：样本在簇边界
- s(i) 接近 -1：可能分错簇

**整体轮廓系数：** S = (1/n) * sum_{i=1}^{n} s(i)

选择使 S 最大的 K。

```python
from sklearn.metrics import silhouette_score

silhouettes = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X)
    silhouettes.append(silhouette_score(X, labels))

plt.plot(K_range, silhouettes, 'ro-')
plt.xlabel('K'); plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.grid(True); plt.show()
```

**3. Gap Statistic**

比较实际数据的WCSS与均匀分布参考数据的WCSS：

    Gap(K) = E*[log(J_K)] - log(J_K)

其中 E* 是参考数据的期望（通过Bootstrap估计）。选择Gap(K) - Gap(K+1) + s_{K+1} 首次非负的最小K。

---

### 6.2 PCA 主成分分析

#### 6.2.1 算法概述

主成分分析（Principal Component Analysis, PCA）是最常用的**无监督降维**算法。它通过正交变换将数据投影到方差最大的方向上，从而用更少的维度捕获数据中最多的信息。

**动机：**
- 高维数据存在**维数灾难**，需要降维
- 高维数据中存在冗余（特征相关），可以压缩
- 降维后便于可视化（降到2D或3D）

#### 6.2.2 PCA完整数学推导

**问题设定：** 给定 n 个 d 维样本 X = {x_1, ..., x_n}，x_i in R^d，将其投影到 k 维子空间（k < d）。

**Step 1: 中心化**

    x_bar = (1/n) * sum_{i=1}^{n} x_i
    x_i' = x_i - x_bar

（PCA要求数据已中心化。如果还需要标准化，先除以标准差。）

**Step 2: 计算协方差矩阵**

    Sigma = (1/n) * X'^T * X'

其中 X' 是中心化后的 n x d 数据矩阵。Sigma 是 d x d 的对称半正定矩阵。

Sigma 的第 (j, k) 元素表示特征 j 和特征 k 的协方差：

    Sigma_{jk} = (1/n) * sum_{i=1}^{n} (x_{ij}' * x_{ik}')

**Step 3: 特征分解**

对协方差矩阵 Sigma 进行特征分解：

    Sigma * v_j = lambda_j * v_j

其中：
- lambda_j >= 0 是第 j 个特征值（按降序排列：lambda_1 >= lambda_2 >= ... >= lambda_d >= 0）
- v_j 是对应的特征向量（主成分方向）

**特征向量的性质：**
- 各特征向量相互正交（因为 Sigma 是对称矩阵）
- 特征向量是单位向量：||v_j|| = 1
- 特征值 lambda_j 表示数据在方向 v_j 上的方差

**Step 4: 选择前k个主成分**

选择前 k 个最大的特征值对应的特征向量：

    V_k = [v_1, v_2, ..., v_k]    (d x k 矩阵)

**方差解释率（Explained Variance Ratio）：**

    方差解释率_j = lambda_j / sum_{i=1}^{d} lambda_i

    累计方差解释率_k = sum_{j=1}^{k} lambda_j / sum_{i=1}^{d} lambda_i

通常选择累计方差解释率 >= 85%~95% 的 k 值。

**Step 5: 投影（降维）**

    z_i = V_k^T * x_i'    (k维向量)

将每个 d 维样本投影到 k 维空间。

**从降维后恢复原始数据（近似）：**

    x_i_hat = V_k * z_i + x_bar    (d维向量，但只有k维的信息)

#### 6.2.3 PCA的另一种推导——最大方差

**目标：** 找到单位向量 w，使得投影后的数据方差最大化。

投影后数据：z_i = w^T * x_i'

投影后方差：

    Var(z) = (1/n) * sum_{i=1}^{n} (w^T * x_i')^2
           = w^T * [(1/n) * X'^T * X'] * w
           = w^T * Sigma * w

约束条件：||w||^2 = 1

使用拉格朗日乘子法：

    L(w, lambda) = w^T * Sigma * w - lambda * (w^T * w - 1)

    dL/dw = 2*Sigma*w - 2*lambda*w = 0
    =>  Sigma * w = lambda * w

这就是特征值方程！最大方差方向就是最大特征值对应的特征向量。

#### 6.2.4 PCA与SVD的关系

实际上，实践中PCA通常通过**奇异值分解（SVD）**计算，而不是显式计算协方差矩阵。

对中心化数据矩阵 X' (n x d) 进行SVD分解：

    X' = U * S * V^T

其中：
- U (n x n)：左奇异向量
- S (n x d)：奇异值对角矩阵
- V (d x d)：右奇异向量（即主成分方向）

协方差矩阵与SVD的关系：

    Sigma = (1/n) * X'^T * X' = (1/n) * V * S^2 * V^T

因此特征值 lambda_j = s_j^2 / n，特征向量 v_j = V 的第 j 列。

**sklearn的PCA实现正是基于SVD**（使用随机SVD加速大规模数据）。

---

### 6.3 Python代码示例

```python
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.datasets import make_blobs, load_digits, load_iris
import matplotlib.pyplot as plt
from matplotlib import cm

# ==================== K-Means ====================

# ========== 1. K-Means基础使用 ==========
X_blobs, y_blobs = make_blobs(n_samples=500, centers=4, n_features=2,
                               random_state=42, cluster_std=1.0)

# K-Means聚类
km = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
labels = km.fit_predict(X_blobs)
print(f"K-Means惯性(Inertia): {km.inertia_:.2f}")
print(f"轮廓系数: {silhouette_score(X_blobs, labels):.4f}")
print(f"各簇样本数: {np.bincount(labels)}")
print(f"质心位置:\n{km.cluster_centers_}")

# ========== 2. K-Means手写实现 ==========
class KMeansScratch:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, init='random'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init

    def _init_centroids(self, X):
        if self.init == 'kmeans++':
            n = X.shape[0]
            centroids = [X[np.random.randint(n)]]
            for _ in range(1, self.n_clusters):
                dists = np.min([np.linalg.norm(X - c, axis=1)**2 for c in centroids], axis=0)
                probs = dists / dists.sum()
                idx = np.random.choice(n, p=probs)
                centroids.append(X[idx])
            return np.array(centroids)
        else:
            indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            return X[indices]

    def fit(self, X):
        self.centroids = self._init_centroids(X)
        for i in range(self.max_iter):
            # 分配
            dists = np.array([np.linalg.norm(X - c, axis=1) for c in self.centroids])
            self.labels = np.argmin(dists, axis=0)
            # 更新
            new_centroids = np.array([
                X[self.labels == k].mean(axis=0) if np.any(self.labels == k) else self.centroids[k]
                for k in range(self.n_clusters)
            ])
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids
        self.inertia_ = sum(np.linalg.norm(X[self.labels == k] - self.centroids[k])**2
                            for k in range(self.n_clusters))
        return self

    def predict(self, X):
        dists = np.array([np.linalg.norm(X - c, axis=1) for c in self.centroids])
        return np.argmin(dists, axis=0)

km_scratch = KMeansScratch(n_clusters=4, init='kmeans++')
km_scratch.fit(X_blobs)
print(f"\n手写K-Means++ 惯性: {km_scratch.inertia_:.2f}")
print(f"手写K-Means++ 轮廓系数: {silhouette_score(X_blobs, km_scratch.labels):.4f}")

# ========== 3. 肘部法则和轮廓系数选K ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 肘部法则
K_range = range(1, 11)
inertias = []
for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_blobs)
    inertias.append(km.inertia_)
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('K'); axes[0].set_ylabel('Inertia (WCSS)')
axes[0].set_title('Elbow Method'); axes[0].grid(True)

# 轮廓系数
K_range_s = range(2, 11)
sil_scores = []
for k in K_range_s:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_blobs)
    sil_scores.append(silhouette_score(X_blobs, labels))
axes[1].plot(K_range_s, sil_scores, 'ro-')
axes[1].set_xlabel('K'); axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Method'); axes[1].grid(True)

plt.tight_layout()
plt.savefig('kmeans_select_k.png'); plt.show()

# ========== 4. 轮廓图（Silhouette Plot） ==========
optimal_k = 4
km = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
labels = km.fit_predict(X_blobs)
sample_sil_values = silhouette_samples(X_blobs, labels)

fig, ax = plt.subplots(figsize=(8, 6))
y_lower = 10
for i in range(optimal_k):
    vals = sorted(sample_sil_values[labels == i])
    y_upper = y_lower + len(vals)
    color = cm.nipy_spectral(float(i) / optimal_k)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, facecolor=color, alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * len(vals), str(i))
    y_lower = y_upper + 10

ax.set_xlabel('Silhouette Coefficient'); ax.set_ylabel('Cluster')
ax.set_title(f'Silhouette Plot (K={optimal_k})')
ax.axvline(x=silhouette_score(X_blobs, labels), color='red', linestyle='--')
plt.tight_layout()
plt.savefig('silhouette_plot.png'); plt.show()


# ==================== PCA ====================

# ========== 5. 手写PCA ==========
class PCAScratch:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        # 方法1：协方差矩阵特征分解
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # 按特征值降序排列
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[idx]
        self.eigenvectors_ = eigenvectors[:, idx]
        self.explained_variance_ratio_ = self.eigenvalues_ / self.eigenvalues_.sum()
        return self

    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.eigenvectors_[:, :self.n_components]

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return Z @ self.eigenvectors_[:, :self.n_components].T + self.mean_

# ========== 6. 手写vs sklearn PCA ==========
digits = load_digits()
X_digits = digits.data  # 64维像素特征
y_digits = digits.target

scaler = StandardScaler()
X_digits_scaled = scaler.fit_transform(X_digits)

# 手写PCA
pca_scratch = PCAScratch(n_components=2)
Z_scratch = pca_scratch.fit_transform(X_digits_scaled)
print(f"手写PCA方差解释率: {pca_scratch.explained_variance_ratio_[:5].round(4)}")

# sklearn PCA
pca = PCA(n_components=2)
Z = pca.fit_transform(X_digits_scaled)
print(f"sklearn PCA方差解释率: {pca.explained_variance_ratio_.round(4)}")

# ========== 7. PCA降维可视化 ==========
plt.figure(figsize=(10, 8))
scatter = plt.scatter(Z[:, 0], Z[:, 1], c=y_digits, cmap='tab10', s=10, alpha=0.7)
plt.colorbar(scatter, label='Digit')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA of Digits Dataset (64D -> 2D)')
plt.grid(True, alpha=0.3)
plt.savefig('pca_digits_2d.png'); plt.show()

# ========== 8. 累计方差解释率 ==========
pca_full = PCA().fit(X_digits_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumvar)+1), cumvar, 'b-o', markersize=3)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.axhline(y=0.90, color='g', linestyle='--', label='90% threshold')
idx_95 = np.argmax(cumvar >= 0.95) + 1
plt.axvline(x=idx_95, color='r', linestyle=':', alpha=0.5)
plt.xlabel('Number of Components'); plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance Ratio')
plt.legend(); plt.grid(True)
plt.savefig('pca_cumvar.png'); plt.show()
print(f"保留95%方差需要 {idx_95} 个主成分（原始64维）")

# ========== 9. PCA + K-Means组合 ==========
# 用PCA降到2维再做K-Means
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_digits_scaled)

km_pca = KMeans(n_clusters=10, init='k-means++', n_init=10, random_state=42)
labels_pca = km_pca.fit_predict(X_2d)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_pca, cmap='tab10', s=10, alpha=0.7)
centers_2d = km_pca.cluster_centers_
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=200, edgecolors='black')
plt.colorbar(scatter, label='Cluster')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.title('K-Means Clustering on PCA-reduced Digits')
plt.grid(True, alpha=0.3)
plt.savefig('pca_kmeans_digits.png'); plt.show()

# 评估聚类效果
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
print(f"\nPCA+K-Means vs 真实标签:")
print(f"  Adjusted Rand Index: {adjusted_rand_score(y_digits, labels_pca):.4f}")
print(f"  Normalized MI: {normalized_mutual_info_score(y_digits, labels_pca):.4f}")
print(f"  Silhouette Score: {silhouette_score(X_2d, labels_pca):.4f}")
```

---

### 6.4 K-Means超参数指南

| 超参数 | sklearn参数 | 说明 | 推荐值 |
|--------|------------|------|--------|
| 簇数K | n_clusters | 聚类数量 | 肘部法则/轮廓系数选择 |
| 初始化 | init | 'k-means++'或'random' | k-means++（默认） |
| 初始化次数 | n_init | 独立运行次数取最优 | 10（默认）|
| 最大迭代 | max_iter | 每次运行的最大迭代数 | 300（默认）|
| 算法 | algorithm | 'lloyd'或'elkan' | elkan更快(低维) |

---

### 6.5 PCA超参数与使用指南

| 超参数 | sklearn参数 | 说明 | 推荐值 |
|--------|------------|------|--------|
| 主成分数 | n_components | 保留的维度数 | 累计方差>=95%的最小值 |
| 是否标准化 | StandardScaler | PCA前标准化 | 特征量纲不同时需要 |
| SVD求解器 | svd_solver | 'auto','full','arpack','randomized' | 大数据用randomized |
| 白化 | whiten | 使各主成分方差为1 | 通常False |

**PCA使用注意事项：**
1. PCA前务必标准化（如果特征量纲不同）
2. PCA是线性降维，非线性关系考虑Kernel PCA或t-SNE
3. PCA后主成分可解释性下降（是原始特征的线性组合）
4. PCA对异常值敏感（可用RobustPCA）

---

### 6.6 优缺点对比

**K-Means：**

| 优点 | 缺点 |
|------|------|
| 简单高效，易于实现 | 需要预先指定K |
| 时间复杂度线性 O(nKT) | 对初始质心敏感（K-Means++缓解） |
| 适合大规模数据 | 只能发现球形簇 |
| 结果易于理解和解释 | 对异常值敏感 |
| Mini-batch变体适合超大数据 | 只保证局部最优 |

**PCA：**

| 优点 | 缺点 |
|------|------|
| 无监督，不需要标签 | 线性降维，无法处理非线性关系 |
| 保留最大方差方向 | 主成分可解释性差 |
| 去除冗余特征和噪声 | 对异常值敏感 |
| 计算效率高（SVD） | 需要先标准化 |
| 有成熟的数学理论支撑 | 方差不等于信息量 |

---

### 6.7 常见面试/考试题

**Q1: K-Means的目标函数是什么？如何优化？**
A: 目标函数是WCSS（簇内平方和）。通过Lloyd's算法迭代优化：(1)固定质心，将样本分配到最近簇 (2)固定分配，更新质心为簇均值。每次迭代目标函数单调递减，保证收敛到局部最优。

**Q2: K-Means一定收敛吗？收敛到全局最优吗？**
A: 一定收敛（目标函数单调递减且下界为0）。但只保证收敛到局部最优，不保证全局最优。因此需要多次随机初始化（n_init参数），选择目标函数最小的结果。

**Q3: K-Means++与随机初始化的区别？**
A: K-Means++选择初始质心时，距离现有质心越远的点越可能被选为新质心，使得初始质心尽量分散。理论保证近似比O(log K)。随机初始化可能导致多个质心在同一簇内。

**Q4: 如何确定PCA需要保留多少主成分？**
A: (1)累计方差解释率>=85%~95% (2)碎石图（Scree Plot）找拐点 (3)交叉验证下游任务性能

**Q5: PCA为什么要去中心化？标准化什么情况下需要？**
A: 去中心化保证PCA找的是方差最大方向（否则主成分会偏向均值方向）。当特征量纲不同（如身高cm vs 体重kg）时必须标准化，否则大量纲特征主导方差。

**Q6: PCA与t-SNE的区别？**
A: (1)PCA是线性降维，t-SNE是非线性 (2)PCA保留全局结构，t-SNE保留局部结构 (3)PCA可逆（近似恢复），t-SNE不可逆 (4)PCA速度快适合大数据，t-SNE慢 (5)PCA适合预处理，t-SNE适合可视化

**Q7: K-Means的K如何选择？**
A: 主要方法：(1)肘部法则——找WCSS曲线拐点 (2)轮廓系数——选最大值 (3)Gap Statistic (4)业务需求直接指定。实践中肘部法则和轮廓系数最常用。

**Q8: K-Means和GMM（高斯混合模型）的区别？**
A: (1)K-Means是硬分配，GMM是软分配（概率） (2)K-Means假设球形等大小簇，GMM允许椭圆形不同大小 (3)GMM是EM算法求解，K-Means是特例 (4)GMM计算量更大但更灵活

---

*本文档持续更新中。参考资料：Stanford CS229 (Andrew Ng)、《统计学习方法》(李航)、《机器学习》(周志华)、sklearn官方文档*
