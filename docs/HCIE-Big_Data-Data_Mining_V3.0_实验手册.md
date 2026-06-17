# HCIE-Big Data-Data Mining V3.0 实验手册

> 华为认证 BigData 系列教程  
> 版本：1.0  
> 华为技术有限公司  
> 版权所有 © 华为技术有限公司 2023

---

## 目录

- [第一部分：Python 数据挖掘案例分析实验指导手册](#第一部分python-数据挖掘案例分析实验指导手册)
  - [实验一：数据预处理与特征工程](#实验一数据预处理与特征工程)
  - [实验二：回归与分类问题建模](#实验二回归与分类问题建模)
  - [实验三：聚类与降维问题建模](#实验三聚类与降维问题建模)
  - [实验四：关联分析与推荐问题建模](#实验四关联分析与推荐问题建模)
  - [实验五：模型评估与优化](#实验五模型评估与优化)
  - [数据挖掘综合实验一：分类案例](#数据挖掘综合实验一分类案例)
  - [数据挖掘综合实验二：回归案例](#数据挖掘综合实验二回归案例)
- [第二部分：PySpark MLlib 实验指导手册](#第二部分pyspark-mllib-实验指导手册)
  - [1. PySpark MLlib 统计分析](#1-pyspark-mllib-统计分析)
  - [2. PySpark MLlib 数据预处理](#2-pyspark-mllib-数据预处理)
  - [3. PySpark MLlib 房价预测](#3-pyspark-mllib-房价预测)
  - [4. PySpark MLlib 欺诈检测](#4-pyspark-mllib-欺诈检测)
  - [5. PySpark MLlib 客户流失预测](#5-pyspark-mllib-客户流失预测)
  - [6. PySpark MLlib 降维聚类分析](#6-pyspark-mllib-降维聚类分析)
  - [7. PySpark MLlib 购物篮数据分析](#7-pyspark-mllib-购物篮数据分析)
  - [8. PySpark MLlib 协同过滤](#8-pyspark-mllib-协同过滤)
- [第三部分：华为大数据平台 MRS 实验指导手册](#第三部分华为大数据平台-mrs-实验指导手册)
  - [MRS 数据湖实验](#mrs-数据湖实验)
- [第四部分：实验环境搭建手册](#第四部分实验环境搭建手册)
- [第五部分：综合大实验指导手册](#第五部分综合大实验指导手册)
  - [1. 鱼的重量回归预测](#1-鱼的重量回归预测)
  - [2. 森林覆盖类型分类预测](#2-森林覆盖类型分类预测)
  - [3. 音乐流派聚类分析](#3-音乐流派聚类分析)
  - [4. 在线零售数据关联分析](#4-在线零售数据关联分析)
  - [5. 书籍推荐协同过滤分析](#5-书籍推荐协同过滤分析)
  - [6. PySpark 广告推荐系统](#6-pyspark-广告推荐系统)

---

## 第一部分：Python 数据挖掘案例分析实验指导手册

> 本书为 HCIE-Big Data-Data Mining V3.0 认证培训教程，适用于准备参加 HCIE-Big Data-Data Mining V3.0 认证的学员或者希望具备数据挖掘技术的读者。

### 华为认证体系介绍

华为认证是华为公司基于"平台+生态"战略，围绕"云-管-端"协同的新 ICT 技术架构，打造的覆盖 ICT（Information and Communications Technology，信息通信技术）全技术领域的认证体系，包含 ICT 基础设施认证、基础软硬件认证、云平台及云服务认证三类认证。

根据 ICT 从业者的学习和进阶需求，华为认证分为工程师级别、高级工程师级别和专家级别三个认证等级。

HCIE-Big Data-Data Mining V3.0 认证定位于大数据挖掘领域，课程包含数据预处理与特征工程、回归与分类问题建模、聚类与降维问题建模、关联分析与推荐问题建模、模型评估与优化、Python 数据挖掘案例分析、PySpark MLlib、华为大数据平台 MRS、华为数据治理中心 DataArts Studio 等内容，旨在推动业界大数据行业的专家型人才培养。

---

### 实验一：数据预处理与特征工程

#### 1.1 实验任务

##### 1.1.1 实验背景

本实验主要目的是学习数据预处理和特征工程相关技术点。前四章所学的数据挖掘相关技术点串联起来，以一个完整的数据挖掘案例来实现数据挖掘的综合应用。

##### 1.1.2 实验目标

- 能够使用 Python 对数据进行初步分析，特征转换。
- 能够使用 Python 对数据常见分布进行转换。
- 能够使用 Python 构造新特征以及特征筛选。
- 具备对数据进行预处理以及特征工程处理的能力。

##### 1.1.3 实验思路

本实验包含了一个完整的数据挖掘前序实验，能更好地帮助初学者深入的理解和掌握整个数据挖掘的流程。

#### 1.2 实验操作步骤

**步骤 1 导入必要的库**

```python
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
```

**步骤 2 读取数据集**

```python
data = pd.read_csv(r'./data/chapter4/bike.csv')
# 查看数据前5行
data.head()
```

输出结果：

|    | datetime            |   season |   holiday |   workingday |   weather |   temp |   atemp |   humidity |   windspeed |   casual |   registered |   count |
|---:|:--------------------|---------:|----------:|-------------:|----------:|-------:|--------:|-----------:|------------:|---------:|-------------:|--------:|
|  0 | 2011-01-01 00:00:00 |        1 |         0 |            0 |         1 |   9.84 |   14.395 |         81 |           0 |        3 |           13 |      16 |
|  1 | 2011-01-01 01:00:00 |        1 |         0 |            0 |         1 |   9.02 |   13.635 |         80 |           0 |        8 |           32 |      40 |
|  2 | 2011-01-01 02:00:00 |        1 |         0 |            0 |         1 |   9.02 |   13.635 |         80 |           0 |        5 |           27 |      32 |
|  3 | 2011-01-01 03:00:00 |        1 |         0 |            0 |         1 |   9.84 |   14.395 |         75 |           0 |        3 |           10 |      13 |
|  4 | 2011-01-01 04:00:00 |        1 |         0 |            0 |         1 |   9.84 |   14.395 |         75 |           0 |        0 |            1 |       1 |

**步骤 3 探索性可视化分析**

使用直方图对 `registered`、`casual`、`count` 进行可视化。

```python
fig, axes = plt.subplots(2, 2)
fig.set_size_inches(12, 10)
sns.distplot(data["registered"], ax=axes[0][0])
sns.distplot(data["casual"], ax=axes[0][1])
sns.distplot(data["count"], ax=axes[1][0])
st.probplot(data["count"], dist='norm', fit=True, plot=axes[1][1])
plt.show()
```

**步骤 4 对数变换**

```python
# 对 count 取对数
data_y = np.log1p(data["count"])
sns.distplot(data_y)
st.probplot(data_y, dist='norm', fit=True, plot=plt)
```

**步骤 5 特征相关性分析**

```python
# 相关系数矩阵
data_corr = data.corr()
# 热力图
sns.heatmap(data_corr, annot=True)
plt.show()
```

**步骤 6 季节与计数的关系**

```python
# 小提琴图
sns.violinplot(data=data, x="season", y="count")
# 箱线图
sns.boxplot(data=data, x="season", y="count")
```

**步骤 7 异常值处理**

使用 3σ 准则检测异常值。

```python
std = data_y.std()
mean = data_y.mean()
print("mean: %s, std: %s" % (mean, std))

a = data_y - data_y.mean()
b = np.abs(a) <= 3 * std
data_y = data_y[b]
data = data[b]
```

输出结果：

```
mean: 5.277153392366841, std: 0.9492463530105473
```

**步骤 8 时间特征提取**

```python
bike = pd.read_csv(r'./data/chapter4/bike.csv')
# 将 datetime 转换为 datetime 格式
bike['datetime'] = pd.to_datetime(bike['datetime'])
# 提取年份
bike['year'] = bike['datetime'].map(lambda x: x.year)
# 提取月份
bike['month'] = bike['datetime'].map(lambda x: x.month)
# 提取日期
bike['day'] = bike['datetime'].map(lambda x: x.day)
# 提取小时
bike['hour'] = bike['datetime'].map(lambda x: x.hour)
```

**步骤 9 数值特征离散化**

```python
# 等宽离散化：湿度
bike['humidity_bin'] = pd.cut(bike['humidity'], 5, labels=range(5))
# 等频离散化：温度
bike['temp_bin'] = pd.qcut(bike['temp'], 5, duplicates='drop', labels=range(5))
# 查看结果
bike[['humidity', 'humidity_bin', 'temp', 'temp_bin']].head()
```

输出结果：

|    |   humidity |   humidity_bin |   temp |   temp_bin |
|---:|-----------:|---------------:|-------:|-----------:|
|  0 |         81 |              3 |   9.84 |          0 |
|  1 |         80 |              3 |   9.02 |          0 |
|  2 |         80 |              3 |   9.02 |          0 |
|  3 |         75 |              2 |   9.84 |          0 |
|  4 |         75 |              2 |   9.84 |          0 |

**步骤 10 构造新特征**

```python
# 提取温差特征
bike['diff_temp'] = bike['atemp'] - bike['temp']
# 提取体感温度与湿度之比特征
bike['temp_hum'] = bike['atemp'] / bike['humidity']
# 温度与风度之比特征
bike['temp_windspeed'] = bike['temp'] / bike['windspeed']
# 是否温差舒适特征（温差绝对值在 0.5 以内）
bike['is_comfort'] = np.where(np.abs(bike['diff_temp']) < 0.5, 1, 0)
```

**步骤 11 数值型特征转换为类别型特征**

```python
# 将湿度离散值作为类别型
df1 = pd.get_dummies(bike['humidity_bin'], prefix='humidity_bin')
# 将季节作为类别型
df2 = pd.get_dummies(bike['season'], prefix='season')
# 拼接数据
data_new = pd.concat([bike, df1, df2], axis=1)
```

**步骤 12 特征筛选——方差选择法**

```python
from sklearn.feature_selection import VarianceThreshold
x = data_new[data_new.columns.drop('count')]
y = data_new['count']
# 方差阈值设置为 0.1
selector = VarianceThreshold(threshold=0.1)
selector.fit(x)
# 保留特征
x.columns[selector.get_support()]
# 被删除特征
x.columns[~selector.get_support()]
```

**步骤 13 特征筛选——相关系数法**

```python
from sklearn.feature_selection import SelectKBest, f_regression
x_new = SelectKBest(f_regression, k=5).fit_transform(x, y)
```

#### 1.3 实验小结

本实验通过数据探索，数据清洗，特征工程为建模准备好数据，一般来说，建模之前的特征工程的好坏往往比建模过程更重要。正所谓"数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已"。

---

### 实验二：回归与分类问题建模

#### 2.1 实验任务

##### 2.1.1 实验背景

本实验是承接前序实验，利用前序实验处理好的数据选择合适的回归或分类算法进行建模。

##### 2.1.2 实验目标

- 具备使用 Python 对数据进行回归建模的能力。
- 具备使用 Python 对数据进行分类建模的能力。
- 掌握线性回归算法、决策树算法、KNN 算法、贝叶斯算法。

#### 2.2 实验操作步骤

**步骤 1 导入必要的库**

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
```

**步骤 2 读取数据集**

```python
data = pd.read_csv(r'./data/chapter4/bike.csv')
data['datetime'] = pd.to_datetime(data['datetime'])
data['year'] = data['datetime'].map(lambda x: x.year)
data['month'] = data['datetime'].map(lambda x: x.month)
data['day'] = data['datetime'].map(lambda x: x.day)
data['hour'] = data['datetime'].map(lambda x: x.hour)
data.drop('datetime', axis=1, inplace=True)
```

**步骤 3 线性回归建模**

```python
x = data[data.columns.drop('count')]
y = data['count']
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
# 线性回归
lr = LinearRegression()
lr.fit(x_train, y_train)
predictions = lr.predict(x_test)
print('MSE: ', mean_squared_error(y_test, predictions))
```

**步骤 4 KNN 回归建模**

```python
knn = KNeighborsRegressor()
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
print('MSE: ', mean_squared_error(y_test, predictions))
```

**步骤 5 决策树回归建模**

```python
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
predictions = dt.predict(x_test)
print('MSE: ', mean_squared_error(y_test, predictions))
```

**步骤 6 分类建模——高斯朴素贝叶斯**

```python
# 提取类别型特征
x = data[['season', 'weather']]
y = data['count']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
nb = GaussianNB()
nb.fit(x_train, y_train)
predictions = nb.predict(x_test)
```

#### 2.3 实验小结

本实验演示了基于回归和分类算法的综合案例，线性回归算法是最基础的回归算法，它通过拟合自变量和因变量之间的线性关系，从而实现对数据的预测。决策树算法是一种简单直观的分类和回归算法，它通过递归地对数据集进行划分，从而构建一棵树状模型。KNN 算法是一种基于实例的学习算法，它通过计算新样本与训练样本之间的距离，从而找到最近的 K 个邻居。贝叶斯算法是一种基于概率统计的分类算法，它通过计算后验概率，从而实现对数据的分类。

---

### 实验三：聚类与降维问题建模

#### 3.1 实验任务

##### 3.1.1 实验背景

本实验是承接前序实验，利用前序实验处理好的数据选择合适的聚类或降维算法进行建模。

##### 3.1.2 实验目标

- 具备使用 Python 对数据进行聚类建模的能力。
- 具备使用 Python 对数据进行降维的能力。
- 掌握 KMeans 算法、PCA 算法、LDA 算法。

#### 3.2 实验操作步骤

**步骤 1 导入必要的库**

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
```

**步骤 2 读取数据集**

```python
data = pd.read_csv(r'./data/chapter4/iris.csv')
# 提取特征和标签
x = data[data.columns[:-1]]
y = data[data.columns[-1]]
```

**步骤 3 KMeans 聚类**

```python
# 数据标准化
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
# KMeans 聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(x_scaled)
# 轮廓系数
score = silhouette_score(x_scaled, kmeans.labels_)
print('Silhouette Score: ', score)
```

**步骤 4 PCA 降维**

```python
# PCA 降维
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)
# 可视化
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=kmeans.labels_)
plt.show()
```

**步骤 5 LDA 降维**

```python
# LDA 降维
lda = LatentDirichletAllocation(n_components=2)
x_lda = lda.fit_transform(x)
```

#### 3.3 实验小结

本实验演示了基于聚类和降维算法的综合案例，KMeans 算法是最基础的聚类算法，它通过迭代地将数据点分配到最近的聚类中心，从而实现对数据的聚类。PCA 算法是最常用的降维算法，它通过线性变换将数据投影到低维空间，从而实现对数据的降维。LDA 算法是一种基于概率模型的降维算法，它通过寻找最优的投影方向，从而实现对数据的降维。

---

### 实验四：关联分析与推荐问题建模

#### 4.1 实验任务

##### 4.1.1 实验背景

本实验是承接前序实验，利用前序实验处理好的数据选择合适的关联分析或推荐算法进行建模。

##### 4.1.2 实验目标

- 具备使用 Python 对数据进行关联分析建模的能力。
- 具备使用 Python 对数据进行推荐建模的能力。
- 掌握 Apriori 算法、FPGrowth 算法、协同过滤算法。

#### 4.2 实验操作步骤

**步骤 1 导入必要的库**

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.metrics.pairwise import cosine_similarity
```

**步骤 2 读取数据集**

```python
data = pd.read_csv(r'./data/chapter4/groceries.csv')
# 查看数据
data.head()
```

**步骤 3 Apriori 算法**

```python
# 频繁项集挖掘
frequent_itemsets = apriori(data, min_support=0.05, use_colnames=True)
# 关联规则
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
```

**步骤 4 FPGrowth 算法**

```python
frequent_itemsets = fpgrowth(data, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
```

**步骤 5 协同过滤推荐**

```python
# 用户-物品评分矩阵
user_item_matrix = data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
# 计算用户相似度
user_similarity = cosine_similarity(user_item_matrix)
```

#### 4.3 实验小结

本实验演示了基于关联分析和推荐算法的综合案例，Apriori 算法是最基础的关联分析算法，它通过迭代地挖掘频繁项集，从而发现数据中的关联规则。FPGrowth 算法是一种高效的关联分析算法，它通过构建频繁模式树，从而避免了 Apriori 算法中的多次扫描数据集。协同过滤算法是一种基于用户行为的推荐算法，它通过计算用户或物品之间的相似度，从而实现对用户的推荐。

---

### 实验五：模型评估与优化

#### 5.1 实验任务

##### 5.1.1 实验背景

本实验是承接前序实验，对前序实验建立的模型进行评估和优化。

##### 5.1.2 实验目标

- 具备使用 Python 对模型进行评估的能力。
- 具备使用 Python 对模型进行优化的能力。
- 掌握交叉验证、网格搜索、模型融合等技术。

#### 5.2 实验操作步骤

**步骤 1 导入必要的库**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
```

**步骤 2 读取数据集**

```python
data = pd.read_csv(r'./data/chapter4/iris.csv')
x = data[data.columns[:-1]]
y = data[data.columns[-1]]
```

**步骤 3 交叉验证**

```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
scores = cross_val_score(dt, x, y, cv=5)
print('Cross-validation scores: ', scores)
print('Mean score: ', scores.mean())
```

**步骤 4 网格搜索**

```python
param_grid = {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(dt, param_grid, cv=5)
grid_search.fit(x, y)
print('Best parameters: ', grid_search.best_params_)
print('Best score: ', grid_search.best_score_)
```

**步骤 5 模型融合——Voting**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

rf = RandomForestClassifier()
lr = LogisticRegression()
svc = SVC()

voting = VotingClassifier(estimators=[('rf', rf), ('lr', lr), ('svc', svc)], voting='hard')
voting.fit(x, y)
```

#### 5.3 实验小结

本实验演示了模型评估与优化的综合案例，交叉验证是一种常用的模型评估方法，它通过将数据集划分为多个子集，从而对模型进行多次评估。网格搜索是一种常用的模型优化方法，它通过遍历给定的参数组合，从而找到最优的参数。模型融合是一种常用的模型优化方法，它通过组合多个模型的预测结果，从而提高模型的性能。

---

### 数据挖掘综合实验一：分类案例

#### 1.1 实验任务

##### 1.1.1 实验背景

本实验主要目的是将前六章所学的数据挖掘相关技术点串联起来，以一个完整的分类案例来实现数据挖掘的综合应用。

##### 1.1.2 实验目标

能够使用 Python 对数据进行初步分析，特征转换，并可以灵活应用集成学习算法。

##### 1.1.3 实验思路

本实验包含了一个完整的数据挖掘综合分类实验，能更好地帮助初学者深入的理解和掌握整个数据挖掘的流程。

#### 1.2 实验操作步骤

**步骤 1 导入必要的库**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import datetime
import warnings
warnings.filterwarnings("ignore")
```

**步骤 2 读取数据集**

```python
df = pd.read_csv(r'./data/classification/adult.csv')
df.head()
```

**步骤 3 数据预处理**

```python
# 对类别型特征进行编码
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])
```

**步骤 4 划分数据集**

```python
x = df.drop('income', axis=1)
y = df['income']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
```

**步骤 5 定义模型评估函数**

```python
def fit_ml_algo(algo, x_train, y_train, x_test, cv):
    model = algo.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    acc = round(accuracy_score(y_test, test_pred) * 100, 2)
    acc_cv = round(cross_val_score(algo, x_train, y_train, cv=cv).mean() * 100, 2)
    return train_pred, test_pred, acc, acc_cv
```

**步骤 6 KNN 分类**

```python
from sklearn.neighbors import KNeighborsClassifier
start_time = time.time()
train_pred_knn, test_pred_knn, acc_knn, acc_cv_knn = fit_ml_algo(
    KNeighborsClassifier(n_neighbors=5), x_train, y_train, x_test, 10)
knn_time = (time.time() - start_time)
print("Accuracy: %s" % acc_knn)
print("Accuracy CV 10-Fold: %s" % acc_cv_knn)
print("Running Time: %s" % datetime.timedelta(seconds=knn_time))
```

**步骤 7 逻辑回归**

```python
from sklearn.linear_model import LogisticRegression
start_time = time.time()
train_pred_log, test_pred_log, acc_log, acc_cv_log = fit_ml_algo(
    LogisticRegression(n_jobs=-1), x_train, y_train, x_test, 10)
log_time = (time.time() - start_time)
print("Accuracy: %s" % acc_log)
print("Accuracy CV 10-Fold: %s" % acc_cv_log)
```

**步骤 8 决策树**

```python
from sklearn.tree import DecisionTreeClassifier
start_time = time.time()
train_pred_dt, test_pred_dt, acc_dt, acc_cv_dt = fit_ml_algo(
    DecisionTreeClassifier(), x_train, y_train, x_test, 10)
dt_time = (time.time() - start_time)
print("Accuracy: %s" % acc_dt)
print("Accuracy CV 10-Fold: %s" % acc_cv_dt)
```

**步骤 9 随机森林**

```python
from sklearn.ensemble import RandomForestClassifier
start_time = time.time()
train_pred_rf, test_pred_rf, acc_rf, acc_cv_rf = fit_ml_algo(
    RandomForestClassifier(n_estimators=100), x_train, y_train, x_test, 10)
rf_time = (time.time() - start_time)
print("Accuracy: %s" % acc_rf)
print("Accuracy CV 10-Fold: %s" % acc_cv_rf)
```

输出结果：

```
Accuracy: 85.04
Accuracy CV 10-Fold: 85.17
Running Time: 0:00:00.892017
```

**步骤 10 GBDT**

```python
from sklearn.ensemble import GradientBoostingClassifier
start_time = time.time()
train_pred_gbt, test_pred_gbt, acc_gbt, acc_cv_gbt = fit_ml_algo(
    GradientBoostingClassifier(), x_train, y_train, x_test, 10)
gbt_time = (time.time() - start_time)
print("Accuracy: %s" % acc_gbt)
print("Accuracy CV 10-Fold: %s" % acc_cv_gbt)
```

输出结果：准确率为 86.35%，耗时 5.698101 秒。

**步骤 11 XGBoost**

```python
import xgboost as xgb
start_time = time.time()
train_pred_xgb, test_pred_xgb, acc_xgb, acc_cv_xgb = fit_ml_algo(
    xgb.XGBClassifier(n_jobs=-1), x_train, y_train, x_test, 10)
```

**步骤 12 LightGBM**

```python
import lightgbm as lgb
start_time = time.time()
train_pred_lgb, test_pred_lgb, acc_lgb, acc_cv_lgb = fit_ml_algo(
    lgb.LGBMClassifier(), x_train, y_train, x_test, 10)
lgb_time = (time.time() - start_time)
```

输出结果：

```
Accuracy: 87.03
Accuracy CV 10-Fold: 86.92
Running Time: 0:00:01.762946
```

**步骤 13 效果对比**

```python
models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 'Random Forest',
              'Decision Tree', 'Gradient Boosting Trees', 'XGBoost', 'LGBM'],
    'Score': [acc_cv_knn, acc_cv_log, acc_cv_rf, acc_cv_dt, acc_cv_gbt, acc_cv_xgb, acc_cv_lgb]
})
models.sort_values(by='Score', ascending=False)
```

**ROC 曲线对比**

```python
plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(10, 10))
models = ['KNN', 'Logistic Regression', 'Random Forest', 'Decision Tree',
          'Gradient Boosting Trees', 'XGBoost', 'LGBM']
probs = [probs_knn, probs_log, probs_rf, probs_dt, probs_gbt, probs_xgb, probs_lgb]
colors = ['blue', 'green', 'red', 'magenta', 'yellow', 'turquoise', 'cyan']

plt.title('Receiver Operating Characteristic')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
```

> 结论：若只考虑准确率一个标准，选择 GBDT 或 Light GBM 为该项目的最优模型。
>
> 思考：请读者尝试使用 Stacking 或 Voting 算法对上述算法进行融合优化。

#### 1.3 实验小结

本实验演示了基于 Bagging 和 Boosting 算法的综合分类案例，Bagging 算法是基础的集成算法，它通过降低基分类器的方差，改善了泛化误差。Boosting 算法加入了权值采样和权重强分类的概念，两者都是通过重采样和弱分类器融合实现的方法。不过 Boosting 会倾向于一直分错的样本，如果样本中有离群的错误样本，Boosting 就会出现效果不好的情况。

---

### 数据挖掘综合实验二：回归案例

#### 2.1 实验任务

##### 2.1.1 实验背景

本实验主要目的是将前六章所学的数据挖掘相关技术点串联起来，以一个完整的回归案例来实现数据挖掘的综合应用。

##### 2.1.2 实验目标

能够使用 Python 对数据进行初步分析，特征转换，并可以灵活应用 Stacking 集成算法。

#### 2.2 实验操作步骤

**步骤 1 导入必要的库**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.simplefilter("ignore")
%matplotlib inline
```

**步骤 2 读取训练数据集和测试数据集**

```python
df_train = pd.read_csv(r'./data/regression/train.csv')
df_test = pd.read_csv(r'./data/regression/test.csv')
pd.set_option('max_columns', 10000)
pd.set_option('max_rows', 500)
df_train['y'].describe()
```

输出结果：

```
count      1460.000000
mean     180921.195890
std       79442.502883
min       34900.000000
25%      129975.000000
50%      163000.000000
75%      214000.000000
max      755000.000000
Name: y, dtype: float64
```

**步骤 3 探索性可视化分析**

```python
# 目标变量 y 分布直方图
sns.distplot(df_train['y'])
# 查看目标变量 y 的偏度和峰度
print("Skewness: %f" % df_train['y'].skew())
print("Kurtosis: %f" % df_train['y'].kurt())
```

输出结果：

```
Skewness: 1.882876
Kurtosis: 6.536282
```

**步骤 4 异常值处理**

```python
# 删除异常值
df_train.drop(df_train[(df_train['x18']<5) & (df_train['y']>200000)].index, inplace=True)
df_train.drop(df_train[(df_train['x20']<1900) & (df_train['y']>400000)].index, inplace=True)
df_train.drop(df_train[(df_train['x20']>1980) & (df_train['y']>700000)].index, inplace=True)
df_train.drop(df_train[(df_train['x39']>6000) & (df_train['y']<200000)].index, inplace=True)
df_train.drop(df_train[(df_train['x47']>4000) & (df_train['y']<200000)].index, inplace=True)
# 重置索引
df_train.reset_index(drop=True, inplace=True)
```

**步骤 5 目标变量 y 处理**

对目标变量 y 进行 log 变换，以满足正态分布的要求。

```python
df_train['y'] = np.log(df_train['y'])
```

**步骤 6 训练数据和测试数据合并**

```python
all_data = pd.concat([df_train, df_test], axis=0)
all_data.reset_index(drop=True, inplace=True)
```

**步骤 7 缺失值填充**

```python
# 类别特征的缺失值用一种新类别插补
str_cols = ["x73", "x75", "x7", "x74", "x58", "x59", "x61", "x64", "x65",
            "x31", "x32", "x33", "x34", "x36", "x26", "x2"]
for col in str_cols:
    all_data[col].fillna("None", inplace=True)

# 数值特征以 0 值插补
num_cols = ["x38", "x39", "x37", "x35", "x48", "x49",
            "x27", "x62", "x63", "x60"]
for col in num_cols:
    all_data[col].fillna(0, inplace=True)

# 剩余特征采用众数插补
other_cols = ["x3", "x43", "x54", "x24", "x25", "x79"]
for col in other_cols:
    all_data[col].fillna(all_data[col].mode()[0], inplace=True)
```

**步骤 8 顺序特征编码**

```python
from sklearn.preprocessing import LabelEncoder

def custom_coding(x):
    if x == 'Ex': r = 0
    elif x == 'Gd': r = 1
    elif x == 'TA': r = 2
    elif x == 'Fa': r = 3
    elif x == 'None': r = 4
    else: r = 5
    return r

cols = ['x32', 'x31', 'x29', 'x28', 'x58', 'x65', 'x64', 'x41', 'x54', 'x73']
for col in cols:
    all_data[col] = all_data[col].apply(custom_coding)
```

**步骤 9 字符特征独热编码**

```python
all_data = pd.get_dummies(all_data)
```

**步骤 10 特征降维**

```python
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=0.001)
lasso_model.fit(x_train, y_train)
FI_lasso = pd.DataFrame({"Feature Importance": lasso_model.coef_},
                        index=all_data.drop(["y"], axis=1).columns)
FI_lasso.sort_values("Feature Importance", ascending=False).round(5)
```

**步骤 11 Stacking 算法**

```python
from mlxtend.regressor import StackingCVRegressor

stack = StackingCVRegressor(
    regressors=(ADA, RF, GBR, ET),
    meta_regressor=ridge, cv=5,
    use_features_in_secondary=False,
    store_train_meta_features=True,
    random_state=42, n_jobs=-1)
score = rmsle_cv(stack)
print("StackingCVRegressor score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
```

输出结果：

```
StackingCVRegressor score: 0.1409 (0.0079)
```

**步骤 12 Voting 算法**

```python
from sklearn.ensemble import VotingRegressor

ereg = VotingRegressor([('stack', stack), ('xgb', xgb), ('lgbm', lgbm)],
                       weights=[14, 3, 3])
ereg.fit(x_train, y_train)
ereg_train_pred = ereg.predict(x_train)
ereg_pred = np.expm1(ereg.predict(x_test))
```

#### 2.3 实验小结

本实验演示了基于 Stacking 算法的综合回归案例，通过数据可视化分析，特征工程，为建模准备好数据，Stacking 是 Kaggle 比赛中最常见的集成学习模型，应用广泛。如果模型效果不佳，可以尝试以下建议：

1. 观察每个子模型，移除那些明显逊色的模型。
2. 继续丰富子模型的种类，比如切换不同的参数的类似模型。
3. 如果数据集非常小，Stacking 不一定有效。
4. 重新检查基学习器的训练，可以尝试用 randomized search 来包括一些新的"准而不同"的模型。
5. 模型事实上提升了，但是准确率（accuracy）不是很好，尤其是当数据严重不平衡的时候，这时可以尝试 ROC 指标。

---

## 第二部分：PySpark MLlib 实验指导手册

> 本实验指导书共包含 8 章实验。

### 1. PySpark MLlib 统计分析

#### 1.1 实验介绍

本实验通过 PySpark 进行 PySpark MLlib 的统计分析实验。

#### 1.2 实验目的

- 掌握 PySpark MLlib 中基本的统计分析方法。
- 掌握 PySpark Shell 中操作 MLlib 的编程技巧。

#### 1.3 实验过程和步骤

**1.3.1 初始化 PySpark 环境**

```python
from pyspark.context import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext('local', 'test')
spark = SparkSession(sc)
```

**1.3.2 摘要统计**

```python
import numpy as np
from pyspark.mllib.stat import Statistics

observations = sc.parallelize([
    (1.0, 10.0, 100.0),
    (2.0, 20.0, 200.0),
    (3.0, 30.0, 300.0)
])

summary = Statistics.colStats(observations)
print(summary.mean())
# 输出: [2.0, 20.0, 200.0]
```

**1.3.3 相关性分析**

```python
from pyspark.mllib.stat import Statistics

seriesX = sc.parallelize([1, 2, 3, 3, 5])
seriesY = sc.parallelize([11, 22, 33, 33, 555])

correlation = Statistics.corr(seriesX, seriesY, "pearson")

data = sc.parallelize([
    (1.0, 10.0, 100.0),
    (2.0, 20.0, 200.0),
    (5.0, 33.0, 366.0)
])
correlMatrix = Statistics.corr(data, method="pearson")
```

**1.3.4 分层抽样**

```python
fractions = {"female": 0.6, "male": 0.4}
rdd = sc.parallelize(fractions.keys()).cartesian(sc.parallelize(range(0, 1000)))
sample = dict(rdd.sampleByKey(False, fractions, 2).groupByKey().collect())
print(len(sample["female"]))  # 596
print(len(sample["male"]))    # 403
```

**1.3.5 随机数生成**

```python
from pyspark.mllib.random import RandomRDDs

u = RandomRDDs.normalRDD(sc, 10000000, 10)
v = u.map(lambda x: 1.0 + 2.0 * x)
v.collect()
```

**1.3.6 核密度估计**

```python
from pyspark.mllib.stat import KernelDensity

sample = sc.parallelize([0.0, 1.0])
kd = KernelDensity()
kd.setSample(sample)
kd.estimate([0.0, 1.0])
# 输出: array([0.3204565, 0.3204565])
```

#### 1.4 实验小结

本实验通过演示了 PySpark MLlib 中统计分析的基础知识点，为基于大数据集群的统计分析提供了理论基础。

---

### 2. PySpark MLlib 数据预处理

#### 2.1 实验介绍

本实验进行数据集的预处理操作，从而为数据分析提供前期的数据准备。

#### 2.2 实验目的

- 掌握 PySpark Mllib 中常用的数据预处理操作。
- 具备基于 PySpark 的大数据处理能力。

#### 2.3 实验过程和步骤

**2.3.1 初始化 PySpark 环境**

```python
from pyspark.context import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext('local', 'test')
spark = SparkSession(sc)
```

**2.3.2 特征提取**

TF-IDF：

```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer

training = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0)
], ["id", "text", "label"])

tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol='features')
lr = LogisticRegression(maxIter=10, regParam=0.001)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
model = pipeline.fit(training)
```

Word2Vec：

```python
from pyspark.mllib.feature import Word2Vec, Word2VecModel

sentence = "a b " * 100 + "a c " * 10
localDoc = [sentence, sentence]
doc = sc.parallelize(localDoc).map(lambda line: line.split(" "))
model = Word2Vec().setVectorSize(10).setSeed(42).fit(doc)
syms = model.findSynonyms("a", 2)
print([s[0] for s in syms])  # ['b', 'c']
```

CountVectorizer：

```python
from pyspark.ml.feature import CountVectorizer

df = spark.createDataFrame(
    [(0, ["a", "b", "c"]), (1, ["a", "b", "b", "c", "a"])],
    ["label", "raw"])
cv = CountVectorizer(inputCol="raw", outputCol="vectors")
model = cv.fit(df)
model.transform(df).show(truncate=False)
```

**2.3.3 特征转换**

RegexTokenizer：

```python
from pyspark.ml.feature import RegexTokenizer

df = spark.createDataFrame([("A B c",)], ["text"])
reTokenizer = RegexTokenizer(inputCol="text", outputCol="words")
reTokenizer.transform(df).head()
# Row(text='A B c', words=['a', 'b', 'c'])
```

StopWordsRemover：

```python
from pyspark.ml.feature import StopWordsRemover

remover = StopWordsRemover(inputCol="raw", outputCol="filtered")
dataSet = spark.createDataFrame([
    (0, ["I", "saw", "the", "red", "balloon"]),
    (1, ["Mary", "had", "a", "little", "lamb"])
], ["id", "raw"])
remover.transform(dataSet).show()
```

PCA：

```python
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors

data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
        (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
        (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
df = spark.createDataFrame(data, ["features"])
pca = PCA(k=3, inputCol="features", outputCol="pca_features")
model = pca.fit(df)
print(model.explainedVariance)
# DenseVector([0.7944, 0.2056])
```

StringIndexer：

```python
from pyspark.ml.feature import StringIndexer

data = [([0, "a"]), ([1, "b"]), ([2, "c"]),
        ([3, "a"]), ([4, "a"]), ([5, "c"])]
df = spark.createDataFrame(data, ["id", "category"])
indexer = StringIndexer(inputCol="category", outputCol="categoryIndex").fit(df)
indexed = indexer.transform(df)
indexed.show()
```

OneHotEncoder：

```python
from pyspark.ml.feature import OneHotEncoder, StringIndexer

data = [([0, "a"]), ([1, "b"]), ([2, "c"]),
        ([3, "a"]), ([4, "a"]), ([5, "c"])]
df = spark.createDataFrame(data, ["id", "category"])
indexer = StringIndexer(inputCol="category", outputCol="categoryIndex").fit(df)
indexed = indexer.transform(df)
encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
encoded = encoder.transform(indexed)
encoded.show()
```

Normalizer：

```python
from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import Vectors

data = [(0, Vectors.dense([1.0, 0.5, -1.0])),
        (1, Vectors.dense([2.0, 1.0, 1.0])),
        (2, Vectors.dense([4.0, 10.0, 2.0]))]
df = spark.createDataFrame(data, ["id", "features"])
normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
l1NormData = normalizer.transform(df)
l1NormData.show()
```

#### 2.4 实验小结

本实验主要介绍了 PySpark MLlib 包所提供的数据处理的工具，从文本数据处理工具到数据处理中常用到的二值化、降维、多项式拓展、数据离散化操作、索引变化、onehot 编码、范数 p-norm 规范化等工具为数据处理中用到的相关方法提供技术基础。

---

### 3. PySpark MLlib 房价预测

#### 3.1 实验介绍

本实验通过 PySpark 进行 PySpark MLlib 的线性回归实验。

#### 3.2 实验目的

了解线性回归算法在实际业务中的应用。

#### 3.3 实验过程和步骤

**3.3.1 初始化 PySpark 环境**

```python
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import Row

sc = SparkContext('local', 'test')
spark = SparkSession(sc)
```

**3.3.2 数据加载**

```python
houses_data = spark.read.format('csv').option('header', 'true').option('inferSchema', 'true').load("data/houses_data.csv")
```

**3.3.3 数据预处理**

```python
rdd = sc.textFile('data/houses_data.csv')
rdd = rdd.map(lambda line: line.split(","))
header = rdd.first()
rdd = rdd.filter(lambda line: line != header)
df = rdd.map(lambda line: Row(street=line[0], city=line[1], zip=line[2],
                               beds=line[4], baths=line[5], sqft=line[6], price=line[9])).toDF()
```

**3.3.4 建立模型**

```python
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.feature import StandardScaler

# 数据标准化
trainingData, testingData = transformedData.randomSplit([.8, .2], seed=1234)
linearModel = LinearRegressionWithSGD.train(trainingData, 1000, .2)
print(linearModel.weights)
# DenseVector([14210.0466, 6661.548, 67422.9673])
```

#### 3.4 实验小结

本章实验采用线性回归算法实现价格预测。其中，迭代次数为 1000，步长为 0.2 这几个参数大家可以自行调整，尝试不同的参数对预测结果的影响。

---

### 4. PySpark MLlib 欺诈检测

#### 4.1 实验介绍

通过实验了解 PySpark MLlib 在信用卡反欺诈案例中的使用。

#### 4.2 实验目的

掌握 PySpark MLlib 建模的基础语法，同时了解数据降维和 GBDT 算法在实际业务中的应用。

#### 4.3 实验过程和步骤

**4.3.1 初始化环境**

```python
from pyspark.context import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext('local', 'test')
spark = SparkSession(sc)
```

**4.3.2 数据加载**

```python
!unzip ./data/creditcard.zip -d ./data
data_O = spark.read.load('data/creditcard.csv', format='csv', header='true', inferSchema='true')
```

**4.3.3 数据预处理**

```python
import pandas as pd
data = data_O.toPandas()
data = data.sample(frac=1)
fraud_df = data.loc[data['Class'] == 1]
non_fraud_df = data.loc[data['Class'] == 0][:492]
normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
new_df = normal_distributed_df.sample(frac=1, random_state=42)
```

**4.3.4 建立模型**

```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import DenseVector

gbt = GBTClassifier(featuresCol="features", maxIter=100, maxDepth=8)
model = gbt.fit(train_data)
predictions = model.transform(test_data)

evaluator = BinaryClassificationEvaluator()
print(evaluator.evaluate(predictions))
# 0.9451945412311265
```

#### 4.4 实验小结

本实验主要介绍了 PCA 降维和数据不均衡处理方法，从而提高了交易欺诈模型的准确率。

---

### 5. PySpark MLlib 客户流失预测

#### 5.1 实验介绍

通过实验了解 PySpark MLlib 在客户流失预测案例中的使用。

#### 5.2 实验目的

掌握 PySpark MLlib 建模的基础语法，同时了解数据预处理和特征工程在实际业务中的应用。

#### 5.3 实验过程和步骤

**5.3.1 初始化 PySpark 环境**

```python
from pyspark.context import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext('local', 'test')
spark = SparkSession(sc)
```

**5.3.2 数据理解**

```python
!unzip ./data/sparkify-event-data.zip -d ./data
event_data = "data/sparkify-event-data.json"
df = spark.read.json(event_data)
df.printSchema()
```

**5.3.3 特征工程与探索性数据分析**

```python
from pyspark.sql import Window
from pyspark.sql.functions import udf, col, concat, count, lit, avg, lag, first, last, when
from pyspark.sql.functions import min as Fmin, max as Fmax, sum as Fsum, round as Fround

windowsession = Window.partitionBy('sessionId').orderBy('ts')
df = df.withColumn("lagged_page", lag(df.page).over(windowsession))
```

**5.3.4 模型评估**

```python
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier

numeric_columns = ['nsongs_perh', 'ntbup_perh', 'ntbdown_perh', 'nfriend_perh',
                   'nadvert_perh', 'nerror_perh', 'upgradedowngrade', 'songratio',
                   'positiveratio', 'negativeratio', 'updownratio', 'trend_songs',
                   'avgsessionitems', 'avgsongs']
numeric_assembler = VectorAssembler(inputCols=numeric_columns, outputCol="numericvectorized")
scaler = StandardScaler(inputCol="numericvectorized", outputCol="numericscaled",
                        withStd=True, withMean=True)
```

**5.3.5 GBDT 分类器**

```python
gb1 = GBTClassifier(maxDepth=5, maxIter=20)
pipeline_gb1 = Pipeline(stages=[numeric_assembler, scaler, total_assembler, gb1])
model_gb1 = pipeline_gb1.fit(train_plus_val)
```

#### 5.4 实验小结

本实验主要介绍了客户预测模型开发过程，通过特征工程和探索性数据分析，从而提高发现确定可能取消帐户并离开服务的用户的准确率。

---

### 6. PySpark MLlib 降维聚类分析

#### 6.1 实验介绍

该数据集为鸢尾花卉数据集（Iris），包含 150 个数据样本，分为 3 类，每类 50 个数据，每个数据包含 4 个属性。

#### 6.2 实验目的

了解 PCA 算法和 KMeans 算法在实际业务中的应用。

#### 6.3 实验过程和步骤

**6.3.1 PCA 降维**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.context import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext('local', 'test')
spark = SparkSession(sc)

iris = load_iris()
X = iris['data']
y = iris['target']
data = pd.DataFrame(X, columns=iris.feature_names)
dataset = spark.createDataFrame(data, iris.feature_names)

assembler = VectorAssembler(inputCols=iris.feature_names, outputCol='features')
df = assembler.transform(dataset).select('features')

scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures',
                        withMean=True, withStd=True).fit(df)
df_scaled = scaler.transform(df)

n_components = 3
pca = PCA(k=n_components, inputCol='scaledFeatures', outputCol='pcaFeatures').fit(df_scaled)
df_pca = pca.transform(df_scaled)
print('Explained Variance Ratio', pca.explainedVariance.toArray())
# [0.72962445 0.22850762 0.03668922]
```

**6.3.2 KMeans 聚类**

```python
from pyspark.ml.clustering import KMeans

# 手肘法确定 k 值
cost = np.zeros(20)
for k in range(2, 20):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = kmeans.fit(df_kmeans.sample(False, 0.1, seed=42))
    cost[k] = model.computeCost(df_kmeans)

# k=12 时增益很小
k = 12
kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(df_kmeans)
centers = model.clusterCenters()
```

#### 6.4 实验小结

本章实验采用 PCA 算法实现降维并用 KMeans 算法实现聚类。其中 k 为 12，是通过手肘法作图观察出来的。

#### 6.5 思考题

请简述手肘法的核心思想？

---

### 7. PySpark MLlib 购物篮数据分析

#### 7.1 实验介绍

在商品推荐中，关联规则推荐使用的比较频繁，它是通过概率来预测有易于理解且准确度比较高的优点。采用 FPGrowth 算法，根据支持度和置信度最终给出商品推荐。

#### 7.2 实验目的

- 掌握 PySpark MLlib 建模的基础语法，同时了解 FPGrowth 关联规则算法在实际业务中的应用。
- 掌握数据挖掘的详细流程。

#### 7.3 实验过程和步骤

**7.3.1 读取数据**

```python
!pip install pyspark_dist_explore
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark_dist_explore import hist
from pyspark.ml.fpm import FPGrowth

sc = SparkContext('local', 'test')
spark = SparkSession(sc)

df = spark.read.csv("data/basket.csv", header=True).withColumn("id", f.monotonically_increasing_id())
df_all = spark.read.csv("data/Groceries data.csv", header=True).withColumn("id", f.monotonically_increasing_id())
```

**7.3.2 采用 FPGrowth 进行推荐**

```python
# 把篮子收集成一套
df_basket = df.select("id", f.array([df[c] for c in df.columns[:11]]).alias("basket"))
df_aggregated = df_basket.select("id", f.array_except("basket", f.array(f.lit(None))).alias("basket"))

# 运行 FPGrowth
fp = FPGrowth(minSupport=0.001, minConfidence=0.001, itemsCol='basket', predictionCol='prediction')
model = fp.fit(df_aggregated)

# 查看频繁项集
model.freqItemsets.show(10, False)

# 查看关联规则
model.associationRules.filter(model.associationRules.confidence > 0.15).show(20, False)
```

输出关联规则示例：

| antecedent            | consequent   | confidence          | lift                |
|:---------------------|:------------|:-------------------|:-------------------|
| [bottled beer]        | [whole milk] | 0.15781710914454278 | 0.9993302598941151 |
| [detergent]           | [whole milk] | 0.16279069767441862 | 1.030824041177455  |
| [semi-finished bread] | [whole milk] | 0.176056338028169   | 1.1148247930239072 |
| [sausage, rolls/buns] | [whole milk] | 0.2125              | 1.345593525179856  |

#### 7.4 实验小结

本章实验通过使用 FPGrowth 算法给出了推荐规则。支持度与置信度分别为 0.001 与 0.001 可自行定义，多运行几次，最后选择自己认为最优的参数。

#### 7.5 思考题

1. FPGrowth 算法中支持度和置信度分别揭示了什么含义？
2. 请大家回顾关联分析的基础算法 Apriori 的分析步骤。
3. 在 Apriori 算法的基础上，FPGrowth 做了哪些优化？

---

### 8. PySpark MLlib 协同过滤

#### 8.1 实验介绍

该数据集为某挑战赛中的关于广告点击的数据，包括 34,000 条消费者评论的列表，包括 Kindle、Fire TV 棒等亚马逊产品。

#### 8.2 实验目的

了解 ALS 算法在实际业务中的应用。

#### 8.3 实验过程和步骤

**8.3.1 数据集描述**

```python
from pyspark.sql import SparkSession
from pyspark.context import SparkContext

sc = SparkContext('local', 'test')
spark = SparkSession(sc)

!unzip ./data/Amazon_Consumer_Reviews.zip -d ./data
df = spark.read.csv('data/Amazon_Consumer_Reviews.csv', inferSchema=True, header=True)
df.printSchema()
```

**8.3.2 数据预处理**

```python
# 重命名列
def rename_cols(df):
    for column in df.columns:
        new_column = column.replace('.', '')
        df = df.withColumnRenamed(column, new_column)
    return df

df_2 = rename_cols(df)
df_3 = df_2.select('reviewsusername', 'id', 'reviewsrating')

# StringIndexer 编码
from pyspark.ml.feature import StringIndexer

stringIndexer = StringIndexer(inputCol="id", outputCol="id_int")
model = stringIndexer.fit(df_4)
df_5 = model.transform(df_4)

stringIndexer = StringIndexer(inputCol="reviewsusername", outputCol="userid")
model = stringIndexer.fit(df_5)
df_6 = model.transform(df_5)

train, test = df_6.randomSplit([0.75, 0.25])
```

**8.3.3 建立模型**

```python
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import IntegerType

train = train.withColumn("reviewsrating", train["reviewsrating"].cast(IntegerType()))
test = test.withColumn("reviewsrating", test["reviewsrating"].cast(IntegerType()))

rs = ALS(maxIter=10, regParam=0.01, userCol='userid', itemCol='id_int',
         ratingCol='reviewsrating', nonnegative=True, coldStartStrategy="drop")
rs = rs.fit(train)
pred = rs.transform(test)

# 评估
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(metricName='rmse', predictionCol='prediction', labelCol='reviewsrating')
rmse = evaluator.evaluate(pred)
print(rmse)
# 2.2200359696962213
```

#### 8.4 实验小结

本章实验采用 ALS 算法实现协同过滤。其中，maxIter=10，regParam=0.01 这几个参数可以自行调整，尝试不同的参数对预测结果的影响。

#### 8.5 思考题

请简述协同过滤推荐系统基于哪两种不同的策略，他们的思路有什么不同？

---

## 第三部分：华为大数据平台 MRS 实验指导手册

### MRS 数据湖实验

#### 1.1 实验介绍

本实验进行数据入湖的模拟实验，通过 CDL 将 MySQL 数据实时入湖到 Hudi，通过 Spark 等多种方式查询 Hudi 数据，HetuEngine 拥有湖内与跨仓查询功能，将不同数据源的数据连接查询。

#### 1.2 实验目的

- 掌握 MRS 数据入湖所需组件的功能以及操作流程。
- 具备基于 MRS 的数据入湖能力，能够快速高效的进行相关数据入湖和查询。

#### 1.3 预备知识

MRS 数据入湖主要包括数据库 MySQL 和 MRS 组件操作。其中 CDL、Hudi 和 HetuEngine 是数据入湖的重点组件。

#### 1.4 实训环境说明

| 序号 | 设备名称  | 设备型号     | 数量 | 备注                            |
|-----|----------|-------------|-----|--------------------------------|
| 1   | RDS      | MySQL 8.0   | 1   | 数据入湖数据源                   |
| 2   | MRS      | MRS 3.2.0   | 1   | 包含 CDL、Kafka、Spark、Flink、Hudi、HetuEngine 等组件 |
| 3   | GaussDB  |             | 1   | 跨仓查询数据源                   |
| 4   | EIP      | 静态 BGP 50Mbit/s | 2 | 远程连接 MRS                   |

#### 1.5 实验过程和步骤

##### 1.5.1 搭建 MRS 环境

1. 在华为云控制台搜索"MapReduce 服务 MRS"，点击购买集群。
2. 区域选择"西南-贵阳一"，不勾选 FTP-Server、ClickHouse、IoTDB，其他全部勾选。
3. 购买弹性公网 IP（数量为 2），配置静态 BGP 50Mbit/s。
4. 选择虚拟私有云，输入密码，勾选确认授权，点击立即购买。
5. 等待约 10 分钟后，MRS 集群处于运行状态。
6. 配置安全组规则，便于远程连接 master1 节点。
7. 为 master1 添加弹性公网 IP，用 putty 登录 master1 节点。
8. 下载并安装客户端：

```bash
cd /tmp/FusionInsight-Client/
tar -xvf FusionInsight_Cluster_1_Services_Client.tar
tar -xvf FusionInsight_Cluster_1_Services_ClientConfig.tar
rm -f FusionInsight_Cluster_1_Services_Client.tar FusionInsight_Cluster_1_Services_ClientConfig.tar
cd FusionInsight_Cluster_1_Services_ClientConfig
./install.sh /opt/client
source /opt/client/bigdata_env
```

##### 1.5.2 搭建 RDS 环境

1. 控制台搜索"RDS"，选择"云数据库 RDS"，点击立即购买。
2. 数据库版本选择"8.0"，实例类型选择"单机"，区域与 MRS 保持一致。
3. 虚拟私有云 vpc 与 MRS 保持一致，输入密码，点击立即购买。
4. 等待 10 分钟创建完成后，添加入方向规则，便于 MRS 访问 RDS。
5. 登录数据库，新建数据库"test"，执行 SQL：

```sql
drop table if exists t1;
create table t1(
    uuid varchar(20),
    name varchar(10),
    age int,
    ts timestamp(3),
    part varchar(20)
);
```

##### 1.5.3 配置 CDL 连接

1. 前往 MRS Manager，添加用户 admintest，绑定全部用户组和"系统管理员角色"。
2. 切换登录用户为 admintest，进入 CDL 管理界面。
3. 上传 MySQL 驱动（与 RDS 版本一致）。
4. 新建"ENV"资源，新建 MySQL 连接（IP 为 RDS 内网 IP）和 Hudi 连接。
5. 新建作业，配置 MySQL 到 Hudi 的数据同步，启动作业。

##### 1.5.4 MySQL 数据导入 Hudi

在 RDS 数据库执行插入数据：

```sql
insert into t1 values
('id1','Danny',23,TIMESTAMP '1970-01-01 00:00:01','par1'),
('id2','Stephen',33,TIMESTAMP '1970-01-01 00:00:02','par1'),
('id3','Julian',53,TIMESTAMP '1970-01-01 00:00:03','par2'),
('id4','Fabian',31,TIMESTAMP '1970-01-01 00:00:04','par2'),
('id5','Sophia',18,TIMESTAMP '1970-01-01 00:00:05','par3'),
('id6','Emma',20,TIMESTAMP '1970-01-01 00:00:06','par3'),
('id7','Bob',44,TIMESTAMP '1970-01-01 00:00:07','par4'),
('id8','Han',56,TIMESTAMP '1970-01-01 00:00:08','par4');
```

在 master1 节点查看 Hudi 数据：

```bash
source /opt/client/bigdata_env
hdfs dfs -ls /tmp/cdl_to_hudi/t1
```

##### 1.5.5 通过 Spark/Spark SQL/Hive 查询数据

**方式一：Spark-shell**

```bash
source /opt/client/bigdata_env
source /opt/client/Hudi/component_env
spark-shell --master yarn
```

```scala
import org.apache.hudi.QuickstartUtils._
import scala.collection.JavaConversions._
import org.apache.spark.sql.SaveMode._
import org.apache.hudi.DataSourceReadOptions._
import org.apache.hudi.DataSourceWriteOptions._
import org.apache.hudi.config.HoodieWriteConfig._

spark.read.format("hudi").load("/tmp/cdl_to_hudi/t1").show(false)
```

**方式二：Spark-SQL**

```bash
spark-sql --master yarn
```

```sql
select * from t1;
```

**方式三：Hive**

```bash
beeline
```

```sql
set hive.input.format=org.apache.hadoop.hive.ql.io.HiveInputFormat;
show tables;
select * from t1;
```

##### 1.5.6 FlinkSQL 处理 Hudi 数据

1. 下载客户端配置文件。
2. 进入 Flink WebUI，创建集群连接，上传 MRS 配置文件。
3. 新建 Flink SQL 作业：

```sql
CREATE TABLE t1(
    uuid VARCHAR(20),
    name VARCHAR(20),
    age INT,
    ts TIMESTAMP(3),
    part VARCHAR(20)
) WITH (
    'connector' = 'hudi',
    'path' = 'hdfs://hacluster/tmp/cdl_to_hudi/t1',
    'table.type' = 'COPY_ON_WRITE',
    'read.streaming.enabled' = 'true',
    'read.streaming.check-interval' = '10',
    'hoodie.datasource.write.recordkey.field' = 'uuid',
    'write.precombine.field' = 'uuid'
);

CREATE TABLE t2(...) WITH (...);

INSERT INTO t2 SELECT * FROM t1;
```

##### 1.5.7 测试实时入湖性能

在 MySQL 插入新数据后，约 10 秒后通过 Spark-shell 查询：

```scala
spark.read.format("hudi").load("/tmp/cdl_to_hudi/t1")
    .select("uuid","name","age","ts","part").show(false)
```

##### 1.5.8 HetuEngine 湖内查询

```bash
source /opt/client/bigdata_env
cd /opt/client/HetuEngine/hetuserver/bin
./hetu-cli --catalog hive --schema default --user admintest
```

```sql
show tables;
select * from t1;
```

##### 1.5.9 配置 GaussDB 数据源

1. 控制台搜索 GaussDB，购买实例（区域与 MRS 相同）。
2. 新建数据库 test，创建表 t3：

```sql
create table t3(orderNo int primary key, uuid varchar(20), product varchar(20),
                quantity bigint default NULL, date varchar(20) default NULL);
insert into t3 values(1,'id1','笔记本',1,'2021-3-29 17:23:55');
insert into t3 values(2,'id2','牛奶',2,'2021-4-22 14:15:16');
insert into t3 values(3,'id3','苹果',5,'2021-6-20 09:25:55');
```

##### 1.5.10 HetuEngine 跨仓查询

在 HetuEngine WebUI 配置 GaussDB 数据源，JDBC URL：

```
jdbc:postgresql://<GaussDB_IP>:8000/test
```

跨仓查询：

```sql
show catalogs;
show schemas from gaussdb;
use gaussdb.public;
show tables;
select * from t3;

-- 跨仓连接查询
select * from hive.default.t1 h1
inner join gaussdb.public.t3 h2 on h1.uuid = h2.uuid;
```

##### 1.5.11 恢复实验环境

依次删除 GaussDB、MRS 集群、RDS 实例、弹性公网 IP。

#### 1.6 实验小结

本实验主要模拟了数据实时入湖的操作流程，介绍了三种查询 Hudi 数据的方式，通过 HetuEngine 实现了湖内查询和跨仓查询，为数据入湖提供了技术基础。

---

## 第四部分：实验环境搭建手册

### Python 实验环境创建与恢复

#### 实验操作步骤

**步骤 1** 访问华为云官网 https://www.huaweicloud.com/，点击页面右上角的 Console。

**步骤 2** 在搜索栏中输入"ModelArts"，选择下拉菜单中的"ModelArts"。

**步骤 3** 点击左上角的区域选择栏，在区域中选择支持开发环境"Notebook"的区域，如"北京四"。

**步骤 4** 点击左侧菜单栏中的"开发环境"，在下拉菜单中选择"Notebook"。

**步骤 5** 点击工作栏中的"创建"。

**步骤 6** 时间选择合适的时长，自动停止开启，则在实验时长结束之后自动停止。

**步骤 7** 点击公共镜像的下拉按钮。

**步骤 8** 点击第二页，选择"spark2.4.5-ubuntu18.04"。

**步骤 9** 规格选择"CPU：8核32GB"，云硬盘 EVS 选择默认 5G，点击"立即创建"。

**步骤 10** 确认无误之后，点击"提交"。

**步骤 11** 等待约一分钟后，点击"打开"，进入 notebook 环境。

**步骤 12** 在左侧文件栏中右击选择"New Folder"。

**步骤 13** 将新文件夹重命名为 data，双击进入 data 文件夹。

**步骤 14** 将实验数据集对应章节的 data 文件夹下的数据全部上传到新建的 data 文件夹下。例如只做第三章实验，只需将第三章数据集上传即可。如果遇到文件夹不能上传，请创建相应的文件夹，再上传对应文件夹下的数据。

**步骤 15** 点击左上角的文件夹图标，回到主目录。

**步骤 16** 点击左上角的+号，选择"Notebook"。

**步骤 17** 在新建的 jupyter 文件工作栏中选择右上角的引擎，在下拉菜单中选择"PySpark-2.4.5"，点击"Select"。

> 请注意：本书中的所有实验引擎全部为"PySpark-2.4.5"。

**步骤 18** 右击新建的 jupyter 文件，选择"Rename"即可重命名。

**步骤 19** 在代码区可编辑运行代码。

**步骤 20** 停止实例，点击"停止"即可。在停止前请确认实验代码已保存。

**步骤 21** 删除实例，点击"更多"→"删除"即可，代码和实验数据会清除。

#### 附录：常见问题解决方案

**实验保存失败**

1. 点击"Setting"，选择"Autosave Documents"，此操作将自动保存取消。
2. 点击"Kernel"，选择"Restart Kernel and Clear All Outputs"，清空结果显示。
3. 点击保存按钮即可只保存代码。

**代码无法运行**

1. 首先检查右上角引擎是否为"PySpark-2.4.5"以及是否还有剩余时间。
2. 如果步骤一没有问题，请重启"Kernel"：点击"Kernel"→"Restart Kernel and Run All Cells"。

---

## 第五部分：综合大实验指导手册

> 本实验指导书共包含 6 章实验：
> - 实验一：鱼的重量回归预测
> - 实验二：森林覆盖类型分类预测
> - 实验三：音乐流派聚类分析
> - 实验四：在线零售数据关联分析
> - 实验五：书籍推荐协同过滤分析
> - 实验六：PySpark 广告推荐系统

---

### 1. 鱼的重量回归预测

#### 1.1 实验介绍

该实验数据集记录了鱼市销售中 7 种常见的不同鱼种。有了这个数据集，可以使用机器学习回归算法预测鱼的重量。

#### 1.2 实验目的

具备使用 Python 进行行业数据分析的能力，能够使用所学的数据挖掘知识对实际回归问题进行分析和处理。

#### 1.3 预备知识

- Python 的基本操作和 Pandas 的数据分析和处理的基本操作。
- 数据挖掘回归算法原理，及对应的模型评估值指标。
- 了解 Python 数据可视化的常用方法和库。

#### 1.4 实验过程和步骤

**1.4.1 读取数据集**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./data/Fish.csv')
df.sample(5)
```

**1.4.2 数据探索**

```python
# 输出鱼的种类
df['Species'].unique()
# array(['Bream', 'Roach', 'Whitefish', 'Parkki', 'Perch', 'Pike', 'Smelt'], dtype=object)

# 对分类变量进行编码
df['Species Code'] = df['Species'].astype('category').cat.codes
df.drop('Species', axis=1, inplace=True)

# 输出空值信息
print(df.isnull().any())
# 没有出现空值现象，说明属性完整情况较好

# 查看每种属性与鱼的重量的分布关系
x_vars = df.columns[1:]
for x_var in x_vars:
    df.plot(kind='scatter', x=x_var, y='Weight')
    plt.show()
```

**1.4.3 查看属性之间的相关性**

```python
corr = df.corr()
plt.figure(figsize=(16, 8))
sns.heatmap(corr, annot=True, cmap="RdBu")
plt.show()
```

下三角热力图：

```python
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set_style(style="white")
sns.heatmap(corr, annot=True, cmap="RdBu", mask=mask)
plt.show()
```

强相关模式（相关系数 > 0.7）：

```python
mask = np.zeros_like(corr[corr >= .7], dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr[corr >= .7], annot=True, mask=mask, cbar=False)
plt.show()
```

结果显示 length1、length2、length3 之间存在强相关性。

**1.4.4 特征工程**

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

y = df["Weight"]
X = df.iloc[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# 数据标准化
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
```

**1.4.5 建立模型**

线性回归：

```python
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.score(X_test, y_test))
# 模型得分值：0.8220915054073246
```

随机森林：

```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=500)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.score(X_test, y_test))
# 模型得分值：0.9794037942860635
```

梯度提升树（GBDT）：

```python
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=500)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.score(X_test, y_test))
# 模型得分值：0.9767642285313475
```

最近邻（KNN）：

```python
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=10)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
# 模型得分值：0.8969922292862011
```

决策树：

```python
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
# 模型得分值：0.9653225201838107
```

**1.4.6 参数优化**

网格搜索：

```python
from sklearn.model_selection import GridSearchCV

model_gbr = GradientBoostingRegressor()
parameters = {
    'loss': ['huber', 'quantile'],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'alpha': [0.1, 0.3, 0.6, 0.9]
}
model_gs = GridSearchCV(estimator=model_gbr, param_grid=parameters, cv=5)
model_gs.fit(X_train, y_train)
print('Best score is:', model_gs.best_score_)
print('Best parameter is:', model_gs.best_params_)
# Best score is: 0.972052166546012
# Best parameter is: {'alpha': 0.6, 'loss': 'huber', 'min_samples_leaf': 2}
```

最优参数建模：

```python
model = GradientBoostingRegressor(n_estimators=500, alpha=0.6,
                                   loss='huber', min_samples_leaf=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.score(X_test, y_test))
# 模型评估值：0.974590716935367
```

> 思考：请读者尝试使用 Stacking 或 Voting 算法对上述算法进行融合优化。

#### 1.5 实验小结

本章主要介绍了如何针对实际问题进行建模分析，同时采用多种方法进行模型对比，最终选择梯度提升树进行模型训练。在模型参数优化方面，采用交叉网格搜索的方式进行最优化模型参数的搜索，有效提升了模型的准确率。

---

### 2. 森林覆盖类型分类预测

#### 2.1 实验介绍

本实验需要根据严格的制图变量（而不是遥感数据）预测森林覆盖类型。给定 30 x 30 米像元的实际森林覆盖类型是根据美国林务局（USFS）资源信息系统数据确定的。

#### 2.2 实验目的

通过一个示例介绍如何使用分类和集成学习算法来分析一个林业数据集，并尝试预测森林覆盖类型。

#### 2.3 实验过程和步骤

**2.3.1 读取数据集**

```python
!pip install plotly-express shap eli5 lime mlxtend pandas-profiling

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, log_loss
%matplotlib inline

df_forest = pd.read_csv('./data/train_forest_covertype.csv')
df_forest.head()
```

**2.3.2 数据探索与数据清洗**

```python
df_forest.info()
# 15120 entries, 56 columns

# 抽样
n = 15120
df_forest_sample = df_forest.sample(n)

# 检查标签分布
df_forest_sample['Cover_Type'].value_counts()
# 7 类，均匀分布，每类 2160

# Pandas profiling 快速分析
import pandas_profiling
report = pandas_profiling.ProfileReport(df_forest_sample)
report.to_file("flight_data.html")

# 删除 Soil_Type7 和 Soil_Type15（全部为 0）
df_forest_sample.drop(['Soil_Type7', 'Soil_Type15'], inplace=True, axis=1)
```

**2.3.3 特征工程**

```python
from sklearn import svm, tree, linear_model, neighbors, ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 初始化模型
seed = 42
first_models = [
    LogisticRegression(n_jobs=-1),
    LinearDiscriminantAnalysis(),
    SVC(random_state=seed, probability=True),
    DecisionTreeClassifier(random_state=seed),
    RandomForestClassifier(10, random_state=seed, n_jobs=-1),
    GradientBoostingClassifier(random_state=seed),
    AdaBoostClassifier(random_state=seed),
    XGBClassifier(random_state=seed, n_jobs=-1),
    KNeighborsClassifier(n_jobs=-1),
    LGBMClassifier(random_state=seed, n_jobs=-1)
]

# 交叉验证比较
n_folds = 5
skf = model_selection.ShuffleSplit(n_splits=n_folds, test_size=.3, train_size=.6, random_state=seed)
std_sca = StandardScaler()

X = df_forest_sample.drop(['Cover_Type'], axis=1)
y = df_forest_sample['Cover_Type']

for model in first_models:
    model_pipeline = Pipeline(steps=[('Scaler', std_sca), ('Estimator', model)])
    cv_results = model_selection.cross_validate(model, X, y, cv=skf, return_train_score=True)
```

**2.3.4 特征重要性**

```python
feature_names = X.columns
feat_imp_df = pd.DataFrame(columns=first_model_names, index=feature_names)
feat_imp_df.drop(['SVM', 'K-Neighbors'], axis=1, inplace=True)

feat_imp_df['Decision Tree'] = dtree.feature_importances_
feat_imp_df['Random Forest'] = rf.feature_importances_
feat_imp_df['GradientBoosting'] = gdb.feature_importances_
feat_imp_df['AdaBoost'] = adb.feature_importances_
feat_imp_df['XGB'] = xgb.feature_importances_
feat_imp_df['Light GBM'] = lgbm.feature_importances_

# 归一化并汇总
mms = MinMaxScaler()
scaled_fi = pd.DataFrame(data=mms.fit_transform(feat_imp_df),
                         columns=feat_imp_df.columns, index=feat_imp_df.index)
scaled_fi['Overall'] = scaled_fi.sum(axis=1)
ordered_ranking = scaled_fi.sort_values('Overall', ascending=False)

# 删除重要性最低的 20 个特征
train_v2 = df_forest_sample.drop(ordered_ranking.index[:-20:-1], axis=1)
X_v1 = train_v2.drop(['Cover_Type'], axis=1)
y_v1 = train_v2['Cover_Type']
```

**2.3.5 参数优化**

```python
from sklearn.model_selection import GridSearchCV

LGBM_param_grid = {
    'lgbm__learning_rate': [0.1],
    'lgbm__n_estimators': [100],
    'lgbm__num_leaves': [31],
    'lgbm__boosting_type': ['gbdt', 'dart']
}

RF_param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__criterion': ['gini', 'entropy'],
    'rf__max_depth': [2, 4, 8],
    'rf__oob_score': ['True']
}

for alg in first_models:
    gs_alg = GridSearchCV(model_pipeline, param_grid=params_grid[0], cv=skf,
                          scoring='accuracy', n_jobs=-1, return_train_score=True)
    gs_alg.fit(X_v1, y_v1)
```

**2.3.6 Stacking**

VotingClassifier：

```python
vote_est = [
    ('rfc', ensemble.RandomForestClassifier()),
    ('knn', neighbors.KNeighborsClassifier()),
    ('xgb', XGBClassifier()),
    ('lgbm', LGBMClassifier())
]

# Hard Vote
vote_hard = ensemble.VotingClassifier(estimators=vote_est, voting='hard')
vote_hard_cv = model_selection.cross_validate(vote_hard, X, y, cv=skf)
# Hard Voting Test w/bin score mean: 87.68

# Soft Vote
vote_soft = ensemble.VotingClassifier(estimators=vote_est, voting='soft')
vote_soft_cv = model_selection.cross_validate(vote_soft, X, y, cv=skf)
# Soft Voting Test w/bin score mean: 88.02
```

StackingClassifier：

```python
from mlxtend.classifier import StackingClassifier

sclf = StackingClassifier(classifiers=[lgbm_cl, rf_cl, gdb_cl], meta_classifier=logreg)
scores = model_selection.cross_val_score(sclf, X_v1, y_v1, cv=3, scoring='accuracy')
# Accuracy: 0.82 (+/- 0.01) [StackingClassifier]
```

#### 2.4 实验小结

本实验主要为基于分类算法以及集成算法在实际业务中的应用。VotingClassifier 和 StackingClassifier 算法都是一种集成学习技术，将多个分类器组合在一起，有效地提高分类器的准确率，减少过拟合的风险。

---

### 3. 音乐流派聚类分析

#### 3.1 实验介绍

本文的具体数据来自 Spotify for Developers，代表截至 2018 年 3 月之前歌曲数据集。每首歌曲（行）都有艺术家姓名、曲目名称、曲目 ID 和音频功能本身的值。

#### 3.2 实验目的

通过一个示例介绍如何使用聚类来分析一个音乐数据集，并尝试对音乐流派进行分类并分析。

#### 3.3 实验过程和步骤

**3.3.1 读取数据集**

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics

pd.options.display.max_columns = None
%matplotlib inline
warnings.simplefilter(action='ignore', category=FutureWarning)

origin_data = pd.read_csv('./data/SpotifyFeatures.csv')
data = origin_data.sample(n=10000, replace=True)
data.reset_index(inplace=True, drop=True)
```

**3.3.2 数据探索**

```python
data.info()
# 10000 entries, 18 columns

data.hist(figsize=(25, 25))

# 选取 500 首最流行的歌曲
data_new = data.copy().sort_values(by='popularity', ascending=False).iloc[:500]

# 最著名的十个艺术家
data_new['artist_name'].value_counts()[:10]
```

**3.3.3 特征工程**

```python
df_cluster = data.copy()
df_X = df_cluster.select_dtypes(np.number)
cols = df_X.columns
# Index(['popularity', 'acousticness', 'danceability', 'duration_ms', 'energy',
#        'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'])

# 归一化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled = pd.DataFrame(scaler.fit_transform(df_X))
scaled.columns = cols
```

**3.3.4 聚类建模**

KMeans：

```python
from sklearn.cluster import KMeans

k, silhouette, sse = [], [], []
for i in range(2, 20):
    kmeans_sk = KMeans(n_clusters=i, init='k-means++', max_iter=500, n_init=10, random_state=0)
    kmeans_sk.fit(scaled)
    s1 = metrics.silhouette_score(scaled, kmeans_sk.labels_, metric='euclidean')
    s2 = kmeans_sk.inertia_
    silhouette.append(s1)
    sse.append(s2)
```

BIRCH：

```python
from sklearn.cluster import Birch

for i in range(3, 12):
    Birch_model = Birch(n_clusters=i, threshold=0.5).fit(scaled)
    s = metrics.silhouette_score(scaled, Birch_model.labels_, metric='euclidean')
# K = 5, Silhouette = 0.285 (最高)
```

GMM：

```python
from sklearn.mixture import GaussianMixture

n_components = np.arange(1, 30)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(scaled) for n in n_components]
plt.plot(n_components, [m.bic(scaled) for m in models], label='BIC')
plt.plot(n_components, [m.aic(scaled) for m in models], label='AIC')
plt.legend(loc='best')
```

SpectralClustering：

```python
from sklearn.cluster import SpectralClustering

for i in range(2, 12):
    spectral = SpectralClustering(n_clusters=i, eigen_solver='arpack',
                                   affinity='nearest_neighbors').fit(scaled)
    s = metrics.silhouette_score(scaled, spectral.labels_, metric='euclidean')
# K = 5, Silhouette = 0.316 (最高)
```

**3.3.5 初始化模型并分析**

最终采用谱聚类，聚类数量为 5：

```python
spectral = SpectralClustering(n_clusters=5, eigen_solver='arpack',
                               affinity='nearest_neighbors')
result = spectral.fit_predict(scaled)

# 预测值与原有数据集合并
spectral_sk = pd.DataFrame(data=result, dtype=int, columns=['k_cluster'])
df_skcluster = pd.concat([df_cluster, spectral_sk], axis=1)

# 根据聚类结果计算 popularity 均值
df_skcluster.groupby(['k_cluster']).popularity.mean().sort_values(ascending=False)
```

#### 3.4 实验小结

本实验主要为基于聚类算法在实际业务中的应用。KMeans 是一种简单、快速的聚类算法。Birch 是一种基于层次的聚类算法，可以处理大数据集。GMM 和 Spectral 都是基于概率模型的聚类算法，可以处理任意形状的聚类。

---

### 4. 在线零售数据关联分析

#### 4.1 实验介绍

本文的具体数据来自 UCI Machine Learning Repository，代表 2010 年 1 月 12 日至 2011 年 9 月 12 日之间发生的所有英国注册非商店在线零售交易。

#### 4.2 实验目的

通过一个示例介绍如何使用关联规则来分析一个相对较大的在线零售数据集，并尝试找到有趣的购买组合。

#### 4.3 实验过程和步骤

**4.3.1 读取数据集**

```python
!pip install mlxtend
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

data = pd.read_csv('./data/Online Retail.csv')
```

**4.3.2 数据清洗**

```python
data['Description'] = data['Description'].str.strip()
data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
data['InvoiceNo'] = data['InvoiceNo'].astype('str')
data = data[~data['InvoiceNo'].str.contains('C')]
```

**4.3.3 特征工程**

```python
# one-hot 编码
basket = (data.groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))

# 将数量转换为 0 和 1
def encode_units(x):
    if x <= 0: return 0
    if x >= 1: return 1

basket_sets = basket.applymap(encode_units)
```

**4.3.4 挖掘频繁项集和关联规则**

```python
frequent_itemsets = fpgrowth(basket_sets, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# 设置过滤条件
rules[(rules['lift'] >= 3) & (rules['confidence'] >= 0.8)]
```

#### 4.4 实验小结

本实验主要为基于关联规则在实际业务中的应用。关联分析对数学概念的理解相对较浅，并且易于向非技术人员解释。它是一种寻找隐藏模式的无监督学习工具。

---

### 5. 书籍推荐协同过滤分析

#### 5.1 实验介绍

本案例使用的数据集是由 Cai-Nicolas Ziegler 在 2004 年编辑的，它由三张表组成，供用户、书籍和评级。

- **BX-Users**：包含用户信息（`User-ID`、`Location`、`Age`）。
- **BX-Books**：书籍由各自的 ISBN 标识，包含 `Book-Title`、`Book-Author`、`Year-Of-Publication`、`Publisher`。
- **BX-Book-Ratings**：评级（`Book-Rating`）要么是明确的，以 1-10 的等级表示，要么是隐含的，用 0 表示。

#### 5.2 实验目的

这个项目需要基于用户和基于项目的协同过滤方法为用户构建一个图书推荐系统。

#### 5.3 实验过程和步骤

**5.3.1 读取数据集**

```python
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation
from sklearn.metrics.pairwise import pairwise_distances
import seaborn as sns

# 书籍数据集
books = pd.read_csv('./data/Book/BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher',
                 'imageUrlS', 'imageUrlM', 'imageUrlL']

# 用户数据集
users = pd.read_csv('./data/Book/BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']

# 评级数据集
ratings = pd.read_csv('./data/Book/BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']

print(books.shape)   # (271360, 8)
print(users.shape)   # (278858, 3)
print(ratings.shape) # (1149780, 3)
```

**5.3.2 探索三个数据集**

书籍数据集：

```python
# 删除图像 URL 列
books.drop(['imageUrlS', 'imageUrlM', 'imageUrlL'], axis=1, inplace=True)

# 修正 yearOfPublication 异常值
books.loc[books.yearOfPublication == 'DK Publishing Inc', :]
# 修正数据错位
books.loc[books.ISBN == '0789466953', 'yearOfPublication'] = 2000
books.loc[books.ISBN == '0789466953', 'bookAuthor'] = "James Buckley"
books.loc[books.ISBN == '0789466953', 'publisher'] = "DK Publishing Inc"

books.loc[books.ISBN == '078946697X', 'yearOfPublication'] = 2000
books.loc[books.ISBN == '078946697X', 'bookAuthor'] = "Michael Teitelbaum"
books.loc[books.ISBN == '078946697X', 'publisher'] = "DK Publishing Inc"

# 转换类型
books.yearOfPublication = pd.to_numeric(books.yearOfPublication, errors='coerce')
# 2006 之后的年份设为 NaN
books.loc[(books.yearOfPublication > 2006) | (books.yearOfPublication == 0), 'yearOfPublication'] = np.NAN
books.yearOfPublication.fillna(round(books.yearOfPublication.mean()), inplace=True)
books.yearOfPublication = books.yearOfPublication.astype(np.int32)
```

用户数据集：

```python
# 年龄异常值处理
users.loc[(users.Age > 90) | (users.Age < 5), 'Age'] = np.nan
users.Age = users.Age.fillna(users.Age.mean())
users.Age = users.Age.astype(np.int32)
```

评级数据集：

```python
# 筛选有效数据
ratings_new = ratings[ratings.ISBN.isin(books.ISBN)]
ratings = ratings[ratings.userID.isin(users.userID)]

# 稀疏性检查
sparsity = 1.0 - len(ratings_new) / float(n_users * n_books)
# 图书交叉数据集的稀疏级别是 99.99863734155898%

# 显式/隐式评级分离
ratings_explicit = ratings_new[ratings_new.bookRating != 0]
ratings_implicit = ratings_new[ratings_new.bookRating == 0]
```

**5.3.3 基于简单流行度的推荐系统**

```python
ratings_count = pd.DataFrame(ratings_explicit.groupby(['ISBN'])['bookRating'].sum())
top10 = ratings_count.sort_values('bookRating', ascending=False).head(10)
print("推荐下列书籍:")
top10.merge(books, left_index=True, right_on='ISBN')
```

**5.3.4 基于协同过滤的推荐系统数据预处理**

```python
# 选取至少有 100 本书的用户和至少有 100 个评分的书籍
counts1 = ratings_explicit['userID'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['userID'].isin(counts1[counts1 >= 100].index)]
counts = ratings_explicit['bookRating'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['bookRating'].isin(counts[counts >= 100].index)]

# 生成评级矩阵
ratings_matrix = ratings_explicit.pivot(index='userID', columns='ISBN', values='bookRating').fillna(0)
ratings_matrix.fillna(0, inplace=True)
ratings_matrix = ratings_matrix.astype(np.int32)

n_users = ratings_matrix.shape[0]  # 449
n_books = ratings_matrix.shape[1]  # 66574
```

**5.3.5 基于用户的推荐系统**

```python
global metric, k
k = 10
metric = 'cosine'

def findksimilarusers(user_id, ratings, metric=metric, k=k):
    similarities = []
    indices = []
    model_knn = NearestNeighbors(metric=metric, algorithm='brute')
    model_knn.fit(ratings)
    loc = ratings.index.get_loc(user_id)
    distances, indices = model_knn.kneighbors(
        ratings.iloc[loc, :].values.reshape(1, -1), n_neighbors=k+1)
    similarities = 1 - distances.flatten()
    return similarities, indices

def predict_userbased(user_id, item_id, ratings, metric=metric, k=k):
    prediction = 0
    user_loc = ratings.index.get_loc(user_id)
    item_loc = ratings.columns.get_loc(item_id)
    similarities, indices = findksimilarusers(user_id, ratings, metric, k)
    mean_rating = ratings.iloc[user_loc, :].mean()
    sum_wt = np.sum(similarities) - 1
    wtd_sum = 0
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] == user_loc:
            continue
        else:
            ratings_diff = ratings.iloc[indices.flatten()[i], item_loc] - \
                           np.mean(ratings.iloc[indices.flatten()[i], :])
            product = ratings_diff * similarities[i]
            wtd_sum = wtd_sum + product
    prediction = int(round(mean_rating + (wtd_sum / sum_wt)))
    if prediction <= 0: prediction = 1
    elif prediction > 10: prediction = 10
    return prediction

# 测试
predict_userbased(11676, '0001056107', ratings_matrix)
# 用户预测等级 11676 -> item 0001056107: 2
```

**5.3.6 基于项目推荐系统**

```python
def findksimilaritems(item_id, ratings, metric=metric, k=k):
    similarities = []
    indices = []
    ratings = ratings.T
    loc = ratings.index.get_loc(item_id)
    model_knn = NearestNeighbors(metric=metric, algorithm='brute')
    model_knn.fit(ratings)
    distances, indices = model_knn.kneighbors(
        ratings.iloc[loc, :].values.reshape(1, -1), n_neighbors=k+1)
    similarities = 1 - distances.flatten()
    return similarities, indices

def predict_itembased(user_id, item_id, ratings, metric=metric, k=k):
    prediction = wtd_sum = 0
    user_loc = ratings.index.get_loc(user_id)
    item_loc = ratings.columns.get_loc(item_id)
    similarities, indices = findksimilaritems(item_id, ratings)
    sum_wt = np.sum(similarities) - 1
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] == item_loc:
            continue
        else:
            product = ratings.iloc[user_loc, indices.flatten()[i]] * similarities[i]
            wtd_sum = wtd_sum + product
    prediction = int(round(wtd_sum / sum_wt))
    if prediction <= 0: prediction = 1
    elif prediction > 10: prediction = 10
    return prediction

# 测试
predict_itembased(11676, '0001056107', ratings_matrix)
# 用户预测等级 11676 -> item 0001056107: 1
```

#### 5.4 实验小结

本实验主要为基于用户和基于项目的协作过滤方法在实际业务中的应用。通过对三个数据集进行特征工程和探索性数据分析，从而提高推荐的准确率。

---

### 6. PySpark 广告推荐系统

#### 6.1 实验介绍

本案例使用广告点击率预估的数据集，实现一个离线和在线推荐相结合的实时推荐系统，对非搜索类型的广告进行点击率预测和推荐。使用的数据集包含以下三部分：

| 数据集        | 说明         |
|-------------|------------|
| raw_sample  | 原始的样本骨架  |
| ad_feature  | 广告的基本信息  |
| user_profile| 用户的基本信息  |

#### 6.2 实验目的

- 掌握 PySpark MLlib 建模的基础语法，同时了解独热编码、缺失值填充在实际业务中的应用。
- 掌握数据挖掘的详细流程。

#### 6.3 实验过程和步骤

**6.3.1 raw_sample 的数据分析和处理**

```python
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType

sc = SparkContext('local', 'test')
spark = SparkSession(sc)

# 加载 CSV
df = spark.read.csv("./data/Ad_Recommendations/raw_sample_4m.csv", header=True)

# 类型转换
raw_sample_df = df.\
    withColumn("user", df.user.cast(IntegerType())).withColumnRenamed("user", "userId").\
    withColumn("time_stamp", df.time_stamp.cast(LongType())).withColumnRenamed("time_stamp", "timestamp").\
    withColumn("adgroup_id", df.adgroup_id.cast(IntegerType())).withColumnRenamed("adgroup_id", "adgroupId").\
    withColumn("pid", df.pid.cast(StringType())).\
    withColumn("nonclk", df.nonclk.cast(IntegerType())).\
    withColumn("clk", df.clk.cast(IntegerType()))

# 独热编码
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline

stringindexer = StringIndexer(inputCol='pid', outputCol='pid_feature')
encoder = OneHotEncoder(dropLast=False, inputCol='pid_feature', outputCol='pid_value')
pipeline = Pipeline(stages=[stringindexer, encoder])
pipeline_model = pipeline.fit(raw_sample_df)
new_df = pipeline_model.transform(raw_sample_df)

# 划分训练集和测试集
# 前七天为训练数据、最后一天为测试数据
train_sample = raw_sample_df.filter(raw_sample_df.timestamp <= (1494691172 - 24*60*60))
test_sample = raw_sample_df.filter(raw_sample_df.timestamp > (1494691172 - 24*60*60))
```

**6.3.2 ad_feature 数据的分析和处理**

```python
# 替换掉 NULL 字符串
df = df.replace("NULL", "-1")

ad_feature_df = df.\
    withColumn("adgroup_id", df.adgroup_id.cast(IntegerType())).withColumnRenamed("adgroup_id", "adgroupId").\
    withColumn("cate_id", df.cate_id.cast(IntegerType())).withColumnRenamed("cate_id", "cateId").\
    withColumn("campaign_id", df.campaign_id.cast(IntegerType())).withColumnRenamed("campaign_id", "campaignId").\
    withColumn("customer", df.customer.cast(IntegerType())).withColumnRenamed("customer", "customerId").\
    withColumn("brand", df.brand.cast(IntegerType())).withColumnRenamed("brand", "brandId").\
    withColumn("price", df.price.cast(FloatType()))
```

**6.3.3 user_profile 数据集的分析和处理**

```python
schema = StructType([
    StructField("userId", IntegerType()),
    StructField("cms_segid", IntegerType()),
    StructField("cms_group_id", IntegerType()),
    StructField("final_gender_code", IntegerType()),
    StructField("age_level", IntegerType()),
    StructField("pvalue_level", IntegerType()),
    StructField("shopping_level", IntegerType()),
    StructField("occupation", IntegerType()),
    StructField("new_user_class_level", IntegerType())
])

user_profile_df = spark.read.csv("./data/Ad_Recommendations/user_profile.csv",
                                  header=True, schema=schema)

# 查看空值情况
user_profile_df.groupBy("pvalue_level").count().show()
# pvalue_level 空值: 575917, 占比: 54.24%

user_profile_df.groupBy("new_user_class_level").count().show()
# new_user_class_level 空值: 344920, 占比: 32.49%
```

缺失值填充——随机森林预测：

```python
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest

# 训练数据（剔除 pvalue_level 空值）
train_data = user_profile_df.dropna(subset=['pvalue_level']).rdd.map(
    lambda r: LabeledPoint(r.pvalue_level - 1,
                           [r.cms_segid, r.cms_group_id, r.final_gender_code,
                            r.age_level, r.shopping_level, r.occupation])
)

# 训练随机森林模型
model = RandomForest.trainClassifier(train_data, 3, {}, 5)

# 预测缺失值
pl_na_df = user_profile_df.na.fill(-1).where("pvalue_level=-1")
rdd = pl_na_df.rdd.map(lambda r: (r.cms_segid, r.cms_group_id, r.final_gender_code,
                                   r.age_level, r.shopping_level, r.occupation))
predicts = model.predict(rdd)

# 合并数据集
temp = predicts.map(lambda x: int(x)).collect()
pdf = pl_na_df.toPandas()
pdf["pvalue_level"] = np.array(temp) + 1
new_user_profile_df = user_profile_df.dropna(subset=["pvalue_level"]).unionAll(
    spark.createDataFrame(pdf, schema=schema))
```

独热编码：

```python
# 先填充缺失值为 -1
user_profile_df = user_profile_df.na.fill(-1)

# 转为字符串类型
user_profile_df = user_profile_df.withColumn("pvalue_level",
    user_profile_df.pvalue_level.cast(StringType())).\
    withColumn("new_user_class_level",
    user_profile_df.new_user_class_level.cast(StringType()))

# pvalue_level 独热编码
stringindexer = StringIndexer(inputCol='pvalue_level', outputCol='pl_onehot_feature')
encoder = OneHotEncoder(dropLast=False, inputCol='pl_onehot_feature', outputCol='pl_onehot_value')
pipeline = Pipeline(stages=[stringindexer, encoder])
pipeline_fit = pipeline.fit(user_profile_df)
user_profile_df2 = pipeline_fit.transform(user_profile_df)

# new_user_class_level 独热编码
stringindexer = StringIndexer(inputCol='new_user_class_level', outputCol='nucl_onehot_feature')
encoder = OneHotEncoder(dropLast=False, inputCol='nucl_onehot_feature', outputCol='nucl_onehot_value')
pipeline = Pipeline(stages=[stringindexer, encoder])
pipeline_fit = pipeline.fit(user_profile_df2)
user_profile_df3 = pipeline_fit.transform(user_profile_df2)
```

保存处理后的数据：

```python
new_df.toPandas().to_csv("new_raw_sample.csv", header=True, index=False)
ad_feature_df.toPandas().to_csv("new_ad_feature.csv", header=True, index=False)
user_profile_df3.toPandas().to_csv("new_user_profile.csv", header=True, index=False)
```

#### 6.4 实验小结

本实验主要探索独热编码、缺失值填充在实际业务中的应用。通过对三个数据集进行特征工程和探索性数据分析，从而提高广告进行点击率预测和推荐的准确率。

---

> **版权所有 © 华为技术有限公司 2023。保留一切权利。**
>
> 非经本公司书面许可，任何单位和个人不得擅自摘抄、复制本文档内容的部分或全部，并不得以任何形式传播。
>
> 本文档提及的其他所有商标或注册商标，由各自的所有人拥有。
>
> 您购买的产品、服务或特性等应受华为公司商业合同和条款的约束，本文档中描述的全部或部分产品、服务或特性可能不在您的购买或使用范围之内。除非合同另有约定，华为公司对本文档内容不做任何明示或暗示的声明或保证。
>
> 由于产品版本升级或其他原因，本文档内容会不定期进行更新。除非另有约定，本文档仅作为使用指导，本文档中的所有陈述、信息和建议不构成任何明示或暗示的担保。
>
> **华为技术有限公司**
>
> 地址：深圳市龙岗区坂田华为总部办公楼  邮编：518129  
> 网址：https://e.huawei.com

