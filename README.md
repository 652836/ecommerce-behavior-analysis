# ecommerce-behavior-analysis

## 一、项目简介
- 本项目采用python+机器学习对数据集进行分析，最后通过tableau对处理后的数据进行可视化呈现。

## 二、数据说明
- 数据集链接：https://www.kaggle.com/datasets/kartikeybartwal/ecommerce-product-recommendation-collaborative

- 本数据集汇集了某个电商平台的用户基本信息、行为习惯和互动数据。它包括用户的年龄、性别、居住地区、收入水平等基本属性，以及他们的兴趣偏好、登录频率、购买行为和平台互动等动态指标。数据说明如下：
![image](https://github.com/user-attachments/assets/c73244f1-8061-4447-944a-eaf1a47893d5)

## 三、分析目标
1) 对用户进行购买行为分析
2) 进行RFM分析
3) 对用户活跃度进行分析
4) 进行个性化推荐预测

## 四、数据预处理
1) 数据导入
```python
      import pandas as pd
      import numpy as np

      df=pd.read_csv('/kaggle/input/ecommerce-product-recommendation-collaborative/user_personalized_features.csv')
```

2) 数据展示和清洗
```python
      # 检查缺失值
      missing_values = df.isnull().sum()
      print(missing_values)
      #df['column'].fillna(df['column'].mean(), inplace=True)
      df.drop(columns=["Unnamed: 0"], inplace=True)  # 删除冗余列
      df = df[df["Income"] > 0]  # 过滤无效收入
      #df.dropna(inplace=True)  # 这里无需删除缺失值
```

## 五、数据分析
1) 购买行为分析
- 分析购买频率、平均订单价值、总消费金额的分布。
- 探索不同用户群体（如不同性别、地区、年龄段）的购买行为差异。
- 识别高价值用户和低价值用户。

```python
      # 2. 购买行为分析
      # 按性别统计购买行为
      purchase_by_gender = df.groupby("Gender").agg(
          avg_purchase_freq=("Purchase_Frequency", "mean"),
          total_spending=("Total_Spending", "sum")
      ).reset_index()
      
      # 按地区统计总消费
      spending_by_location = df.groupby("Location")["Total_Spending"].sum().reset_index()
```

2) RFM分析
### 2.1 RFM模型介绍
RFM模型是衡量客户价值和客户创利能力的重要工具和手段，该模型通过一个客户的最近一次消费时间、消费频率、消费金额三项指标来描述该客户的价值状况。在电商领域，RFM模型可帮助企业了解客户的购买行为和购买偏好，从而识别高价值客户、潜在回头客或低活跃度用户。这有助于电商企业定制个性化的营销策略。

RFM模型通过三个关键指标来描述客户的价值状况，这三个指标分别是：

- R（Recency）：最近一次消费时间。它表示用户最后一次下单时间距今天有多长时间。这个指标与用户流失和复购直接相关，如果客户最近消费过，那么他们更有可能再次消费。
- F（Frequency）：消费频率。它表示用户在固定的时间段内消费了多少次。这个指标反映了用户的消费活跃度，消费频率越高的客户，对商家的忠诚度通常也越高。
- M（Monetary）：消费金额。它表示用户在固定的周期内在平台上花费了多少钱。这个指标直接反映了用户对公司贡献的价值，消费金额越高的客户，通常被认为价值越大。
这三个指标共同构成了RFM模型，帮助商家更好地理解和评估客户的价值，从而制定更精准的营销策略，提高客户留存率，促进客户消费，最终实现业务增长。同时，通过RFM模型，企业可以将客户进行细分，针对不同群体的客户采取不同的营销策略，实现精准营销。

### 2.2.  RFM分析实现
RFM计算：最近一次登录、购买频率、总消费金额。
用户分群：高价值客户、潜力客户、一般客户、流失风险客户。

**代码在文件中一致给出，得到下述结果：**
![image](https://github.com/user-attachments/assets/fc7e0142-a957-421a-974b-03d52f347232)

3） 用户活跃度分析
- 活跃度定义：结合最近登录天数和在网站上的停留时间，将最近7天内登录且停留时间超过300分钟的用户定义为高活跃用户，超过30天未登录且停留时间低于100分钟的为低活跃用户。中间状态的用户则为普通用户

**得到结果：**


4) 个性化预测
- 个性化推荐预测需要使用机器学习模型，根据用户的兴趣和产品类别偏好，预测他们可能感兴趣的其他产品类别,这里分别采用k-neighbor模型和apriroi模型针对特征"Interests", "Product_Category_Preference"进行分析预测：



## 数据可视化


## 结论

