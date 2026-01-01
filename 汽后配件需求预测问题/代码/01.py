import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

warnings.filterwarnings("ignore")

# 读取csv文件
address = r'D:/数学建模/数学建模/代码/第三次/历史配件订单表.csv'
data = pd.read_csv(address, encoding='gbk')

# 预处理
# 1、查看缺失值比例
print("查看缺失值比例：\n", data.isnull().sum()/len(data))
'''无缺失值'''

# 2、对当天该仓库对该配件的需求量进行加和处理
data['日期'] = pd.to_datetime(data['日期'])
df = data.groupby(['仓库编码', '配件编码', pd.Grouper(key='日期', freq='D')])['需求量'].sum().reset_index()
df.set_index('日期', inplace=True)

# 绘制所有配件的需求量叠加图
pivot_df = df.pivot(columns='配件编码', values='需求量')
pivot_df.plot(figsize=(10, 5), title='所有配件的需求量叠加图')
plt.xlabel('日期')
plt.ylabel('需求量')
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
plt.legend(title='配件编码')
plt.show()

# 3、异常值处理
# 筛选需要处理的列
cols = ['需求量']

# 初始化离群点计数Series
outlier_counts_before = pd.Series(index=cols, dtype=int)
outlier_counts_after = pd.Series(index=cols, dtype=int)

# 处理离群点
for col in cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 识别离群点
    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
    outlier_counts_before[col] = outliers.sum()

    # 剔除离群点
    df = df[~outliers]

# 假设我们只关注一个特定的配件sku003
df_sku003 = df[df['配件编码'] == 'sku003']

# 提取要预测的列
df_sku003_demand = df_sku003[['需求量']]
df_sku003_demand.index = pd.to_datetime(df_sku003_demand.index)

# 拆分数据集,最后一个月为测试集，其他为训练集
last_month_start = df_sku003_demand.index.max() - pd.DateOffset(months=1) + pd.Timedelta(days=1)
train = df_sku003_demand.loc[:last_month_start - pd.Timedelta(days=1)]
test = df_sku003_demand.loc[last_month_start:]

# 使用SARIMAX模型
# 假设季节性周期为月（m=12），这里需要手动调整p, d, q, P, D, Q参数
# 这些参数可以通过ACF和PACF图来初步估计
model = SARIMAX(train['需求量'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False,
                enforce_invertibility=False)
results = model.fit()

# 预测
predict_demand = results.predict(start=test.index.min(), end=test.index.max(), dynamic=False)

# 绘制时间序列与预测数据
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['需求量'], label='训练集需求量', marker='o', linestyle='-')
plt.plot(test.index, test['需求量'], label='测试集需求量', marker='o', linestyle='-')
plt.plot(test.index, predict_demand, label='预测需求量', color='yellow', marker='^', linestyle='-')
plt.xticks(rotation=45)
plt.xlabel('日期')
plt.ylabel('需求量')
plt.title('SKU003 需求量预测')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
plt.show()


# 计算SMAPE（这里不计算WMAPE，因为SMAPE更常用于季节性数据）
def mean_symmetric_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))


# 计算SMAPE
smape = mean_symmetric_absolute_percentage_error(test['需求量'], predict_demand)
print(f"SMAPE: {smape:.2f}%")