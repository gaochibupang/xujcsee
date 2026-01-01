import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")
from pmdarima import auto_arima

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

    # 使用中位数替换离群点
    median_val = df[col].median()
    df.loc[outliers, col] = median_val

# 假设我们只关注一个特定的配件sku008
df_sku003 = df[df['配件编码'] == 'sku127']

# 提取要预测的列
df_sku003_demand = df_sku003[['需求量']]
df_sku003_demand.index = pd.to_datetime(df_sku003_demand.index)

# 获取数据集的起始和结束日期
start_date = df_sku003_demand.index.min()
end_date = df_sku003_demand.index.max()

# 使用前一个日期的值填充缺失的日期
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
df_sku003_demand = df_sku003_demand.reindex(all_dates, method='ffill')

################################################ 可视化 周期
# 设置图形大小和字体
plt.figure(figsize=(15, 10))
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置字体为黑体

# 设置第一个子图：第一个月
ax1 = plt.subplot(3, 1, 1)  # 3行1列的第一个位置
first_month_data = df_sku003_demand[df_sku003_demand.index < start_date + pd.DateOffset(months=1)]
ax1.plot(first_month_data.index, first_month_data['需求量'], label='SKU003 第一个月需求量')
ax1.set_title('SKU003 第一个月需求量')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
ax1.set_xlabel('日期')
ax1.set_ylabel('需求量')
ax1.grid(True)
ax1.legend()

# 设置第二个子图：第一季度
ax2 = plt.subplot(3, 1, 2)  # 3行1列的第二个位置
first_quarter_data = df_sku003_demand[df_sku003_demand.index < start_date + pd.DateOffset(months=3)]
ax2.plot(first_quarter_data.index, first_quarter_data['需求量'], label='SKU003 第一季度需求量')
ax2.set_title('SKU003 第一季度需求量')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
ax2.set_xlabel('日期')
ax2.set_ylabel('需求量')
ax2.grid(True)
ax2.legend()

# 设置第三个子图：第一年
ax3 = plt.subplot(3, 1, 3)  # 3行1列的第三个位置
first_year_data = df_sku003_demand[df_sku003_demand.index < start_date + pd.DateOffset(years=1)]
ax3.plot(first_year_data.index, first_year_data['需求量'], label='SKU003 第一年需求量')
ax3.set_title('SKU003 第一年需求量')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
ax3.set_xlabel('日期')
ax3.set_ylabel('需求量')
ax3.grid(True)
ax3.legend()
plt.tight_layout()
plt.show()

############################################ 可视化 ACF图、PACF图判断拖尾截尾
# ACF图
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置字体为黑体
plot_acf(df_sku003_demand['需求量'], lags=40)
plt.title('SKU003 需求量的自相关函数(ACF)')
plt.xlabel('滞后阶数')
plt.ylabel('自相关系数')
plt.grid(True)
plt.show()

# PACF图
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置字体为黑体
plot_pacf(df_sku003_demand['需求量'], lags=40)
plt.title('SKU003 需求量的偏自相关函数(PACF)')
plt.xlabel('滞后阶数')
plt.ylabel('偏自相关系数')
plt.grid(True)
plt.show()

########################################### 建模

# 拆分数据集,最后1个月为测试集，其他为训练集
last_month_start = end_date - pd.offsets.MonthBegin(1)
train = df_sku003_demand.loc[:last_month_start - pd.Timedelta(days=1)]
test = df_sku003_demand.loc[last_month_start:]

# 使用auto_arima自动选择最优参数
stepwise_model = auto_arima(train['需求量'], start_p=0, start_q=0,
                            max_p=3, max_q=3,   # m是季节性周期，如果数据没有季节性，可以设置为1或省略
                            start_P=0, seasonal=True,  # 季节性参数，这里假设没有季节性
                            d=None, D=None, trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)

# 预测
predict_demand = stepwise_model.predict(n_periods=len(test))
print(predict_demand)
# 绘制时间序列与预测数据
plt.figure(figsize=(12, 6))
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置字体为黑体
plt.plot(test.index, test['需求量'], label='测试集需求量', marker='o', linestyle='-')
predict_dates = pd.date_range(start=test.index.min(), periods=len(predict_demand), freq='D')
plt.plot(predict_dates, predict_demand, label='预测需求量', color='yellow', marker='^', linestyle='-')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# 计算WMAPE
def mean_weighted_absolute_percentage_error(y_true, y_pred, weights=None):
    if weights is None:
        weights = np.ones_like(y_true)
    y_true, y_pred, weights = np.array(y_true), np.array(y_pred), np.array(weights)
    diff = np.abs((y_true - y_pred) / y_true)
    norm_diff = diff * weights
    norm_diff = norm_diff[y_true != 0]  # 避免除以零
    return np.average(norm_diff, weights=weights[y_true != 0])

# 计算SMAPE
def mean_symmetric_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))

# 创建一个包含真实值和预测值的 DataFrame
results_df = pd.DataFrame({
    '需求量': test['需求量'],
    '预测需求量': predict_demand
}, index=test.index)

# 计算 WMAPE 和 SMAPE
wmape = mean_weighted_absolute_percentage_error(results_df['需求量'], results_df['预测需求量'])
smape = mean_symmetric_absolute_percentage_error(results_df['需求量'], results_df['预测需求量'])

print(f"WMAPE: {wmape:.2f}%")
print(f"SMAPE: {smape:.2f}%")

