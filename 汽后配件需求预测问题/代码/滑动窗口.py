import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
import statsmodels.api as sm
import itertools
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.svm import SVR

# 读取csv文件
address = r'D:/数学建模/数学建模/代码/第三次/历史配件订单表.csv'
data = pd.read_csv(address, encoding='gbk')

# 预处理
# 1、查看缺失值比例
print("查看缺失值比例：", data.isnull().sum() / len(data))
'''无缺失值'''

# 2、对当天该仓库对该配件的需求量进行加和处理
data['日期'] = pd.to_datetime(data['日期'])
df = data.groupby(['仓库编码', '配件编码', pd.Grouper(key='日期', freq='D')])['需求量'].sum().reset_index()
df.set_index('日期', inplace=True)

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

# 假设我们只关注一个特定的配件sku003
df_sku003 = df[df['配件编码'] == 'sku003']

# 提取要预测的列
df_sku003_demand = df_sku003[['需求量']]
df_sku003_demand.index = pd.to_datetime(df_sku003_demand.index)

# 获取数据集的起始和结束日期
start_date = df_sku003_demand.index.min()
end_date = df_sku003_demand.index.max()

# 使用前一个日期的值填充缺失的日期
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
df_sku003_demand = df_sku003_demand.reindex(all_dates, method='ffill')

# 拆分数据集，最后3个月为测试集，其他为训练集
last_month_start = end_date - pd.offsets.MonthBegin(3) + pd.Timedelta(days=1)
train = df_sku003_demand.loc[:last_month_start - pd.Timedelta(days=1)]
test = df_sku003_demand.loc[last_month_start:]

# SARIMA建模
def SARIMA_search(data):
    p = q = range(0, 3)
    s = [7]  # 周期
    d = [0, 1]
    PDQs = list(itertools.product(p, d, q, s))
    pdq = list(itertools.product(p, d, q))

    params = []
    seasonal_params = []
    results = []

    for param in pdq:
        for seasonal_param in PDQs:
            mod = sm.tsa.SARIMAX(data, order=param, seasonal_order=seasonal_param,
                                 enforce_stationarity=False, enforce_invertibility=False)
            result = mod.fit()
            params.append(param)
            seasonal_params.append(seasonal_param)
            results.append(result.aic)

    grid = pd.DataFrame({
        "pdq": params,
        "PDQs": seasonal_params,
        "aic": results
    })

    # 返回具有最低AIC值的行
    best_row = grid[grid["aic"] == grid["aic"].min()]
    return best_row

# 使用SARIMA_search找到最佳参数
best_params = SARIMA_search(df_sku003_demand.dropna())

# 从结果中提取最佳参数
best_p, best_d, best_q = best_params.iloc[0]['pdq']
best_P, best_D, best_Q, best_s = best_params.iloc[0]['PDQs']

# 使用最佳参数建立模型
model = sm.tsa.SARIMAX(df_sku003_demand, order=(best_p, best_d, best_q),
                       seasonal_order=(best_P, best_D, best_Q, best_s))
SARIMA_m = model.fit()

# 滑动窗口预测
window_size = 30
forecast_steps = len(test)
predictions = []

for i in range(forecast_steps):
    # 定义训练数据的结束点
    end_of_train = test.index[i] - pd.Timedelta(days=1)
    # 截取训练数据
    train_data = df_sku003_demand.loc[:end_of_train]
    # 重新拟合模型
    model = sm.tsa.SARIMAX(train_data, order=(best_p, best_d, best_q),
                           seasonal_order=(best_P, best_D, best_Q, best_s))
    fitted_model = model.fit()
    # 进行一步预测
    pred = fitted_model.get_forecast(steps=1)
    predictions.append(pred.predicted_mean.iloc[-1])

# 将预测结果转换为Series
predict_demand = pd.Series(predictions, index=test.index)

# 计算WMAPE和SMAPE
def mean_weighted_absolute_percentage_error(y_true, y_pred, weights=None):
    if weights is None:
        weights = np.ones_like(y_true)
    y_true, y_pred, weights = np.array(y_true), np.array(y_pred), np.array(weights)
    diff = np.abs((y_true - y_pred) / y_true)
    norm_diff = diff * weights
    norm_diff = norm_diff[y_true != 0]  # 避免除以零
    return np.average(norm_diff, weights=weights[y_true != 0])

def mean_symmetric_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))

results_df = pd.DataFrame({
    '需求量': test['需求量'],
    '预测需求量': predict_demand
}, index=test.index)

wmape = mean_weighted_absolute_percentage_error(results_df['需求量'], results_df['预测需求量'])
smape = mean_symmetric_absolute_percentage_error(results_df['需求量'], results_df['预测需求量'])

print(f"WMAPE: {wmape:.2f}%")
print(f"SMAPE: {smape:.2f}%")

# 绘制时间序列与预测数据
plt.figure(figsize=(12, 6))
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置字体为黑体
plt.plot(test.index, test['需求量'], label='测试集需求量', marker='o', linestyle='-')
plt.plot(test.index, predict_demand, label='预测需求量', color='yellow', marker='^', linestyle='-')
plt.xticks(rotation=45)
plt.legend()
plt.show()
