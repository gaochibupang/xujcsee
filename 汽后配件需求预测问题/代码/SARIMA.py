import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller as ADF  # 使用别名ADF避免与前面的重复导入冲突
import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
import itertools

# 读取csv文件
address = r'D:/数学建模/数学建模/代码/第三次/历史配件订单表.csv'
data = pd.read_csv(address, encoding='gbk')

# 预处理
# 1、查看缺失值比例
print("查看缺失值比例：\n", data.isnull().sum()/len(data))
'''无缺失值'''

# 2、对当天该仓库对该配件的需求量进行加和处理，若不存在订单，则当天需求量为0
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

    # 剔除离群点
    df = df[~outliers]

######################################## 时间序列分析
# 假设我们只关注一个特定的配件sku003
df_sku003 = df[df['配件编码'] == 'sku003']
# 提取要预测的列
df_sku003_demand = df_sku003[['需求量']]
df_sku003_demand.index = pd.to_datetime(df_sku003_demand.index)

# 获取数据集的起始和结束日期
start_date = df_sku003_demand.index.min()
end_date = df_sku003_demand.index.max()

# 用0填充缺失的日期
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
df_sku003_demand = df_sku003_demand.reindex(all_dates, fill_value=0)

# 绘制时间序列图
plt.figure(figsize=(15, 5))
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.plot(df_sku003_demand.index, df_sku003_demand['需求量'], label='SKU003需求量')
plt.title('SKU003训练集 需求量时间序列')
plt.xlabel('日期')
plt.ylabel('需求量')
plt.grid(True)
plt.legend()
#plt.show()

# 分解时序
# STL（Seasonal and Trend decomposition using Loess）是一个非常通用和稳健强硬的分解时间序列的方法
decompostion=sm.tsa.STL(df_sku003_demand).fit()#statsmodels.tsa.api:时间序列模型和方法
decompostion.plot()
# 趋势效益
trend=decompostion.trend
# 季节效应
seasonal=decompostion.seasonal
# 随机效应
residual=decompostion.resid

#平稳性检验
#自定义函数用于ADF检查平稳性
def test_stationarity(timeseries,alpha):#alpha为检验选取的显著性水平
    adf=ADF(timeseries)
    p=adf[1]#p值
    critical_value=adf[4]["5%"]#在95%置信区间下的临界的ADF检验值
    test_statistic=adf[0]#ADF统计量
    if p<alpha and test_statistic<critical_value:
        print("ADF平稳性检验结果：在显著性水平%s下，数据经检验平稳"%alpha)
        return True
    else:
        print("ADF平稳性检验结果：在显著性水平%s下，数据经检验不平稳"%alpha)
        return False

#原始数据平稳性检验
test_stationarity(df_sku003_demand,1e-3)

# 搜索法定阶
def SARIMA_search(data):
    p = q = range(0, 3)
    s = [12]  # 周期为12
    d = [1]  # 做了一次季节性差分
    PDQs = list(itertools.product(p, d, q, s))  # itertools.product()得到的是可迭代对象的笛卡儿积
    pdq = list(itertools.product(p, d, q))  # list是python中是序列数据结构，序列中的每个元素都分配一个数字定位位置
    params = []
    seasonal_params = []
    results = []
    grid = pd.DataFrame()
    for param in pdq:
        for seasonal_param in PDQs:
            # 建立模型
            mod = sm.tsa.SARIMAX(data, order=param, seasonal_order=seasonal_param, \
                                 enforce_stationarity=False, enforce_invertibility=False)
            # 实现数据在模型中训练
            result = mod.fit()
            print("ARIMA{}x{}-AIC:{}".format(param, seasonal_param, result.aic))
            # format表示python格式化输出，使用{}代替%
            params.append(param)
            seasonal_params.append(seasonal_param)
            results.append(result.aic)
    grid["pdq"] = params
    grid["PDQs"] = seasonal_params
    grid["aic"] = results
    print(grid[grid["aic"] == grid["aic"].min()])

SARIMA_search(df_sku003_demand.dropna())

#建立模型
model=sm.tsa.SARIMAX(df_sku003_demand,order=(0,1,2),seasonal_order=(2,1,2,12))
SARIMA_m=model.fit()
print(SARIMA_m.summary())

#模型预测
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
#获取预测结果，自定义预测误差
def PredictionAnalysis(data,model,start,dynamic=False):
    pred=model.get_prediction(start=start,dynamic=dynamic,full_results=True)
    pci=pred.conf_int()#置信区间
    pm=pred.predicted_mean#预测值
    truth=data[start:]#真实值
    pc=pd.concat([truth,pm,pci],axis=1)#按列拼接
    pc.columns=['true','pred','up','low']#定义列索引
    print("1、MSE:{}".format(mse(truth,pm)))
    print("2、RMSE:{}".format(np.sqrt(mse(truth,pm))))
    print("3、MAE:{}".format(mae(truth,pm)))
    return pc


# 绘制预测结果
def PredictonPlot(pc):
    plt.figure(figsize=(10, 8))
    plt.fill_between(pc.index, pc['up'], pc['low'], color='grey', alpha=0.15, label='Confidence Interval')
    plt.plot(pc['true'], label='Actual Demand')
    plt.plot(pc['pred'], label='Predicted Demand')
    plt.legend()
    plt.title('Demand Forecast and Actual Demand')
    plt.xlabel('日期')
    plt.ylabel('需求量')
    plt.grid(True)
    plt.show()


# 预测未来
forecast = SARIMA_m.get_forecast(steps=60)
fig, ax = plt.subplots(figsize=(20, 16))
df_sku003_demand.plot(ax=ax, label="Historical Demand")
forecast.predicted_mean.plot(ax=ax, label="Forecasted Demand", color='r')
ax.fill_between(forecast.conf_int().index, forecast.conf_int().iloc[:, 0],
                forecast.conf_int().iloc[:, 1], color='grey', alpha=0.15, label='Confidence Interval')
ax.legend(loc="best", fontsize=20)
ax.set_xlabel("Date", fontsize=20)
ax.set_ylabel("Demand", fontsize=18)
plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.show()


'''

# 非季节性ACF和PACF
plot_acf(df['需求量'], lags=40)
plt.show()
plot_pacf(df['需求量'], lags=40)
plt.show()

'''