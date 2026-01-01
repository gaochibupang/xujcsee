import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import periodogram
from statsmodels.tsa.stattools import adfuller as ADF
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 读取csv文件
address = r'D:/数学建模/数学建模/代码/第三次/历史配件订单表.csv'
data = pd.read_csv(address, encoding='gbk')

# 查看缺失值比例
print("查看缺失值比例：\n", data.isnull().sum()/len(data))
'''无缺失值'''

# 对当天该仓库对该配件的需求量进行加和处理
data['日期'] = pd.to_datetime(data['日期'])
df = data.groupby(['仓库编码', '配件编码', pd.Grouper(key='日期', freq='D')])['需求量'].sum().reset_index()
df.set_index('日期', inplace=True)

# 1、绘制所有配件的需求量叠加图
pivot_df = df.pivot(columns='配件编码', values='需求量')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False   # 正确显示负号
pivot_df.plot(figsize=(10, 5), title='所有配件的需求量叠加图')
plt.xlabel('日期')
plt.ylabel('需求量')
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
plt.legend(title='配件编码')
plt.show()

# 假设我们先关注一个特定的配件sku108
df_sku008 = df[df['配件编码'] == 'sku216']

# 提取要预测的列
df_sku008_demand = df_sku008[['需求量']]
df_sku008_demand.index = pd.to_datetime(df_sku008_demand.index)

# 获取数据集的起始和结束日期
start_date = df_sku008_demand.index.min()
end_date = df_sku008_demand.index.max()

# 用0填充缺失的日期
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
df_sku008_demand = df_sku008_demand.reindex(all_dates, fill_value=0)

# 2、使用periodogram来检测季节性
freq, spec = periodogram(df_sku008_demand['需求量'])

# 找到频谱中的最大值对应的频率，这可以指示数据的季节性周期
max_freq = freq[spec.argmax()]
print(f"Detected seasonal period: {1 / max_freq} (if data is sampled daily)")

# 也可以使用seasonal_decompose来分解时间序列并观察季节性成分
result = seasonal_decompose(df_sku008_demand['需求量'], model='additive', period=int(1 / max_freq))
result.seasonal.plot()
plt.title('季节性成分观察')
plt.show()

################################### 具有季节性，使用SARIMA模型

# 3、平稳性检验
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
test_stationarity(df_sku008_demand,1e-3)

################### 平稳时间序列

# 分解时序
# STL（Seasonal and Trend decomposition using Loess）是一个非常通用和稳健强硬的分解时间序列的方法
decompostion=sm.tsa.STL(df_sku008_demand).fit()#statsmodels.tsa.api:时间序列模型和方法
decompostion.plot()
# 趋势效益
trend=decompostion.trend
# 季节效应
seasonal=decompostion.seasonal
# 随机效应
residual=decompostion.resid
plt.show()