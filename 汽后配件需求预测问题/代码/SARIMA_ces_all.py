import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from tslearn.clustering import KShape
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.preprocessing import MinMaxScaler

os.environ["OMP_NUM_THREADS"] = "3"  # 设置环境变量以避免内存泄漏

# 读取csv文件
address = r'D:/数学建模/数学建模/代码/第三次/历史配件订单表.csv'
data = pd.read_csv(address, encoding='gbk')

# 对当天该仓库对该配件的需求量进行加和处理
data['日期'] = pd.to_datetime(data['日期'])
df = data.groupby(['仓库编码', '配件编码', pd.Grouper(key='日期', freq='D')])['需求量'].sum().reset_index()

# 设定索引为日期和配件编码
df.set_index(['日期', '配件编码'], inplace=True)

# 数据透视以获取每个配件每天的需求量，并用1填充缺失值
demand_pivot = df.pivot_table(index='日期', columns='配件编码', values='需求量').fillna(1)


def replace_outliers_with_median(df, threshold=3):
    """
    用中位数替代DataFrame中每列的异常值。

    参数:
    df : DataFrame
        输入的数据框架。
    threshold : float
        用于计算异常值的标准差倍数阈值。

    返回:
    DataFrame
        异常值被中位数替代后的DataFrame。
    """
    # 计算每列的中位数
    median = df.median()

    # 复制原始DataFrame以避免直接修改
    df_clean = df.copy()

    # 遍历每一列
    for col in df_clean.columns:
        # 计算当前列的平均值和标准差
        col_mean = df_clean[col].mean()
        col_std = df_clean[col].std()

        # 标记为True的位置是异常值
        is_outlier = np.abs(df_clean[col] - col_mean) >= (threshold * col_std)

        # 用中位数替代异常值
        df_clean.loc[is_outlier, col] = median[col]

    return df_clean

# 调用函数替换异常值
demand_pivot = replace_outliers_with_median(demand_pivot)
print(demand_pivot)

# 将demand_pivot存进csv文件
demand_pivot.to_csv('demand_pivot.csv', encoding='utf_8_sig')


# 由于轮廓系数可能不适用于时间序列数据，我们将尝试使用它，但请注意其局限性
def calculate_silhouette_scores(X, max_clusters=10):
    silhouette_avg = []
    for n_clusters in range(2, max_clusters + 1):
        clusterer = KShape(n_clusters=n_clusters, verbose=False)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg.append(silhouette_score(X, cluster_labels, metric='sqeuclidean'))
    return silhouette_avg


# 将 DataFrame 转换为 numpy 数组，每个时间序列是一行
X = demand_pivot.values

# 计算轮廓系数
silhouette_scores = calculate_silhouette_scores(X, max_clusters=10)  # 尝试 2 到 10 个聚类

# 绘制轮廓系数图
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Coefficient for Different Number of Clusters')
plt.grid(True)
plt.show()

# 选择最佳聚类数（轮廓系数最高的那个）
best_n_clusters = np.argmax(silhouette_scores) + 2  # 因为 range 是从 2 开始的
print(f"Best number of clusters: {best_n_clusters}")

# 使用最佳聚类数重新聚类
model = KShape(n_clusters=best_n_clusters, verbose=True)
y_pred = model.fit_predict(X)


# 创建一个字典来存储每个聚类的数据
clusters_data = {}
for i in range(best_n_clusters):
    clusters_data[i] = demand_pivot.iloc[y_pred == i]

# 绘制每个聚类的时间序列图
plt.figure(figsize=(17, 12))
for i, (cluster_id, cluster_df) in enumerate(clusters_data.items()):
    plt.subplot(best_n_clusters, 1, i + 1)  # 调整子图的位置
    for col in cluster_df.columns:
        plt.plot(cluster_df.index, cluster_df[col], label=col)
    plt.title(f'Cluster {cluster_id + 1}')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1, 0.95), loc='upper left')  # 将图例放在外面
plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
plt.show()


# 每个聚类中随机选择的配件
print("Random Accessories per Cluster:")
for i, (cluster_id, cluster_df) in enumerate(clusters_data.items()):
    random_column = random.choice(list(cluster_df.columns))
    print(f"Cluster {cluster_id + 1}: {random_column}")

'''
Cluster 1: sku127
最佳pdq: 1 0 1
最佳PDQs: 0 1 1 30
WMAPE: 0.61%
SMAPE: 49.35%
最终WMAPE: 0.47%
最终SMAPE: 33.96%

Cluster 2: sku216
最佳pdq: 1 0 2
最佳PDQs: 0 1 2 7
WMAPE: 0.42%
SMAPE: 36.99%
最终WMAPE: 0.23%
最终SMAPE: 22.99%
放入SARIMA模型训练，得到最佳参数'''



# 假设 clusters_data 已经被正确分割，且每个聚类包含时间序列数据
# 这里以第一个聚类为例，进行模型训练和预测
# 初始化一个空的DataFrame来收集所有SKU的预测结果
all_predictions_df = pd.DataFrame()
# 第一个聚类数据
cluster_0_data = clusters_data[0]

# 假设每个时间序列数据都以 '配件编码' 作为列名
for sku in cluster_0_data.columns:
    # 提取时间序列数据
    ts_data = cluster_0_data[sku]

    # 转换为适合SARIMA模型的时间序列格式
    ts_data.index = pd.to_datetime(ts_data.index)
    ts_data.name = sku

    # 划分训练集和测试集
    start_date = pd.to_datetime('2022-01-01')
    end_date = pd.to_datetime('2023-07-31')
    # 使用前一个日期的值填充缺失的日期
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    ts_data = ts_data.reindex(all_dates, method='ffill')
    ts_data = ts_data.fillna(1)
    split_date = end_date - pd.offsets.MonthBegin(1)
    train_data = ts_data.loc[:split_date - pd.Timedelta(days=1)]
    test_data = ts_data.loc[split_date:]

    # SARIMA模型参数
    # 第一个聚类最佳pdq: 1 0 1, 最佳PDQs: 0 1 1 30
    order = (1, 0, 1)
    seasonal_order = (0, 1, 1, 30)

    # 建立SARIMA模型
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    # 预测未来31天
    forecast = results.get_forecast(steps=31)
    future_dates = pd.date_range(start=split_date + pd.Timedelta(days=1), periods=31, freq='D')
    future_demand = forecast.predicted_mean

    # 计算残差并训练SVR模型
    residuals = test_data - results.predict(start=split_date, end=test_data.index[-1], dynamic=False)
    X_train = np.arange(len(residuals)).reshape(-1, 1)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
    svr_model.fit(X_train_scaled, residuals)

    # 预测未来31天的残差
    X_future = np.arange(len(future_demand)).reshape(-1, 1)
    X_future_scaled = scaler.transform(X_future)
    predicted_residuals = svr_model.predict(X_future_scaled)

    # 结合预测结果和残差
    final_future_predictions = future_demand + predicted_residuals

    # 绘制结果图（可选）
    plt.figure(figsize=(12, 6))
    plt.plot(ts_data.index, ts_data, label='历史需求量')
    plt.plot(future_dates, final_future_predictions, label='未来31天预测需求量', color='green')
    plt.title(f'SKU {sku} 历史与未来31天需求量预测')
    plt.xlabel('日期')
    plt.ylabel('需求量')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    #plt.show()

    # 将当前SKU的预测结果保存到DataFrame中
    sku_predictions_df = pd.DataFrame(final_future_predictions, index=future_dates, columns=['预测需求量'])
    sku_predictions_df['SKU'] = sku  # 添加SKU列以标识这些预测属于哪个SKU

    # 将当前SKU的预测结果追加到all_predictions_df中
    all_predictions_df = pd.concat([all_predictions_df, sku_predictions_df], ignore_index=True)

    # 现在，all_predictions_df包含了所有SKU的预测结果
# 保存到CSV文件
all_predictions_df.to_csv('all_predictions.csv', index=False)

# 第二个聚类
# 初始化一个空的DataFrame来收集所有SKU的预测结果
all_predictions_df2 = pd.DataFrame()
# 第2个聚类数据
cluster_1_data = clusters_data[1]

# 假设每个时间序列数据都以 '配件编码' 作为列名
for sku in cluster_1_data.columns:
    # 提取时间序列数据
    ts_data = cluster_1_data[sku]

    # 转换为适合SARIMA模型的时间序列格式
    ts_data.index = pd.to_datetime(ts_data.index)
    ts_data.name = sku

    # 划分训练集和测试集
    start_date = pd.to_datetime('2022-01-01')
    end_date = pd.to_datetime('2023-07-31')
    # 使用前一个日期的值填充缺失的日期
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    ts_data = ts_data.reindex(all_dates, method='ffill')
    ts_data = ts_data.fillna(1)
    split_date = end_date - pd.offsets.MonthBegin(1)
    train_data = ts_data.loc[:split_date - pd.Timedelta(days=1)]
    test_data = ts_data.loc[split_date:]

    # SARIMA模型参数
    # 第一个聚类最佳pdq: 1 0 1, 最佳PDQs: 0 1 1 30
    order = (1, 0, 2)
    seasonal_order = (0, 1, 2, 7)

    # 建立SARIMA模型
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    # 预测未来31天
    forecast = results.get_forecast(steps=31)
    future_dates = pd.date_range(start=split_date + pd.Timedelta(days=1), periods=31, freq='D')
    future_demand = forecast.predicted_mean

    # 计算残差并训练SVR模型
    residuals = test_data - results.predict(start=split_date, end=test_data.index[-1], dynamic=False)
    X_train = np.arange(len(residuals)).reshape(-1, 1)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
    svr_model.fit(X_train_scaled, residuals)

    # 预测未来31天的残差
    X_future = np.arange(len(future_demand)).reshape(-1, 1)
    X_future_scaled = scaler.transform(X_future)
    predicted_residuals = svr_model.predict(X_future_scaled)

    # 结合预测结果和残差
    final_future_predictions = future_demand + predicted_residuals

    # 绘制结果图（可选）
    plt.figure(figsize=(12, 6))
    plt.plot(ts_data.index, ts_data, label='历史需求量')
    plt.plot(future_dates, final_future_predictions, label='未来31天预测需求量', color='green')
    plt.title(f'SKU {sku} 历史与未来31天需求量预测')
    plt.xlabel('日期')
    plt.ylabel('需求量')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    #plt.show()

    # 将当前SKU的预测结果保存到DataFrame中
    sku_predictions_df2 = pd.DataFrame(final_future_predictions, index=future_dates, columns=['预测需求量'])
    sku_predictions_df2['SKU'] = sku  # 添加SKU列以标识这些预测属于哪个SKU

    # 将当前SKU的预测结果追加到all_predictions_df中
    all_predictions_df2 = pd.concat([all_predictions_df2, sku_predictions_df2], ignore_index=True)

    # 现在，all_predictions_df包含了所有SKU的预测结果
# 保存到CSV文件
all_predictions_df2.to_csv('all_predictions2.csv', index=False)