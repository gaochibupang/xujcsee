import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
data = pd.read_csv("E:/数据采集与预处理/数据采集与预处理/12_pandas案例/第二次/代码/data1_time.csv")
data = pd.DataFrame(data, columns=['start_time', 'last_time'])
print(data.columns)
# 特征缩放
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# DBSCAN参数搜索
num_random_search = 100
eps_values = np.linspace(0.01, 0.15, num_random_search)
min_samples_values = np.arange(1, 2)
best_silhouette = -np.inf
best_params = {'eps': None, 'min_samples': None}

for eps in eps_values:
    for min_samples in min_samples_values:
        # 使用当前参数进行DBSCAN聚类
        dbscan_result = DBSCAN(eps=eps, min_samples=min_samples).fit(data_scaled)
        cluster_labels = dbscan_result.labels_

        # 如果聚类结果不是全为-1（即存在非噪声点）
        if len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(data, cluster_labels)
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_params['eps'] = eps
                best_params['min_samples'] = min_samples

            # 输出最佳参数和轮廓系数
print("Best Parameters:", best_params)
print("Best Silhouette Score:", best_silhouette)

# 使用最佳参数进行聚类
final_dbscan = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples']).fit(data_scaled)
final_cluster_labels = final_dbscan.labels_

# 将缩放后的数据转换回 DataFrame 形式（但注意，这里的值已经是缩放后的，不适合直接解释）
data_scaled_df = pd.DataFrame(data_scaled, columns=['start_time', 'last_time'])

# 使用原始数据来计算红灯时长和周期
data_with_labels = pd.DataFrame(data, columns=['start_time', 'last_time'])
data_with_labels['cluster'] = final_cluster_labels

# 使用原始数据来可视化聚类结果
plt.figure(figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
sns.scatterplot(data=data_with_labels, x='start_time', y='last_time',
                hue=pd.Categorical.from_codes(final_cluster_labels, categories=np.unique(final_cluster_labels)),
                palette='rainbow')
plt.title("聚类结果散点图")
plt.xlabel("Start Time")
plt.ylabel("Last Time")
plt.show()


##################################### 红绿灯周期 #################################
# 找出每个簇中的时间点'last_time'（作为周期结束的代表）
# 因为每个簇代表一波车流
# 首先，我们创建一个字典来存储每个簇的 'last_time' 的最大值和最小值
cluster_last_time_extremes = {}
for cluster in np.unique(final_cluster_labels):
    if cluster != -1:
        cluster_data = data_with_labels[data_with_labels['cluster'] == cluster]
        cluster_last_time_extremes[cluster] = {'min': cluster_data['last_time'].min(),
                                               'max': cluster_data['last_time'].max()}
        # cluster_last_times[cluster] = cluster_data['last_time'].median()  # 选择2：使用中位数作为代表

    # 计算簇与簇之间'last_time'的差（作为周期），使用第二个簇的最小值减去第一个簇的最大值
periods = []
clusters = sorted(cluster_last_time_extremes.keys())
for i in range(len(clusters) - 1):
    diff = cluster_last_time_extremes[clusters[i + 1]]['min'] - cluster_last_time_extremes[clusters[i]]['max']
    # diff = cluster_last_times[clusters[i + 1]] - cluster_last_times[clusters[i]]  # 选择2
    periods.append(diff)

# 绘制周期直方图
plt.figure(figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.hist(periods, bins='auto', alpha=0.7, edgecolor='black')
plt.xlabel('周期')
plt.ylabel('频率')
plt.title('周期直方图')
plt.grid(axis='y', alpha=0.75)
plt.show()


############################################## 红灯时间 ##########################################
"""方法2：等待时间不可能超过一个红绿灯周期，剔除超过红绿灯周期的红灯时间数据，计算均值作为红灯时间"""
# 读取数据（注意：这里假设文件是Excel格式，且列名正确）
df = pd.read_csv("E:/数据采集与预处理/数据采集与预处理/12_pandas案例/第二次/代码/data1_time.csv")
df = pd.DataFrame(df, columns=['time_diff'])

diff_all = 105  # 运行所得周期：104,95,105,88,88

# 筛选出'time_diff'小于等于红绿灯周期的数据
filtered_data = df[df['time_diff'] <= diff_all]

# 计算筛选后数据的均值
mean_value = filtered_data['time_diff'].mean()
print("红灯时间：", mean_value)

green = diff_all - mean_value
print("绿灯时间：", green)

# 结果：
"A1: 红灯时间：72, 绿灯时间：31"
"A2: 红灯时间：71, 绿灯时间：24"
"A3: 红灯时间：73, 绿灯时间：32"
"A4: 红灯时间：65, 绿灯时间：23"
"A5: 红灯时间：70, 绿灯时间：18"