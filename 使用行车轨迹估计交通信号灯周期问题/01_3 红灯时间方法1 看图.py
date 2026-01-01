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

###################################### 红灯时长 ##################################################
# 计算每个簇的红灯时长平均值
red_light_durations = {}
for cluster in np.unique(final_cluster_labels):
    if cluster != -1:  # 忽略噪声点
        cluster_data = data_with_labels[data_with_labels['cluster'] == cluster]
        red_light_durations[cluster] = cluster_data['last_time'].subtract(cluster_data['start_time']).mean()

    # 输出每个簇的红灯时长
print("Red Light Durations for each cluster:", red_light_durations)

# 绘制红灯时长直方图
plt.figure(figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
cluster_durations = list(red_light_durations.values())
plt.hist(cluster_durations, bins='auto', alpha=0.7, edgecolor='black')
plt.xlabel('红灯时长')
plt.ylabel('频率')
plt.title('红灯时长直方图')
plt.grid(axis='y', alpha=0.75)
plt.show()

"""方法1：看图：频率最高的数据的中位数，选择78作为红灯时间"""

"""方法2：剔除超过红绿灯周期的红灯时间数据，计算均值作为红灯时间"""

"""方法......"""

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

"""   ...... A1周期取104,频率最高的数据的中位数   """
"""   计算得A1绿灯时间为104-74=30   """

"""A2: 95-72=23"""       # 频率最高的数据的中位数
"""A3: 105-90=15"""      # 频率最高的数据的中位数
"""A4: 88-75=13"""       # 综合考虑，红灯时间必定小于周期，所以红灯时间取频率第二且红灯时间小于周期的数据的中位数
"""A5: 88-71=17"""       # 综合考虑，红灯时间必定小于周期，所以红灯时间取频率第二且红灯时间小于周期的数据的中位数

