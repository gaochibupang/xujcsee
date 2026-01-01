import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据（注意：这里假设文件是Excel格式，且列名正确）  
data = pd.read_csv("E:/数据采集与预处理/数据采集与预处理/12_pandas案例/第二次/代码/data1_time.csv")

# 数据预处理：删除缺失值，并假设列名已经正确设置为Time_point1, Time_point2, Time_point3  

data = pd.DataFrame(data, columns=['start_time', 'last_time'])
print(data.columns)
# 特征缩放（注意：在Python中，我们通常对整个DataFrame进行缩放，而不是单独对每列）  
scaler = StandardScaler()
data = scaler.fit_transform(data)

# DBSCAN参数搜索  
num_random_search = 100
eps_values = np.linspace(0.01, 0.2, num_random_search)
min_samples_values = np.arange(1, 11)
best_silhouette = -np.inf
best_params = {'eps': None, 'min_samples': None}

for eps in eps_values:
    for min_samples in min_samples_values:
        # 使用当前参数进行DBSCAN聚类  
        dbscan_result = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
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
final_dbscan = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples']).fit(data)
final_cluster_labels = final_dbscan.labels_

data = pd.DataFrame(data, columns=['start_time', 'last_time'])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='start_time', y='last_time',
                hue=pd.Categorical.from_codes(final_cluster_labels, categories=np.unique(final_cluster_labels)),
                palette='rainbow')
plt.title("DBSCAN Clustering Result of Al")
plt.xlabel("start_time")
plt.ylabel("last_time")
plt.show()

# 将缩放后的数据转换回原始形式
data_original = scaler.inverse_transform(data)

# 创建一个DataFrame，包含原始数据和聚类标签
data_with_labels = pd.DataFrame(data_original, columns=['start_time', 'last_time'])
data_with_labels['cluster'] = final_cluster_labels

# 计算每个簇的红灯时长
red_light_durations = []
for cluster in data_with_labels['cluster'].unique():
    if cluster != -1:  # 忽略噪声点
        cluster_data = data_with_labels[data_with_labels['cluster'] == cluster]
        min_start_time = cluster_data['start_time'].mean()
        max_last_time = cluster_data['last_time'].mean()
        red_light_duration = max_last_time - min_start_time  # 通过取最大结束时间和最小开始时间的差值，我们可以确保得到的结果是最长的红灯时长
        red_light_durations.append(red_light_duration)

# 输出每个簇的红灯时长
print("Red Light Durations for each cluster:", red_light_durations)


# 画出红灯时长直方图
plt.figure(figsize=(10, 6))
plt.hist(red_light_durations, bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of Red Light Durations")
plt.xlabel("Red Light Duration")
plt.ylabel("Frequency")
plt.show()

# 计算平均红灯时长
avg_red_light_duration = np.mean(red_light_durations)
print("平均红灯时长：", avg_red_light_duration)


