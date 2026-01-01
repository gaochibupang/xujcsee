import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 读取csv文件
address = r'E:/数据采集与预处理/数据采集与预处理/12_pandas案例/第二次/代码/v_low.csv'
v_low = pd.read_csv(address, encoding='gbk')

# 获取所有独特的车辆 ID
ID = v_low['vehicle_id'].unique()

######################################### 重复问题1步骤 ######################################
# 停车检测
speed_threshold = 1  # 速度阈值，单位m/s
start_stop_times = {}  # 存储开始停止的时间点
last_stop_times = {}  # 存储停车的最后一个时间点（速度为0的时刻）

for current_vehicle in ID:
    vehicle_data = v_low[v_low['vehicle_id'] == current_vehicle]
    low_speed_indices = np.where(vehicle_data['speed'] <= speed_threshold)[0]

    if len(low_speed_indices) > 0:
        # 开始停止的时间点是速度首次降至或低于阈值的时刻
        start_stop_time = vehicle_data.iloc[low_speed_indices[0]]['time']

        # 查找最后一个速度为0的索引
        zero_speed_indices = np.where(vehicle_data['speed'] == 0)[0]
        # 筛选出低速行驶期间内速度为0的索引
        filtered_zero_speed_indices = zero_speed_indices[
            (zero_speed_indices >= low_speed_indices[0]) & (zero_speed_indices <= low_speed_indices[-1])]

        if len(filtered_zero_speed_indices) > 0:
            # 如果有速度为0的时刻，取最后一个
            last_stop_time = vehicle_data.iloc[filtered_zero_speed_indices[-1]]['time']
        else:
            # 如果没有速度为0的时刻，取特殊值来表示没有停车
            last_stop_time = None

        # 存储结果
        start_stop_times[current_vehicle] = start_stop_time
        last_stop_times[current_vehicle] = last_stop_time

# 步骤3: 筛选有效的停车数据
min_stop_duration = 10  # 最小停车时长，单位秒
filtered_data = []
for vehicle_id, start_time in start_stop_times.items():
    last_time = last_stop_times.get(vehicle_id)
    # 输出结果
    #print(f"Vehicle {vehicle_id}: Started stopping at {start_time}, last stop (speed=0) time at {last_time}")
    # 筛选有效的停车数据，剔除停车时长<10的数据
    if last_time is not None:
        # 计算时间差
        time_diff = last_time - start_time
        if time_diff >= min_stop_duration:
            # 将时间差（或其他特征）添加到列表中
            filtered_data.append((start_time, last_time))


# ******************************* 计算红灯时间及周期 （文件01_3）******************
# 读取数据
data = pd.DataFrame(filtered_data, columns=['start_time', 'last_time'])
# 特征缩放
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# DBSCAN参数搜索
num_random_search = 100
eps_values = np.linspace(0.01, 4, num_random_search)
min_samples_values = np.arange(2, 10)
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
print("最佳参数和轮廓系数:", best_params)
print("Best Silhouette Score:", best_silhouette)

# 使用最佳参数进行聚类
final_dbscan = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples']).fit(data_scaled)
final_cluster_labels = final_dbscan.labels_

# 将缩放后的数据转换回 DataFrame 形式（但注意，这里的值已经是缩放后的，不适合直接解释）
data_scaled_df = pd.DataFrame(data_scaled, columns=['start_time', 'last_time'])

# 使用原始数据来计算红灯时长和周期
data_with_labels = pd.DataFrame(data, columns=['start_time', 'last_time'])
data_with_labels['cluster'] = final_cluster_labels


###################################### 红灯时长 ##################################################
# 计算每个簇的红灯时长平均值
red_light_durations = {}
for cluster in np.unique(final_cluster_labels):
    if cluster != -1:  # 忽略噪声点
        cluster_data = data_with_labels[data_with_labels['cluster'] == cluster]
        red_light_durations[cluster] = cluster_data['last_time'].subtract(cluster_data['start_time']).mean()

# 计算过滤后簇的红灯时长中位数
average_red_light_duration = np.median(list(red_light_durations.values()))
print("红灯时间：", average_red_light_duration)

'''效果很差
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

# 使用 numpy 的 histogram 函数找到频率最高的周期
counts, bin_edges = np.histogram(periods, bins='auto')
max_index = np.argmax(counts)  # 找到计数最多的bin的索引
most_common_period_bin = 0.5 * (bin_edges[max_index] + bin_edges[max_index + 1])  # 计算该bin的中心点作为周期的代表
print(f"红绿灯周期: {most_common_period_bin}")  # 最常见的周期 bin 相对应的中位数
'''

# 计算红灯时间相对误差
original = 71  # 74, 72, 90, 75, 71

relative_error = (abs(average_red_light_duration - original) / original)
print('\n低流量数据红灯时间的相对误差为：', relative_error)


# 结果：
# A1原本红灯时间：74
"低流量: 红灯时间：81.125, 相对误差为： 0.09628378378378379"
"中流量: 红灯时间：77.708, 相对误差为： 0.05011261261261255"
"高流量: 红灯时间：63.75, 相对误差为： 0.13851351351351351"

# A2原本红灯时间：72
"低流量: 红灯时间：80, 相对误差为： 0.1111111111111111"
"中流量: 红灯时间：72.846, 相对误差为： 0.011752136752136662"
"高流量: 红灯时间：48, 相对误差为： 0.3333333333333333"

# A3原本红灯时间：90
"低流量: 红灯时间：78.666, 相对误差为： 0.12592592592592602"
"中流量: 红灯时间：81.604, 相对误差为： 0.09328845369237039"
"高流量: 红灯时间：71.666, 相对误差为： 0.20370370370370366"

# A4原本红灯时间：75
"低流量: 红灯时间：76, 相对误差为： 0.013333333333333334"
"中流量: 红灯时间：77.329, 相对误差为： 0.031058201058201007"
"高流量: 红灯时间：78, 相对误差为： 0.04"

# A5原本红灯时间：71
"低流量: 红灯时间：102.333, 相对误差为： 0.44131455399061026"
"中流量: 红灯时间：80.381, 相对误差为： 0.13212147887323958"
"高流量: 红灯时间：78.5, 相对误差为： 0.1056338028169014"