import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 读取csv文件
address = r'E:/数据采集与预处理/数据采集与预处理/12_pandas案例/第二次/代码/D4.csv'  # 改变C1,C2,C3,C4,C5，C6即可
data = pd.read_csv(address, encoding='gbk')

# ********************** 计算速度 时间 （文件01_2）***********************
# 获取所有独特的车辆 ID
ID = data['vehicle_id'].unique()

#  步骤一：计算车辆速度
for current_vehicle in ID:
    # 提取当前车辆的数据
    vehicle_data = data[data['vehicle_id'] == current_vehicle]

    # 对时间进行排序（确保数据按时间顺序）
    vehicle_data = vehicle_data.sort_values(by='time')

    # 提取 x, y 坐标和时间
    t = vehicle_data['time'].values
    x = vehicle_data['x'].values
    y = vehicle_data['y'].values

    # 计算空间距离差
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)

    # 计算速度
    v = np.sqrt(dx ** 2 + dy ** 2) / dt

    # 将速度添加到数据集中
    data.loc[data['vehicle_id'] == current_vehicle, 'speed'] = np.concatenate(([0], v))


# 步骤2: 停车检测
speed_threshold = 1  # 速度阈值，单位m/s
start_stop_times = {}  # 存储开始停止的时间点
last_stop_times = {}  # 存储停车的最后一个时间点（速度为0的时刻）

for current_vehicle in ID:
    vehicle_data = data[data['vehicle_id'] == current_vehicle]
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
eps_values = np.linspace(0.01, 0.15, num_random_search)
min_samples_values = np.arange(1, 10)
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

# 使用原始数据来可视化聚类结果略


###################################### 红灯时长 ##################################################
# 计算每个簇的红灯时长
red_light_durations = {}
for cluster in np.unique(final_cluster_labels):
    if cluster != -1:  # 忽略噪声点
        cluster_data = data_with_labels[data_with_labels['cluster'] == cluster]
        red_light_durations[cluster] = cluster_data['last_time'].subtract(cluster_data['start_time']).mean()


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


##################################### 滚动平均 ###############################
# 应用滚动平均
rolling_window = 5  # 可以根据数据调整窗口大小
rolling_mean_periods = pd.Series(periods).rolling(window=rolling_window).mean()

# 可视化滚动平均
plt.figure(figsize=(10, 6))
plt.plot(periods, marker='o', label='Periods')
plt.plot(rolling_mean_periods, label='Rolling Mean of Periods')
plt.xlabel('Index')
plt.ylabel('Period Duration')
plt.title('Rolling Mean of Traffic Light Cycles')
plt.legend()
plt.grid(True)
plt.show()

########################################### CUMSUM算法检测突变点 ########################
# 计算累积和
cumulative_sum = np.cumsum(periods - np.mean(periods))

# 设定阈值
threshold = 2 * np.std(cumulative_sum)

# 检测突变点
change_points = np.where(np.abs(cumulative_sum) > threshold)[0]

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(periods, marker='o', label='Periods')
plt.plot(cumulative_sum, label='Cumulative Sum', linestyle='--')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.axhline(y=-threshold, color='r', linestyle='--')
for cp in change_points:
    plt.axvline(x=cp, color='r', linestyle=':', label='Change Point' if cp == change_points[0] else '')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Traffic Light Cycles with Change Points Detected by CUMSUM')
plt.legend()
plt.grid(True)
plt.show()

# 获取突变点对应的前一个簇的索引
previous_cluster_indices = [clusters[cp - 1] if cp > 0 else None for cp in change_points]
# 过滤掉第一个突变点（如果它是列表中的第一个点），因为它没有前一个簇
previous_cluster_indices = [idx for idx in previous_cluster_indices if idx is not None]
# 获取这些簇的'last_time'
change_point_times = [cluster_last_time_extremes[idx]['max'] for idx in previous_cluster_indices]
# 输出突变点所在的时间点
print("周期切换时刻:", change_point_times)

'''接下来：
根据突变点分割周期时长数据：使用突变点将周期时长数据（periods）分割成多个子列表。
计算每个部分的平均周期时长：对分割后的每个子列表计算平均值。
计算每个部分的红灯时间平均时长：对于每个部分，首先根据突变点的时间找到对应的簇，然后计算这些簇中红灯时间的中位数。'''

# 根据突变点分割周期时长数据
split_indices = [0] + list(change_points) + [len(periods) - 1]
periods_segments = [periods[split_indices[i]:split_indices[i + 1]]
                    for i in range(len(split_indices) - 1)]

# 打印分割后的周期时长数据
print('\n分割后每部分的周期时长数据:')
for i, segment in enumerate(periods_segments):
    print(f"Segment {i + 1}: {segment}")

'''观察每个Segment的周期时长，发现周期时长数值上呈现较明显的倍数关系，认为由于车流量较稀疏，连续相邻的几个信
号灯周期内均无车辆通过路口，从而导致估计出的某一周期内的红灯和绿灯总时长较长。因此每个Segment取最小周期作为
这部分的周期时长。若最小周期时长大于200,则除其对于100的倍数。'''

print('\n每部分周期时长：')
# 每个部分的最小周期时长
min_periods = [np.min(segment) if segment else np.nan for segment in periods_segments]
# 打印结果
for i, min_period in enumerate(min_periods):
    if min_period > 600:
        min_period = min_period / 6
    elif min_period > 500:
        min_period = min_period / 5
    elif min_period > 400:
        min_period = min_period / 4
    elif min_period > 300:
        min_period = min_period / 3
    elif min_period > 200:
        min_period = min_period / 2
    elif min_period > 150:
        min_period = min_period / 2
    print(f"Segment {i + 1}的周期时长: {min_period}")

############################### 计算每个部分的红灯时间 ############################################
# 映射突变点到簇索引
change_point_cluster_indices = [clusters[cp - 1] for cp in change_points]

# 计算每个部分的红灯时间
median_red_light_durations = []
for i in range(len(periods_segments)):
    if i == 0:
        # 第一个部分，从第一个簇开始到第一个突变点之前的簇
        start_cluster = clusters[0]
        end_cluster = change_point_cluster_indices[0]
    elif i == len(periods_segments) - 1:
        # 最后一个部分，从最后一个突变点之后的簇到最后一个簇
        start_cluster = change_point_cluster_indices[-1]
        end_cluster = clusters[-1]
    else:
        # 中间部分，从当前突变点之前的簇到下一个突变点之前的簇
        start_cluster = change_point_cluster_indices[i - 1]
        end_cluster = change_point_cluster_indices[i]

        # 提取该部分的簇
    clusters_in_segment = range(start_cluster, end_cluster + 1)

    # 计算红灯时间
    segment_red_light_durations = [red_light_durations[cluster] for cluster in clusters_in_segment if
                                   cluster in red_light_durations]
    if segment_red_light_durations:
        median_red_light_duration = np.median(segment_red_light_durations) # 取中位数
        if median_red_light_duration > 100:
            median_red_light_duration = np.min(segment_red_light_durations)
    else:
        median_red_light_duration = np.nan

    median_red_light_durations.append(median_red_light_duration)

# 打印结果
print('\n红灯时长')
for i, median_red_light in enumerate(median_red_light_durations):
    print(f"Segment {i + 1}的红灯时长: {median_red_light}")


