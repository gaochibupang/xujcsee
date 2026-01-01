import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist

# 读取csv文件
address = r'E:/数据采集与预处理/数据采集与预处理/12_pandas案例/第二次/附件/附件3/C1.csv'
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
            filtered_data.append((last_time))

####################################### 计算红绿灯周期变换的时刻 ################################
data = pd.DataFrame(filtered_data, columns=['last_time'])

# 特征缩放
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
dbscan = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples']).fit(data)
clusters = dbscan.labels_

# 将缩放后的数据转换回原始形式
data = scaler.inverse_transform(data)

# 计算每一簇之间的距离作为红绿灯周期
unique_clusters = np.unique(clusters)
periods = []
transition_times = []

for cluster in unique_clusters:
    if cluster != -1:
        cluster_data = data[clusters == cluster]
        distances = pdist(cluster_data, metric='euclidean')
        period = np.nanmean(distances) if len(distances) > 0 else np.nan
        periods.append(period)
        transition_time = cluster_data[-1][-1] + period
        transition_times.append(transition_time)


# 计算红绿灯周期
red_green_light_periods = np.diff(transition_times)
# 去掉红绿灯周期中的空值
red_green_light_periods = red_green_light_periods[~np.isnan(red_green_light_periods)]
print("所有红绿灯周期：", red_green_light_periods)




print("红绿灯周期变换的时刻：", transition_times)

# 去掉nan值
transition_times = [x for x in transition_times if not np.isnan(x)]

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(transition_times, marker='o', linestyle='-')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置字体为黑体
plt.title('红绿灯周期变换的时刻')
plt.xlabel('序号')
plt.ylabel('时间')
plt.grid(True)
plt.xticks(range(len(transition_times)), labels=[f'Cycle {i+1}' for i in range(len(transition_times))], rotation=45)
plt.tight_layout()
plt.show()


# 求红绿灯周期的中位数
median_period = np.median(red_green_light_periods)
print("红绿灯周期的中位数：", median_period)