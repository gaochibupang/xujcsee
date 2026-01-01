import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

"""定位误差为最小影响因素"""

# 读取csv文件
address = r'E:/数据采集与预处理/数据采集与预处理/12_pandas案例/第二次/附件/附件2/B2.csv'
data = pd.read_csv(address, encoding='gbk')

########################################### 1、IQR离群点检测 ###########################################
# 筛选需要处理的列
cols = ['x', 'y']

# 初始化离群点计数Series
outlier_counts_before = pd.Series(index=cols, dtype=int)
outlier_counts_after = pd.Series(index=cols, dtype=int)

# 处理离群点
for col in cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 识别离群点
    outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
    outlier_counts_before[col] = outliers.sum()

    # 剔除离群点
    data = data[~outliers]

    # 再次检查离群点
    outliers_after = (data[col] < lower_bound) | (data[col] > upper_bound)
    outlier_counts_after[col] = outliers_after.sum()

# 打印原始和处理后的异常值计数
print("原始离群点计数:", outlier_counts_before)
print("处理后离群点计数:", outlier_counts_after)

# data.to_csv('dingweiwucha.csv', index=False)

######################################## 2、数据插值和平滑 #############################################
# 查看缺失值比例
print("查看缺失值比例：\n", data.isnull().sum()/len(data))
'''无缺失值'''
# 如果存在，用插值填补缺失值
data_interpolated = data.ffill()  # 使用前一个值填充

# 平滑数据（使用移动平均）
window_size = 6
data_smoothed = data_interpolated.copy()
data_smoothed['x_smoothed'] = data_interpolated['x'].rolling(window=window_size, min_periods=1).mean()
data_smoothed['y_smoothed'] = data_interpolated['y'].rolling(window=window_size, min_periods=1).mean()

# 剔除离群点后的轨迹图
df = pd.DataFrame(data_smoothed, columns=['time', 'vehicle_id', 'x', 'y'])  # 创建DataFrame，并指定列名
fig, ax = plt.subplots()  # 创建一个新的图形和坐标轴
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置字体

# 遍历每个车辆ID，绘制轨迹
for vehicle_id, group in df.groupby('vehicle_id'):
    ax.plot(group['x'], group['y'], label=f'Vehicle {vehicle_id}')

ax.legend()  # 添加图例
ax.set_xlabel('X Position')  # 设置坐标轴标签
ax.set_ylabel('Y Position')
ax.set_title('剔除离群点后所有车辆轨迹图')  # 设置标题
plt.show()  # 显示图形

df.to_csv('df.csv', index=False)
data_smoothed.to_csv('data_smoothed.csv', index=False)

###################### 3、使用平滑后的数据计算红灯周期及信号灯周期（重复问题1中的模型） #######################

# ********************** 计算速度 时间 （文件01_2）***********************
# 获取所有独特的车辆 ID
ID = data_smoothed['vehicle_id'].unique()

#  步骤一：计算车辆速度
for current_vehicle in ID:
    # 提取当前车辆的数据
    vehicle_data = data_smoothed[data_smoothed['vehicle_id'] == current_vehicle]

    # 对时间进行排序（确保数据按时间顺序）
    vehicle_data = vehicle_data.sort_values(by='time')

    # 提取 x, y 坐标和时间
    t = vehicle_data['time'].values
    x = vehicle_data['x_smoothed'].values
    y = vehicle_data['y_smoothed'].values

    # 计算空间距离差
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)

    # 计算速度
    v = np.sqrt(dx ** 2 + dy ** 2) / dt

    # 将速度添加到数据集中
    data_smoothed.loc[data_smoothed['vehicle_id'] == current_vehicle, 'speed'] = np.concatenate(([0], v))

# 保存带有速度列的数据集到CSV文件
#data_smoothed.to_csv('02_1_data_speed.csv', index=False)

# 步骤2: 停车检测
speed_threshold = 1  # 速度阈值，单位m/s
start_stop_times = {}  # 存储开始停止的时间点
last_stop_times = {}  # 存储停车的最后一个时间点（速度为0的时刻）

for current_vehicle in ID:
    vehicle_data = data_smoothed[data_smoothed['vehicle_id'] == current_vehicle]
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
min_stop_duration = 1  # 最小停车时长，单位秒
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

# 使用原始数据来可视化聚类结果
plt.figure(figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
sns.scatterplot(data=data_with_labels, x='start_time', y='last_time',
                hue=pd.Categorical.from_codes(final_cluster_labels, categories=np.unique(final_cluster_labels)),
                palette='rainbow')
plt.title("聚类结果散点图")
plt.xlabel("Start Time")
plt.ylabel("Last Time")
#plt.show()

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

# 绘制红灯时长直方图
plt.figure(figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
cluster_durations = list(red_light_durations.values())
plt.hist(cluster_durations, bins='auto', alpha=0.7, edgecolor='black')
plt.xlabel('红灯时长')
plt.ylabel('频率')
plt.title('红灯时长直方图')
plt.grid(axis='y', alpha=0.75)
#plt.show()


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
#plt.show()

# 使用 numpy 的 histogram 函数找到频率最高的周期
counts, bin_edges = np.histogram(periods, bins='auto')
max_index = np.argmax(counts)  # 找到计数最多的bin的索引
most_common_period_bin = 0.5 * (bin_edges[max_index] + bin_edges[max_index + 1])  # 计算该bin的中心点作为周期的代表
if most_common_period_bin > 500:
    most_common_period_bin = most_common_period_bin/5
elif most_common_period_bin > 400:
    most_common_period_bin = most_common_period_bin / 4
elif most_common_period_bin > 300:
    most_common_period_bin = most_common_period_bin / 3
elif most_common_period_bin > 200:
    most_common_period_bin = most_common_period_bin / 2
elif most_common_period_bin > 150:
    most_common_period_bin = most_common_period_bin / 2
print(f"红绿灯周期: {most_common_period_bin}")  # 最常见的周期 bin 相对应的中位数

# 计算绿灯时间
green = most_common_period_bin - average_red_light_duration
print(f"绿灯时间: {green}")

# 调整window_size，得到最佳Best Silhouette Score，最终结果：
"B1: 红灯时间：70, 绿灯时间：27"
"B2: 红灯时间：76, 绿灯时间：24"
"B3: 红灯时间：76, 绿灯时间：60"
"B4: 红灯时间：84, 绿灯时间：52"
"B5: 红灯时间：70, 绿灯时间：46"
