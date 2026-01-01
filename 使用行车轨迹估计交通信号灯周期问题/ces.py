from random import random

import numpy as np
import pandas as pd
from annotated_types import Len
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 读取csv文件
address = r'E:/数据采集与预处理/数据采集与预处理/12_pandas案例/第二次/附件/附件1/A1.csv'  # 改变A1,A2,A3,A4,A5即可
data = pd.read_csv(address, encoding='gbk')


# 抽样比例范围
sample_rates = range(30, 91)
max_red_light_times ={'抽样比例':[],'红灯时间':[]}
for p in sample_rates:
    # 随机选择车辆数据
    sample_vehicle_ids= random.sample(list(data.keys()), int(Len(data.keys()) *p/ 100))
    red_light_periods=[]
    for vehicle_id in sample_vehicle_ids:
        ID = data['vehicle_id'].unique()
        # ********************** 计算速度 时间 （文件01_2）***********************
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

        # 保存带有速度列的数据集到CSV文件
        # data_smoothed.to_csv('02_1_data_speed.csv', index=False)

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
            # print(f"Vehicle {vehicle_id}: Started stopping at {start_time}, last stop (speed=0) time at {last_time}")
            # 筛选有效的停车数据，剔除停车时长<10的数据
            if last_time is not None:
                # 计算时间差
                time_diff = last_time - start_time
                if time_diff >= min_stop_duration:
                    # 将时间差（或其他特征）添加到列表中
                    filtered_data.append((start_time, last_time))

        # 读取数据
        data = pd.DataFrame(filtered_data, columns=['start_time', 'last_time'])
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

        # 使用最佳参数进行聚类
        final_dbscan = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples']).fit(data_scaled)
        final_cluster_labels = final_dbscan.labels_

        # 将缩放后的数据转换回 DataFrame 形式（但注意，这里的值已经是缩放后的，不适合直接解释）
        data_scaled_df = pd.DataFrame(data_scaled, columns=['start_time', 'last_time'])

        # 使用原始数据来计算周期
        data_with_labels = pd.DataFrame(data, columns=['start_time', 'last_time'])
        data_with_labels['cluster'] = final_cluster_labels

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

        # 使用 numpy 的 histogram 函数找到频率最高的周期
        counts, bin_edges = np.histogram(periods, bins='auto')
        max_index = np.argmax(counts)  # 找到计数最多的bin的索引
        most_common_period_bin = 0.5 * (bin_edges[max_index] + bin_edges[max_index + 1])  # 计算该bin的中心点作为周期的代表
        print(f"红绿灯周期: {most_common_period_bin}")  # 最常见的周期 bin 相对应的中位数




