import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def calculate_traffic_light_period(data, sample_proportions):
    # 获取所有独特的车辆 ID
    ID = data['vehicle_id'].unique()

    # 步骤一：计算车辆速度
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

    # 读取数据
    data = pd.DataFrame(filtered_data, columns=['start_time', 'last_time'])
    # 特征缩放
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # DBSCAN参数搜索
    num_random_search = 100
    eps_values = np.linspace(0.01, 0.15, num_random_search)
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

    # 使用最佳参数进行聚类
    final_dbscan = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples']).fit(data_scaled)
    final_cluster_labels = final_dbscan.labels_

    # 将缩放后的数据转换回 DataFrame 形式（但注意，这里的值已经是缩放后的，不适合直接解释）
    data_scaled_df = pd.DataFrame(data_scaled, columns=['start_time', 'last_time'])

    # 使用原始数据来计算周期
    data_with_labels = pd.DataFrame(data, columns=['start_time', 'last_time'])
    data_with_labels['cluster'] = final_cluster_labels

    ###################################### 红灯时长 ##################################################
    # 计算每个簇的红灯时长平均值
    red_light_durations = {}
    for cluster in np.unique(final_cluster_labels):
        if cluster != -1:  # 忽略噪声点
            cluster_data = data_with_labels[data_with_labels['cluster'] == cluster]
            red_light_durations[cluster] = cluster_data['last_time'].subtract(cluster_data['start_time']).mean()

    # 输出每个簇的红灯时长
    # print("Red Light Durations for each cluster:", red_light_durations)

    # 计算过滤后簇的红灯时长最大值（由于样本量过少会导致值偏低，在此取最大值）
    average_red_light_duration = np.max(list(red_light_durations.values()))
    #print("红灯时间：", average_red_light_duration)
    return average_red_light_duration

def main():
    sample_proportions = np.arange(0.3, 1.0, 0.01)
    results = []
    for proportion in sample_proportions:
        sampled_data = data.sample(frac=proportion, random_state=42)  # 随机抽样比例为proportion的数据子集
        period = calculate_traffic_light_period(sampled_data, sample_proportions)
        results.append({'sample_proportion': proportion, 'period': period})
    # 将结果保存到CSV文件中
    result_df = pd.DataFrame(results)
    result_df.to_csv("02_样本红灯时间.csv", index=False)

if __name__ == "__main__":
    file_path = r'E:/数据采集与预处理/数据采集与预处理/12_pandas案例/第二次/附件/附件1/A5.csv'  # 改变A1,A2,A3,A4,A5即可
    data = pd.read_csv(file_path, encoding='gbk')
    main()
