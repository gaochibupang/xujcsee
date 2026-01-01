import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 读取csv文件
address = r'E:/数据采集与预处理/数据采集与预处理/12_pandas案例/第二次/附件/附件3/C1.csv'  # 改变A1,A2,A3,A4,A5即可
data1 = pd.read_csv(address, encoding='gbk')


# 获取所有独特的车辆 ID
ID = data1['vehicle_id'].unique()

#  步骤一：计算车辆速度
for current_vehicle in ID:
    # 提取当前车辆的数据
    vehicle_data = data1[data1['vehicle_id'] == current_vehicle]

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

    # 绘制速度时间图
    plt.plot(t[1:], v, label=f'Vehicle {current_vehicle}速度时间图')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    # plt.show()

    # 将速度添加到数据集中
    data1.loc[data1['vehicle_id'] == current_vehicle, 'speed'] = np.concatenate(([0], v))

# 保存带有速度列的数据集到CSV文件
data1.to_csv('data1_speed.csv', index=False)

# 步骤2: 停车检测
speed_threshold = 1  # 速度阈值，单位m/s
start_stop_times = {}  # 存储开始停止的时间点
last_stop_times = {}  # 存储停车的最后一个时间点（速度为0的时刻）

for current_vehicle in ID:
    vehicle_data = data1[data1['vehicle_id'] == current_vehicle]
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
    print(f"Vehicle {vehicle_id}: Started stopping at {start_time}, last stop (speed=0) time at {last_time}")
    # 筛选有效的停车数据，剔除停车时长<10的数据
    if last_time is not None:
        # 计算时间差
        time_diff = last_time - start_time
        if time_diff >= min_stop_duration:
            # 将时间差（或其他特征）添加到列表中
            filtered_data.append((vehicle_id, start_time, last_time, time_diff))

#print(filtered_data)
# 写入CSV文件
with open('data1_time.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # 写入表头
    writer.writerow(['vehicle_id', 'start_time', 'last_time', 'time_diff'])
    # 写入数据
    for row in filtered_data:
        writer.writerow(row)
