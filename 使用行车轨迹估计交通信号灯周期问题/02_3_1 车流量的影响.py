import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# 读取csv文件
address = r'E:/数据采集与预处理/数据采集与预处理/12_pandas案例/第二次/附件/附件1/A5.csv'  # 改变A1,A2,A3,A4,A5即可
data = pd.read_csv(address, encoding='gbk')


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

    # 绘制速度时间图
    plt.plot(t[1:], v, label=f'Vehicle {current_vehicle}速度时间图')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    # plt.show()

    # 将速度添加到数据集中
    data.loc[data['vehicle_id'] == current_vehicle, 'speed'] = np.concatenate(([0], v))

# 保存带有速度列的数据集到CSV文件
# data.to_csv('data_speed.csv', index=False)

'''观察速度时间图，将车流量按最大速度分为三组：12及以下,12~16,16以上'''

# 计算每辆车的最大速度
max_speeds = data.groupby('vehicle_id')['speed'].max()

# 根据最大速度将数据分为三组
v_low = data[data['vehicle_id'].map(max_speeds) <= 12]
v_middle = data[(data['vehicle_id'].map(max_speeds) > 12) & (data['vehicle_id'].map(max_speeds) <= 16)]
v_high = data[data['vehicle_id'].map(max_speeds) > 16]

# 将这三组数据分别保存到csv文件中
v_low.to_csv('v_low.csv', index=False)
v_middle.to_csv('v_middle.csv', index=False)
v_high.to_csv('v_high.csv', index=False)

