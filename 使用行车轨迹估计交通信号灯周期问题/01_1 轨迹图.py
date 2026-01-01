import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 读取csv文件
address = r'D:/数学建模/数学建模/附件/附件1/A1.csv'  # 改变A1,A2,A3,A4,A5即可
data1 = pd.read_csv(address, encoding='gbk')


# 创建DataFrame
df = pd.DataFrame(data1, columns=['time', 'vehicle_id', 'x', 'y'])

# 为每个车辆ID绘制单独的轨迹图
for vehicle_id, group in df.groupby('vehicle_id'):
    # 创建一个新的图形和坐标轴
    fig, ax = plt.subplots()

    # 设置字体为黑体
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 绘制轨迹
    ax.plot(group['x'], group['y'], label=f'Vehicle {vehicle_id}')

    # 添加图例
    ax.legend()

    # 设置坐标轴标签
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')

    # 设置标题
    ax.set_title(f'车辆 {vehicle_id} 轨迹图')

    # 显示图形
    plt.show()