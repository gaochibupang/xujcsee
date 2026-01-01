import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# 读取csv文件
address = r'E:/数据采集与预处理/数据采集与预处理/12_pandas案例/第二次/附件/附件4/D.csv'
data = pd.read_csv(address, encoding='gbk')


#################################### 1、画出轨迹图确定路口形状 ############################################
########################################### IQR离群点检测
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

# 绘制轨迹图
data = pd.DataFrame(data, columns=['time', 'vehicle_id', 'x', 'y'])  # 创建DataFrame，并指定列名
fig, ax = plt.subplots()  # 创建一个新的图形和坐标轴
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置字体
# 遍历每个车辆ID，绘制轨迹
for vehicle_id, group in data.groupby('vehicle_id'):
    ax.plot(group['x'], group['y'], label=f'Vehicle {vehicle_id}')
ax.legend(loc='upper right')  # 添加图例
ax.set_xlabel('X Position')  # 设置坐标轴标签
ax.set_ylabel('Y Position')
ax.set_title('所有车辆轨迹图')  # 设置标题
plt.show()  # 显示图形


'''已知一个十字路口共有12种方向轨迹，所以先使用K-means对轨迹进行聚类，将数据分为12组，代表一个路口的不同方向。'''

################################### 2、使用K-means对轨迹进行聚类 ################################
# 提取起点和终点（这里简化处理，仅取第一点和最后一点）
start_end_points = data.groupby('vehicle_id').agg({'x': ['first', 'last'], 'y': ['first', 'last']})
start_end_points.columns = ['x_start', 'x_end', 'y_start', 'y_end']

# 将DataFrame转换为NumPy数组
features = start_end_points.values

# 使用K-means聚类
k = 12  # 将方向轨迹分为12类
kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
start_end_points['cluster'] = kmeans.labels_

# 绘制每类车辆的轨迹图
fig, axes = plt.subplots(3, 4, figsize=(13, 8))
for i, ax in enumerate(axes.flatten()):
    if i < k:
        cluster_data = data[data['vehicle_id'].isin(start_end_points[start_end_points['cluster'] == i].index)]
        ax.plot(cluster_data['x'], cluster_data['y'], '.', label=f'Cluster {i+1}')
        ax.set_title(f'Cluster {i+1} Trajectories')
        ax.legend()
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()

# 将['x_start', 'x_end', 'y_start', 'y_end']分别保存到CSV
# for i in range(k):
    # cluster_data = start_end_points[start_end_points['cluster'] == i+1]
    # cluster_data.to_csv(f'cluster_{i+1}.csv', index=False)

# 将每类数据分别保存到CSV，包含车辆ID
for i in range(k):
    # 首先，从原始数据中筛选出属于当前聚类的车辆ID
    cluster_vehicle_ids = start_end_points[start_end_points['cluster'] == i].index

    # 然后，使用这些车辆ID从原始数据中筛选出完整的轨迹数据
    cluster_data = data[data['vehicle_id'].isin(cluster_vehicle_ids)]

    # 保存这个聚类的轨迹数据到CSV文件，D1.csv~D12.csv共12个
    cluster_data.to_csv(f'D{i+1}.csv', index=False)

'''之后将这12个文件用问题1中的模型计算红绿灯周期及红灯时间，得到最终结果。
仅考虑红绿灯正常、合理运行时的情况，不考虑晚高峰、交通拥堵、交通管制、交警辅佐等特殊情况。'''

# 最终结果：
"D1: 红灯时间：86, 绿灯时间：54, 红绿灯周期: 140"
"D2: 红灯时间：104, 绿灯时间：36, 红绿灯周期: 140"
"D3: 红灯时间：104, 绿灯时间：37, 红绿灯周期: 141"
"D4: 红灯时间：91, 绿灯时间：50, 红绿灯周期: 141"
"D5: 红灯时间：87, 绿灯时间：53, 红绿灯周期: 140"
"D6: 红灯时间：99, 绿灯时间：37, 红绿灯周期: 136"
"D7: 红灯时间：117, 绿灯时间：20, 红绿灯周期: 137"
"D8: 红灯时间：103, 绿灯时间：37, 红绿灯周期: 140"
"D9: 红灯时间：104, 绿灯时间：34, 红绿灯周期: 138"
"D10: 红灯时间：86, 绿灯时间：55, 红绿灯周期: 141"
"D11: 红灯时间：96, 绿灯时间：43, 红绿灯周期: 139"
"D12: 红灯时间：91, 绿灯时间：49, 红绿灯周期: 140"
