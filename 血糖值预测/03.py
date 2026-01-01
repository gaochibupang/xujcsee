import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from datetime import datetime

from sklearn.decomposition import PCA
import numpy as np

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error  # MAE
from sklearn.metrics import mean_squared_error  # MSE

# 读取csv文件
address = r'E:/数据采集与预处理/数据采集与预处理/12_pandas案例/D/有血糖值的检测数据.csv'
data = pd.read_csv(address, encoding='gbk')

'''数据清洗及预处理'''
# 查看变量
# print(data.columns.tolist())

# 肝功能指标（*r-谷氨酰基转换酶、*丙氨酸氨基转换酶、*天门冬氨酸氨基转换酶、*总蛋白、*球蛋白、*碱性磷酸酶、白蛋白、白球比例）
# 乙肝相关指标（乙肝e抗体、乙肝e抗原、乙肝核心抗体、乙肝表面抗体、乙肝表面抗原）
# 血脂指标（低密度脂蛋白胆固醇、总胆固醇、高密度脂蛋白胆固醇、甘油三酯）
# 血常规指标（红细胞体积分布宽度、红细胞压积、红细胞平均体积、红细胞平均血红蛋白浓度、红细胞平均血红蛋白量、红细胞计数、血红蛋白、白细胞计数、淋巴细胞%、中性粒细胞%、单核细胞%、嗜酸细胞%、嗜碱细胞%、血小板体积分布宽度、血小板平均体积、血小板比积、血小板计数）
# 肾功能指标（尿素、尿酸、肌酐）
# 血糖指标（血糖）
# 其他/基本信息(id、体检日期、年龄、性别)                                          '''


# 查看数据类型
#print("查看数据类型:")
#print(data.info())
# 转换非数值型数据：性别和体检日期
# 使'男性'编码为2，'女性'编码为1
label_encoder = LabelEncoder()
data['性别'] = label_encoder.fit_transform(data['性别'])
# 将体检日期转换为日期时间格式，并转换为天数差
data['体检日期'] = pd.to_datetime(data['体检日期'], format='%d/%m/%Y', errors='coerce')
reference_date = datetime(2017, 10, 1)
data['体检日期'] = (data['体检日期'] - reference_date).dt.days
#print("再次查看数据类型:")
#print(data.info())

# 1. 缺失值处理
#print("1、缺失值处理")
# 查看缺失值比例
#print("查看缺失值比例", data.isnull().sum()/len(data))

# 对每列使用中位数填充缺失值
numeric_cols = data.columns
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

# 再次查看缺失值
#print("再次查看缺失值数量:", data.isnull().sum())

# 2. 异常值处理
#print("2、异常值处理")
'''
# 绘制箱线图（不太好看）
plt.figure(figsize=(30, 10))
data.boxplot()
plt.title('Boxplot for Data')
plt.ylabel('Values')
plt.show()'''

'''说明：由于在处理缺失值时乙肝相关指标大量缺失，为保证数据准确，在此不对其进行异常值处理。'''

# 列出不需要处理异常值的变量名
exclude_cols = ['乙肝e抗体', '乙肝e抗原', '乙肝核心抗体', '乙肝表面抗体', '乙肝表面抗原']
# 筛选出数值型列，并排除不需要处理的列
numeric_cols = data.columns.difference(exclude_cols)
# 初始化异常值计数Series
outlier_counts_before = pd.Series(index=numeric_cols, dtype=int)
outlier_counts_after = pd.Series(index=numeric_cols, dtype=int)

# 处理异常值
for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 识别异常值
    outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
    outlier_counts_before[col] = outliers.sum()

    # 用中位数替换异常值
    data[col] = data[col].mask(outliers, data[col].median())

    # 验证处理后是否还有异常值
    outliers_after = (data[col] < lower_bound) | (data[col] > upper_bound)
    outlier_counts_after[col] = outliers_after.sum()

# 打印原始和处理后的异常值计数
#print("原始异常值计数:", outlier_counts_before)
#print("处理后异常值计数:", outlier_counts_after)

# 查看“血糖”分布情况，绘制分布直方图
plt.figure(figsize=(10, 6))  # 设置图形的大小
plt.hist(data['血糖'], bins=10, color='skyblue', alpha=0.7)  # bins参数控制直方图的柱子数量
plt.title('血糖分布直方图')  # 设置图形的标题
plt.xlabel('血糖值')  # 设置x轴标签
plt.ylabel('频数')  # 设置y轴标签
plt.grid(axis='y', alpha=0.75)  # 显示y轴的网格线，并设置透明度
#plt.show()

# 保存处理后的数据
#data.to_csv('train.csv', index=False)

# 3、数据变换
#print("3、数据变换")
# 标准化 zscore
data_scale = zscore(data, axis=0)

'''描述性统计分析'''

# 1、查看数据统计量：计数（count）、平均值（mean）、标准差（std）、最小值（min）、第一四分位数（25%或Q1）、中位数（50%或Q2）、第三四分位数（75%或Q3）和最大值（max）
print("1、查看数据统计量：")
print(data.describe())

# 2、相关性分析（使用标准化后的数据data_scale，后续建模可能也需要使用标准化后的数据）
#print("2、相关性分析")
# 计算相关性矩阵
corr_matrix = data_scale.corr()
# 打印相关性矩阵
#print(corr_matrix)
# 可视化 绘制热图
plt.figure(figsize=(14, 10))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False
sns.heatmap(corr_matrix, vmax=0.9, square=True, cmap='coolwarm')
plt.title('Correlation Matrix')
#plt.show()

# 找出与'血糖'相关性最强的几个变量（取绝对值后最大的6个）
blood_sugar_corr = corr_matrix['血糖'].abs().sort_values(ascending=False)
top_correlated_features = blood_sugar_corr[1:7]  # 排除自身，取前6个

# 打印这些变量的名称和它们与'血糖'的相关性
print("与'血糖'相关性最强的变量（前6个，不包括'血糖'自身）:")
print(top_correlated_features)

# 选取这些变量绘制散点图（使用data）
# 设置子图的布局为2行3列
nrows, ncols = 2, 3
# 绘制散点图
plt.figure(figsize=(18, 10))
for i, var in enumerate(top_correlated_features.index):
    # 计算当前子图的位置（行和列）
    row = i // ncols
    col = i % ncols

    # 创建一个子图
    plt.subplot(nrows, ncols, i + 1)

    # 绘制散点图
    x = data[var]
    y = data['血糖']
    plt.scatter(x, y, s=5, alpha=0.5)

    plt.xlabel(var)
    plt.ylabel('血糖')
    plt.title(f'{var} 与 血糖 的散点图')
    plt.grid(True)

plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
#plt.show()

# 可以看出：各变量与“血糖”之间无明显线性关系......

print("**********************************   第二题   **************************************************")
# 剔除干扰变量
# 剔除干扰变量
data = data.drop(['id', '体检日期'], axis=1)

# 合并相关变量
#print(data.columns.tolist())
data['肾'] = data['尿素'] + data['肌酐'] + data['尿酸']
data['乙肝抗体'] = data['乙肝e抗体'] + data['乙肝表面抗体'] + data['乙肝核心抗体']
data['乙肝抗原'] = data['乙肝e抗原'] + data['乙肝表面抗原']
data['低密度脂蛋白胆固醇/高密度脂蛋白胆固醇'] = data['低密度脂蛋白胆固醇'] / data['高密度脂蛋白胆固醇']
data['总酶'] = data['*天门冬氨酸氨基转换酶'] + data['*丙氨酸氨基转换酶'] + data['*碱性磷酸酶'] + data[
    '*r-谷氨酰基转换酶']

data = data.drop(columns=['尿素', '肌酐', '尿酸',
                          '低密度脂蛋白胆固醇', '高密度脂蛋白胆固醇',
                          '乙肝e抗体', '乙肝e抗原', '乙肝表面抗原', '乙肝表面抗体', '乙肝核心抗体',
                          '*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶', '*碱性磷酸酶', '*r-谷氨酰基转换酶'])

# 相关性分析
# 找出与'血糖'相关性最弱的几个变量
data_scale_2 = zscore(data, axis=0)  # 标准化
corr_matrix = data_scale_2.corr()   # 计算相关性矩阵
blood_sugar_corr = corr_matrix['血糖'].abs().sort_values(ascending=True)
# 剔除与'血糖'相关性最低的10个变量
num_to_drop = 10
variables_to_drop = blood_sugar_corr.index[1:num_to_drop + 1]  # +1是因为我们从索引1开始（排除了'血糖'）

# 剔除变量
data = data.drop(columns=variables_to_drop)

# 主成分分析
# print(data.columns.tolist())
# 去除'血糖'列，因为我们只对其他变量进行PCA
X = data.drop('血糖', axis=1)
y = data['血糖']  # 保存'血糖'列作为目标变量

# 标准化
X_scaled = zscore(X, axis=0)

# 计算协方差矩阵
covariance_matrix = np.cov(X_scaled, rowvar=False)
#print("协方差矩阵:\n", covariance_matrix)

# 计算相关系数矩阵
correlation_matrix = np.corrcoef(X_scaled, rowvar=False)
#print("相关系数矩阵:\n", correlation_matrix)

# 执行PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 获取特征值（的方差）和特征向量
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_
#print("特征值（的方差）:\n", eigenvalues)
#print("特征向量:\n", eigenvectors)
# 计算主成分贡献率
explained_variance_ratio = pca.explained_variance_ratio_
print("主成分贡献率:\n", explained_variance_ratio)
# 计算累计贡献率
cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
print("累计贡献率:\n", cumulative_explained_variance_ratio)

# 找到累计贡献率超过80%时的主成分数量m
threshold = 0.80
m = np.where(cumulative_explained_variance_ratio >= threshold)[0][0] + 1

# 从X_pca中选取前m个主成分
X_pca_reduced = X_pca[:, :m]

# 打印结果
print(f"为了保留至少{threshold * 100:.2f}%的变异性，需要选择前{m}个主成分。")
print("降维后的数据（只保留前{}个主成分）:\n".format(m), X_pca_reduced)
# 每个特征在第m个主成分上的载荷值
print('每个特征在第m个主成分上的载荷值:')
for i in range(m):
    print(f"第{i + 1}个主成分:\n{pca.components_[i]}")

# 查看这些主成分的解释方差比（贡献率）
print("降维后的主成分贡献率:\n", explained_variance_ratio[:m])

# 绘制条形图
plt.figure(figsize=(10, 6))
plt.bar(range(1, m + 1), explained_variance_ratio[:m], color='blue', alpha=0.7)
plt.title('Explained Variance Ratio of Selected Principal Components')
plt.xlabel('Principal Component Number')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.xticks(range(1, m + 1))  # 确保 x 轴标签正确
#plt.show()

# 特征值选取
# 原始特征值
original_feature_names = X.columns

# 每个主成分中的特征按载荷绝对值（影响程度）降序排列
for i in range(m):
    # 获取第i个主成分的特征向量（载荷）
    loadings = pca.components_[i]
    # 获取载荷绝对值最大的特征的索引，并排序（按绝对值降序）
    sorted_indices = np.argsort(np.abs(loadings))[::-1]
    # 对应的原始特征名
    sorted_feature_names = original_feature_names[sorted_indices]
    # 打印结果
    print(f"（按绝对值降序）第{i + 1}个主成分中载荷绝对值最大的特征:")
    for j, name in enumerate(sorted_feature_names, 1):
        print(f"{j}. {name} (载荷值: {loadings[sorted_indices[j - 1]]:.3f})")

# 每个主成分中影响最大（载荷绝对值最大）的特征
# 初始化一个空列表来存储每个主成分中最重要的特征
most_important_features_list = []

for i in range(m):
    # 找到第i个主成分中载荷绝对值最大的特征的索引
    max_loading_index = np.argmax(np.abs(pca.components_[i]))
    most_important_features = original_feature_names[max_loading_index]
    most_important_features_list.append(most_important_features)
    print(f"第{i + 1}个主成分中最重要（载荷绝对值最大）的原始特征: {most_important_features}")

# 转换为集合以去除重复项，然后转回列表以保持顺序
unique_most_important_features = list(dict.fromkeys(most_important_features_list))

# 打印最重要的原始特征
print(f"\n总共有 {len(unique_most_important_features)} 个最重要（载荷绝对值最大）的原始特征。它们是：")
for feature in unique_most_important_features:
    print(feature)

# 使用Pandas选择这些特征
X_new = X[unique_most_important_features]

# 打印新数据集
print(X_new.head())  # 查看新数据集的前几行
print(y.head())
# 如果需要，可以将新数据集保存到CSV文件或其他格式
# X_new.to_csv('new_data.csv', index=False)
merged_df = pd.concat([X_new, y], axis=1)

# 保存到CSV
merged_df.to_csv('merged_data.csv', index=False)  # index=False表示不保存行索引

print("**********************************   第三题   **************************************************")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# 定义参数网格
param_grid = {'n_estimators': [50, 70, 90, 100, 200],
    #'max_features': ['sqrt', 'log2'],
    #'max_depth': [50, 60, 70, 80],
    #'min_samples_split': [2, 5, 10],
    #'min_samples_leaf': [1, 2, 4]
    }

# 初始化随机森林回归模型
rf_model = RandomForestRegressor()

# 初始化GridSearchCV对象，设置交叉验证的折数和其他参数，并启用并行计算
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
                           n_jobs=-1)  # n_jobs=-1表示使用所有可用的CPU核心

# 使用训练数据拟合GridSearchCV对象
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)

# 计算训练集预测结果的平均绝对误差（MAE）(随机森林模型)
predictions1 = grid_search.predict(X_train)
predictions = grid_search.predict(X_test)

MAE = mean_absolute_error(y_train, predictions1)
print(f'Training Mean Absolute Error: {MAE}')
# 计算模型预测值和真实值之间的均方误差（MSE）(随机森林模型)
MSE = mean_squared_error(y_train, predictions1)
print(f'Training Mean Squared Error: {MSE}')

MAE = mean_absolute_error(y_test, predictions)
print(f'Test Mean Absolute Error: {MAE}')
# 计算模型预测值和真实值之间的均方误差（MSE）(随机森林模型)
MSE = mean_squared_error(y_test, predictions)
print(f'Test Mean Squared Error: {MSE}')

print(grid_search.best_score_)

plt.figure(figsize=(10, 6))
plt.scatter(y_train, predictions1, color='blue', label='Predicted vs Actual')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4, label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Comparison of Actual and Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

# 绘制预测值与实际值的对比图
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4, label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Comparison of Actual and Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

# 计算残差
residuals = y_test - predictions

# 绘制残差图
plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals, color='red', alpha=0.5)
plt.hlines(y=0, xmin=predictions.min(), xmax=predictions.max(), color='black', lw=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()