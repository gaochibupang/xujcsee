import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from datetime import datetime

# 读取csv文件
address = r'E:/数据采集与预处理/数据采集与预处理/12_pandas案例/D/有血糖值的检测数据.csv'
data = pd.read_csv(address, encoding='gbk')

'''数据清洗及预处理'''
# 查看变量
print(data.columns.tolist())

# 肝功能指标（*r-谷氨酰基转换酶、*丙氨酸氨基转换酶、*天门冬氨酸氨基转换酶、*总蛋白、*球蛋白、*碱性磷酸酶、白蛋白、白球比例）
# 乙肝相关指标（乙肝e抗体、乙肝e抗原、乙肝核心抗体、乙肝表面抗体、乙肝表面抗原）
# 血脂指标（低密度脂蛋白胆固醇、总胆固醇、高密度脂蛋白胆固醇、甘油三酯）
# 血常规指标（红细胞体积分布宽度、红细胞压积、红细胞平均体积、红细胞平均血红蛋白浓度、红细胞平均血红蛋白量、红细胞计数、血红蛋白、白细胞计数、淋巴细胞%、中性粒细胞%、单核细胞%、嗜酸细胞%、嗜碱细胞%、血小板体积分布宽度、血小板平均体积、血小板比积、血小板计数）
# 肾功能指标（尿素、尿酸、肌酐）
# 血糖指标（血糖）
# 其他/基本信息(id、体检日期、年龄、性别)                                          '''

# 剔除无关变量
# data = data.drop(['id', '体检日期'], axis=1)

# 查看数据类型
print("查看数据类型:")
print(data.info())
# 转换非数值型数据：性别和体检日期
# 使'男性'编码为2，'女性'编码为1
label_encoder = LabelEncoder()
data['性别'] = label_encoder.fit_transform(data['性别'])
# 将体检日期转换为日期时间格式，并转换为天数差
data['体检日期'] = pd.to_datetime(data['体检日期'], format='%d/%m/%Y', errors='coerce')
reference_date = datetime(2017, 10, 1)
data['体检日期'] = (data['体检日期'] - reference_date).dt.days
print("再次查看数据类型:")
print(data.info())

# 1. 缺失值处理
print("1、缺失值处理")
# 查看缺失值比例
print("查看缺失值比例", data.isnull().sum()/len(data))

# 对每列使用中位数填充缺失值
numeric_cols = data.columns
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

# 再次查看缺失值
print("再次查看缺失值数量:", data.isnull().sum())

#data.to_csv('train6666.csv', index=False)
# 2. 异常值处理
print("2、异常值处理")
'''
# 绘制箱线图
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

    # 剔除异常值
    data = data[(data['血糖'] >= lower_bound) & (data['血糖'] <= upper_bound)]

    # 验证处理后是否还有异常值
    outliers_after = (data[col] < lower_bound) | (data[col] > upper_bound)
    outlier_counts_after[col] = outliers_after.sum()

# 打印原始和处理后的异常值计数
print("原始异常值计数:", outlier_counts_before)
print("处理后异常值计数:", outlier_counts_after)

# 保存处理后的数据
#data.to_csv('train.csv', index=False)

# 3、数据变换
print("3、数据变换")
# 标准化 zscore
data_scale = zscore(data, axis=0)


'''描述性统计分析'''

# 1、查看数据统计量：计数（count）、平均值（mean）、标准差（std）、最小值（min）、第一四分位数（25%或Q1）、中位数（50%或Q2）、第三四分位数（75%或Q3）和最大值（max）
print("1、查看数据统计量：")
print(data.describe())

# 2、相关性分析（使用标准化后的数据data_scale，后续建模可能也需要使用标准化后的数据）
print("2、相关性分析")
# 计算相关性矩阵
corr_matrix = data_scale.corr()
# 打印相关性矩阵
print(corr_matrix)
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

# 主成分分析

