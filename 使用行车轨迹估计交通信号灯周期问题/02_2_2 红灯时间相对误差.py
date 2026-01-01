import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# 计算相对误差
def calculate_relative_errors(original, sample_periods):
    return [abs(sample - original) / original for sample in sample_periods]

# 不同样本比例下的相对误差数据
def relative_error_data(file_path):
    original_red_light_duration = 71  # 74,72,90,75,71
    data = pd.read_csv(file_path)
    sample_proportions = data['sample_proportion']
    periods = data['period']
    relative_errors = calculate_relative_errors(original_red_light_duration, periods)

    # 创建一个新的DataFrame，将样本比例和相对误差存储在其中
    result_df = pd.DataFrame({'sample_proportion': sample_proportions, 'relative_error': relative_errors})
    # 将结果保存为CSV文件
    result_df.to_csv('relative_errors.csv', index=False)

if __name__ == "__main__":
    file_path = '02_样本红灯时间.csv'
    relative_error_data(file_path)

