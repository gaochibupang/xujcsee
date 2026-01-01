import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
import itertools
from sklearn.svm import SVR

# 读取csv文件
address = r'D:/数学建模/数学建模/代码/第三次/历史配件订单表.csv'
data = pd.read_csv(address, encoding='GBK')


# 初始化一个空的DataFrame来存储所有配件的预测结果
all_predictions = pd.DataFrame()

# 2、对当天该仓库对该配件的需求量进行加和处理
data['日期'] = pd.to_datetime(data['日期'])
df = data.groupby(['仓库编码', '配件编码', pd.Grouper(key='日期', freq='D')])['需求量'].sum().reset_index()
df.set_index('日期', inplace=True)
# df.to_csv('df.csv', encoding='utf_8_sig')

# 遍历所有配件
for sku in range(3, 905):  # sku003至sku904
    sku_code = f'sku{sku:03d}'
    df_sku = df[df['配件编码'] == sku_code]

    cols = ['需求量']

    # 初始化离群点计数Series
    outlier_counts_before = pd.Series(index=cols, dtype=int)
    outlier_counts_after = pd.Series(index=cols, dtype=int)

    # 处理离群点
    for col in cols:
        Q1 = df_sku[col].quantile(0.25)
        Q3 = df_sku[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 识别离群点
        outliers = (df_sku[col] < lower_bound) | (df_sku[col] > upper_bound)
        outlier_counts_before[col] = outliers.sum()

        # 使用均值替换离群点
        median_val = df_sku[col].mean()
        df_sku.loc[outliers, col] = median_val

    # 提取要预测的列
    df_sku003_demand = df_sku[['需求量']]
    df_sku003_demand.index = pd.to_datetime(df_sku003_demand.index)

    start_date = pd.to_datetime('2022-01-01')
    end_date = pd.to_datetime('2023-07-31')

    # 使用前一个日期的值填充缺失的日期
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    df_sku003_demand = df_sku003_demand.reindex(all_dates, method='ffill')

    # 用1填充剩下日期的缺失值
    df_sku003_demand = df_sku003_demand.fillna(1)

    ######################################## 建模
    # 拆分数据集,最后1个月为测试集，其他为训练集
    last_month_start = end_date - pd.offsets.MonthBegin(1)
    train = df_sku003_demand.loc[:last_month_start - pd.Timedelta(days=1)]
    test = df_sku003_demand.loc[last_month_start:]

    # 分解时序
    # STL（Seasonal and Trend decomposition using Loess）是一个非常通用和稳健强硬的分解时间序列的方法
    decompostion = sm.tsa.STL(df_sku003_demand).fit()  # statsmodels.tsa.api:时间序列模型和方法
    decompostion.plot()
    # 趋势效益
    trend = decompostion.trend
    # 季节效应
    seasonal = decompostion.seasonal
    # 随机效应
    residual = decompostion.resid
    plt.show()


    # 搜索法定阶
    def SARIMA_search(data):
        p = q = range(0, 3)
        s = [7]  # 周期
        d = [0, 1]
        PDQs = list(itertools.product(p, d, q, s))
        pdq = list(itertools.product(p, d, q))

        params = []
        seasonal_params = []
        results = []

        for param in pdq:
            for seasonal_param in PDQs:
                mod = sm.tsa.SARIMAX(data, order=param, seasonal_order=seasonal_param,
                                     enforce_stationarity=False, enforce_invertibility=False)
                result = mod.fit()
                params.append(param)
                seasonal_params.append(seasonal_param)
                results.append(result.aic)

        grid = pd.DataFrame({
            "pdq": params,
            "PDQs": seasonal_params,
            "aic": results
        })

        # 返回具有最低AIC值的行
        best_row = grid[grid["aic"] == grid["aic"].min()]
        return best_row


    # 使用SARIMA_search找到最佳参数
    best_params = SARIMA_search(df_sku003_demand.dropna())

    # 从结果中提取最佳参数
    best_p, best_d, best_q = best_params.iloc[0]['pdq']
    best_P, best_D, best_Q, best_s = best_params.iloc[0]['PDQs']

    # 使用最佳参数建立模型
    model = sm.tsa.SARIMAX(df_sku003_demand, order=(best_p, best_d, best_q),
                           seasonal_order=(best_P, best_D, best_Q, best_s))
    SARIMA_m = model.fit()

    # 打印模型摘要
    print(SARIMA_m.summary())
    print('最佳pdq:', best_p, best_d, best_q)
    print('最佳PDQs:', best_P, best_D, best_Q, best_s)

    # 模型检验
    fig = SARIMA_m.plot_diagnostics(figsize=(15, 12))  # plot_diagnostics对象允许我们快速生成模型诊断并调查任何异常行为
    plt.show()

    # 模型预测
    predict_demand = SARIMA_m.predict(start=test.index.min() + pd.Timedelta(days=1),
                                      end=test.index.max() + pd.Timedelta(days=1), dynamic=False)


    # print(predict_demand)

    # 计算WMAPE
    def mean_weighted_absolute_percentage_error(y_true, y_pred, weights=None):
        if weights is None:
            weights = np.ones_like(y_true)
        y_true, y_pred, weights = np.array(y_true), np.array(y_pred), np.array(weights)
        diff = np.abs((y_true - y_pred) / y_true)
        norm_diff = diff * weights
        norm_diff = norm_diff[y_true != 0]  # 避免除以零
        return np.average(norm_diff, weights=weights[y_true != 0])


    # 计算SMAPE
    def mean_symmetric_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return 100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))


    # 创建一个包含真实值和预测值的 DataFrame
    results_df = pd.DataFrame({
        '需求量': test['需求量'],
        '预测需求量': predict_demand
    }, index=test.index)

    # 计算 WMAPE 和 SMAPE
    wmape = mean_weighted_absolute_percentage_error(results_df['需求量'], results_df['预测需求量'])
    smape = mean_symmetric_absolute_percentage_error(results_df['需求量'], results_df['预测需求量'])

    #################################### 模型优化
    # 1. 计算残差
    residuals = test['需求量'] - predict_demand

    # 2. 训练SVR模型
    X_train = np.arange(len(residuals)).reshape(-1, 1)
    svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
    svr_model.fit(X_train, residuals)

    # 3. 预测残差
    X_test = np.arange(len(test)).reshape(-1, 1)
    predicted_residuals = svr_model.predict(X_test)

    # 4. 结合预测结果
    final_predictions = predict_demand + predicted_residuals

    # 计算最终预测结果的WMAPE和SMAPE
    results_df['最终预测需求量'] = final_predictions
    wmape_final = mean_weighted_absolute_percentage_error(results_df['需求量'], results_df['最终预测需求量'])
    smape_final = mean_symmetric_absolute_percentage_error(results_df['需求量'], results_df['最终预测需求量'])

    ################################################# 数据预测
    # 1. 预测未来30天的数据
    forecast = SARIMA_m.get_forecast(steps=31)
    future_dates = pd.date_range(start=test.index.max(), periods=32, freq='D')[1:]  # 生成未来30天的日期
    future_demand = forecast.predicted_mean

    # 2. 预测未来31天的残差
    X_future = np.arange(len(future_demand)).reshape(-1, 1)
    predicted_residuals = svr_model.predict(X_future)

    # 3. 结合预测结果和残差
    final_future_predictions = future_demand + predicted_residuals

    # 存储预测结果
    sku_predictions = pd.DataFrame({
        '日期': future_dates,
        '预测需求量': final_future_predictions,
        '配件编码': sku_code
    })
    all_predictions = pd.concat([all_predictions, sku_predictions], ignore_index=True)

# 存储所有配件的预测结果到CSV文件
all_predictions.to_csv('all_sku_predictions.csv', encoding='utf_8_sig', index=False)

