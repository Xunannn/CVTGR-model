import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# 读取数据
data = pd.read_csv(r'D:\MASTER\my_paper\paper_2\Processed data\perturbation testing\XTG-Qingyi10.csv')

# 获取特征和目标列
features = data.iloc[:, :-1]  # 假设最后一列为目标列
targets = data.iloc[:, -1]   # 假设最后一列为目标列

# 处理缺失值
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# 创建随机森林回归模型
rf = RandomForestRegressor()

# 训练模型
rf.fit(features, targets)

# 预测数据
predicted_sums = rf.predict(features)

# 创建包含预测结果的 DataFrame
predicted_data = pd.DataFrame({'Predicted Sum': predicted_sums})

# 将预测结果保存到 CSV 文件
predicted_data.to_csv(r'D:\MASTER\my_paper\paper_2\Result\Perturbation Testing\XAJ-TCN-GRU\Pre_Qingyi10.csv', index=False)

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

rmse = np.sqrt(mean_squared_error(targets, predicted_sums))
print('Test RMSE: %.3f' % rmse)
# 计算MAE
mae = mean_absolute_error(targets, predicted_sums)
print("MAE: %.3f" % mae)

# 计算MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

mape = mean_absolute_percentage_error(targets, predicted_sums)
print("MAPE: %.3f" % mape)
# 计算R2
print('r2_score: %.3f' % r2_score(targets, predicted_sums))
# 计算平均值
mean_observed = np.mean(targets)
mean_simulated = np.mean(predicted_sums)
# 计算RRMSE
rrmse = rmse / mean_observed
print('RRMSE: %.3f' % rrmse)
# 计算RMAE
rmae = mae / mean_observed
print('RMAE: %.3f' % rmae)
# 计算KGE
r = np.corrcoef(targets, predicted_sums)[0,1]  # 相关系数
# 计算标准差
std_observed = np.std(targets)
std_simulated = np.std(predicted_sums)
# 计算KGE
kge = 1 - np.sqrt((r - 1)**2 + (mean_simulated/mean_observed - 1)**2 + (std_simulated/std_observed - 1)**2)
print('KGE: %.3f' % kge)