import numpy as np
import matplotlib.pyplot as plt
from PyEMD import CEEMDAN
import pandas as pd

# 读取数据集
dataset_path = r'D:\MASTER\DATASET\JialingRiver\MIC_FeatureSelection\FS_Wusheng.csv'
dataset = pd.read_csv(dataset_path)
print('读取数据集完成')

# 提取输入变量（特征）和目标变量
X = dataset.iloc[:, :-1].values  # 输入变量
y = dataset.iloc[:, -1].values   # 目标变量

# 创建 CEEMDAN 对象
ceemdan = CEEMDAN()

# 对每个变量进行 CEEMDAN 分解
imfs_all = []  # 创建一个空列表，用于存储每个变量的分解结果
for var in range(X.shape[1]):
    imfs_var = ceemdan(X[:, var])
    imfs_all.append(imfs_var)

# 对目标变量进行 CEEMDAN 分解
imfs_target = ceemdan(y)

# 创建一个空的DataFrame来存储结果
df_imfs = pd.DataFrame()
df_residuals = pd.DataFrame()
print('分解完成')

# 遍历每个变量的IMF和残差
for var in range(len(imfs_all)):
    imfs_var = imfs_all[var]

    # 将IMF值分别添加到DataFrame中
    for i, imf in enumerate(imfs_var):
        df_imfs[f'Variable {var+1} IMF {i+1}'] = imf

    # 将残差值添加到DataFrame中
    df_residuals[f'Variable {var+1} Residual'] = X[:, var] - np.sum(imfs_var, axis=0)

# 将目标变量的IMF值添加到DataFrame中
for i, imf in enumerate(imfs_target):
    df_imfs[f'Target IMF {i+1}'] = imf

# 将目标变量的残差值添加到DataFrame中
df_residuals['Target Residual'] = y - np.sum(imfs_target, axis=0)

# 将DataFrame保存为CSV文件
df_imfs.to_csv(r'D:\MASTER\my_essay\DATA\CEEMDAN\Wusheng-imfs.csv', index=False)
df_residuals.to_csv(r'D:\MASTER\my_essay\DATA\CEEMDAN\Wusheng-residuals.csv', index=False)




# # 可视化分解后的 IMF
# num_imfs = 5  # IMF的数量
#
# # 绘制每个变量的每个 IMF
# plt.figure(figsize=(30, 20))
# for var in range(X.shape[1]):
#     plt.subplot(X.shape[1], num_imfs + 2, var * (num_imfs + 2) + var + 1, label=f'Original {var + 1}')
#     plt.plot(X[:, var], 'b')
#     plt.title(f'Variable {var+1}')
#     plt.xlabel('Time')
#
#     imfs_var = imfs_all[var]
#     for i, imf in enumerate(imfs_var):
#         plt.subplot(X.shape[1], num_imfs + 2, var * (num_imfs + 2) + i + 2, label=f'IMF {i+1}')
#         plt.plot(imf, 'r')
#         plt.title(f'IMF {i+1}')
#         plt.xlabel('Time')
#
#     # 添加残差模态图
#     plt.subplot(X.shape[1], num_imfs + 2, var * (num_imfs + 2) + num_imfs + 2, label=f'Residual {var + 1}')
#     plt.plot(imfs_var[-1], 'g')
#     plt.title(f'Residual {var+1}')
#     plt.xlabel('Time')
#
# plt.tight_layout()
# plt.show()





