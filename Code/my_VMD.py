import numpy as np
import pandas as pd
from vmdpy import VMD

# 读取数据集
df = pd.read_csv(r'D:\MASTER\my_essay\DATA\CEEMDAN\Wusheng\IMF1.csv')
data = np.array(df)

# 参数设置
alpha = 10  # moderate bandwidth constraint
tau = 0  # noise-tolerance (no strict fidelity enforcement)
K = 5  # number of modes
DC = 0  # no DC part imposed
init = 2  # initialize omegas uniformly
tol = 1e-7

# 分解数据列
imfs_all = []
for column in range(data.shape[1]):
    column_data = data[:, column]
    u, u_hat, omega = VMD(column_data, alpha, tau, K, DC, init, tol)
    imfs_all.append(u)

# 获取最长的分解结果长度
max_length = max([imfs.shape[1] for imfs in imfs_all])

# 创建一个空的DataFrame来存储结果
df_imfs = pd.DataFrame()

# 将每个列的分解结果添加到DataFrame中
for column, imfs in enumerate(imfs_all):
    for i in range(imfs.shape[0]):
        imf = imfs[i]
        imf_padded = np.pad(imf, (0, max_length - imf.shape[0]), mode='constant', constant_values=np.nan)
        df_imfs[f'Column {column+1} IMF {i+1}'] = imf_padded

# 将DataFrame保存为CSV文件
df_imfs.to_csv(r'D:\MASTER\my_essay\DATA\CEEMDAN-VMD\Wusheng-IMF1\Vimfs1.csv', index=False)

print("分解结果已保存到CSV文件中。")










