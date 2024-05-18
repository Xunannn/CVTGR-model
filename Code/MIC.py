import pandas as pd
from minepy import MINE

# 读取数据
data = pd.read_csv(r'D:\MASTER\my_paper\paper_2\Processed data\DL\original-Getang.csv')

# 构建MINE对象
mine = MINE(alpha=0.6, c=15)

# 初始化最大信息系数和对应的特征
mic_scores = []
mic_features = []

# 遍历所有特征
for col in data.columns:
    # 计算特征与目标变量之间的最大信息系数
    mine.compute_score(data[col], data['runoff'])
    mic_score = mine.mic()

    # 将最大信息系数和对应的特征存入列表
    mic_scores.append(mic_score)
    mic_features.append(col)

# 将最大信息系数和对应的特征转化为DataFrame格式
mic_df = pd.DataFrame({'feature': mic_features, 'mic_score': mic_scores})

# 按照最大信息系数从大到小排序
mic_df.sort_values(by='mic_score', ascending=False, inplace=True)

# 选取最大信息系数前K个特征作为重要特征
K = 10
selected_features = mic_df['feature'][:K]

print('Selected features:')
print(selected_features)
