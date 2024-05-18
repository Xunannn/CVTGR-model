import numpy as np
import pandas as pd
from scipy.stats import norm  # 用于标准正态分布
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # 用于日期格式化

data = pd.read_csv(r"D:\MASTER\my_paper\paper_2\Result\XAJ-TCN-GRU\Pre_Wuding.csv")
# 移除所有包含 NaN 的行
data = data.dropna()
pre = data.iloc[:, 0].values
observed = data.iloc[:, 1].values
# print("pre:",pre)
# 计算标准误差
std_error = np.std(observed - pre)
print("std_error:", std_error)
data2 = pd.read_csv(r"D:\MASTER\my_paper\paper_2\Result\XAJ-TCN-GRU\Pre_Chu.csv")
pre2 = data2.iloc[:, 0].values
observed2 = data2.iloc[:, 1].values
std_error2 = np.std(observed2 - pre2)
# print("observed2:", observed2)
data3 = pd.read_csv(r"D:\MASTER\my_paper\paper_2\Result\XAJ-TCN-GRU\Pre_Jianxi.csv")
pre3 = data3.iloc[:, 0].values
observed3 = data3.iloc[:, 1].values
std_error3 = np.std(observed3 - pre3)
# print("observed2:", observed2)
data4 = pd.read_csv(r"D:\MASTER\my_paper\paper_2\Result\XAJ-TCN-GRU\Pre_Qingyi.csv")
pre4 = data4.iloc[:, 0].values
observed4 = data4.iloc[:, 1].values
std_error4 = np.std(observed4 - pre4)
# 使用正态分布的分位数（norm.ppf）来计算不同置信水平下的上下置信界限
# 对于漓江数据，分别计算了90%、80%和70%的置信区间
z = norm.ppf(0.95)
lower_bound90 = pre[:] - z * std_error
upper_bound90 = pre[:] + z * std_error
print(len(lower_bound90))
print(len(upper_bound90))
# print("Upper Bound 90:", upper_bound90)
# print("Lower Bound 90:", lower_bound90)
# print("Observed Data:", observed)
z = norm.ppf(0.9)
lower_bound80 = pre[:] - z * std_error
upper_bound80 = pre[:] + z * std_error
z = norm.ppf(0.8)
lower_bound70 = pre[:] - z * std_error
upper_bound70 = pre[:] + z * std_error
print(len(upper_bound70))

z2 = norm.ppf(0.99998)
lower_bound902 = pre2[:] - z2 * std_error2
upper_bound902 = pre2[:] + z2 * std_error2
print(len(upper_bound902))
z2 = norm.ppf(0.995)
lower_bound802 = pre2[:] - z2 * std_error2
upper_bound802 = pre2[:] + z2 * std_error2
z2 = norm.ppf(0.99)
lower_bound702 = pre2[:] - z2 * std_error2
upper_bound702 = pre2[:] + z2 * std_error2

z3 = norm.ppf(0.995)
lower_bound903 = pre3[:] - z3 * std_error3
upper_bound903 = pre3[:] + z3 * std_error3
print(len(upper_bound903))
z3 = norm.ppf(0.9)
lower_bound803 = pre3[:] - z3 * std_error3
upper_bound803 = pre3[:] + z3 * std_error3
z3 = norm.ppf(0.8)
lower_bound703 = pre3[:] - z3 * std_error3
upper_bound703 = pre3[:] + z3 * std_error3
#
z4 = norm.ppf(0.995)
lower_bound904 = pre4[:] - z4 * std_error4
upper_bound904 = pre4[:] + z4 * std_error4
print(len(upper_bound904))
z4 = norm.ppf(0.9)
lower_bound804 = pre4[:] - z4 * std_error4
upper_bound804 = pre4[:] + z4 * std_error4
z4 = norm.ppf(0.8)
lower_bound704 = pre4[:] - z4 * std_error4
upper_bound704 = pre4[:] + z4 * std_error4

plt.figure(figsize=(10, 5))  # 8英寸宽，6英寸高
# 创建一个数组 x 包含 0 到 1415 的整数
x = np.arange(0, 998)
# 配置字体和坐标轴刻度的方向
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
print(len(x))


# ----------------α=0.05---------------
# 创建一个图形，包含4个子图
# plt.subplot(2, 2, 1)
# ax = plt.gca()   # 获取当前子图的轴对象
# # 设置 x 轴的日期格式为 'YYYY-MM-DD'，并指定日期间隔为 15 天
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# ax.xaxis.set_major_locator(mdates.DayLocator(interval=190))
# # 指定 x 轴刻度标签
# xticklabels = ['2020/12/6','2021/6/14','2021/12/21','2022/6/29','2023/1/15','2023/7/24']
# # 将 x 轴的刻度标签设置为指定的标签，并旋转标签文字以适应布局
# ax.set_xticklabels(xticklabels, rotation=0, fontsize=8)
# ax.yaxis.set_tick_params(labelsize=8)
# plt.title("Wuding River")   # 设置子图的标题
# # 使用 fill_between 方法绘制预测区间，alpha=0.6 用于设置透明度，color 设置颜色
# plt.fill_between(x, upper_bound90, lower_bound90, alpha=0.6, color='deepskyblue')
# plt.plot(observed, color='grey', linewidth=0.8)
# plt.ylabel("streamflow (m³/s)", fontsize=11)
# plt.legend(["Observed", "Prediction interval"], fontsize=7, loc='upper right')
# plt.text(30, 500, 'α=0.05')    # 在子图内部添加文本，用于标记 α 值
#
# plt.subplot(2, 2, 2)
# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# ax.xaxis.set_major_locator(mdates.DayLocator(interval=190))
# xticklabels = ['2020/12/6','2021/6/14','2021/12/21','2022/6/29','2023/1/15','2023/7/24']
# ax.set_xticklabels(xticklabels, rotation=0, fontsize=8)
# ax.yaxis.set_tick_params(labelsize=8)
# plt.title("Chu River")
# plt.fill_between(x, upper_bound902, lower_bound902, alpha=0.6, color='deepskyblue')
# plt.plot(observed2, color='grey', linewidth=0.8)
# plt.ylabel("streamflow (m³/s)", fontsize=11)
# plt.legend(["Observed", "Prediction interval"], fontsize=7, loc='upper right')
# plt.text(30, 400, 'α=0.05')
#
# plt.subplot(2, 2, 3)
# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# ax.xaxis.set_major_locator(mdates.DayLocator(interval=190))
# xticklabels = ['2020/12/6','2021/6/14','2021/12/21','2022/6/29','2023/1/15','2023/7/24']
# ax.set_xticklabels(xticklabels, rotation=0, fontsize=8)
# ax.yaxis.set_tick_params(labelsize=8)
# plt.title("Jianxi River")
# plt.fill_between(x, upper_bound903, lower_bound903, alpha=0.6, color='deepskyblue')
# plt.plot(observed3, color='grey', linewidth=0.8)
# plt.ylabel("streamflow (m³/s)", fontsize=11)
# plt.legend(["Observed", "Prediction interval"], fontsize=7, loc='upper right')
# plt.text(30, 750, 'α=0.05')
#
# plt.subplot(2, 2, 4)
# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# ax.xaxis.set_major_locator(mdates.DayLocator(interval=190))
# xticklabels = ['2020/12/6','2021/6/14','2021/12/21','2022/6/29','2023/1/15','2023/7/24']
# ax.set_xticklabels(xticklabels, rotation=0, fontsize=8)
# ax.yaxis.set_tick_params(labelsize=8)
# plt.title("Qingyi River")
# plt.fill_between(x, upper_bound904, lower_bound904, alpha=0.6, color='deepskyblue')
# plt.plot(observed4, color='grey', linewidth=0.8)
# plt.ylabel("streamflow (m³/s)", fontsize=11)
# plt.legend(["Observed", "Prediction interval"], fontsize=7, loc='upper right')
# plt.text(30, 1000, 'α=0.05')

# ----------------α=0.1---------------
# plt.subplot(2, 2, 1)
# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# ax.xaxis.set_major_locator(mdates.DayLocator(interval=190))
# xticklabels = ['2020/12/6','2021/6/14','2021/12/21','2022/6/29','2023/1/15','2023/7/24']
# ax.set_xticklabels(xticklabels, rotation=0, fontsize=8)
# ax.yaxis.set_tick_params(labelsize=8)
# plt.title("Wuding River")
# plt.fill_between(x, upper_bound80, lower_bound80, alpha=0.6, color='orange')
# plt.plot(observed, color='grey', linewidth=0.8)
# plt.ylabel("streamflow (m³/s)", fontsize=11)
# plt.legend(["Observed", "Prediction interval"], fontsize=7, loc='upper right')
# plt.text(30, 500, 'α=0.1')
#
# plt.subplot(2, 2, 2)
# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# ax.xaxis.set_major_locator(mdates.DayLocator(interval=190))
# xticklabels = ['2020/12/6','2021/6/14','2021/12/21','2022/6/29','2023/1/15','2023/7/24']
# ax.set_xticklabels(xticklabels, rotation=0, fontsize=8)
# ax.yaxis.set_tick_params(labelsize=8)
# plt.title("Chu River")
# plt.fill_between(x, upper_bound802, lower_bound802, alpha=0.6, color='orange')
# plt.plot(observed2, color='grey', linewidth=0.8)
# plt.ylabel("streamflow (m³/s)", fontsize=11)
# plt.legend(["Observed", "Prediction interval"], fontsize=7, loc='upper right')
# plt.text(30, 400, 'α=0.1')
#
# plt.subplot(2, 2, 3)
# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# ax.xaxis.set_major_locator(mdates.DayLocator(interval=190))
# xticklabels = ['2020/12/6','2021/6/14','2021/12/21','2022/6/29','2023/1/15','2023/7/24']
# ax.set_xticklabels(xticklabels, rotation=0, fontsize=8)
# ax.yaxis.set_tick_params(labelsize=8)
# plt.title("Jianxi River")
# plt.fill_between(x, upper_bound803, lower_bound803, alpha=0.6, color='orange')
# plt.plot(observed3, color='grey', linewidth=0.8)
# plt.ylabel("streamflow (m³/s)", fontsize=11)
# plt.legend(["Observed", "Prediction interval"], fontsize=7, loc='upper right')
# plt.text(30, 750, 'α=0.1')
#
# plt.subplot(2, 2, 4)
# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# ax.xaxis.set_major_locator(mdates.DayLocator(interval=190))
# xticklabels = ['2020/12/6','2021/6/14','2021/12/21','2022/6/29','2023/1/15','2023/7/24']
# ax.set_xticklabels(xticklabels, rotation=0, fontsize=8)
# ax.yaxis.set_tick_params(labelsize=8)
# plt.title("Qingyi River")
# plt.fill_between(x, upper_bound804, lower_bound804, alpha=0.6, color='orange')
# plt.plot(observed4, color='grey', linewidth=0.8)
# plt.ylabel("streamflow (m³/s)", fontsize=11)
# plt.legend(["Observed", "Prediction interval"], fontsize=7, loc='upper right')
# plt.text(30, 1000, 'α=0.1')
# #
# # ----------------α=0.2---------------

plt.subplot(2, 2, 1)
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=190))
xticklabels = ['2020/12/6','2021/6/14','2021/12/21','2022/6/29','2023/1/15','2023/7/24']
ax.set_xticklabels(xticklabels, rotation=0, fontsize=8)
ax.yaxis.set_tick_params(labelsize=8)
plt.title("Wuding River")
plt.fill_between(x, upper_bound70, lower_bound70, alpha=0.6, color='red')
plt.plot(observed, color='grey', linewidth=0.8)
plt.ylabel("streamflow (m³/s)", fontsize=11)
plt.legend(["Observed", "Prediction interval"], fontsize=7, loc='upper right')
plt.text(30, 500, 'α=0.2')

plt.subplot(2, 2, 2)
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=190))
xticklabels = ['2020/12/6','2021/6/14','2021/12/21','2022/6/29','2023/1/15','2023/7/24']
ax.set_xticklabels(xticklabels, rotation=0, fontsize=8)
ax.yaxis.set_tick_params(labelsize=8)
plt.title("Chu River")
plt.fill_between(x, upper_bound702, lower_bound702, alpha=0.6, color='red')
plt.plot(observed2, color='grey', linewidth=0.8)
plt.ylabel("streamflow (m³/s)", fontsize=11)
plt.legend(["Observed", "Prediction interval"], fontsize=7, loc='upper right')
plt.text(30, 400, 'α=0.2')

plt.subplot(2, 2, 3)
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=190))
xticklabels = ['2020/12/6','2021/6/14','2021/12/21','2022/6/29','2023/1/15','2023/7/24']
ax.set_xticklabels(xticklabels, rotation=0, fontsize=8)
ax.yaxis.set_tick_params(labelsize=8)
plt.title("Jianxi River")
plt.fill_between(x, upper_bound703, lower_bound703, alpha=0.6, color='red')
plt.plot(observed3, color='grey', linewidth=0.8)
plt.ylabel("streamflow (m³/s)", fontsize=11)
plt.legend(["Observed", "Prediction interval"], fontsize=7, loc='upper right')
plt.text(30, 750, 'α=0.2')

plt.subplot(2, 2, 4)
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=190))
xticklabels = ['2020/12/6','2021/6/14','2021/12/21','2022/6/29','2023/1/15','2023/7/24']
ax.set_xticklabels(xticklabels, rotation=0, fontsize=8)
ax.yaxis.set_tick_params(labelsize=8)
plt.title("Qingyi River")
plt.fill_between(x, upper_bound704, lower_bound704, alpha=0.6, color='red')
plt.plot(observed4, color='grey', linewidth=0.8)
plt.ylabel("streamflow (m³/s)", fontsize=11)
plt.legend(["Observed", "Prediction interval"], fontsize=7, loc='upper right')
plt.text(30, 1000, 'α=0.2')
# #
# #
plt.subplots_adjust(top=0.945,
bottom=0.141,
left=0.061,
right=0.987,
hspace=0.294,
wspace=0.157)
plt.savefig(r'D:\MASTER\my_paper\paper_2\figure\区间预测0.2.png', dpi=300)
plt.show()


# sum = 0
# naw = 0
# lb = lower_bound90
# ub = upper_bound90
# ymax = np.max(observed)
# ymin = np.min(observed)
# for i in range(998):
#     if (observed[i] >= lb[i] and observed[i] <= ub[i]):
#         sum += 1
#     naw += (ub[i] - lb[i]) / (ymax - ymin)
# print("picp90= ", sum / 998, "，naw90= ", naw / 998)
#
# sum = 0
# naw = 0
# lb = lower_bound80
# ub = upper_bound80
# ymax = np.max(observed)
# ymin = np.min(observed)
# for i in range(998):
#     if (observed[i] >= lb[i] and observed[i] <= ub[i]):
#         sum += 1
#     naw += (ub[i] - lb[i]) / (ymax - ymin)
# print("picp80= ", sum / 998, "，naw80= ", naw / 998)
#
# sum = 0
# naw = 0
# lb = lower_bound70
# ub = upper_bound70
# ymax = np.max(observed)
# ymin = np.min(observed)
# for i in range(998):
#     if (observed[i] >= lb[i] and observed[i] <= ub[i]):
#         sum += 1
#     naw += (ub[i] - lb[i]) / (ymax - ymin)
# print("picp70= ", sum / 998, "，naw70= ", naw / 998)