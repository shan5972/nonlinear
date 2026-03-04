import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 生成时序数据
time = pd.date_range('2023-01-01', periods=24, freq='H')
fifo_latency = np.random.normal(100, 20, 24)
drf_latency = np.random.normal(80, 15, 24)
rl_latency = np.random.normal(60, 10, 24)

plt.figure(figsize=(12,6))
sns.set_style("whitegrid")

# 绘制渐变区域
plt.fill_between(time, fifo_latency, alpha=0.3, color='#2A9D8F', label='FIFO')
plt.fill_between(time, drf_latency, alpha=0.3, color='#E9C46A', label='DRF')
plt.fill_between(time, rl_latency, alpha=0.3, color='#E76F51', label='RL')

# 添加趋势线
sns.lineplot(x=time, y=fifo_latency, color='#2A9D8F', linewidth=2.5)
sns.lineplot(x=time, y=drf_latency, color='#E9C46A', linewidth=2.5)
sns.lineplot(x=time, y=rl_latency, color='#E76F51', linewidth=2.5)

plt.title('Latency Trend under Bursty Workloads', fontsize=14, pad=20)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Latency (ms)', fontsize=12)
plt.legend(title='Algorithm', title_fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
