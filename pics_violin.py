import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据（假设数据列名为 drf, fs, pri, li）
df = pd.read_excel("uti_rate.xlsx")

# 将数据转换为长格式（便于seaborn绘制分组图）
data_long = df.melt(var_name="Algorithm", value_name="Value")

# 设置图形样式
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

# 绘制小提琴图（背景分布）
sns.violinplot(
    x="Algorithm",
    y="Value",
    data=data_long,
    palette="pastel",  # 浅色填充
    inner=None,        # 不显示内部图形
    saturation=0.5,    # 降低饱和度
    width=0.8          # 控制宽度
)

# 叠加箱线图（调整位置不重叠）
sns.boxplot(
    x="Algorithm",
    y="Value",
    data=data_long,
    width=0.3,         # 缩窄箱线图宽度
    palette="dark",     # 深色系
    boxprops=dict(alpha=0.8),  # 透明度
    linewidth=1.5,      # 边框线宽
    flierprops=dict(markerfacecolor="red", markersize=5)  # 异常点样式
)

# 标注标题和轴标签
plt.title("CPU utilization", fontsize=14)
plt.xlabel("Algorithms", fontsize=12)
plt.ylabel("Numerical distribution", fontsize=12)

# 添加图例（手动创建）
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='gray', lw=2, label='Box Plot (Median/Quartiles)'),
    Line2D([0], [0], color='skyblue', lw=4, alpha=0.5, label='Violin plot (density distribution))')
]
plt.legend(handles=legend_elements, loc="upper right")

plt.tight_layout()
plt.show()
