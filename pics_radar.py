import plotly.express as px
import pandas as pd

metrics = ['Модуль упругости', 'Задержка', 'Использование ресурсов', 'Jain’s fairness index','Скорость фрагментации ресурсов']
algorithms = ['Lyapunov', 'DRF', 'Priority','FilterScore']
values = [
    [0.6167, 0.2663, 0.5459, 0.6489,0.7726],  # Lyapunov
    [0.2385, 13.1076, 0.4538, 0.6121,0.8189],  # DRF
    [0.7013, 1.4942, 0.4463, 0.5882,0.1907],     # Priority
    [0.1782, 17.5362, 0.4647, 0.6579,0.3663]     # FS
]

# 创建DataFrame
df = pd.DataFrame(values, index=algorithms, columns=metrics)

# 归一化函数（按列处理）
def min_max_normalize(column):
    min_val = column.min()
    max_val = column.max()
    # 处理极差为0的情况（所有值相同）
    if max_val == min_val:
        return pd.Series([0.5]*len(column), index=column.index)
    return (column - min_val) / (max_val - min_val)

# 应用归一化
df_normalized = df.apply(min_max_normalize)

# 转换为长格式
df_long = df_normalized.reset_index().melt(
    id_vars='index',
    var_name='metric',
    value_name='value'
)

# 绘制雷达图
fig = px.line_polar(
    df_long,
    r='value',
    theta='metric',
    line_close=True,
    color='index',  # 根据算法名称区分颜色
    color_discrete_sequence=['#2A9D8F','#E9C46A','#E76F51','#82B0D2'],  # 指定颜色
    template="plotly_dark"
)

fig.update_traces(fill='toself', line=dict(width=4))
fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),  # 固定范围到[0,1]
    font=dict(family="Arial", size=14),
    title="Multi-Dimensional Algorithm Comparison (Normalized)",
    width=800,
    height=600
)

fig.show()
