import matplotlib.pyplot as plt
import numpy as np

# 模拟数据
alpha_values = [30, 0]
eta_x = np.linspace(0.5, 3.5, 100)
jain_index_fair = [1 - (eta - 0.5) / 3 for eta in eta_x]
jain_index_opt = [1 - (eta - 0.5)**2 / 9 for eta in eta_x]

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制折线图
plt.plot(eta_x, jain_index_fair, 'r--', label=r'$\alpha$-fair based Tradeoff')
plt.plot(eta_x, jain_index_opt, 'b--', label=r'Opt. Tradeoff (Proc. Z)')

# 添加特殊点和注释
delta_eta = 0.33
delta_j = 0.2
eta1 = 2.0
jain1_fair = 1 - (eta1 - 0.5) / 3
jain1_opt = 1 - (eta1 - 0.5)**2 / 9

eta2 = eta1 + delta_eta
jain2_fair = 1 - (eta2 - 0.5) / 3
jain2_opt = 1 - (eta2 - 0.5)**2 / 9

plt.scatter([eta1, eta2], [jain1_fair, jain2_fair], color='red', marker='o')
plt.scatter([eta1, eta2], [jain1_opt, jain2_opt], color='blue', marker='o')

plt.annotate(f'Δη = {delta_eta*100:.0f}%', xy=(eta1, jain1_opt), xytext=(2.2, 0.7),
             arrowprops=dict(facecolor='green', shrink=0.05))
plt.annotate(f'ΔJ = {delta_j*100:.0f}%', xy=(eta1, jain1_fair), xytext=(2.2, 0.6),
             arrowprops=dict(facecolor='red', shrink=0.05))

# 设置坐标轴标签和标题
plt.xlabel('Sum Rate Efficiency, η(x), Mbit/s')
plt.ylabel("Jain's Index")
plt.title('Jain\'s Index vs Sum Rate Efficiency')

# 设置图例
plt.legend()

# 设置网格
plt.grid(True)

# 设置坐标轴范围
plt.xlim(0.5, 3.5)
plt.ylim(0.4, 1.0)

# 添加α值标签
plt.text(1, 0.95, r'$\alpha = 30$', fontsize=10, color='blue')
plt.text(3.2, 0.42, r'$\alpha = 0$', fontsize=10, color='red')

# 显示图形
plt.show()
