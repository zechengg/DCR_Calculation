import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import solve_bvp

# get the avalanche probability

# 假设数据文件的每一行格式为: x, y, line_id, alpha_n, alpha_p
data = pd.read_csv('data_files/alpha.txt', sep='\s+', header=None, names=['x', 'y', 'line_id', 'alpha_n', 'alpha_p'],skiprows=8)
data = data.to_numpy()
x = data[:, 0]*1e-6
y = data[:, 1]*1e-6
line_ids = data[:, 2]
alpha_n = data[:, 3]
alpha_p = data[:, 4]

# 找出所有不同的线段编号
unique_line_ids = np.unique(line_ids)

# 用于存储每条线段的结果
results = {}
# 对每个线段进行处理
for line_id in unique_line_ids:
    # 获取当前线段的数据索引
    indices = np.where(line_ids == line_id)[0]
    # 提取当前线段的数据
    x_segment = x[indices]
    y_segment = y[indices]
    alpha_n_segment = alpha_n[indices]
    alpha_p_segment = alpha_p[indices]
    # 计算线段相邻两点的距离
    distances = np.sqrt(np.diff(x_segment)**2 + np.diff(y_segment)**2)
    l = np.concatenate(([0], np.cumsum(distances)))

# 初始猜测

    n_initial = 0  # Example initial value for n at x = 0
    p_final = 0  # Example final value for p at x = 1
    y_guess = np.zeros((2, l.size))
    y_guess[0, :] = np.linspace(n_initial, 1, l.size)  # Linear guess for n
    y_guess[1, :] = np.linspace(1, p_final, l.size)  # Linear guess for p


    # 定义微分方程
    def ode(l_internal, y):
        n, p = y
        alpha_n_interp = np.interp(l_internal, l, alpha_n_segment)
        alpha_p_interp = np.interp(l_internal, l, alpha_p_segment)
        dn_dl = (1 - n) * alpha_n_interp * (n + p - n * p)
        dp_dl = -(1 - p) * alpha_p_interp * (n + p - n * p)

        return np.vstack((dn_dl, dp_dl))


    # 定义边界条件
    def bc(ya, yb):
        # 根据问题定义适当的边界条件
        return np.array([ya[0] - n_initial, yb[1] - p_final])


    # 使用 solve_bvp 求解
    solution = solve_bvp(
        ode,
        bc,
        l,
        y_guess,
        tol=1e-6,  # 减小容差
        max_nodes=5000,  # 增加最大节点数
        verbose=2 )
    # 存储结果
    n_solution = solution.sol(l)[0]
    p_solution = solution.sol(l)[1]

    # 将结果存入字典
    results[line_id] = {
        'l': l,
        'n_solution': n_solution,
        'p_solution': p_solution
    }

# 设置字体大小
fontsize = 16
title_fontsize = 24

# 绘制 Pn(x) 和 Pp(x) 的图
plt.figure(figsize=(10, 5))
for line_id, result in results.items():
    l = result['l']
    n_solution = result['n_solution']
    p_solution = result['p_solution']
    plt.plot(l, n_solution, label=f'Pn (Line {line_id})', linestyle='--')
    plt.plot(l, p_solution, label=f'Pp (Line {line_id})', linestyle='-')
plt.xlabel('x (μm)', fontsize=fontsize)
plt.ylabel('Value (100%)', fontsize=fontsize)
plt.title('Solutions for Pn(x) and Pp(x)', fontsize=title_fontsize)
plt.legend(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
ax = plt.gca()
offset_text = ax.xaxis.get_offset_text()
offset_text.set_size(fontsize)
offset_text = ax.yaxis.get_offset_text()
offset_text.set_size(fontsize)
plt.grid(True)

plt.show()