import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_bvp


# 定义一个函数来读取数据文件
def read_data(file_path):
    """
    从指定路径读取数据文件，并过滤出特定范围的数据。

    参数:
        file_path (str): 数据文件的路径。

    返回:
        numpy.ndarray: 过滤后的数据数组。
    """
    # 使用 pandas 读取数据文件，设置分隔符为空白字符，跳过前9行，没有表头
    data = pd.read_csv(file_path, sep='\s+', header=None, skiprows=9)
    # 过滤 x 在 [0, 1.3] 和 y 在 [1, 1.3] 范围内的数据
    #filtered_data = data[(data[0] >= 0) & (data[0] <= 1.3) & (data[1] >= 1) & (data[1] <= 1.26)]
    # 将过滤后的 Pandas DataFrame 转换为 NumPy 数组
    data_array = data.to_numpy()
    return data_array


# 定义一个函数将坐标映射到整数索引
def map_coordinates_to_indices(coords):
    """
    将给定的一维坐标数组映射到唯一的整数索引。

    参数:
        coords (numpy.ndarray): 坐标数组。

    返回:
        tuple: 包含两个元素的元组，第一个是整数索引数组，第二个是唯一的坐标值。
    """
    # 获取唯一坐标值及它们在原数组中的位置
    unique_coords, inverse = np.unique(coords, return_inverse=True)
    # 将逆向索引重新组织为原始坐标数组的形状
    indices = inverse.reshape(coords.shape)
    return indices, unique_coords



# 读取数据
alpha_n_data = read_data('semi.iig1.alpha_n.txt')
alpha_p_data = read_data('semi.iig1.alpha_p.txt')

# 获取唯一坐标值并映射到整数索引
x_coords = alpha_n_data[:, 0]
y_coords = alpha_n_data[:, 1]

x_indices, unique_x = map_coordinates_to_indices(x_coords)
y_indices, unique_y = map_coordinates_to_indices(y_coords)

max_x = len(unique_x)
max_y = len(unique_y)


# 将数据转换成网格形式
alpha_n_grid = np.full((max_y, max_x), np.nan)
alpha_p_grid = np.full((max_y, max_x), np.nan)

for data, grid in zip([alpha_n_data, alpha_p_data], [alpha_n_grid, alpha_p_grid]):
    for i, row in enumerate(data):
        x, y, value = x_indices[i], y_indices[i], row[2]
        if not np.isnan(value):
            grid[y, x] = value

# 设置全局字体大小
plt.rcParams.update({'font.size': 8})  # 将全局字体大小设置为8


#################################################################################################
#################################################################################################


def derivatives(x, y, alpha_n, alpha_p):
    n, p = y
    alpha_n_x = np.interp(x, x_values, alpha_n)
    alpha_p_x = np.interp(x, x_values, alpha_p)

    dn_dx = (1 - n) * alpha_n_x * (n + p - n * p)
    dp_dx = -(1 - p) * alpha_p_x * (n + p - n * p)

    return np.vstack((dn_dx, dp_dx))


# Boundary conditions function
def boundary_conditions(ya, yb):
    return np.array([ya[0] - n_initial, yb[1] - p_final])

# Define the derivatives for n and p
# Define the domain and initial conditions
alpha_n = alpha_n_grid[:, 0]  # Example values
alpha_p = alpha_p_grid[:, 0]  # Example values
x_values = np.linspace(min(unique_y)* 1e-6, max(unique_y)* 1e-6, alpha_n.shape[0])
n_initial = 0  # Example initial value for n at x = 0
p_final = 0  # Example final value for p at x = 1


# Initial guess for n and p (linear guess)
y_guess = np.zeros((2, x_values.size))
y_guess[0, :] = np.linspace(n_initial, 1, x_values.size)  # Linear guess for n
y_guess[1, :] = np.linspace(1, p_final, x_values.size)  # Linear guess for p

# Solve the boundary value problem
solution = solve_bvp(
    lambda x, y: derivatives(x, y, alpha_n, alpha_p),
    boundary_conditions,
    x_values,
    y_guess,
    tol=1e-6,  # 减小容差
    max_nodes=5000,  # 增加最大节点数
    verbose=2  # 查看详细信息

)



# Extract the solutions
n_solution = solution.sol(x_values)[0]
p_solution = solution.sol(x_values)[1]

fontsize = 16; title_fontsize = 24
# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(x_values, alpha_n, label='αn', color='blue')
plt.plot(x_values, alpha_p, label='αp', color='red')
plt.xlabel('x(μm)', fontsize=fontsize)
plt.ylabel('αn αp(1/m)',fontsize=fontsize)
plt.title('Ionization coefficient electron(1/m)',fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
ax = plt.gca()
offset_text = ax.xaxis.get_offset_text()
offset_text.set_size(fontsize)
offset_text = ax.yaxis.get_offset_text()
offset_text.set_size(fontsize)
# 设置字体大小
#offset_text.set_position((1.05, 1))  # 可选：调整位置
plt.grid(True)
#plt.show()


# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(x_values, n_solution, label='Pn(x)', color='blue')
plt.plot(x_values, p_solution, label='Pp(x)', color='red')
plt.xlabel('x(μm)',fontsize=fontsize)
plt.ylabel('Value(100%)',fontsize=fontsize)
plt.title('Solutions for Pn(x) and Pp(x)',fontsize=title_fontsize)
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