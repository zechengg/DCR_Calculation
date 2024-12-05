import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_bvp, simps
from scipy.interpolate import griddata
from scipy.integrate import dblquad

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
Rsrh_data = read_data('semi.Rsrh.txt')
Rbtbt_data = read_data('semi.udg1.Gn.txt')

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
Rsrh_grid = np.full((max_y, max_x), np.nan)
Rbtbt_grid = np.full((max_y, max_x), np.nan)

for data, grid in zip([alpha_n_data, alpha_p_data, Rsrh_data, Rbtbt_data], [alpha_n_grid, alpha_p_grid, Rsrh_grid, Rbtbt_grid]):
    for i, row in enumerate(data):
        x, y, value = x_indices[i], y_indices[i], row[2]
        if not np.isnan(value):
            grid[y, x] = value

# 设置全局字体大小
plt.rcParams.update({'font.size': 8})  # 将全局字体大小设置为8

# 假设 alpha_n_grid, alpha_p_grid, x_indices, y_indices, alpha_n_data_filtered, alpha_p_data_filtered 已经定义

fig, axs = plt.subplots(1, 2, figsize=(6, 3), dpi=300)

# 绘制 αn 的图像
im1 = axs[0].imshow(alpha_n_grid, extent=[min(x_coords), max(x_coords), min(y_coords), max(y_coords)],origin='lower', aspect='auto', cmap='plasma')
axs[0].scatter(x_indices, y_indices, c=alpha_n_data[:, 2], cmap='plasma', s=5)
axs[0].set_title('αn', fontsize=10)  # 直接设置标题的字体大小

fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

# 绘制 αp 的图像
im2 = axs[1].imshow(alpha_p_grid, extent=[min(unique_x), max(unique_x), min(unique_y), max(unique_y)], origin='lower', aspect='auto', cmap='plasma')
axs[1].scatter(x_indices, y_indices, c=alpha_p_data[:, 2], cmap='plasma', s=5)
axs[1].set_title('αp', fontsize=10)  # 直接设置标题的字体大小
fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

#plt.tight_layout()
#plt.show()



#######################################################################################################
#######################################################################################################
# Generate example values for alpha_n and alpha_p
alpha_n = alpha_n_grid  # Example values
alpha_p = alpha_p_grid  # Example values

# Define the domain based on the shape of alpha_n and alpha_p
x_values = np.linspace(min(unique_x)* 1e-6, max(unique_x)* 1e-6, alpha_n.shape[1])
y_values = np.linspace(min(unique_y)* 1e-6, max(unique_y)* 1e-6, alpha_n.shape[0])

# Define the initial conditions
n_initial = 0  # Example initial value for n at y = 0
p_final = 0  # Example final value for p at y = 1

# Initial guess for n and p (linear guess)
y_guess = np.zeros((2, y_values.size))
y_guess[0, :] = np.linspace(n_initial, 0.5, y_values.size)  # Linear guess for n
y_guess[1, :] = np.linspace(0.5, p_final, y_values.size)  # Linear guess for p


# Define the partial derivatives for n and p with respect to y
def derivatives(y, Y, x, alpha_n, alpha_p):
    n, p = Y
    alpha_n_y = np.interp(y, y_values, alpha_n[:,x])
    alpha_p_y = np.interp(y, y_values, alpha_p[:,x])
    dn_dy = (1 - n) * alpha_n_y * (n + p - n * p)
    dp_dy = -(1 - p) * alpha_p_y * (n + p - n * p)
    return np.vstack((dn_dy, dp_dy))


# Boundary conditions function
def boundary_conditions(Ya, Yb):
    na, pa = Ya
    nb, pb = Yb
    return np.array([na - n_initial, pb - p_final])


# Store solutions
n_solutions = []
p_solutions = []

# Solve the boundary value problem for each x
for x in range(len(x_values)):
    solution = solve_bvp(
        lambda y, Y: derivatives(y, Y, x, alpha_n, alpha_p),
        boundary_conditions,
        y_values,
        y_guess,
        tol=1e-5,  # 减小容差
        max_nodes=50000,  # 增加最大节点数
        #verbose=2  # 查看详细信息
    )

    if not solution.success:
        print(
            f"Warning: The solution did not converge for x={x_values[x]}. Try adjusting the initial guess or boundary conditions.")

    n_solutions.append(solution.sol(y_values)[0])
    p_solutions.append(solution.sol(y_values)[1])

# Convert to numpy arrays for plotting
n_solutions = np.array(n_solutions)
p_solutions = np.array(p_solutions)

# Plot the results
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

X, Y = x_values, y_values
min_val = min(n_solutions.min(), p_solutions.min())
max_val = max(n_solutions.max(), p_solutions.max())
levels = np.linspace(min_val, max_val, 20)
contour_n = ax[0].contourf(X, Y, n_solutions.T, levels=levels, cmap='viridis')
plt.colorbar(contour_n, ax=ax[0])
ax[0].set_title('Solution for n(x, y)')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')

contour_p = ax[1].contourf(X, Y, p_solutions.T, levels=levels, cmap='viridis')
plt.colorbar(contour_p, ax=ax[1])
ax[1].set_title('Solution for p(x, y)')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')

plt.tight_layout()
plt.show()

###########################################################################################
###########################################################################################
values_srh = (n_solutions.T + p_solutions.T - n_solutions.T * p_solutions.T) * (-Rsrh_grid * 1e-18)
values_btbt = (n_solutions.T + p_solutions.T - n_solutions.T * p_solutions.T) * (Rbtbt_grid * 1e-18)

# 假设网格间距是 dx 和 dy
dx =  (max(unique_x) - min(unique_x)) / (max_x - 1)  # x 方向的间距
dy =  (max(unique_y) - min(unique_y)) / (max_y - 1)
# 计算积分
integral_srh = simps(simps(values_srh, dx=dx), dx=dy)
integral_btbt = simps(simps(values_btbt, dx=dx), dx=dy)

print(f"SRH积分结果: {integral_srh}")
print(f"BTBT积分结果: {integral_btbt}")