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
# 创建网格化坐标
X, Y = np.meshgrid(unique_x, unique_y)
# 绘制图像
fig, axs = plt.subplots(1, 2, figsize=(7, 3), dpi=300)

# 绘制 αn 的图像
pc1 = axs[0].pcolormesh(X, Y, Rsrh_grid, shading='auto', cmap='viridis')
#axs[0].set_title(r'$\alpha_n$ (Pcolormesh)', fontsize=12)
axs[0].set_title(r'$R_{srh}$ ', fontsize=12)
axs[0].set_xlabel('X (μm)')
axs[0].set_ylabel('Y (μm)')
axs[0].ticklabel_format(axis='both', style='sci', scilimits=(0,0), useOffset=True)
cb1 = fig.colorbar(pc1, ax=axs[0], fraction=0.046, pad=0.04)

# 绘制 αp 的图像
pc2 = axs[1].pcolormesh(X, Y, Rbtbt_grid, shading='auto', cmap='viridis')
axs[1].set_title(r'$R_{btbt}$ ', fontsize=12)
axs[1].set_xlabel('X (μm)')
axs[1].set_ylabel('Y (μm)')
cb2 = fig.colorbar(pc2, ax=axs[1], fraction=0.046, pad=0.04)

# 调整布局
plt.tight_layout()

# 显示图像
plt.show()
