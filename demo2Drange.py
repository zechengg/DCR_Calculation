import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取TXT文件
file_path = 'semi.iig1.alpha_p.txt'  # 替换为你的文件名
data = pd.read_csv(file_path, sep='\s+', header=None, skiprows=10)  # 使用sep='\s+'替代delim_whitespace

# 将数据转换为NumPy数组
data_array = data.to_numpy()

# 假设文件中有三列数据
x = data_array[:, 0]  # 第一列
y = data_array[:, 1]  # 第二列
z = data_array[:, 2]  # 第三列

# 过滤条件
x_mask = (x >= 0) & (x <= 1.3)  # x 在 (0, 1.3) 之间
y_mask = (y >= 1) & (y <= 1.3)  # y 在 (1, 1.3) 之间

# 结合两个条件
mask = x_mask & y_mask

# 应用过滤条件
x_filtered = x[mask]
y_filtered = y[mask]
z_filtered = z[mask]

# 检查过滤后的数据
print("Filtered x:", x_filtered)
print("Filtered y:", y_filtered)
print("Filtered z:", z_filtered)

# 绘制散点图
plt.figure(figsize=(10, 6))
scatter = plt.scatter(x_filtered, y_filtered, c=z_filtered, cmap='viridis', edgecolor='k', s=50)  # s 是点的大小
plt.colorbar(scatter, label='Value')  # 添加颜色条
plt.title('Scatter Plot of Filtered Data')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.xlim(0, 1.3)  # 设置x轴范围
plt.ylim(1, 1.3)  # 设置y轴范围
plt.grid()
plt.show()