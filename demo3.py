import BV_calculate as dp
import matplotlib.pyplot as plt
import numpy as np

# 读取和处理第一个文件
X, Y, αn = dp.read_and_process_file('semi.iig1.alpha_n.txt')
print(X)
# 读取和处理第二个文件
X, Y, αp = dp.read_and_process_file('semi.iig1.alpha_p.txt')

# 创建网格

# 将网格数据展平，以便于 scatter 函数使用
X_flat = X.flatten()
Y_flat = Y.flatten()
αn_flat = αn.flatten()
αp_flat = αp.flatten()

# 绘制αn散点图
plt.figure(figsize=(12, 9))
sc = plt.scatter(X_flat, Y_flat, c=αn_flat, cmap='viridis', s=50)
plt.title('Scatter Plot of αn with Uniform Sampling')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.colorbar(sc, label='αn Value')
plt.show()

# 绘制αp散点图
plt.figure(figsize=(12, 9))
sc = plt.scatter(X_flat, Y_flat, c=αp_flat, cmap='plasma', s=50)
plt.title('Scatter Plot of αp with Uniform Sampling')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.colorbar(sc, label='αp Value')
plt.show()