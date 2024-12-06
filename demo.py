import numpy as np

# 假设有两个浮点数列表
list1 = [1.0, 2.0, 3.0]
list2 = [4.0, 5.0, 6.0]

# 将列表转换为 NumPy 数组
array1 = np.array(list1)
array2 = np.array(list2)

# 元素级别的加法
add_result = array1 + array2* array1
print("Addition:", add_result)

# 元素级别的减法
sub_result = array1 - array2
print("Subtraction:", sub_result)

# 元素级别的乘法
mul_result = array1 * array2
print("Multiplication:", mul_result)

# 元素级别的除法
div_result = array1 / array2
print("Division:", div_result)