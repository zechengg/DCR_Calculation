# 定义输入和输出文件路径
input_file_path = 'alpha.txt'
output_file_path = 'points_coordinate.txt'

# 初始化一个计数器来跟踪行号
line_count = 0

# 打开输入文件进行读取，并创建或打开输出文件进行写入
with open(input_file_path, 'r', encoding='utf-8') as infile, \
        open(output_file_path, 'w', encoding='utf-8') as outfile:
    # 逐行读取文件
    for line in infile:
        line_count += 1

        # 如果当前行号大于等于9，则开始处理
        if line_count >= 9:
            # 分割每一行的数据（默认以空格为分隔符）
            columns = line.strip().split()

            # 检查是否至少有两列数据
            if len(columns) >= 2:
                # 只取前两列并用制表符连接起来
                new_line = '\t'.join(columns[:2]) + '\n'
                # 写入新的文件
                outfile.write(new_line)