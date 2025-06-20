import re

# --- 配置 ---
input_filename = 'contra_data_clean_noipad/fine_test_Ru copy.txt'      # 输入文件名
output_filename = 'contra_data_clean_noipad/fine_test_Ru.txt' # 输出过滤后的文件名
# 要移除的模式中的数字
# 规则: 移除 *-N_*.npy, 其中 N 是下面列表中的任意一个数字
numbers_to_remove = ['3', '4', '16', '17'] 

# --- 主程序 ---

# 1. 构建一个正则表达式
# 这个表达式会匹配任何以 "-数字_" 结尾的模式，其中“数字”是我们要移除的数字之一
# 例如: '-(3|4|15|16)_'
pattern_to_remove = re.compile(r'-(' + '|'.join(numbers_to_remove) + r')_')

# 2. 读取并过滤文件名
print(f"正在从 {input_filename} 读取文件名...")
try:
    with open(input_filename, 'r', encoding='utf-8') as f_in:
        # 读取所有行，并去除每行末尾的换行符
        all_filenames = [line.strip() for line in f_in if line.strip()]

    # 使用列表推导式进行过滤
    # re.search(pattern, string) 会在string中寻找pattern，如果找到就不是None
    # 我们保留那些 re.search 返回 None 的文件名
    filtered_filenames = [
        name for name in all_filenames 
        if not pattern_to_remove.search(name)
    ]

    # 3. 将过滤后的结果写入新文件
    with open(output_filename, 'w', encoding='utf-8') as f_out:
        for name in filtered_filenames:
            f_out.write(name + '\n')

    print("-" * 30)
    print(f"处理完成！")
    print(f"总共处理了 {len(all_filenames)} 个文件名。")
    print(f"移除了 {len(all_filenames) - len(filtered_filenames)} 个文件名。")
    print(f"过滤后的列表已保存到: {output_filename}")

except FileNotFoundError:
    print(f"错误：找不到输入文件 '{input_filename}'。请确保文件存在于正确的位置。")