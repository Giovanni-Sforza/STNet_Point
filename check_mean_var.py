import numpy as np
import random
import os
import matplotlib.pyplot as plt

def analyze_particle_data(file_list_path, npy_files_directory, num_samples=100):
    """
    从npy文件列表中随机抽取样本，计算全局均值和标准差，
    并可视化每个文件内部特征的均值和方差分布。

    参数:
    file_list_path (str): 记录了npy文件名的txt文件的路径。
    npy_files_directory (str): 存放npy文件的目录路径。
    num_samples (int): 想要随机抽取的npy文件数量。
    """
    # 为了在matplotlib中正确显示中文和负号
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是一个常用的中文字体
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 读取TXT文件中的文件名列表
    try:
        with open(file_list_path, 'r') as f:
            all_filenames = [line.strip() for line in f]
    except FileNotFoundError:
        print(f"错误：找不到文件列表 '{file_list_path}'")
        return

    # 2. 随机抽取文件名
    if len(all_filenames) < num_samples:
        print(f"警告：文件总数 ({len(all_filenames)}) 小于指定的抽样数量 ({num_samples})。将使用所有文件。")
        sampled_filenames = all_filenames
        num_samples = len(all_filenames)
    else:
        sampled_filenames = random.sample(all_filenames, num_samples)

    # 3. 读取数据，同时计算全局和单个文件的统计量
    all_data_for_global_calc = []
    intra_file_means = []
    intra_file_vars = []

    print(f"开始处理 {len(sampled_filenames)} 个随机抽样的文件...")
    for filename in sampled_filenames:
        file_path = os.path.join(npy_files_directory, filename)
        try:
            data = np.load(file_path) # 加载.npy文件. [6]
            if data.shape != (600, 10):
                print(f"警告：文件 '{filename}' 的形状为 {data.shape}，非预期的 (600, 10)，已跳过。")
                continue

            # 用于计算全局统计量
            all_data_for_global_calc.append(data)

            # 计算该文件内部的均值和方差
            mean_in_file = np.mean(data, axis=0) # 沿粒子维度计算，得到(10,)的均值. [11, 12, 13]
            var_in_file = np.var(data, axis=0)   # 沿粒子维度计算，得到(10,)的方差
            intra_file_means.append(mean_in_file)
            intra_file_vars.append(var_in_file)

        except FileNotFoundError:
            print(f"警告：找不到文件 '{file_path}'，已跳过。")
        except Exception as e:
            print(f"读取文件 '{file_path}' 时出错: {e}")

    if not all_data_for_global_calc:
        print("错误：未能成功加载任何数据。请检查文件路径和文件内容。")
        return

    # --- 4. 计算并打印全局统计量 ---
    combined_data = np.vstack(all_data_for_global_calc)
    global_mean = np.mean(combined_data, axis=0)
    global_std = np.std(combined_data, axis=0) # 使用numpy.std()计算标准差. [16]

    print("\n" + "="*40)
    print("全局统计结果")
    print(f"从 {len(all_data_for_global_calc)} 个文件中加载的总粒子数: {combined_data.shape[0]}")
    print(f"特征维度: {combined_data.shape[1]}")
    print("\n特征的全局均值 (Global Mean):")
    print(global_mean)
    print("\n特征的全局标准差 (Global Std):")
    print(global_std)
    print("="*40 + "\n")


    # --- 5. 可视化单个文件的统计量分布 ---
    if intra_file_means and intra_file_vars:
        # 将列表转换为Numpy数组，方便绘图
        # 此时两个数组的shape都为 (num_samples, 10)
        means_array = np.array(intra_file_means)
        vars_array = np.array(intra_file_vars)

        # 创建一个包含两个子图的图表
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))

        # 绘制均值的箱形图
        axes[0].boxplot(means_array)
        axes[0].set_title(f'{num_samples}个随机文件内部的【特征均值】分布', fontsize=16)
        axes[0].set_ylabel('特征均值 (Mean)', fontsize=12)
        axes[0].set_xlabel('特征维度 (Feature Index)', fontsize=12)
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # 绘制方差的箱形图
        axes[1].boxplot(vars_array)
        axes[1].set_title(f'{num_samples}个随机文件内部的【特征方差】分布', fontsize=16)
        axes[1].set_ylabel('特征方差 (Variance)', fontsize=12)
        axes[1].set_xlabel('特征维度 (Feature Index)', fontsize=12)
        axes[1].grid(True, linestyle='--', alpha=0.6)

        # 调整子图间距并显示图像
        plt.tight_layout()
        print("正在生成统计分布图...")
        plt.show()
        plt.savefig("check_mean_var.png",dpi=300)

# --- 使用示例 ---
if __name__ == '__main__':
    # 请将以下路径替换为您的实际路径
    FILE_LIST_PATH = "ml_new2/ml_preprocessed_data_5class_new2/fine_train_Ru.txt"  # <-- 修改这里：您的txt文件名
    NPY_FILES_DIRECTORY = "ml_new2/ml_preprocessed_data_5class_new2/method_3"  # <-- 修改这里：您的npy文件所在的文件夹路径

    # 运行核心功能
    analyze_particle_data(FILE_LIST_PATH, NPY_FILES_DIRECTORY, num_samples=1000)