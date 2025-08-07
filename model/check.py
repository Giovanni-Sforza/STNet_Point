import torch

# 替换为您的 .pth 文件路径
file_path = 'experiments/PPC_Ru_test1_20250806-220852/ckpt_latest.pth'

try:
    # 加载模型的状态字典
    # 使用 map_location='cpu' 可以确保即使模型是在 GPU 上训练的，也能在只有 CPU 的环境上成功加载
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))

    # 有些情况下，保存的文件可能是一个包含 state_dict 和其他信息的字典
    # 常见键有 'model', 'state_dict', 'model_state_dict'
    # 如果直接加载的不是 state_dict，请尝试检查并提取
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    # 如果还有其他可能的键，可以在这里添加

    print(f"成功加载文件: {file_path}")
    print("-" * 50)
    print("层名 -> 参数形状:")
    print("-" * 50)

    # 遍历 state_dict 并打印每一层的名字和形状
    for layer_name, param in state_dict.items():
        print(f"{layer_name} -> {param.shape}")

    print("-" * 50)

except FileNotFoundError:
    print(f"错误：文件未找到，请检查路径 '{file_path}'")
except Exception as e:
    print(f"加载文件时发生错误: {e}")