import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from typing import Tuple
try:
    from model.set_transformer import SetTransformer_point,DeepSet,SetTransformer
except ImportError:
    print("Warning: Could not import from 'set_transformer'.")

class STDT(nn.Module):
    def __init__(self, num_classes: int, output_dim: int = 512, input_feature_dim: int = 3, num_events: int = 32, feature_out=False):
        super(STDT, self).__init__()
        self.output_dim = output_dim
        self.input_feature_dim = input_feature_dim
        self.num_events = num_events
        self.num_classes = num_classes
        self.feature_out = feature_out

        self.dim_input= self.input_feature_dim +3
        self.point_backends = SetTransformer_point(dim_input = self.dim_input,num_outputs=1,dim_output=self.output_dim)
        
        self.DeepSet = SetTransformer(dim_input=self.output_dim,num_outputs=1,dim_output=self.output_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.output_dim, 256, bias=False), # 增大了第一层的宽度
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5), # 增加了Dropout率，因为分类器可能更容易过拟合
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(128, self.num_classes)
        )
    def forward(self, xyz_cluster):
        """
        对一个事件簇进行特征提取、池化和分类。

        Args:
            xyz_cluster (torch.Tensor): 一个批次的事件簇坐标数据。
                                        Shape: [B, E, N, 3]
                                        B: batch size (批次大小)
                                        E: num_events (每个簇的事件数)
                                        N: num_points (每个事件的点数)
            feat_cluster (torch.Tensor): 一个批次的事件簇特征数据。
                                         Shape: [B, E, C_in, N]
        
        Returns:
            torch.Tensor: 分类的logits输出, Shape: [B, num_classes]
            或
            Tuple[torch.Tensor, torch.Tensor]: 如果 self.feature_out 为 True,
                                               返回 (logits, pooled_features)
        """
        event_features = []
        # 遍历簇中的每一个事件
        device = next(self.parameters()).device
        #print(device)
        for i in range(self.num_events):
            # 提取当前批次中所有样本的第 i 个事件的数据
            # xyz_i shape: [B, N, 3]
            # feat_i shape: [B, C_in, N]
            xyz_i = xyz_cluster[:, i, :, :].to(device).contiguous()
            
            
            # 使用第 i 个独立的 PointNet 进行特征提取
            # h_i shape: [B, output_dim, 1]
            #h_i = self.pointnet_backends[i](xyz_i, feat_i)
            h_i = self.point_backends(xyz_i)
            # 去掉最后一个维度，变为 [B, output_dim]
            #h_i_squeezed = h_i.squeeze(-1)
            #print("event_features",i,h_i_squeezed.shape)
            event_features.append(h_i)
            
        
        # 将特征列表堆叠成一个张量
        # stacked_features shape: [B, E, output_dim]
        stacked_features = torch.stack(event_features, dim=1)
        #print("stacked_features: ", stacked_features.shape)
        B = stacked_features.size(0)
        
        output = self.DeepSet(stacked_features)
        cls_output = self.classifier(output)
        #print(cls_output.shape)
        #pooled_features_transformer = self.attention_pooling(transformer_output_norm.permute(1,0,2).contiguous())
        #pooled_features = torch.cat([mean_features, std_features], dim=1)
        # 将池化后的特征送入分类器
        #pooled_features = mean_features
        
        
        if self.feature_out:
            pass
            print("have not write")
        else:
            return cls_output

def create_model(config):
    """ Create model with given config, including coord_dim """
    model_name = config.get('model_name', 'STDT')
    if model_name == 'STDT':
        model = STDT(
            input_feature_dim=config.get('input_feature_dim', 3), # Use a clearer name
            num_events = config.get("num_events",32),
            output_dim=config.get('output_dim', 512),
            num_classes=config.get('num_classes', 5),
            feature_out=config.get('feature_out', False)
        )
    return model

# --- 使用示例 ---
if __name__ == '__main__':
    # --- 模型超参数 ---
    BATCH_SIZE = 4       # 一次处理4个事件簇
    NUM_EVENTS = 128      # 每个簇包含32个事件
    NUM_POINTS = 600    # 每个事件有1024个粒子
    INPUT_DIM = 3        # 每个粒子的输入特征是三维动量 (px, py, pz)
    OUTPUT_DIM = 512     # PointNet输出的特征维度
    NUM_CLASSES = 5      # 分类任务的数量 (例如 Ru vs Zr)

    # --- 创建模型 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = STDT(
        num_classes=NUM_CLASSES,
        output_dim=OUTPUT_DIM,
        input_feature_dim=0,
        num_events=NUM_EVENTS,
        feature_out=False 
    ).to(device)

    print(f"模型总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # --- 创建模拟输入数据 ---
    # 粒子坐标 (这里用随机数模拟，实际应为动量)
    # Shape: [B, E, N, 3]
    mock_xyz = torch.randn(BATCH_SIZE, NUM_EVENTS, NUM_POINTS, 3).to(device).contiguous()
    


    print("\n--- 输入数据维度 ---")
    print(f"坐标 (xyz_cluster): {mock_xyz.shape}")
    #print(f"特征 (feat_cluster): {mock_features.shape}")

    # --- 模型前向传播 ---
    logits = model(mock_xyz)
    
    print("\n--- 输出数据维度 ---")
    print(f"分类Logits: {logits.shape}")

    # --- 检查输出 ---
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
    print("\n模型运行成功，输出维度正确！")