import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from typing import Tuple

try:
    from extensions.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModule
except ImportError:
    print("Warning: Could not import from 'extension.pointnet2_modules'.")

class PointNet(nn.Module):
    def __init__(self,  output_dim: int = 512, input_feature_dim: int = 3):
        super().__init__()
        self.output_dim = output_dim
        self.input_feature_dim = input_feature_dim
        

        
        # 解释: 最后一层通常是全局抽象，聚合所有特征。
        #self.sa0 = PointnetSAModule(npoint=512, radius=0.2, nsample=8, mlp=[self.input_feature_dim,  32, 64])
        self.sa1 = PointnetSAModule(npoint=128, radius=0.25, nsample=4, mlp=[ self.input_feature_dim, 64, 128])
        # 估算: r=0.4的球体积约0.268。期望点数 = 0.268 * 215 ≈ 57个。这下足够nsample=32了。
        self.sa2 = PointnetSAModule(npoint=32, radius=0.4, nsample=4, mlp=[128, 128, 256])
        self.sa3 = PointnetSAModule(npoint=None, radius=None, nsample=None, mlp=[256, 256, self.output_dim])

    def forward(self,xyz_A,feat_A):
        """
        xyz: [batchsize,Npoint,xyz]
        feature: [batchsize,feature,Npoint]
        """
        """l0_xyz_A, l0_feat_A = self.sa0(xyz_A, feat_A)
        l1_xyz_A, l1_feat_A = self.sa1(l0_xyz_A, l0_feat_A)
        l2_xyz_A, l2_feat_A = self.sa2(l1_xyz_A, l1_feat_A)
        l3_xyz_A, l3_feat_A = self.sa3(l2_xyz_A, l2_feat_A)"""
        l0_xyz_A, l0_feat_A = self.sa1(xyz_A, feat_A)
        l2_xyz_A, l2_feat_A = self.sa2(l0_xyz_A, l0_feat_A)
        l3_xyz_A, l3_feat_A = self.sa3(l2_xyz_A, l2_feat_A)
        return l3_feat_A

class SelfAttentionPooling(nn.Module):
    """
    自注意力池化层
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.input_dim = input_dim
        # 这是一个简化的自注意力实现，它学习一个查询向量来为每个输入特征计算重要性分数
        self.attention_weights_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, E, D]
                              B: 批次大小 (batch_size)
                              E: 事件数量 (num_events or sequence_length)
                              D: 特征维度 (feature_dimension)
        Returns:
            torch.Tensor: 池化后的张量，形状为 [B, D]
        """
        # 计算每个事件的注意力分数
        # attention_scores shape: [B, E, 1]
        attention_scores = self.attention_weights_layer(x)

        # 使用 softmax 将分数转换为权重 (在事件维度上)
        # attention_weights shape: [B, E, 1]
        attention_weights = F.softmax(attention_scores, dim=1)

        # 将权重应用到原始特征上 (输入x在此处充当 "Value")
        # 利用广播机制进行逐元素相乘: (B, E, D) * (B, E, 1) -> (B, E, D)
        weighted_features = x * attention_weights

        # 沿着事件维度求和，得到最终的池化特征
        # pooled_features shape: [B, D]
        pooled_features = torch.sum(weighted_features, dim=1)

        return pooled_features

class PointPoolingClassifier(nn.Module):
    def __init__(self, num_classes: int, output_dim: int = 512, input_feature_dim: int = 3, num_events: int = 32, feature_out=False,
            transformer_layers: int = 4, transformer_heads: int = 8):
        super(PointPoolingClassifier, self).__init__()
        self.output_dim = output_dim
        self.input_feature_dim = input_feature_dim
        self.num_events = num_events
        self.num_classes = num_classes
        self.feature_out = feature_out

        # 创建 self.num_events 个独立的PointNet模块
        # 使用 nn.ModuleList 将它们正确注册为模型的子模块
        #self.pointnet_backends = nn.ModuleList(
        #    [PointNet(output_dim=self.output_dim, input_feature_dim=self.input_feature_dim) for _ in range(self.num_events)]
        #)
        self.pointnet_backends = PointNet(output_dim=self.output_dim, input_feature_dim=self.input_feature_dim)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.output_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.output_dim, # 必须与 PointNet 的输出维度一致
            nhead=transformer_heads,   # 多头注意力的头数
            dim_feedforward=self.output_dim * 4, # 前馈网络的隐藏层维度，通常是 d_model 的4倍
            dropout=0.1
        )
        
        # 将多个 encoder_layer 堆叠起来
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=transformer_layers
        )
        # 分类头，对池化后的特征进行分类
        pooled_feature_dim = self.output_dim 
        self.post_transformer_norm = nn.LayerNorm(self.output_dim)
        self.final_norm = nn.LayerNorm(pooled_feature_dim)


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
 
    def forward(self, xyz_cluster, feat_cluster):
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
            Tuple[torch.Tensor, torch.Tensor]: 如果 self.feature_out 为 True，
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
            feat_i = feat_cluster[:, i, :, :].to(device).contiguous()
            
            # 使用第 i 个独立的 PointNet 进行特征提取
            # h_i shape: [B, output_dim, 1]
            #h_i = self.pointnet_backends[i](xyz_i, feat_i)
            h_i = self.pointnet_backends(xyz_i, feat_i)
            # 去掉最后一个维度，变为 [B, output_dim]
            h_i_squeezed = h_i.squeeze(-1)
            #print("event_features",i,h_i_squeezed.shape)
            event_features.append(h_i_squeezed)
            
        
        # 将特征列表堆叠成一个张量
        # stacked_features shape: [B, E, output_dim]
        stacked_features = torch.stack(event_features, dim=1)
        #print("stacked_features: ", stacked_features.shape)
        B = stacked_features.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # shape: [B, 1, output_dim]
        x = torch.cat((cls_tokens, stacked_features), dim=1)  # shape: [B, E+1, output_dim]
        # trnasformer requirs: [E,B,D]
        transformer_output = self.transformer_encoder(x.permute(1,0,2).contiguous())
        transformer_output_norm = self.post_transformer_norm(transformer_output) 
        cls_output = transformer_output_norm[0]  # [B, D]
        #print(cls_output.shape)
        #pooled_features_transformer = self.attention_pooling(transformer_output_norm.permute(1,0,2).contiguous())
        #pooled_features = torch.cat([mean_features, std_features], dim=1)
        # 将池化后的特征送入分类器
        #pooled_features = mean_features
        logits = self.classifier(cls_output)
        
        if self.feature_out:
            return logits, cls_output
        else:
            return logits

def create_model(config):
    """ Create model with given config, including coord_dim """
    model_name = config.get('model_name', 'PPC')
    if model_name == 'PPC':
        model = PointPoolingClassifier(
            input_feature_dim=config.get('input_feature_dim', 3), # Use a clearer name
            num_events = config.get("num_events",32),
            output_dim=config.get('output_dim', 512),
            num_classes=config.get('num_classes', 5),
            feature_out=config.get('feature_out', False),
            transformer_layers = config.get("transformer_layers",3),
            transformer_heads = config.get("transformer_heads",4)
        )
    return model

# --- 使用示例 ---
if __name__ == '__main__':
    # --- 模型超参数 ---
    BATCH_SIZE = 4       # 一次处理4个事件簇
    NUM_EVENTS = 344      # 每个簇包含32个事件
    NUM_POINTS = 1024    # 每个事件有1024个粒子
    INPUT_DIM = 3        # 每个粒子的输入特征是三维动量 (px, py, pz)
    OUTPUT_DIM = 512     # PointNet输出的特征维度
    NUM_CLASSES = 2      # 分类任务的数量 (例如 Ru vs Zr)

    # --- 创建模型 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = PointPoolingClassifier(
        num_classes=NUM_CLASSES,
        output_dim=OUTPUT_DIM,
        input_feature_dim=INPUT_DIM,
        num_events=NUM_EVENTS,
        feature_out=True # 设置为True来获取中间特征用于分析
    ).to(device)

    print(f"模型总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # --- 创建模拟输入数据 ---
    # 粒子坐标 (这里用随机数模拟，实际应为动量)
    # Shape: [B, E, N, 3]
    mock_xyz = torch.randn(BATCH_SIZE, NUM_EVENTS, NUM_POINTS, 3).to(device).contiguous()
    
    # 粒子特征 (PointNet++中，通常将坐标也作为初始特征)
    # Shape: [B, E, 3, N] -> PyTorch的卷积层需要通道维度在前
    mock_features = mock_xyz.permute(0, 1, 3, 2).to(device).contiguous()

    print("\n--- 输入数据维度 ---")
    print(f"坐标 (xyz_cluster): {mock_xyz.shape}")
    print(f"特征 (feat_cluster): {mock_features.shape}")

    # --- 模型前向传播 ---
    logits, pooled_feat = model(mock_xyz, mock_features)
    
    print("\n--- 输出数据维度 ---")
    print(f"分类Logits: {logits.shape}")
    print(f"池化后特征: {pooled_feat.shape}")

    # --- 检查输出 ---
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
    assert pooled_feat.shape == (BATCH_SIZE, OUTPUT_DIM)
    print("\n模型运行成功，输出维度正确！")