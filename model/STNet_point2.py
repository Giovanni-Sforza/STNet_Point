import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from typing import Tuple

try:
    from extensions.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModule
    from extensions.pointops.functions.pointops import knn
except ImportError:
    print("Warning: Could not import from 'extension.pointnet2_modules'.")


# --------------------------------------------------------------------------
# 2. STNet核心模块_3D
# --------------------------------------------------------------------------
class TeLU(nn.Module):
    def __init__(self):
        super(TeLU, self).__init__()
        pass
    def forward(self,input):
        return input * torch.tanh(torch.exp(input))

def point_conv1d_1x1(in_channel: int, out_channel: int, bn: bool = True) -> nn.Module:
    """A shared MLP layer implemented with a 1x1 1D convolution."""
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel, 1, bias=not bn),
        nn.BatchNorm1d(out_channel) if bn else nn.Identity(),
        TeLU(),
        nn.Dropout(0.4)
    )


class TFF_Point(nn.Module):
    """
    Temporal Feature Fusion for point cloud features that are already aligned
    (e.g., global features with npoint=1).
    """
    def __init__(self, in_channel: int, out_channel: int):
        super(TFF_Point, self).__init__()
        self.catconvA = point_conv1d_1x1(in_channel * 2, in_channel)
        self.catconvB = point_conv1d_1x1(in_channel * 2, in_channel)
        self.catconv = point_conv1d_1x1(in_channel * 2, out_channel)
        self.convA = nn.Conv1d(in_channel, 1, 1)
        self.convB = nn.Conv1d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, featA: torch.Tensor, featB: torch.Tensor) -> torch.Tensor:
        feat_diff = featA - featB
        feat_diffA = self.catconvA(torch.cat([feat_diff, featA], dim=1))
        feat_diffB = self.catconvB(torch.cat([feat_diff, featB], dim=1))

        A_weight = self.sigmoid(self.convA(feat_diffA))
        B_weight = self.sigmoid(self.convB(feat_diffB))

        featA_weighted = A_weight * featA
        featB_weighted = B_weight * featB

        x = self.catconv(torch.cat([featA_weighted, featB_weighted], dim=1))
        return x


class TFF_Point_SpatiallyAware(nn.Module):
    """Spatially-Aware Temporal Feature Fusion for Point Clouds using FPModule for alignment."""
    def __init__(self, in_channel: int, out_channel: int, fp_mlp: List[int]):
        super(TFF_Point_SpatiallyAware, self).__init__()
        self.fp_aligner = PointnetFPModule(mlp=fp_mlp)
        
        self.catconvA = point_conv1d_1x1(in_channel * 2, in_channel)
        self.catconvB = point_conv1d_1x1(in_channel * 2, in_channel)
        self.catconv = point_conv1d_1x1(in_channel * 2, out_channel)
        self.convA = nn.Conv1d(in_channel, 1, 1)
        self.convB = nn.Conv1d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pcdA: tuple, pcdB: tuple) -> torch.Tensor:
        xyz_A, featA = pcdA
        xyz_B, featB = pcdB
        
        # Align featB to the coordinate system of A
        featB_on_A = self.fp_aligner(xyz_A, xyz_B, None, featB)

        feat_diff = featA - featB_on_A

        feat_diffA = self.catconvA(torch.cat([feat_diff, featA], dim=1))
        feat_diffB = self.catconvB(torch.cat([feat_diff, featB_on_A], dim=1))

        A_weight = self.sigmoid(self.convA(feat_diffA))
        B_weight = self.sigmoid(self.convB(feat_diffB))
        
        featA_weighted = A_weight * featA
        featB_weighted = B_weight * featB_on_A

        x = self.catconv(torch.cat([featA_weighted, featB_weighted], dim=1))
        
        return x


class FullDualPathAttentionTFF(nn.Module):
    """
    【已优化】一个完整的、空间感知的、双通道全注意力TFF模块。
    融合了Pre-Norm, 残差连接, 和可维护的代码结构。
    """
    def __init__(self, 
                 in_channel: int, 
                 out_channel: int, 
                 num_heads: int = 8):
        super().__init__()
        
        assert in_channel % num_heads == 0, "in_channel must be divisible by num_heads"
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_heads = num_heads
        self.head_dim = in_channel // num_heads

        # Q, K, V 投影层
        self.q_proj = nn.Conv1d(in_channel, in_channel, 1)
        self.k_proj = nn.Conv1d(in_channel, in_channel, 1)
        self.v_proj = nn.Conv1d(in_channel, in_channel, 1)
        
        # 空间偏置生成器
        self.spatial_bias_mlp = nn.Sequential(
            nn.Conv2d(1, num_heads // 2, 1, bias=False),
            nn.BatchNorm2d(num_heads // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_heads // 2, num_heads, 1)
        )
        
        # 差异性通道的投影
        self.q_proj_dissim = nn.Conv1d(in_channel, in_channel, 1)
        self.k_proj_dissim = nn.Conv1d(in_channel, in_channel, 1)

        # 归一化层
        self.norm_sim = nn.LayerNorm(in_channel)
        self.norm_dissim = nn.LayerNorm(in_channel)
        self.norm_input = nn.LayerNorm(in_channel)

        # 融合模块
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(in_channel * 3, in_channel * 2, 1, bias=False),
            nn.BatchNorm1d(in_channel * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(in_channel * 2, out_channel, 1)
        )
        
        # 残差连接 (使用 nn.Identity() 简化)
        if in_channel == out_channel:
            self.residual_proj = nn.Identity()
        else:
            self.residual_proj = nn.Conv1d(in_channel, out_channel, 1)
            

    def _calculate_scores(self, pcdA, pcdB):
        """私有方法，用于计算两种注意力分数和必要的中间变量"""
        xyz_A, featA = pcdA
        xyz_B, featB = pcdB
        B, C, N = featA.shape

        featA_norm = self.norm_input(featA.transpose(1, 2)).transpose(1, 2)
        
        dist_matrix = torch.cdist(xyz_A, xyz_B, p=2)
        spatial_bias = self.spatial_bias_mlp(-dist_matrix.unsqueeze(1))
        
        # 相似性通道计算
        q_A_sim = self.q_proj(featA_norm).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k_B_sim = self.k_proj(featB).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        sim_scores = torch.matmul(q_A_sim, k_B_sim.transpose(-2, -1)) / (self.head_dim ** 0.5) + spatial_bias
        
        # 差异性通道计算
        q_A_dissim = self.q_proj_dissim(featA_norm)
        k_B_dissim = self.k_proj_dissim(featB)
        feat_dist_matrix = torch.cdist(q_A_dissim.transpose(1, 2), k_B_dissim.transpose(1, 2), p=2)
        dissim_scores = feat_dist_matrix.unsqueeze(1) + spatial_bias
        
        # 返回所有需要的值
        return featA_norm, self.v_proj(featB), sim_scores, dissim_scores

    def forward(self, 
                pcdA: Tuple[torch.Tensor, torch.Tensor], 
                pcdB: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        
        xyz_A, featA = pcdA
        B, C, N = featA.shape
        
        # 1. 计算注意力分数和权重
        featA_norm, v_B_raw, sim_scores, dissim_scores = self._calculate_scores(pcdA, pcdB)
        
        sim_weights = F.softmax(sim_scores, dim=-1)
        dissim_weights = F.softmax(dissim_scores, dim=-1)
        
        # 2. 计算上下文向量
        v_B = v_B_raw.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        
        context_sim = torch.matmul(sim_weights, v_B).permute(0, 1, 3, 2).contiguous().view(B, C, N)
        context_sim = self.norm_sim(context_sim.transpose(1, 2)).transpose(1, 2)
        
        context_dissim = torch.matmul(dissim_weights, v_B).permute(0, 1, 3, 2).contiguous().view(B, C, N)
        context_dissim = self.norm_dissim(context_dissim.transpose(1, 2)).transpose(1, 2)

        # 3. 融合与残差连接
        fused_feat = self.fusion_mlp(
            torch.cat([featA_norm, context_sim, context_dissim], dim=1)
        )
        
        output = fused_feat + self.residual_proj(featA)
        
        return output

    def get_attention_maps(self, pcdA, pcdB) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回注意力权重矩阵，用于可视化分析"""
        _, _, sim_scores, dissim_scores = self._calculate_scores(pcdA, pcdB)
        
        sim_weights = F.softmax(sim_scores, dim=-1)
        dissim_weights = F.softmax(dissim_scores, dim=-1)
        
        return sim_weights, dissim_weights



class SparseDualPathAttentionTFF(nn.Module): # 建议改名
    """
    一个基于KNN的、空间感知的、双通道（相似性+差异性）稀疏注意力TFF模块。
    使用了高效的邻居特征收集方法。
    """
    def __init__(self, 
                 in_channel: int, 
                 out_channel: int, 
                 num_heads: int = 8,
                 k_neighbors: int = 64):
        super().__init__()
        
        assert in_channel % num_heads == 0, "in_channel must be divisible by num_heads"
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_heads = num_heads
        self.head_dim = in_channel // num_heads
        self.k_neighbors = k_neighbors

        # Q, K, V 投影层
        self.q_proj = nn.Conv1d(in_channel, in_channel, 1)
        self.k_proj = nn.Conv1d(in_channel, in_channel, 1)
        self.v_proj = nn.Conv1d(in_channel, in_channel, 1)
        
        # 空间偏置生成器
        #self.spatial_bias_mlp = nn.Sequential(
        #    nn.Linear(1, num_heads // 2),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(num_heads // 2, num_heads)
        #)
        
        # 差异性通道的投影
        self.q_proj_dissim = nn.Conv1d(in_channel, in_channel, 1)
        self.k_proj_dissim = nn.Conv1d(in_channel, in_channel, 1)

        # 归一化层
        self.norm_sim = nn.LayerNorm(in_channel)
        self.norm_dissim = nn.LayerNorm(in_channel)
        self.norm_input = nn.LayerNorm(in_channel)

        # 融合模块
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(in_channel * 3, in_channel * 2, 1, bias=False),
            nn.BatchNorm1d(in_channel * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(in_channel * 2, out_channel, 1)
        )
        
        # 残差连接
        if in_channel == out_channel:
            self.residual_proj = nn.Identity()
        else:
            self.residual_proj = nn.Conv1d(in_channel, out_channel, 1)

    def _gather_neighbors(self, features, knn_idx):
        """
        【已优化】高效地根据KNN索引提取邻居特征。
        Args:
            features: (B, H, N, D)
            knn_idx: (B, N, K)
        Returns:
            neighbors: (B, H, N, K, D)
        """
        B, H, N, D = features.shape
        K = knn_idx.shape[2]
        
        # 创建一个批次索引，用于高级索引
        batch_indices = torch.arange(B, device=features.device).view(B, 1, 1).expand(B, N, K)
        
        # features (B, H, N, D) -> (B, N, H, D)
        features_permuted = features.permute(0, 2, 1, 3)
        
        # 使用高级索引一次性提取所有邻居
        # gathered_features 的形状为 (B, N, K, H, D)
        gathered_features = features_permuted[batch_indices, knn_idx]
        
        # 转换回期望的维度顺序 (B, H, N, K, D)
        return gathered_features.permute(0, 3, 1, 2, 4)

    def forward(self, pcdA, pcdB) -> torch.Tensor:
        xyz_A, featA = pcdA
        xyz_B, featB = pcdB
        B, C, N = featA.shape

        featA_norm = self.norm_input(featA.transpose(1, 2)).transpose(1, 2)
        
        # 使用你的高效KNN库
        knn_idx, knn_dists = knn(xyz_A, xyz_B, self.k_neighbors)
        
        #spatial_bias = self.spatial_bias_mlp((-knn_dists).unsqueeze(-1)).permute(0, 3, 1, 2)

        # 多头变换
        q_A_sim = self.q_proj(featA_norm).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k_B_sim = self.k_proj(featB).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        v_B = self.v_proj(featB).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        # 提取邻居特征
        k_B_neighbors = self._gather_neighbors(k_B_sim, knn_idx)
        v_B_neighbors = self._gather_neighbors(v_B, knn_idx)
        
        # 相似性通道
        sim_scores = torch.sum(q_A_sim.unsqueeze(3) * k_B_neighbors, dim=-1) / (self.head_dim ** 0.5)
        #sim_scores += spatial_bias
        sim_weights = F.softmax(sim_scores, dim=-1)
        
        context_sim = torch.sum(sim_weights.unsqueeze(-1) * v_B_neighbors, dim=3)
        context_sim = self.norm_sim(context_sim.permute(0, 2, 1, 3).reshape(B, N, C)).transpose(1, 2)

        # 差异性通道
        q_A_dissim = self.q_proj_dissim(featA_norm).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k_B_dissim = self.k_proj_dissim(featB).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k_B_dissim_neighbors = self._gather_neighbors(k_B_dissim, knn_idx)
        
        feat_dist = torch.norm(q_A_dissim.unsqueeze(3) - k_B_dissim_neighbors, dim=-1, p=2)
        dissim_scores = feat_dist #+ spatial_bias
        dissim_weights = F.softmax(dissim_scores, dim=-1)
        
        context_dissim = torch.sum(dissim_weights.unsqueeze(-1) * v_B_neighbors, dim=3)
        context_dissim = self.norm_dissim(context_dissim.permute(0, 2, 1, 3).reshape(B, N, C)).transpose(1, 2)

        # 融合与残差
        fused_feat = self.fusion_mlp(torch.cat([featA_norm, context_sim, context_dissim], dim=1))
        output = fused_feat + self.residual_proj(featA)
        
        return output


class GlobalFeatureDualAttention(nn.Module):
    """
    一个专门用于融合全局特征的双通道注意力模块。

    该模块移除了所有空间坐标依赖，使其适用于处理像PointNet++输出的
    全局特征向量。它包含两个并行的注意力路径：
    1. 相似性通道：基于标准的缩放点积注意力，捕捉特征的相似性。
    2. 差异性通道：基于特征间的L2距离，显式地捕捉特征的差异性。
    
    输入是两个特征张量，通常形状为 (Batch, Channels, 1)。
    """
    def __init__(self, 
                 in_channel: int, 
                 out_channel: int, 
                 num_heads: int = 8):
        """
        Args:
            in_channel (int): 输入特征的通道数。
            out_channel (int): 输出特征的通道数。
            num_heads (int): 注意力头的数量。
        """
        super().__init__() # Corrected super call
        
        assert in_channel % num_heads == 0, "in_channel 必须能被 num_heads 整除"
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_heads = num_heads
        self.head_dim = in_channel // num_heads

        # --- 通用层 ---
        self.q_proj = nn.Conv1d(in_channel, in_channel, 1, bias=False)
        self.k_proj = nn.Conv1d(in_channel, in_channel, 1, bias=False)
        self.v_proj = nn.Conv1d(in_channel, in_channel, 1, bias=False)
        
        # --- 差异性通道专用投影 ---
        self.q_proj_dissim = nn.Conv1d(in_channel, in_channel, 1, bias=False)
        self.k_proj_dissim = nn.Conv1d(in_channel, in_channel, 1, bias=False)

        # --- 归一化层 ---
        self.norm_input = nn.LayerNorm(in_channel)
        self.norm_sim = nn.LayerNorm(in_channel)
        self.norm_dissim = nn.LayerNorm(in_channel)

        # --- 融合与输出 ---
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(in_channel * 3, in_channel * 2, 1, bias=False),
            nn.BatchNorm1d(in_channel * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(in_channel * 2, out_channel, 1)
        )
        
        # 残差连接
        if in_channel == out_channel:
            self.residual_proj = nn.Identity()
        else:
            self.residual_proj = nn.Conv1d(in_channel, out_channel, 1)

    def _calculate_scores(self, featA: torch.Tensor, featB: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        私有方法，计算两种注意力分数。
        """
        B, C, N = featA.shape # N 预期为 1
        
        featA_norm = self.norm_input(featA.transpose(1, 2)).transpose(1, 2)
        
        # --- 相似性通道计算 (Standard Attention) ---
        q_A_sim = self.q_proj(featA_norm).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k_B_sim = self.k_proj(featB).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        sim_scores = torch.matmul(q_A_sim, k_B_sim.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # --- 差异性通道计算 (Feature Distance) ---
        # 按照您的澄清，直接使用L2距离，不取负值。
        # softmax会给距离大的（更不相似）特征分配更高的权重。
        q_A_dissim = self.q_proj_dissim(featA_norm)
        k_B_dissim = self.k_proj_dissim(featB)
        
        feat_dist_matrix = torch.cdist(q_A_dissim.transpose(1, 2), k_B_dissim.transpose(1, 2), p=2)
        
        # 扩展到多头，使其与 sim_scores 形状一致
        dissim_scores = feat_dist_matrix.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        return featA_norm, self.v_proj(featB), sim_scores, dissim_scores

    def forward(self, featA: torch.Tensor, featB: torch.Tensor) -> torch.Tensor:
        """
        Args:
            featA (torch.Tensor): 第一个全局特征 (Query), 形状 (B, C, 1).
            featB (torch.Tensor): 第二个全局特征 (Key, Value), 形状 (B, C, 1).
            
        Returns:
            torch.Tensor: 融合后的特征，形状 (B, out_channel, 1).
        """
        B, C, N = featA.shape
        
        featA_norm, v_B_raw, sim_scores, dissim_scores = self._calculate_scores(featA, featB)
        
        sim_weights = F.softmax(sim_scores, dim=-1)
        dissim_weights = F.softmax(dissim_scores, dim=-1)
        
        v_B = v_B_raw.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        
        context_sim = torch.matmul(sim_weights, v_B)
        context_sim = context_sim.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        context_sim = self.norm_sim(context_sim).transpose(1, 2)
        
        context_dissim = torch.matmul(dissim_weights, v_B)
        context_dissim = context_dissim.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        context_dissim = self.norm_dissim(context_dissim).transpose(1, 2)

        fused_feat = self.fusion_mlp(
            torch.cat([featA_norm, context_sim, context_dissim], dim=1)
        )
        
        output = fused_feat + self.residual_proj(featA)
        
        return output

    def get_attention_maps(self, featA: torch.Tensor, featB: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回两种注意力的权重图，用于分析。"""
        _, _, sim_scores, dissim_scores = self._calculate_scores(featA, featB)
        
        sim_weights = F.softmax(sim_scores, dim=-1)
        dissim_weights = F.softmax(dissim_scores, dim=-1)
        
        return sim_weights, dissim_weights
    


class SelfAttentionBlock_Point(nn.Module):
    """Self-Attention Block for Point Cloud Features."""
    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels):
        super().__init__()
        self.key_project = point_conv1d_1x1(key_in_channels, transform_channels)
        self.query_project = point_conv1d_1x1(query_in_channels, transform_channels)
        self.value_project = point_conv1d_1x1(key_in_channels, transform_channels)
        self.out_project = point_conv1d_1x1(transform_channels, out_channels)
        self.transform_channels = transform_channels

    def forward(self, query_feats, key_feats, value_feats):
        query = self.query_project(query_feats).permute(0, 2, 1)
        key = self.key_project(key_feats)
        value = self.value_project(value_feats).permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.transform_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value).permute(0, 2, 1).contiguous()
        context = self.out_project(context)
        return context


class SFF_Point(nn.Module):
    """Spatial/Scale Feature Fusion for Point Clouds using FPModule."""
    def __init__(self, in_channel: int, out_channel: int, fp_mlp: List[int]):
        super(SFF_Point, self).__init__()
        self.fp_module = PointnetFPModule(mlp=fp_mlp)
        
        self.attention = SelfAttentionBlock_Point(
            key_in_channels=fp_mlp[-1],
            query_in_channels=fp_mlp[-1],
            transform_channels=fp_mlp[-1] // 2,
            out_channels=fp_mlp[-1]
        )
        self.catconv = point_conv1d_1x1(fp_mlp[-1] * 2, out_channel)

    def forward(self, small_pcd: tuple, big_pcd: tuple) -> torch.Tensor:
        xyz_small, feat_small = small_pcd
        xyz_big, feat_big = big_pcd
        
        fused_feat = self.fp_module(xyz_big, xyz_small, feat_big, feat_small)
        refined_feat = self.attention(fused_feat, fused_feat, fused_feat)
        out = self.catconv(torch.cat([refined_feat, fused_feat], dim=1))
        return out

class GFF_Module(nn.Module):
    """
    Global Feature Fusion Module.
    Uses self-attention to refine a target global feature using a context global feature.
    This is inspired by the SFF module but operates on global vectors (N=1).
    """
    def __init__(self, feature_channels: int):
        super(GFF_Module, self).__init__()
        self.attention = SelfAttentionBlock_Point(
            key_in_channels=feature_channels,
            query_in_channels=feature_channels,
            transform_channels=feature_channels // 2,
            out_channels=feature_channels
        )
        self.mlp = nn.Sequential(
            point_conv1d_1x1(feature_channels * 2, feature_channels),
            point_conv1d_1x1(feature_channels, feature_channels)
        )

    def forward(self, target_feat: torch.Tensor, context_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target_feat (torch.Tensor): The feature to be refined, e.g., rt1_global. [B, C, 1]
            context_feat (torch.Tensor): The guiding feature, e.g., rt4_global. [B, C, 1]
        Returns:
            refined_feat (torch.Tensor): The refined target feature. [B, C, 1]
        """
        # The query comes from the target, key/value from the context.
        # This means "what information from the context is useful for me (the target)?"
        refined_by_attention = self.attention(query_feats=target_feat, key_feats=context_feat, value_feats=context_feat)
        
        # Fuse the original target feature with its refined version
        fused_feat = torch.cat([target_feat, refined_by_attention], dim=1)
        
        # Further process with an MLP
        out_feat = self.mlp(fused_feat)
        
        return out_feat

class Point_Classifier_Head(nn.Module):
    """Classifier Head for Point Cloud Global Feature."""
    def __init__(self, in_channel: int, num_class: int):
        super(Point_Classifier_Head, self).__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(in_channel, 128, bias=False),
            #nn.BatchNorm1d(512),
            #nn.ReLU(inplace=True),
            #nn.Dropout(0.3),
            #nn.Linear(512, 256),
            nn.BatchNorm1d(128),
            TeLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_class)
        )

    def forward(self, global_feature_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_feature_vector (torch.Tensor): A 2D tensor of shape [BatchSize, In_Channel].
        """
        # 假设输入已经是2D的全局特征向量，直接送入MLP
        return self.mlp_head(global_feature_vector)


# --------------------------------------------------------------------------
# 3.  STNet 
# --------------------------------------------------------------------------
class STNet_Point_Classifier_Advanced(nn.Module):
    def __init__(self, num_class: int, output_dim: int = 64, input_feature_dim: int = 0, feature_out=False,pretrain=False):
        super(STNet_Point_Classifier_Advanced, self).__init__()
        self.output_dim = output_dim

        # === 1. Backbone ===
        """self.sa1 = PointnetSAModule(npoint=512, radius=0.1, nsample=16, mlp=[input_feature_dim, 32, 32, 64])
        self.sa2 = PointnetSAModule(npoint=128, radius=0.2, nsample=32, mlp=[64, 64, 64, 128])
        self.sa3 = PointnetSAModule(npoint=64, radius=0.4, nsample=64, mlp=[128, 128, 128, 256])
        self.sa4 = PointnetSAModule(npoint=None, radius=None, nsample=None, mlp=[256, 256, 512, 1024])
        channel_list = [64, 128, 256, 1024]"""
        #self.sa0 = PointnetSAModule(npoint=512, radius=0.35, nsample=16, mlp=[input_feature_dim, 32, 32, 64])
        #self.sa1 = PointnetSAModule(npoint=256, radius=0.45, nsample=32, mlp=[64, 64, 64, 128])
        # 估算: r=0.4的球体积约0.268。期望点数 = 0.268 * 215 ≈ 57个。这下足够nsample=32了。
        #self.sa2 = PointnetSAModule(npoint=128, radius=0.55, nsample=64, mlp=[128, 128, 256])
        # 解释: 在更稀疏的点上（256个），用更大的半径(0.6)捕捉更大尺度结构。
        self.sa3 = PointnetSAModule(npoint=None, radius=None, nsample=None, mlp=[256, 256, 512])
        # 解释: 最后一层通常是全局抽象，聚合所有特征。
        #self.sa0 = PointnetSAModule(npoint=512, radius=0.2, nsample=16, mlp=[input_feature_dim, 32, 32, 64])
        self.sa1 = PointnetSAModule(npoint=128, radius=0.25, nsample=32, mlp=[input_feature_dim, 64, 64, 128])
        # 估算: r=0.4的球体积约0.268。期望点数 = 0.268 * 215 ≈ 57个。这下足够nsample=32了。
        self.sa2 = PointnetSAModule(npoint=32, radius=0.3, nsample=64, mlp=[128, 128, 256])
        # 解释: 在更稀疏的点上（256个），用更大的半径(0.6)捕捉更大尺度结构。

        #channel_list = [128, 256, 1024]
        channel_list = [64,128, 256, 512]
        # === 2. TFF Modules ===
        
        #self.tff1 = TFF_Point_SpatiallyAware(channel_list[0], output_dim, fp_mlp=[channel_list[0]]*3)
        #self.tff2 = TFF_Point_SpatiallyAware(channel_list[1], output_dim, fp_mlp=[channel_list[1]]*3)
        #self.tff3 = TFF_Point_SpatiallyAware(channel_list[2], output_dim, fp_mlp=[channel_list[2]]*3)
        #self.tff3 = TFF_Point(channel_list[2], output_dim)

        #self.daf0 = FullDualPathAttentionTFF(channel_list[0], output_dim,4)
        self.daf1 = FullDualPathAttentionTFF(channel_list[1], output_dim,4)
        self.daf2 = FullDualPathAttentionTFF(channel_list[2], output_dim,4)
        #self.daf3 = FullDualPathAttentionTFF(channel_list[2], output_dim,4)
        #self.tff3 = TFF_Point(channel_list[3], output_dim)
        self.tff3 = GlobalFeatureDualAttention(channel_list[3], output_dim,8)
        # === 3. Global Pooling & Fusion Modules ===
        #self.sa_fusion0 = PointnetSAModule(npoint=None, radius=None, nsample=None, mlp=[output_dim]*3)
        self.sa_fusion1 = PointnetSAModule(npoint=None, radius=None, nsample=None, mlp=[output_dim]*3)
        self.sa_fusion2 = PointnetSAModule(npoint=None, radius=None, nsample=None, mlp=[output_dim]*3)
        #self.sa_fusion3 = PointnetSAModule(npoint=None, radius=None, nsample=None, mlp=[output_dim]*3)
        
        # SFF-inspired Global Feature Fusion Modules
        self.gff1 = GFF_Module(output_dim)
        self.gff2 = GFF_Module(output_dim)
        #self.gff0 = GFF_Module(output_dim)

        # === 4. Final Classifier Head ===
        classifier_in_channel = output_dim * 3
        self.classifier = Point_Classifier_Head(classifier_in_channel, num_class)

        self.pretrain = pretrain
        self.feature_out = feature_out

    def forward(self, pcdA: tuple, pcdB: tuple=None) -> torch.Tensor:
        

        # --- Backbone ---
        #l0_xyz_A, l0_feat_A = self.sa0(xyz_A, feat_A)
        #l0_xyz_B, l0_feat_B = self.sa0(xyz_B, feat_B)
        if self.pretrain:
            xyz_A, feat_A = pcdA
            l1_xyz_A, l1_feat_A = self.sa1(xyz_A, feat_A); l2_xyz_A, l2_feat_A = self.sa2(l1_xyz_A, l1_feat_A); l3_xyz_A, l3_feat_A = self.sa3(l2_xyz_A, l2_feat_A)
            return l3_feat_A
        else: 
            xyz_A, feat_A = pcdA; xyz_B, feat_B = pcdB
            l1_xyz_A, l1_feat_A = self.sa1(xyz_A, feat_A); l2_xyz_A, l2_feat_A = self.sa2(l1_xyz_A, l1_feat_A); l3_xyz_A, l3_feat_A = self.sa3(l2_xyz_A, l2_feat_A)#; l4_xyz_A, l4_feat_A = self.sa4(l3_xyz_A, l3_feat_A)
            l1_xyz_B, l1_feat_B = self.sa1(xyz_B, feat_B); l2_xyz_B, l2_feat_B = self.sa2(l1_xyz_B, l1_feat_B); l3_xyz_B, l3_feat_B = self.sa3(l2_xyz_B, l2_feat_B)#; l4_xyz_B, l4_feat_B = self.sa4(l3_xyz_B, l3_feat_B)
            
            # --- Temporal Fusion ---
            #rt1 = self.tff1((l1_xyz_A, l1_feat_A), (l1_xyz_B, l1_feat_B)); rt2 = self.tff2((l2_xyz_A, l2_feat_A), (l2_xyz_B, l2_feat_B)); rt3 = self.tff3((l3_feat_A), ( l3_feat_B))#; rt4 = self.tff4(l4_feat_A, l4_feat_B)
            #rt0 = self.daf0((l0_xyz_A, l0_feat_A), (l0_xyz_B, l0_feat_B))
            rt1 = self.daf1((l1_xyz_A, l1_feat_A), (l1_xyz_B, l1_feat_B)); rt2 = self.daf2((l2_xyz_A, l2_feat_A), (l2_xyz_B, l2_feat_B)); rt3 = self.tff3(l3_feat_A, l3_feat_B)#; rt4 = self.tff4(l4_feat_A, l4_feat_B)
            # --- Global Pooling ---
            #_, rt0_global = self.sa_fusion0(l0_xyz_A, rt0)
            _, rt1_global = self.sa_fusion1(l1_xyz_A, rt1); _, rt2_global = self.sa_fusion2(l2_xyz_A, rt2)
            """_, rt3_global = self.sa_fusion3(l3_xyz_A, rt3); rt4_global = rt4"""
            rt3_global = rt3
            # --- SFF-inspired Global Feature Interaction ---
            #new_rt0_global = self.gff0(target_feat=rt0_global, context_feat=rt3_global)
            new_rt1_global = self.gff1(target_feat=rt1_global, context_feat=rt3_global)
            new_rt2_global = self.gff2(target_feat=rt2_global, context_feat=rt3_global)
            #new_rt3_global = self.gff3(target_feat=rt3_global, context_feat=rt4_global)

            # --- Final Concatenation ---
            final_comparison_feature = torch.cat([new_rt1_global, new_rt2_global, rt3_global], dim=1)

            # --- Classification ---
            # Reshape for Linear layer if needed
            if final_comparison_feature.dim() == 3:
                final_comparison_feature = final_comparison_feature.squeeze(-1)

            if self.feature_out:
                return final_comparison_feature
            else: 
                # 现在 final_comparison_feature 保证是 2D 的 [B, C]
                logits = self.classifier(final_comparison_feature)
                return logits

def create_model(config):
    """ Create model with given config, including coord_dim """
    model_name = config.get('model_name', 'STNet_Point')
    if model_name == 'STNet_Point':
        model = STNet_Point_Classifier_Advanced(
            input_feature_dim=config.get('input_feature_dim', 2), # Use a clearer name
            output_dim=config.get('output_dim', 512),
            num_class=config.get('num_class', 4),
            feature_out=config.get('feature_out', False),
            pretrain = config.get("pretrain",False)
        )
    return model



# --------------------------------------------------------------------------
# 5. 主函数测试
# --------------------------------------------------------------------------
if __name__ == '__main__':
    # --- Configuration ---
    NUM_CLASSES = 10
    NUM_POINTS = 800 # As per your requirement
    BATCH_SIZE = 4
    INPUT_FEAT_DIM = 2 
    output_dim = 512 # A smaller intermediate channel size for a deeper network

    # --- Model Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STNet_Point_Classifier_Advanced(
        num_class=NUM_CLASSES, 
        output_dim=1024,
        input_feature_dim=2
    ).to(device)
    
    print(f"Model STNet_Point_Classifier_Deep initialized on {device}.")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Dummy Data Creation ---
    print(f"\nCreating dummy data for batch size {BATCH_SIZE}, {NUM_POINTS} points...")
    xyz_A = torch.randn(BATCH_SIZE, NUM_POINTS, 3).to(device)
    features_A = torch.randn(BATCH_SIZE, INPUT_FEAT_DIM, NUM_POINTS).to(device) if INPUT_FEAT_DIM > 0 else None
    pcdA = (xyz_A, features_A)
    
    xyz_B = torch.randn(BATCH_SIZE, NUM_POINTS, 3).to(device)
    features_B = torch.randn(BATCH_SIZE, INPUT_FEAT_DIM, NUM_POINTS).to(device) if INPUT_FEAT_DIM > 0 else None
    pcdB = (xyz_B, features_B)
    
    # --- Forward and Backward Pass Test ---
    print("Performing a forward and backward pass test...")
    try:
        model.train()
        logits = model(pcdA, pcdB)
        
        print("\n--- Forward Pass Successful! ---")
        print(f"Input pcdA xyz shape: {pcdA[0].shape}")
        print(f"Output logits shape: {logits.shape}")
        
        expected_shape = (BATCH_SIZE, NUM_CLASSES)
        assert logits.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {logits.shape}"
        print("Output shape is correct.")
        
        loss = F.cross_entropy(logits, torch.randint(0, NUM_CLASSES, (BATCH_SIZE,)).to(device))
        loss.backward()
        print("Backward pass successful.")
        
    except Exception as e:
        print(f"\n--- An error occurred during the test ---")
        import traceback
        traceback.print_exc()


    """# 在测试/推理阶段
    model.eval()
    xyz_A.requires_grad_() # 开启输入点的梯度计算

    logits = model(pcdA, pcdB)
    score = logits.max() # 取最大logit值
    score.backward() # 反向传播计算梯度

    saliency = xyz_A.grad.norm(dim=-1) # 计算每个点梯度的模长
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min()) # 归一化到[0, 1]

    # 现在可以用 saliency 作为颜色来可视化点云A了
    # 例如，使用 open3d 库
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz_A.detach().cpu().numpy()[0])
    # colors = plt.cm.jet(saliency.detach().cpu().numpy()[0])[:, :3]
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])"""