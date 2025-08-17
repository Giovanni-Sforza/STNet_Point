import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class PCN(nn.Module):
    def __init__(self, num_classes=2,num_dimension=3,ortho_reg=0.001,drop_rate=0.3, **kwargs):
        super().__init__()

        
        # 核心参数
        self.num_classes = num_classes
        self.num_dimension = num_dimension
        self.ortho_reg = ortho_reg
        self.drop_rate = drop_rate
        
        # 网络组件
        self._build_network()
        
        # 初始化参数
        self._initialize_weights()
        self.ortho_loss = 0.0

    def _build_network(self):
        #构建网络结构
        # TNet变换模块
        self.input_tnet = TNet(3, self.ortho_reg)
        self.feature_tnet = TNet(32, self.ortho_reg)
        
        # 特征提取序列
        self.conv_layers = nn.Sequential(
            ConvBN(3, 32),
            ConvBN(32, 32),
            ConvBN(32, 64),
            ConvBN(64, 512)
        )
        
        # 分类头
        self.cls_head = nn.Sequential(
            DenseBN(512, 256),
            nn.Dropout(self.drop_rate),
            DenseBN(256, 128),
            nn.Dropout(self.drop_rate),
            nn.Linear(128, self.num_classes)
        )
        self.dropout = nn.Dropout(self.drop_rate)
        # 特征输出头
        #self.feat_head = nn.Identity()

    def _initialize_weights(self):
        #初始化分类头权重
        init.normal_(self.cls_head[-1].weight, std=0.01)
        init.constant_(self.cls_head[-1].bias, 0)



    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path, map_location="cpu")
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            #del base_ckpt[k]


        incompatible = self.load_state_dict(base_ckpt, strict=False)

    def forward(self, pts, return_feature=False):
        
        '''前向传播
        :param pts: 输入点云 [B, N, 3]
        :param return_feature: 是否返回特征'''
        
        B, N, C = pts.shape
        self.ortho_loss = 0.0
        
        # 输入变换
        trans, loss = self.input_tnet(pts.permute(0, 2, 1))
        self.ortho_loss += loss
        x = torch.bmm(pts, trans)  # [B, N, 3]
        
        # 第一卷积层
        x = x.permute(0, 2, 1)  # [B, 3, N]
        x = self.conv_layers[0](x)
        #x = self.dropout(x)
        # 特征变换
        trans_feat, loss = self.feature_tnet(x)
        self.ortho_loss += loss
        x = torch.bmm(x.permute(0, 2, 1), trans_feat).permute(0, 2, 1)
        #x = self.dropout(x)
        # 后续卷积层
        x = self.conv_layers[1](x)
        x = self.conv_layers[2](x)
        x = self.conv_layers[3](x)
        # [batch, feature, Npoint]
        
        # 全局特征
        x = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)  # [B, 512]
        
        #if return_feature:
        #    return self.feat_head(x)
        #_ = self.cls_head(x)
        #print(_.max(dim=1))
        #return self.cls_head(x)
        return x

class TNet(nn.Module):
    #空间变换网络
    def __init__(self, dim, reg_weight):
        super().__init__()
        self.dim = dim
        self.reg_weight = reg_weight
        
        self.conv_layers = nn.Sequential(
            ConvBN(dim, 32),
            ConvBN(32, 64),
            ConvBN(64, 512),
            nn.AdaptiveMaxPool1d(1)
        )
        self.mlp = nn.Sequential(
            DenseBN(512, 256),
            DenseBN(256, 128)
        )
        self.transform = nn.Linear(128, dim*dim)
        self._init_weights()

    def _init_weights(self):
        #正交矩阵初始化
        init.constant_(self.transform.weight, 0)
        eye = torch.eye(self.dim).flatten()
        init.constant_(self.transform.bias, 0)
        self.transform.bias.data.copy_(eye)

    def forward(self, x):
        B = x.size(0)
        x = self.conv_layers(x).squeeze(-1)
        x = self.mlp(x)
        
        transform = self.transform(x).view(B, self.dim, self.dim)
        
        # 正交正则计算
        identity = torch.eye(self.dim, device=x.device)
        ortho_loss = torch.norm(
            torch.bmm(transform, transform.transpose(1,2)) - identity,
            p='fro', dim=(1,2)
        ).mean() * self.reg_weight
        
        return transform, ortho_loss

class ConvBN(nn.Module):
    #带批归一化的1D卷积
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm1d(out_ch, momentum=0.1)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        
        # 初始化
        init.kaiming_normal_(self.conv.weight)
        init.constant_(self.conv.bias, 0)
        init.constant_(self.bn.weight, 1)
        init.constant_(self.bn.bias, 0)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DenseBN(nn.Module):
    #带批归一化的全连接层
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim, momentum=0.1)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        
        # 初始化
        init.kaiming_normal_(self.fc.weight)
        init.constant_(self.fc.bias, 0)
        init.constant_(self.bn.weight, 1)
        init.constant_(self.bn.bias, 0)

    def forward(self, x):
        return self.act(self.bn(self.fc(x)))
    
def create_model(config):
    """ Create model with given config, including coord_dim """
    model_name = config.get('model_name', 'PCN')
    if model_name == 'PCN':
        model = PCN(
            num_classes=config.get('num_classes', 2)# Use a clearer name
        )
    return model

if __name__ == "__main__":
    # --- 测试模块 ---
    
    # 1. 定义超参数
    batch_size = 4      # 批处理大小
    num_points = 1024   # 每个点云的点数
    num_features = 3    # 每个点的维度 (x, y, z)
    num_classes = 10    # 假设有10个分类类别
    
    # 2. 检查是否有可用的GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")
    
    # 3. 实例化模型并移动到指定设备
    model = PCN(num_classes=num_classes).to(device)
    print("模型 PCN 实例化成功。")
    # print(model) # 如果需要可以取消注释来查看模型结构
    
    # 4. 创建一个随机的虚拟输入张量来模拟一批点云数据
    #    形状为 [batch_size, num_points, num_features]
    dummy_input = torch.randn(batch_size, num_points, num_features).to(device)
    print(f"创建虚拟输入数据，形状为: {dummy_input.shape}")
    
    # 5. 测试标准的前向传播（获取分类结果）
    print("\n--- 测试分类模式 ---")
    try:
        # 将模型置于评估模式
        model.eval()
        with torch.no_grad(): # 在测试时不需要计算梯度
            logits = model(dummy_input)
        print("前向传播成功！")
        print(f"分类Logits的输出形状: {logits.shape}")
        # 验证输出形状是否正确
        expected_shape = (batch_size, num_classes)
        assert logits.shape == expected_shape, f"输出形状错误！期望: {expected_shape}, 得到: {logits.shape}"
        print(f"输出形状正确，为 [批量大小, 类别数]。")
        print(f"正交损失值为: {model.ortho_loss.item()}")
        
    except Exception as e:
        print(f"前向传播失败: {e}")

    # 6. 测试返回特征的前向传播
    print("\n--- 测试特征提取模式 ---")
    try:
        # 将模型置于训练模式，以测试包含Dropout等层的行为
        model.train()
        features = model(dummy_input, return_feature=True)
        print("特征提取前向传播成功！")
        # 根据网络结构，最后的全局特征维度是512
        expected_feature_shape = (batch_size, 512)
        print(f"特征向量的输出形状: {features.shape}")
        assert features.shape == expected_feature_shape, f"特征输出形状错误！期望: {expected_feature_shape}, 得到: {features.shape}"
        print(f"特征输出形状正确，为 [批量大小, 特征维度]。")
        print(f"正交损失值为: {model.ortho_loss.item()}")

    except Exception as e:
        print(f"特征提取前向传播失败: {e}")