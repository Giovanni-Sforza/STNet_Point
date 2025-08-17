import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        if self.num_outputs == 1 :
            X = X.squeeze(1)
        return X

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=True):
        super(SetTransformer, self).__init__()
        self.num_outputs = num_outputs
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        if self.num_outputs ==1 :
            return self.dec(self.enc(X)).squeeze(1)
        return self.dec(self.enc(X))

class SetTransformer_point(nn.Module):
    def __init__(
        self,
        dim_input=3,
        num_outputs=1,
        dim_output=128,
        num_inds=32,
        dim_hidden=128,
        num_heads=4,
        ln=True,
    ):
        super(SetTransformer_point, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        return self.dec(self.enc(X)).squeeze(1)

# ==================== 测试模块 ====================
if __name__ == "__main__":
    
    # 1. 定义模型和输入的超参数
    batch_size = 4      # 批处理大小
    set_size = 10       # 每个集合中的元素数量
    dim_input = 64      # 每个输入元素的特征维度
    
    num_outputs = 1     # 期望的输出向量数量 (对于集合分类任务通常为1)
    dim_output = 10     # 每个输出向量的维度 (例如，预测10个类别的logits)
    
    num_inds = 16       # ISAB模块中的引导点数量
    dim_hidden = 128    # 模型内部的隐藏维度
    num_heads = 4       # 多头注意力的头数
    ln = True           # 是否使用LayerNorm

    # 2. 实例化Set Transformer模型
    print("Initializing Set Transformer model...")
    model = SetTransformer(
        dim_input=dim_input,
        num_outputs=num_outputs,
        dim_output=dim_output,
        num_inds=num_inds,
        dim_hidden=dim_hidden,
        num_heads=num_heads,
        ln=ln
    )
    print(model)
    
    # 3. 创建一个虚拟的输入张量
    # 形状为 (batch_size, set_size, dim_input)
    # 代表一个批次的数据，每批包含4个集合，每个集合有10个元素，每个元素是64维的向量
    input_set = torch.randn(batch_size, set_size, dim_input)

    # 4. 执行前向传播
    output = model(input_set)

    # 5. 打印输入和输出的形状以进行验证
    print("\n--- Forward Pass Test ---")
    print(f"Input shape:  {input_set.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {num_outputs}, {dim_output})")
    assert output.shape == (batch_size, num_outputs, dim_output)
    print("Shape verification successful!")

    # 6. 额外测试：置换不变性 (Permutation Invariance)
    # 创建一个输入顺序被打乱的集合
    # torch.randperm(n) 会生成一个从0到n-1的随机排列
    permuted_indices = torch.randperm(set_size)
    permuted_input_set = input_set[:, permuted_indices, :]
    
    # 使用相同的模型进行前向传播
    permuted_output = model(permuted_input_set)
    
    # 检查两次输出是否（近似）相等
    # 由于浮点数计算的细微差异，我们使用 allclose 进行比较，设置一个小的容忍度 atol
    is_invariant = torch.allclose(output, permuted_output, atol=1e-6)

    print("\n--- Permutation Invariance Test ---")
    print(f"Input set elements were permuted.")
    # print("Original output (first batch element):\n", output[0])
    # print("Permuted input output (first batch element):\n", permuted_output[0])
    print(f"Is the model output (approximately) the same? {is_invariant}")
    assert is_invariant
    print("Permutation invariance test successful!")