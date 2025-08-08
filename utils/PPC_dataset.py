import torch
from torch.utils.data import Dataset
import numpy as np
import os
import re
import random
from typing import List, Tuple, Dict

class EventClusterDataset(Dataset):
    """
    一个为事件簇设计的PyTorch Dataset。

    在每个epoch开始时，它会将所有事件按类别分组，随机打乱，
    然后组成同质（同一类别）的事件簇。

    Args:
        data_dir (str): 存放 .npy 事件文件的根目录。
        file_list_path (str): 一个 .txt 文件，每行包含一个 .npy 的文件名。
        cluster_size (int): 每个事件簇中包含的事件数量。
        preload (bool): 是否将所有数据预加载到内存中。对于大型数据集，建议设为False。
    """
    def __init__(self, data_dir: str, file_list_path: str, cluster_size: int, preload: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.cluster_size = cluster_size
        self.preload = preload
        
        self.data_cache = {}
        self.clusters = []

        print("Initializing dataset...")
        # 1. 读取并按类别预处理所有文件名
        self.class_files = self._prepare_file_lists(file_list_path)
        
        # 2. 如果选择预加载，将所有npy文件读入内存
        if self.preload:
            print("Preloading all .npy files into RAM. This may take a while...")
            for class_label, filenames in self.class_files.items():
                for filename in filenames:
                    file_path = os.path.join(self.data_dir, filename)
                    self.data_cache[filename] = np.load(file_path)
            print("Preloading complete.")

        # 3. 为第一个epoch创建初始的事件簇
        print("Creating initial clusters for the first epoch...")
        self.shuffle_and_recluster()
        print(f"Dataset initialized. Found {len(self)} clusters.")


    def _parse_event_filename(self, filename: str) -> Tuple[str, int, int]:
        """
        解析文件名 <point_count>-<class_label>_<filename_core>.npy
        返回 (filename_core, point_count, class_label)
        """
        name_without_ext = os.path.splitext(filename)[0]
        # 注意: 您的原始函数假定文件名在字符串的开头。如果路径中包含目录，
        # 需要先获取基本名称。os.path.basename可以处理这个问题。
        base_name = os.path.basename(name_without_ext)
        match = re.match(r'^\d+-(\d+)_(\d+)$', base_name)
        if match:
            class_label = int(match.group(1)) -1 
            filename_core = match.group(2)
            # 这里我们不返回 point_count 因为在数据加载时会动态获取
            return filename_core, class_label
        else:
            raise ValueError(f"Filename {filename} does not match expected format '<...>-<class_label>_<core>.npy'")

    def _prepare_file_lists(self, file_list_path: str) -> Dict[int, List[str]]:
        """读取文件列表，解析并按类别分组"""
        class_files: Dict[int, List[str]] = {}
        with open(file_list_path, 'r') as f:
            all_filenames = [line.strip() for line in f if line.strip()]
        
        for filename in all_filenames:
            try:
                _, class_label = self._parse_event_filename(filename)
                if class_label not in class_files:
                    class_files[class_label] = []
                class_files[class_label].append(filename)
            except ValueError as e:
                print(f"Skipping file due to parsing error: {e}")
                
        # 打印统计信息
        print("Found files per class:")
        for label, files in sorted(class_files.items()):
            print(f"  - Class {label}: {len(files)} files")
        
        return class_files

    def shuffle_and_recluster_cluster(self):
        """
        核心功能：为新的epoch重新打乱和创建簇。
        这个函数应该在每个epoch开始前被外部的训练循环调用。
        """
        self.clusters = []
        
        # 对每个类别独立进行操作
        for class_label, filenames in self.class_files.items():
            # 1. 随机打乱该类别的事件列表
            random.shuffle(filenames)
            
            # 2. 计算可以形成多少个完整的簇
            num_clusters_for_class = len(filenames) // self.cluster_size
            
            # 3. 创建簇
            for i in range(num_clusters_for_class):
                start_idx = i * self.cluster_size
                end_idx = start_idx + self.cluster_size
                cluster_filenames = filenames[start_idx:end_idx]
                
                # 将 (文件名列表, 类别标签) 元组加入总列表
                self.clusters.append((cluster_filenames, class_label))
        
        # 4. 最后，打乱所有创建好的簇的顺序
        # 这确保了模型训练时不会按顺序看到所有类别0的簇，然后是类别1的簇
        random.shuffle(self.clusters)

    def shuffle_and_recluster(self):
        """
        更高级的核心功能：为新的epoch重新创建簇。

        与旧方法（洗牌后切片）不同，此方法通过重复从事件池中随机抽样
        来创建簇，确保每个epoch中簇的内部构成都是全新的随机组合。
        """
        self.clusters = []
        
        # 对每个类别独立进行操作
        for class_label, filenames in self.class_files.items():
            
            # 1. 创建一个该类别所有文件名的可变副本（“事件池”）
            #    我们不能直接修改 self.class_files[class_label]，因为它在下个epoch还需要
            pool = list(filenames)
            
            # 2. 当事件池中的文件还足够组成一个完整的簇时，持续循环
            while len(pool) >= self.cluster_size:
                
                # 3. 从池中随机抽样（无放回），组成一个新的簇
                #    random.sample 是实现此功能的完美工具
                cluster_filenames = random.sample(pool, self.cluster_size)
                
                # 4. 将新创建的簇加入到总列表中
                self.clusters.append((cluster_filenames, class_label))
                
                # 5. 从事件池中移除已经被抽样的文件
                #    这是一个高效的实现方式
                for f in cluster_filenames:
                    pool.remove(f)
        
        # 6. 最后，像以前一样，打乱所有创建好的簇的最终顺序
        #    这确保了模型训练时不会按顺序看到所有类别0的簇，然后是类别1的簇
        random.shuffle(self.clusters)
    def __len__(self) -> int:
        """返回当前epoch中簇的总数"""
        return len(self.clusters)


    '''@staticmethod
    def _transfer(cor):
        """x = (cor[:,0])/0.54
        y = (cor[:,1])/0.55
        z = (cor[:,2])/0.48"""
        points = cor[:,5:8]
        a_cartesian_points =  cor[:,0:3]/0.5#torch.stack([x, y, z], dim=-1)
        return a_cartesian_points,points
        '''
    @staticmethod
    def _transfer(cor):
        r = cor[:,0]
        phi = cor[:, 1]
        z = cor[:, 2]#/1.7

        x = r * torch.cos(phi)#/0.16
        y = r * torch.sin(phi)#/0.16
        z = r*torch.sinh(z)

        #fake_归一化
        x = x/4.5
        y = y/4.5
        z = z/16
        # 3. 组装成笛卡尔坐标点云
        # a_cartesian_points 的 shape 也是 (N, 3)
        a_cartesian_points = torch.stack([x, y, z], dim=-1)
        points = torch.stack([r,phi,z],dim=-1)
        return a_cartesian_points,points

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        获取一个事件簇的数据。

        Args:
            idx (int): 簇的索引。

        Returns:
            A tuple: `((xyz_cluster, feat_cluster), label)`
            - xyz_cluster (torch.Tensor): 簇内所有事件的坐标数据。
                                          Shape: [E, N, 3], E=cluster_size
            - feat_cluster (torch.Tensor): 簇内所有事件的特征数据。
                                           Shape: [E, 3, N]
            - label (torch.Tensor): 该簇的类别标签。
        """
        cluster_filenames, label = self.clusters[idx]
        
        xyz_data = []
        features_data = []
        
        for filename in cluster_filenames:
            # 从缓存或磁盘加载数据
            if self.preload and filename in self.data_cache:
                event_data = self.data_cache[filename]
            else:
                file_path = os.path.join(self.data_dir, filename)
                event_data = np.load(file_path)

            # 假设 .npy 文件存储的是 (N, 3) 的 (px, py, pz) 数据
            # 你需要根据你的PointNet++输入调整这里
            # 例如，如果需要归一化，可以在这里做
            
            event_tensor = torch.from_numpy(event_data).float()
            xyz, points = self._transfer(event_tensor)
            features = points.t()
            # xyz: [N, 3]
            #xyz =  torch.from_numpy(xyz_npy)
            # features: [3, N] (PointNet++ 通常需要通道在前的格式)
            #features = points_npy.from_numpy(xyz_npy).t()
            
            xyz_data.append(xyz)
            features_data.append(features)
        
        # 使用 torch.stack 将事件列表组合成一个簇张量
        xyz_cluster = torch.stack(xyz_data, dim=0)
        feat_cluster = torch.stack(features_data, dim=0)
        #print(xyz_cluster.shape)
        #print(feat_cluster.shape)
        # 返回数据和标签
        return (xyz_cluster, feat_cluster), torch.tensor(label, dtype=torch.long)


# --- 使用示例和测试 ---
if __name__ == '__main__':
    # --- 1. 创建一个临时的、仿真的数据环境 ---
    print("--- Setting up a mock data environment ---")
    DATA_DIR = './temp_data'
    LIST_DIR = './temp_lists'
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LIST_DIR, exist_ok=True)
    
    FILE_LIST_PATH = os.path.join(LIST_DIR, 'files.txt')
    
    # 创建一些仿真的npy文件和文件列表
    # 类别1有100个事件，类别2有110个事件
    all_files = []
    for i in range(100):
        filename = f"1024-1_{i:06d}.npy"
        np.save(os.path.join(DATA_DIR, filename), np.random.rand(1024, 3))
        all_files.append(filename)
    for i in range(110):
        filename = f"1024-2_{i:06d}.npy"
        np.save(os.path.join(DATA_DIR, filename), np.random.rand(1024, 3))
        all_files.append(filename)
        
    with open(FILE_LIST_PATH, 'w') as f:
        for fname in all_files:
            f.write(fname + '\n')
            
    print("Mock environment created.\n")
    
    # --- 2. 初始化 Dataset ---
    CLUSTER_SIZE = 16
    dataset = EventClusterDataset(data_dir=DATA_DIR, file_list_path=FILE_LIST_PATH, cluster_size=CLUSTER_SIZE)
    
    # 验证簇的数量：
    # Class 1: 100 // 16 = 6 clusters
    # Class 2: 110 // 16 = 6 clusters
    # Total = 12 clusters
    print(f"\nExpected number of clusters: {(100 // CLUSTER_SIZE) + (110 // CLUSTER_SIZE)}")
    print(f"Actual number of clusters in dataset: {len(dataset)}")
    assert len(dataset) == (100 // CLUSTER_SIZE) + (110 // CLUSTER_SIZE)

    # --- 3. 从 Dataset 中获取一个样本 ---
    print("\n--- Getting one sample from the dataset ---")
    (xyz, feat), label = dataset[0]
    print(f"Sample 0 - XYZ cluster shape: {xyz.shape}")      # 应为 [16, 1024, 3]
    print(f"Sample 0 - Feat cluster shape: {feat.shape}")     # 应为 [16, 3, 1024]
    print(f"Sample 0 - Label: {label}")
    assert xyz.shape == (CLUSTER_SIZE, 1024, 3)
    assert feat.shape == (CLUSTER_SIZE, 3, 1024)
    first_item_label = label
    
    # --- 4. 模拟新 Epoch：调用 shuffle_and_recluster ---
    print("\n--- Simulating a new epoch by reshuffling ---")
    dataset.shuffle_and_recluster()
    (xyz_new, feat_new), label_new = dataset[0]
    print(f"After reshuffle, Sample 0 - Label: {label_new}")
    print("(The label might be different, showing the cluster order has changed)")

    # --- 5. 与 DataLoader 一起使用 ---
    print("\n--- Demonstrating usage with DataLoader ---")
    # 注意：在实际训练循环中，每次循环前都要调用 shuffle_and_recluster
    
    # Epoch 1
    print("\n--- Epoch 1 ---")
    dataset.shuffle_and_recluster()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False) # shuffle=False因为dataset自己管理了
    
    for i, ((xyz_batch, feat_batch), label_batch) in enumerate(data_loader):
        print(f"Batch {i}:")
        print(f"  - XYZ batch shape: {xyz_batch.shape}")  # 应为 [4, 16, 1024, 3]
        print(f"  - Feat batch shape: {feat_batch.shape}") # 应为 [4, 16, 3, 1024]
        print(f"  - Labels batch shape: {label_batch.shape}") # 应为 [4]
        print(f"  - Labels in batch: {label_batch.tolist()}")
        if i == 0: break # 只演示第一个batch
        
    # Epoch 2
    print("\n--- Epoch 2 ---")
    dataset.shuffle_and_recluster() # 再次调用
    data_loader_2 = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    # ... 继续训练 ...
    print("DataLoader demonstration complete.")

    # --- 6. 清理临时文件 ---
    print("\n--- Cleaning up temporary files ---")
    import shutil
    shutil.rmtree(DATA_DIR)
    shutil.rmtree(LIST_DIR)
    print("Cleanup complete.")
"""
### 如何在您的训练循环中使用

#您的训练代码结构看起来会是这样：

# 1. 在训练开始前，创建数据集实例
dataset = EventClusterDataset(data_dir='path/to/your/npy', 
                              file_list_path='path/to/your/list.txt', 
                              cluster_size=32) # 或 16

model = PointPoolingClassifier(...)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 2. 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    print(f"--- Starting Epoch {epoch+1}/{num_epochs} ---")
    
    # 3. 每个epoch开始时，调用此方法！
    dataset.shuffle_and_recluster()
    
    # 4. 创建DataLoader
    # 注意：这里的 shuffle 设为 False，因为我们已经在dataset层面手动完成了更复杂的打乱。
    # DataLoader自身的shuffle只会打乱簇的顺序，而我们的方法连簇的构成都重新生成了。
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    
    model.train()
    for (xyz_cluster_batch, feat_cluster_batch), labels_batch in data_loader:
        # 将数据移动到GPU
        xyz_cluster_batch = xyz_cluster_batch.to(device)
        feat_cluster_batch = feat_cluster_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        logits = model(xyz_cluster_batch, feat_cluster_batch)
        
        # 计算损失
        loss = F.cross_entropy(logits, labels_batch)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # ... (打印日志，评估等) ...

    # ... (验证循环) ..."""