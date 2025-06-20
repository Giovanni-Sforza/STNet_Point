import os
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Sampler, Dataset
from collections import defaultdict
from dataset import SinglePointCloudDataset
import torch
class SiamesePointCloudBatchSampler(Sampler):
    """
   孪生网络的点云批处理采样器 - 从相邻类别中选择批次
    
    args:
        dataset: 包含点云数据的数据集实例
        bucket_info_dir: 包含桶信息的.txt文件的目录
        max_points_per_batch: 每个批次中最大允许的点总数
        shuffle: 是否打乱每个桶内的样本顺序
    """
    
    def __init__(
        self, 
        dataset: Dataset,
        bucket_info_dir: str,
        max_points_per_batch: int = 1000000,  # 可以根据GPU内存调整
        shuffle: bool = True
    ):
        self.dataset = dataset
        self.max_points_per_batch = max_points_per_batch
        self.shuffle = shuffle
        
        # 从桶信息文件中读取数据
        self.buckets = self._load_buckets(bucket_info_dir)
        
        # 按类别组织数据集索引
        self.class_to_indices = self._organize_by_class()
        
        # 生成成对批次
        self.paired_batches = self._create_paired_batches()
        
    def _load_buckets(self, bucket_info_dir: str) -> Dict[int, List[int]]:
        """
        加载桶信息，返回 {点数: [数据集索引列表]} 的字典
        """
        buckets = defaultdict(list)
        
        # 获取所有txt文件
        bucket_files = [f for f in os.listdir(bucket_info_dir) if f.endswith('.txt')]
        
        for bucket_file in bucket_files:
            # 提取点数信息 (xxx-aaa.txt)
            try:
                points_count = int(bucket_file.split('-')[0])
            except (ValueError, IndexError):
                print(f"警告: 无法从文件名 {bucket_file} 解析点数信息，跳过")
                continue
                
            # 读取该桶中的文件列表
            with open(os.path.join(bucket_info_dir, bucket_file), 'r') as f:
                npy_filenames = [line.strip() for line in f if line.strip()]
                
            # 获取数据集中对应的索引
            for npy_filename in npy_filenames:
                if npy_filename in self.dataset.file_to_id:
                    idx = self.dataset.file_to_id[npy_filename]
                    # 使用点数作为桶的标识
                    buckets[points_count].append(idx)
                else:
                    print(f"警告: 在数据集中找不到文件 {npy_filename}")
        
        return buckets
    
    def _organize_by_class(self) -> Dict[int, List[int]]:
        """
        按类别组织数据集索引
        
        将所有索引按类别分组，便于后续成对批次生成
        返回: {类别: [数据集索引列表]}
        """
        class_to_indices = defaultdict(list)
        
        for idx, file_info in self.dataset.id_to_file_info.items():
            class_label = file_info['class']
            class_to_indices[class_label].append(idx)
        
        return class_to_indices
    
    def _create_paired_batches(self) -> List[Tuple[List[int], List[int]]]:
        """
        创建成对的批次，确保来自相邻类别
        
        生成规则：
        1. 从第0、1类开始，第2、3类，以此类推
        2. 每对批次可以来自不同的桶
        3. 批次大小受max_points_per_batch控制
        """
        paired_batches = []
        
        # 获取所有类别并排序
        sorted_classes = sorted(self.class_to_indices.keys())
        
        # 成对处理类别
        for i in range(0, len(sorted_classes)-1, 2):
            class_a = sorted_classes[i]
            class_b = sorted_classes[i+1]
            
            # 获取这两个类别的索引
            indices_a = self.class_to_indices[class_a]
            indices_b = self.class_to_indices[class_b]
            
            # 如果需要，打乱索引
            if self.shuffle:
                random.shuffle(indices_a)
                random.shuffle(indices_b)
            
            # 计算每个批次的最大样本数
            # 这里我们尝试让两个批次尽可能平衡
            # 根据两个类别的点云点数来调整
            def get_batch_size(indices):
                # 如果索引为空，返回0
                if not indices:
                    return 0
                
                # 获取第一个样本的点数
                first_sample_info = self.dataset.id_to_file_info[indices[0]]
                points_count = first_sample_info['point_count']
                
                # 计算批次大小
                return max(1, self.max_points_per_batch // points_count)
            
            batch_size_a = get_batch_size(indices_a)
            batch_size_b = get_batch_size(indices_b)
            
            # 限制批次大小不超过可用样本数
            batch_size_a = min(batch_size_a, len(indices_a))
            batch_size_b = min(batch_size_b, len(indices_b))
            
            # 分割成批次
            for j in range(0, min(len(indices_a), len(indices_b)), max(batch_size_a, batch_size_b)):
                batch_a = indices_a[j:j+batch_size_a]
                batch_b = indices_b[j:j+batch_size_b]
                
                # 只有当两个批次都非空时才添加
                if batch_a and batch_b:
                    paired_batches.append((batch_a, batch_b))
        
        # 如果需要，打乱成对批次的顺序
        if self.shuffle:
            random.shuffle(paired_batches)
        
        return paired_batches
    
    def __iter__(self):
        """返回成对批次的迭代器"""
        if self.shuffle:  # 如果每个 epoch 都要重新打乱
            self.paired_batches = self._create_paired_batches()
        
        for batch_a, batch_b in self.paired_batches:
            # 可以选择是否平铺批次，这里我们返回成对批次
            yield batch_a, batch_b
    
    def __len__(self):
        """返回成对批次的总数"""
        return len(self.paired_batches)
    
class SiameseCollator:
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __call__(self, batch_pairs):
        batch_a = [self.dataset[idx] for pair in batch_pairs for idx in pair[0]]
        batch_b = [self.dataset[idx] for pair in batch_pairs for idx in pair[1]]
         # 提取点云数据和标签
        point_clouds_a = torch.stack([torch.tensor(item[0], dtype=torch.float32) for item in batch_a])
        labels_a = torch.tensor([item[1] for item in batch_a], dtype=torch.long)
        
        point_clouds_b = torch.stack([torch.tensor(item[0], dtype=torch.float32) for item in batch_b])
        labels_b = torch.tensor([item[1] for item in batch_b], dtype=torch.long)
        
        # 收集原始文件名 (如果需要)
        filenames_a = [item[2] for item in batch_a]
        filenames_b = [item[2] for item in batch_b]
        return point_clouds_a, labels_a, point_clouds_b, labels_b

    # 使用方式
    #collate_fn = SiameseCollator(dataset)

# 使用示例
def get_siamese_dataloader_example():
    from torch.utils.data import DataLoader
    
    # 假设你已经有了 SinglePointCloudDataset 实例
    dataset = SinglePointCloudDataset(
        data_info_dir="/path/to/bucket/info",
        some_npy_root_dir="/path/to/npy/files"
    )
    
    # 创建孪生批处理采样器
    batch_sampler = SiamesePointCloudBatchSampler(
        dataset=dataset,
        bucket_info_dir="/path/to/bucket/info",
        max_points_per_batch=1000000,  # 根据你的GPU内存调整
        shuffle=True
    )
    
    # 创建 DataLoader
    # 注意：这里需要自定义 collate_fn 来处理成对批次
    collate_fn = SiameseCollator(dataset)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=4,  # 根据你的CPU核心数调整
    )
    
    return dataloader

