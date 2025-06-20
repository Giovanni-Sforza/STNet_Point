import os
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Sampler, Dataset
from collections import defaultdict
from dataset import SinglePointCloudDataset

class DynamicPointCloudBatchSampler(Sampler):
    """
    动态点云批处理采样器 - 根据点云中的点数决定批次大小，
    确保每个批次只从同一个桶中读取数据。
    
    参数:
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
        # 为每个桶创建批次
        self.batches = self._create_batches()
        
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
                    buckets[points_count].append(idx)
                else:
                    print(f"警告: 在数据集中找不到文件 {npy_filename}")
        
        return buckets
    
    def _create_batches(self) -> List[List[int]]:
        """
        为每个桶创建批次，确保每个批次的总点数不超过最大限制
        """
        all_batches = []
        
        for points_count, indices in self.buckets.items():
            # 如果需要，打乱每个桶内的样本顺序
            if self.shuffle:
                random.shuffle(indices)
            
            # 计算每个批次可以包含的最大样本数
            # 点云数据的总点数 = 每个点云的点数 × 点云数量
            max_samples_per_batch = max(1, self.max_points_per_batch // points_count)
            
            # 创建批次
            for i in range(0, len(indices), max_samples_per_batch):
                batch = indices[i:i + max_samples_per_batch]
                all_batches.append(batch)
        
        # 如果需要，打乱不同桶之间的批次顺序
        if self.shuffle:
            random.shuffle(all_batches)
            
        return all_batches
    
    def __iter__(self):
        """返回批次迭代器"""
        if self.shuffle:  # 如果每个 epoch 都要重新打乱
            self.batches = self._create_batches()
        
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        """返回批次总数"""
        return len(self.batches)





if __name__ == "__main__":
    # 使用示例
    def get_dataloader_example():
        from torch.utils.data import DataLoader
        
        # 假设你已经有了 SinglePointCloudDataset 实例
        dataset = SinglePointCloudDataset(
            data_info_dir="/path/to/bucket/info",
            some_npy_root_dir="/path/to/npy/files"
        )
        
        # 创建动态批处理采样器
        batch_sampler = DynamicPointCloudBatchSampler(
            dataset=dataset,
            bucket_info_dir="/path/to/bucket/info",
            max_points_per_batch=1000000,  # 根据你的GPU内存调整
            shuffle=True
        )
        
        # 创建 DataLoader
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=4,  # 根据你的CPU核心数调整
            # 注意：当使用 batch_sampler 时，不能再设置 batch_size 和 shuffle
        )
        
        return dataloader


    # 使用示例：如何遍历数据加载器
    def training_loop_example(dataloader):
        for batch_data, batch_labels, batch_filenames in dataloader:
            # batch_data: 点云数据
            # batch_labels: 类别标签
            # batch_filenames: 原始文件名
            
            # 处理批次数据...
            pass