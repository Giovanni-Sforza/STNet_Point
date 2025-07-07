import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from collections import defaultdict
import bisect # 用于高效查找
import random
class TrulyMemoryEfficientDataset(Dataset):
    def __init__(self,
                 root_dir_source1: str,
                 root_dir_source2: str,
                 split_file_source1: str,
                 split_file_source2: str,
                 transform=None,
                 mode='train',
                 max_pairs_per_class=None,
                 sampling_strategy='random',
                 seed=42):
        
        self.root_dir_s1 = root_dir_source1
        self.root_dir_s2 = root_dir_source2
        self.transform = transform
        self.mode = mode
        self.max_pairs_per_class = max_pairs_per_class
        self.sampling_strategy = sampling_strategy
        self.seed = seed
        self.current_epoch = 0
        
        # 按类别分组文件名
        s1_files_by_class = self._group_files_by_class(split_file_source1, self.root_dir_s1)
        s2_files_by_class = self._group_files_by_class(split_file_source2, self.root_dir_s2)

        # 只存储必要的元信息，不存储实际配对
        self.class_info = []
        self.s1_files_by_class = {}
        self.s2_files_by_class = {}
        self.cumulative_counts = []
        
        total_virtual_pairs = 0
        sorted_s1_classes = sorted(s1_files_by_class.keys())
        
        for class1 in sorted_s1_classes:
            target_class2 = class1 + 13
            if target_class2 in s2_files_by_class:
                files1 = s1_files_by_class[class1]
                files2 = s2_files_by_class[target_class2]
                
                if len(files1) == 0 or len(files2) == 0:
                    continue
                
                # 计算这个类别组应该产生多少配对
                if self.sampling_strategy == 'full':
                    num_pairs = len(files1) * len(files2)
                elif self.sampling_strategy == 'sequential':
                    num_pairs = min(len(files1), len(files2))
                elif self.sampling_strategy == 'random':
                    if mode == 'train':
                        max_pairs = max_pairs_per_class or min(1000, len(files1) * len(files2))
                    else:
                        max_pairs = max_pairs_per_class or min(100, len(files1) * len(files2))
                    num_pairs = min(max_pairs, len(files1) * len(files2))
                
                if num_pairs > 0:
                    # 只存储元信息
                    self.class_info.append({
                        'class1': class1,
                        'class2': target_class2,
                        'num_files1': len(files1),
                        'num_files2': len(files2),
                        'num_pairs': num_pairs
                    })
                    
                    # 只存储参与配对的文件列表
                    self.s1_files_by_class[class1] = files1
                    self.s2_files_by_class[target_class2] = files2
                    
                    total_virtual_pairs += num_pairs
                    self.cumulative_counts.append(total_virtual_pairs)

        if total_virtual_pairs == 0:
            raise ValueError("No valid pairs found based on the rule 'class1 == class2 - 13'.")

        print(f"Dataset initialized: {total_virtual_pairs} virtual pairs, "
              f"{len(self.class_info)} class groups, mode='{mode}'")

    def set_epoch(self, epoch):
        """设置epoch用于动态随机化"""
        self.current_epoch = epoch

    def _group_files_by_class(self, split_file, root_dir):
        files_by_class = defaultdict(list)
        with open(split_file, 'r') as f:
            filenames = [line.strip() for line in f if line.strip()]
        for fname in filenames:
            try:
                class_label = self._parse_event_filename(fname)
                if os.path.exists(os.path.join(root_dir, fname)):
                    files_by_class[class_label].append(fname)
            except ValueError as e:
                print(f"Warning: Skipping file {fname} due to parsing error: {e}")
        return files_by_class

    def _parse_event_filename(self, full_filename):
        name_without_ext = os.path.splitext(full_filename)[0]
        match = re.match(r'^(\d+)-(\d+)_(\d+)$', name_without_ext)
        if match:
            return int(match.group(2))
        raise ValueError(f"Filename {full_filename} does not match expected format.")

    def __len__(self):
        return self.cumulative_counts[-1] if self.cumulative_counts else 0

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. 找到idx属于哪个类别组（与原代码相同）
        group_idx = bisect.bisect_right(self.cumulative_counts, idx)
        start_idx = self.cumulative_counts[group_idx - 1] if group_idx > 0 else 0
        local_idx = idx - start_idx
        
        # 2. 获取类别组信息
        class_info = self.class_info[group_idx]
        class1 = class_info['class1']
        class2 = class_info['class2']
        num_files1 = class_info['num_files1']
        num_files2 = class_info['num_files2']
        num_pairs = class_info['num_pairs']
        
        # 3. 动态计算文件索引（关键：不存储配对，而是动态计算）
        if self.sampling_strategy == 'full':
            # 全遍历：直接从local_idx解码
            idx1 = local_idx // num_files2
            idx2 = local_idx % num_files2
            
        elif self.sampling_strategy == 'sequential':
            # 顺序配对
            idx1 = local_idx
            idx2 = local_idx
            
        elif self.sampling_strategy == 'random':
            # 动态随机配对
            if self.mode in ['val']:
                # 测试模式：确定性随机
                pair_seed = hash(f"{class1}_{class2}_{local_idx}_{self.seed}") % (2**31)
                pair_random = random.Random(pair_seed)
                idx1 = pair_random.randint(0, num_files1 - 1)
                idx2 = pair_random.randint(0, num_files2 - 1)
            else:
                # 训练模式：考虑epoch的随机性
                pair_seed = hash(f"{class1}_{class2}_{local_idx}_{self.seed}_{self.current_epoch}") % (2**31)
                pair_random = random.Random(pair_seed)
                
                if num_pairs == num_files1 * num_files2:
                    # 如果要全部配对，就按网格解码
                    idx1 = local_idx // num_files2
                    idx2 = local_idx % num_files2
                else:
                    # 随机选择
                    idx1 = pair_random.randint(0, num_files1 - 1)
                    idx2 = pair_random.randint(0, num_files2 - 1)
        
        # 4. 获取文件名和加载数据
        event_filename_s1 = self.s1_files_by_class[class1][idx1]
        event_filename_s2 = self.s2_files_by_class[class2][idx2]
        
        # 5. 加载数据
        event_data1_path = os.path.join(self.root_dir_s1, event_filename_s1)
        event_data1 = np.load(event_data1_path)
        
        event_data2_path = os.path.join(self.root_dir_s2, event_filename_s2)
        event_data2 = np.load(event_data2_path)

        # 6. 解析文件信息
        def parse_full_info(filename):
            name_without_ext = os.path.splitext(filename)[0]
            match = re.match(r'^(\d+)-(\d+)_(\d+)$', name_without_ext)
            return int(match.group(1)), int(match.group(2)), match.group(3)

        point_count_s1, class_label_s1, _ = parse_full_info(event_filename_s1)
        point_count_s2, class_label_s2, _ = parse_full_info(event_filename_s2)

        sample = {
            'event1': torch.from_numpy(event_data1).float(),
            'class_label1': torch.tensor(class_label_s1, dtype=torch.long),
            'point_count1': torch.tensor(point_count_s1, dtype=torch.long),

            'event2': torch.from_numpy(event_data2).float(),
            'class_label2': torch.tensor(class_label_s2, dtype=torch.long),
            'point_count2': torch.tensor(point_count_s2, dtype=torch.long),

            'filename1': event_filename_s1,
            'filename2': event_filename_s2,
        }
        
        if self.transform:
            sample = self.transform(sample)

        return sample