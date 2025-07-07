import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from collections import defaultdict
import bisect # 用于高效查找
import random
import logging


class PairedEventContrastiveDatasetV3(Dataset):
    def __init__(self,
                 root_dir_source1: str, # event_source_1 文件夹的完整路径
                 root_dir_source2: str, # event_source_2 文件夹的完整路
                 split_file_source1: str, # 记录source1数据文件名的txt文件完整路径
                 split_file_source2: str, # 记录source2数据文件名的txt文件完整路径
                 shared_mask_from_source1: bool = True, # 新增：如果mask共享，是否从source1的文件名推断mask名
                 transform=None):
        """
        Args:
            root_dir_source1 (string): source1 事件数据文件夹的完整路径。
            root_dir_source2 (string): source2 事件数据文件夹的完整路径。
            split_file_source1 (string): 记录source1数据文件名列表的txt文件完整路径。
            split_file_source2 (string): 记录source2数据文件名列表的txt文件完整路径。
            transform (callable, optional): 可选的转换操作。
        """
        self.root_dir_s1 = root_dir_source1
        self.root_dir_s2 = root_dir_source2
        self.shared_mask_from_source1 = shared_mask_from_source1
        self.transform = transform

        # 读取文件名列表 (txt文件中只包含文件名，不包含路径)
        with open(split_file_source1, 'r') as f:
            self.event_filenames_s1 = [line.strip() for line in f if line.strip()]
        with open(split_file_source2, 'r') as f:
            self.event_filenames_s2 = [line.strip() for line in f if line.strip()]

        if len(self.event_filenames_s1) != len(self.event_filenames_s2):
            raise ValueError(f"Mismatch in number of files listed in {split_file_source1} and {split_file_source2}")

        # 健全性检查
        for fname1, fname2 in zip(self.event_filenames_s1, self.event_filenames_s2):
            path1 = os.path.join(self.root_dir_s1, fname1)
            path2 = os.path.join(self.root_dir_s2, fname2)
            if not os.path.exists(path1):
                raise FileNotFoundError(f"Event file not found: {path1}")
            if not os.path.exists(path2):
                raise FileNotFoundError(f"Event file not found: {path2}")

            core_name1, _, _ = self._parse_event_filename(fname1)




    def _parse_event_filename(self, full_filename):
        """
        解析文件名 <point_count>-<class_label>_<filename_core>.npy
        其中 filename_core 是纯数字。
        返回 (filename_core, point_count, class_label)
        """
        name_without_ext = os.path.splitext(full_filename)[0]
        # 正则表达式: <数字>-<数字>_<数字>.npy
        match = re.match(r'^(\d+)-(\d+)_(\d+)$', name_without_ext)
        if match:
            point_count = int(match.group(1))
            class_label = int(match.group(2))
            filename_core = match.group(3) # filename_core 是字符串形式的数字
            return filename_core, point_count, class_label
        else:
            raise ValueError(f"Filename {full_filename} does not match expected format '<digits>-<digits>_<digits>.npy'")

    def __len__(self):
        return len(self.event_filenames_s1)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        event_filename_s1 = self.event_filenames_s1[idx]
        event_filename_s2 = self.event_filenames_s2[idx]

        # --- 解析 Source 1 ---
        core_name_s1, point_count_s1, class_label_s1 = self._parse_event_filename(event_filename_s1)
        event_data1_path = os.path.join(self.root_dir_s1, event_filename_s1)
        event_data1 = np.load(event_data1_path)

        # --- 解析 Source 2 ---
        # (即使mask共享，我们也需要解析s2的事件文件名以获取其元数据)
        core_name_s2, point_count_s2, class_label_s2 = self._parse_event_filename(event_filename_s2)
        event_data2_path = os.path.join(self.root_dir_s2, event_filename_s2)
        event_data2 = np.load(event_data2_path)

        # --- 加载 Mask ---
       

        
        # 转换为张量 (event data)
        event_data1_tensor = torch.from_numpy(event_data1).float()
        event_data2_tensor = torch.from_numpy(event_data2).float()

        sample = {
            'event1': event_data1_tensor,
            'class_label1': torch.tensor(class_label_s1, dtype=torch.long),
            'point_count1': torch.tensor(point_count_s1, dtype=torch.long),

            'event2': event_data2_tensor,
            'class_label2': torch.tensor(class_label_s2, dtype=torch.long),
            'point_count2': torch.tensor(point_count_s2, dtype=torch.long),

            'filename1': event_filename_s1,
            'filename2': event_filename_s2,
            'core_name_s1': core_name_s1, # 返回核心ID可能有用
            'core_name_s2': core_name_s2
        }
        
        if self.shared_mask_from_source1:
            sample['shared_mask_core_name'] = core_name_s1

        if self.transform:
            sample = self.transform(sample)

        return sample

# --- 示例 Config (模拟从外部配置文件读取) ---
class Config:
    def __init__(self):
        # 基础路径设置
        self.project_root = "dummy_contrastive_data_v3" # 假设这是你的项目或数据存放的根目录

        # 数据源1的路径
        self.source1_name = "event_set_alpha"
        self.s1_event_dir = os.path.join(self.project_root, self.source1_name)
        self.s1_mask_dir = os.path.join(self.s1_event_dir, "mask")

        # 数据源2的路径
        self.source2_name = "event_set_beta"
        self.s2_event_dir = os.path.join(self.project_root, self.source2_name)
        self.s2_mask_dir = os.path.join(self.s2_event_dir, "mask")
        
        # Split 文件路径 (这些文件通常在 project_root 或一个专门的 splits 文件夹下)
        self.train1_split_file = os.path.join(self.project_root, "train1.txt")
        self.train2_split_file = os.path.join(self.project_root, "train2.txt")
        self.test1_split_file = os.path.join(self.project_root, "test1.txt")
        self.test2_split_file = os.path.join(self.project_root, "test2.txt")

        # 是否共享mask，并从source1推断
        self.use_shared_mask_from_s1 = True




class MemoryEfficientRandomPairedDataset(Dataset):
    def __init__(self,
                 root_dir_source1: str,
                 root_dir_source2: str,
                 split_file_source1: str,
                 split_file_source2: str,
                 transform=None,
                 mode='train',  # 新增：'train', 'test', 'val'
                 max_pairs_per_class=None,  # 新增：限制每个类别的最大配对数
                 sampling_strategy='random',  # 新增：'random', 'sequential', 'full'
                 seed=42):  # 新增：随机种子
        
        self.root_dir_s1 = root_dir_source1
        self.root_dir_s2 = root_dir_source2
        self.transform = transform
        self.mode = mode
        self.max_pairs_per_class = max_pairs_per_class
        self.sampling_strategy = sampling_strategy
        
        # 设置随机种子以确保可重现性
        if seed is not None:
            random.seed(seed)
        self.base_seed = seed
        # 1. 按类别分组文件名
        self.s1_files_by_class = self._group_files_by_class(split_file_source1, self.root_dir_s1)
        self.s2_files_by_class = self._group_files_by_class(split_file_source2, self.root_dir_s2)

        # 2. 根据模式创建配对策略
        self.pairs = []  # 存储实际的配对 (class1, idx1, class2, idx2)
        
        # 按类别排序以保证确定性
        
        self.sorted_s1_classes = sorted( self.s1_files_by_class.keys())

        for class1 in self.sorted_s1_classes:
            target_class2 = class1 + 13
            if target_class2 in self.s2_files_by_class:
                files1 = self.s1_files_by_class[class1]
                files2 = self.s2_files_by_class[target_class2]
                
                if len(files1) == 0 or len(files2) == 0:
                    continue

                # 根据不同策略生成配对
                class_pairs = self._generate_pairs_for_class(
                    class1, target_class2, len(files1), len(files2)
                )
                self.pairs.extend(class_pairs)

        # 存储文件信息
        self.s1_files_by_class = {k: v for k, v in self.s1_files_by_class.items() 
                                  if any(pair[0] == k for pair in self.pairs)}
        self.s2_files_by_class = {k: v for k, v in self.s2_files_by_class.items() 
                                  if any(pair[2] == k for pair in self.pairs)}

        if len(self.pairs) == 0:
            raise ValueError("No valid pairs found based on the rule 'class1 == class2 - 13'.")

        print(f"Dataset initialized in '{mode}' mode. "
              f"Generated {len(self.pairs)} pairs using '{sampling_strategy}' strategy.")
    def shuffle_pairs(self, epoch: int):
        """在每个epoch开始时调用，以重新生成随机配对。"""
        # 使用基础种子和epoch号来创建一个新的、可复现但每个epoch都不同的种子
        if self.base_seed is not None:
            random.seed(self.base_seed + epoch)

        self.pairs = []
        for class1 in self.sorted_s1_classes:
            target_class2 = class1 + 13
            if target_class2 in self.s2_files_by_class:
                files1 = self.s1_files_by_class[class1]
                files2 = self.s2_files_by_class[target_class2]
                
                if len(files1) == 0 or len(files2) == 0:
                    continue

                class_pairs = self._generate_pairs_for_class(
                    class1, target_class2, len(files1), len(files2)
                )
                self.pairs.extend(class_pairs)

        if len(self.pairs) == 0 and epoch == 0: # 只在第一次检查时报错
            raise ValueError("No valid pairs found based on the rule 'class1 == class2 - 13'.")

        print(f"Epoch {epoch}: Dataset pairs regenerated in '{self.mode}' mode. "
              f"Generated {len(self.pairs)} pairs using '{self.sampling_strategy}' strategy.")


    def _generate_pairs_for_class(self, class1, class2, num_files1, num_files2):
        """根据策略生成类别内的配对"""
        pairs = []
        
        if self.sampling_strategy == 'full':
            # 全遍历（原来的行为）
            for i in range(num_files1):
                for j in range(num_files2):
                    pairs.append((class1, i, class2, j))
                    
        elif self.sampling_strategy == 'sequential':
            # 顺序配对：每个文件1与对应位置的文件2配对
            min_files = min(num_files1, num_files2)
            for i in range(min_files):
                pairs.append((class1, i, class2, i))
                
        elif self.sampling_strategy == 'random':
            # 随机采样配对
            if self.mode == 'train':
                # 训练时可以用更多配对
                max_pairs = self.max_pairs_per_class or min(1000, num_files1 * num_files2)
            else:
                # 测试/验证时用较少配对
                max_pairs = self.max_pairs_per_class or min(100, num_files1 * num_files2)
            
            # 生成随机配对
            actual_pairs = min(max_pairs, num_files1 * num_files2)
            
            if actual_pairs == num_files1 * num_files2:
                # 如果需要的配对数等于总数，直接全遍历
                for i in range(num_files1):
                    for j in range(num_files2):
                        pairs.append((class1, i, class2, j))
            else:
                # 随机采样
                sampled_pairs = set()
                while len(sampled_pairs) < actual_pairs:
                    i = random.randint(0, num_files1 - 1)
                    j = random.randint(0, num_files2 - 1)
                    sampled_pairs.add((i, j))
                
                for i, j in sampled_pairs:
                    pairs.append((class1, i, class2, j))
        
        return pairs

    def _parse_event_filename(self, full_filename):
        name_without_ext = os.path.splitext(full_filename)[0]
        match = re.match(r'^(\d+)-(\d+)_(\d+)$', name_without_ext)
        if match:
            return int(match.group(2))
        raise ValueError(f"Filename {full_filename} does not match expected format.")

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

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 直接从预生成的配对列表中获取
        class1, idx1, class2, idx2 = self.pairs[idx]
        
        # 获取文件名
        event_filename_s1 = self.s1_files_by_class[class1][idx1]
        event_filename_s2 = self.s2_files_by_class[class2][idx2]
        
        # 加载数据
        event_data1_path = os.path.join(self.root_dir_s1, event_filename_s1)
        event_data1 = np.load(event_data1_path)
        
        event_data2_path = os.path.join(self.root_dir_s2, event_filename_s2)
        event_data2 = np.load(event_data2_path)

        # 解析文件信息
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
    
class VotingPairedDataset(Dataset):
    """
    A Dataset for validation that implements the voting strategy.
    For each sample from source1, it pairs it with N random samples from
    the corresponding class in source2.
    """
    def __init__(self,
                 root_dir_source1: str,
                 root_dir_source2: str,
                 split_file_source1: str,
                 split_file_source2: str,
                 num_votes: int = 50,
                 seed: int = 42):
        
        self.root_dir_s1 = root_dir_source1
        self.root_dir_s2 = root_dir_source2
        self.num_votes = num_votes
        
        if seed is not None:
            random.seed(seed)

        # 1. Load source1 files, which are the primary samples to be evaluated.
        with open(split_file_source1, 'r') as f:
            self.s1_files = [line.strip() for line in f if line.strip()]

        # 2. Load and group source2 files by class to serve as voters.
        self.s2_files_by_class = self._group_files_by_class(split_file_source2, self.root_dir_s2)

        logging.info(f"VotingPairedDataset initialized. Found {len(self.s1_files)} primary samples.")
        if not self.s2_files_by_class:
            raise ValueError("Source 2 (voters) file list is empty or could not be grouped.")

    def _parse_event_filename(self, full_filename):
        # This utility function should be consistent across your project
        name_without_ext = os.path.splitext(full_filename)[0]
        match = re.match(r'^(\d+)-(\d+)_(\d+)$', name_without_ext)
        if match:
            return int(match.group(2))
        raise ValueError(f"Filename {full_filename} does not match expected format for class parsing.")

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

    def __len__(self):
        return len(self.s1_files)

    def __getitem__(self, idx):
        # 1. Get the primary event1 file
        event1_filename = self.s1_files[idx]
        event1_path = os.path.join(self.root_dir_s1, event1_filename)
        event1_data = torch.from_numpy(np.load(event1_path)).float()
        
        # 2. Determine its class and the target class for voters
        s1_class = self._parse_event_filename(event1_filename)
        target_s2_class = s1_class + 13

        # 3. Get candidate voters from source2
        candidate_voters = self.s2_files_by_class.get(target_s2_class, [])
        if not candidate_voters:
            # If no voters, we must return something of the correct shape.
            # We'll return a dummy voter batch and a flag.
            logging.warning(f"No voters found for s1_class {s1_class} (target s2_class {target_s2_class}).")
            dummy_voter_data = torch.zeros_like(event1_data).unsqueeze(0) # Shape (1, num_points, features)
            return {
                'event1_data': event1_data,
                'event1_class': s1_class,
                'voters_data': dummy_voter_data,
                'has_voters': False
            }

        # 4. Sample N voters
        num_to_sample = min(self.num_votes, len(candidate_voters))
        voter_filenames = random.sample(candidate_voters, num_to_sample)
        
        # 5. Load all voter data and stack them into a tensor
        voters_data_list = [
            torch.from_numpy(np.load(os.path.join(self.root_dir_s2, fname))).float()
            for fname in voter_filenames
        ]
        voters_data = torch.stack(voters_data_list) # Shape (N, num_points, features)

        return {
            'event1_data': event1_data,        # Tensor [num_points, features]
            'event1_class': s1_class,          # int
            'voters_data': voters_data,        # Tensor [N, num_points, features]
            'has_voters': True                 # bool
        }
    

class EpochShuffledPairedDataset(Dataset):
    """
    A PyTorch Dataset that implements the epoch-based shuffling and pairing strategy.
    
    In each epoch:
    1. For each class, it takes the list of Ru files and Zr files.
    2. It shuffles both lists independently.
    3. It pairs them one-to-one.
    4. All pairs from all classes are combined to form the data for one epoch.
    """
    def __init__(self, root_dir_ru, root_dir_zr, split_file_ru, split_file_zr):
        self.root_dir_ru = root_dir_ru
        self.root_dir_zr = root_dir_zr
        
        logging.info("Initializing EpochShuffledPairedDataset...")
        
        # 1. Load all file paths and group by class
        self.ru_files_by_class = self._load_and_group_files(split_file_ru)
        self.zr_files_by_class = self._load_and_group_files(split_file_zr)
        
        self.classes = sorted(self.ru_files_by_class.keys())
        logging.info(f"Found {len(self.classes)} classes: {self.classes}")
        for c in self.classes:
            logging.info(f"  Class {c}: {len(self.ru_files_by_class[c])} Ru files, {len(self.zr_files_by_class[c])} Zr files.")
            if c not in self.zr_files_by_class:
                logging.warning(f"Class {c} has Ru files but no Zr files. This class will be skipped.")

        # 2. This will hold the pairs for the current epoch
        self.pairs = []
        
        # 3. Create the initial set of pairs for the first epoch
        self.create_epoch_pairs()

    def _parse_class_from_filename(self, filename):
        """
        Parses the base class ID (e.g., 1 to 4) from the filename.
        Example: '1024-1_...npy' -> class 1
                 '1024-14_...npy' -> class 1 (since 14 % 13 = 1, and label starts from 1)
        Adjust the logic if your file naming is different.
        """
        name_without_ext = os.path.splitext(filename)[0]
        match = re.match(r'^\d+-(\d+)_', name_without_ext)
        if match:
            # Assumes original labels are 1-4 for Ru and 14-17 for Zr
            original_label = int(match.group(1))
            # Normalize to a base class (e.g., 1-4)
            # This logic assumes your 4 classes are labeled 1,2,3,4 for Ru and 14,15,16,17 for Zr
            # If labels are different (e.g., 1-4 for both), simplify this.
            # Let's assume a simple case where we map both to a common base ID
            if 1 <= original_label <= 4: # Assuming Ru classes
                return original_label
            elif 14 <= original_label <= 17: # Assuming Zr classes are Ru_class + 13
                return original_label - 13
            else:
                 # A more robust approach if labels are just 1,2,3,4 for BOTH Ru and Zr
                 return (original_label - 1) % 4 + 1
        return -1 # Invalid format

    def _load_and_group_files(self, split_file):
        files_by_class = defaultdict(list)
        try:
            with open(split_file, 'r') as f:
                filenames = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logging.error(f"Split file not found: {split_file}")
            return files_by_class

        for fname in filenames:
            class_id = self._parse_class_from_filename(fname)
            if class_id != -1:
                files_by_class[class_id].append(fname)
        return files_by_class

    def create_epoch_pairs(self):
        """
        This is the core method. It shuffles and pairs the data for a new epoch.
        """
        logging.info("Creating new pairs for the next epoch...")
        self.pairs = []
        
        for class_id in self.classes:
            if class_id not in self.zr_files_by_class:
                continue
                
            ru_list = self.ru_files_by_class[class_id]
            zr_list = self.zr_files_by_class[class_id]
            
            # Shuffle copies of the lists
            random.shuffle(ru_list)
            random.shuffle(zr_list)
            
            # Pair them up to the length of the shorter list
            num_pairs_for_class = min(len(ru_list), len(zr_list))
            
            for i in range(num_pairs_for_class):
                ru_file = ru_list[i]
                zr_file = zr_list[i]
                
                # The label should be the base class ID. We subtract 1 for 0-indexing.
                label = class_id -1
                
                self.pairs.append((ru_file, zr_file, label))

        # Shuffle the combined list of all pairs from all classes
        random.shuffle(self.pairs)
        logging.info(f"Created a total of {len(self.pairs)} pairs for this epoch.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ru_fname, zr_fname, label = self.pairs[idx]
        
        # Load data from files
        try:
            ru_path = os.path.join(self.root_dir_ru, ru_fname)
            zr_path = os.path.join(self.root_dir_zr, zr_fname)
            
            event1_data = np.load(ru_path)
            event2_data = np.load(zr_path)

            return {
                'event1': torch.from_numpy(event1_data).float(),
                'event2': torch.from_numpy(event2_data).float(),
                # We return the base class label directly.
                # In your _process_spt_batch, you used (label-1)%13. 
                # Here we pass the 0-indexed class label.
                # Since your classes are 1,2,3,4 -> this will be 0,1,2,3
                'class_label1': torch.tensor(label, dtype=torch.long), 
                'class_label2': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            logging.error(f"Error loading data for index {idx} ({ru_fname}, {zr_fname}): {e}")
            # Return a dummy sample or re-raise
            # For simplicity, we can try getting the next valid sample
            return self.__getitem__((idx + 1) % len(self))

# --- 示例使用 ---
if __name__ == '__main__':
    cfg = Config() # 加载配置

    # 1. 创建一些假的npy数据和txt文件用于测试
    print("Creating dummy data...")
    # 使用cfg中的路径
    os.makedirs(cfg.s1_event_dir, exist_ok=True)
    os.makedirs(cfg.s1_mask_dir, exist_ok=True)
    os.makedirs(cfg.s2_event_dir, exist_ok=True)
    os.makedirs(cfg.s2_mask_dir, exist_ok=True)

    num_dummy_files_train = 3
    num_dummy_files_test = 2
    max_points = 200
    num_features = 64

    train1_list = []
    train2_list = []
    test1_list = []
    test2_list = []

    def create_dummy_files_for_set_v3(prefix, num_files, file_list_s1_ref, file_list_s2_ref, shared_core_id_start=0):
        for i in range(num_files):
            # 假设共享mask，所以 core_id 对于配对的文件是相同的
            shared_core_id = f"{shared_core_id_start + i:06d}" # 例如 "000000", "000001"

            pc1 = np.random.randint(90, max_points)
            cl1 = np.random.randint(0, 5)
            fname1 = f"{pc1}-{cl1}_{shared_core_id}.npy" # 使用共享核心ID
            file_list_s1_ref.append(fname1)

            data1 = np.random.rand(max_points, num_features).astype(np.float32)
            data1[pc1:] = 0
            np.save(os.path.join(cfg.s1_event_dir, fname1), data1)
            
            # 如果共享mask，我们只需要创建一个mask文件，基于shared_core_id
            # 假设mask也放在s1的mask目录下 (或者一个统一的共享mask目录)
            mask_shared = np.zeros(max_points, dtype=bool)
            # Mask的真实长度可以基于pc1, pc2的某种组合，或者一个固定的值
            # 这里简单假设它基于pc1的长度，或者一个预定义的最大真实长度
            mask_true_len = pc1 # 或者 min(pc1,pc2) 或 max(pc1,pc2) 或一些其他逻辑
            mask_shared[:mask_true_len] = True
            if not os.path.exists(os.path.join(cfg.s1_mask_dir, f"{shared_core_id}.npy")): # 避免重复创建
                 np.save(os.path.join(cfg.s1_mask_dir, f"{shared_core_id}.npy"), mask_shared)


            pc2 = np.random.randint(90, max_points)
            cl2 = np.random.randint(0, 5)
            fname2 = f"{pc2}-{cl2}_{shared_core_id}.npy" # 使用相同的共享核心ID
            file_list_s2_ref.append(fname2)

            data2 = np.random.rand(max_points, num_features).astype(np.float32)
            data2[pc2:] = 0
            np.save(os.path.join(cfg.s2_event_dir, fname2), data2)
            
            # 如果不是共享mask (cfg.use_shared_mask_from_s1 = False), 那么需要为s2也创建mask
            if not cfg.use_shared_mask_from_s1:
                mask_s2 = np.zeros(max_points, dtype=bool)
                mask_s2[:pc2] = True
                np.save(os.path.join(cfg.s2_mask_dir, f"{shared_core_id}.npy"), mask_s2)


    create_dummy_files_for_set_v3("train", num_dummy_files_train, train1_list, train2_list, shared_core_id_start=0)
    create_dummy_files_for_set_v3("test", num_dummy_files_test, test1_list, test2_list, shared_core_id_start=num_dummy_files_train)


    with open(cfg.train1_split_file, 'w') as f:
        for item in train1_list: f.write(f"{item}\n")
    with open(cfg.train2_split_file, 'w') as f:
        for item in train2_list: f.write(f"{item}\n")
    with open(cfg.test1_split_file, 'w') as f:
        for item in test1_list: f.write(f"{item}\n")
    with open(cfg.test2_split_file, 'w') as f:
        for item in test2_list: f.write(f"{item}\n")

    print(f"Dummy data created in {cfg.project_root}")

    # 2. 创建 Dataset 实例 (例如，训练集)
    train_dataset = PairedEventContrastiveDatasetV3(
        root_dir_source1=cfg.s1_event_dir,
        root_dir_source2=cfg.s2_event_dir,
        split_file_source1=cfg.train1_split_file,
        split_file_source2=cfg.train2_split_file,
        shared_mask_from_source1=cfg.use_shared_mask_from_s1
    )
    print(f"\nTrain Dataset size: {len(train_dataset)}")

    # 3. 获取一个样本
    if len(train_dataset) > 0:
        sample_data = train_dataset[0]
        print("\nSample 0 from train_dataset:")
        print(f"  Event 1: {sample_data['filename1']}, Core1: {sample_data['core_name_s1']}, Class1: {sample_data['class_label1']}, Points1: {sample_data['point_count1']}")
        print(f"  Mask 1 shape: {sample_data['mask1'].shape}, Num True: {torch.sum(sample_data['mask1'])}")
        print(f"  Event 2: {sample_data['filename2']}, Core2: {sample_data['core_name_s2']}, Class2: {sample_data['class_label2']}, Points2: {sample_data['point_count2']}")
        print(f"  Mask 2 shape: {sample_data['mask2'].shape}, Num True: {torch.sum(sample_data['mask2'])}")
        if 'shared_mask_core_name' in sample_data:
             print(f"  Shared Mask Core Name: {sample_data['shared_mask_core_name']}")
        print(f"  Are mask1 and mask2 the same object (if shared)? {sample_data['mask1'] is sample_data['mask2']}")


    # 4. 创建 DataLoader 实例
    batch_size = 2
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print("\nIterating through Train DataLoader...")
    for i_batch, batch_sample in enumerate(train_dataloader):
        print(f"\nBatch {i_batch}:")
        print(f"  Event 1 filenames: {batch_sample['filename1']}")
        print(f"  Event 2 filenames: {batch_sample['filename2']}")
        print(f"  Mask1 batch shape: {batch_sample['mask1'].shape}")
        # 如果共享mask，mask1和mask2的tensor id会相同
        # print(f"  Mask1 ids: {[id(m) for m in batch_sample['mask1']]}")
        # print(f"  Mask2 ids: {[id(m) for m in batch_sample['mask2']]}")

        # 验证共享mask在batch中是否仍然是同一个对象（如果batching没有深拷贝）
        if cfg.use_shared_mask_from_s1:
            for i in range(batch_sample['mask1'].size(0)):
                if not (batch_sample['mask1'][i] is batch_sample['mask2'][i] or torch.equal(batch_sample['mask1'][i], batch_sample['mask2'][i])):
                    print(f"    Warning: Masks in batch for sample {i} are not identical when they should be shared!")
                    break
            else:
                print("    Shared masks in batch appear consistent.")


        # 模型调用将是:
        # output1 = model(batch_sample['event1'], mask=batch_sample['mask1'])
        # output2 = model(batch_sample['event2'], mask=batch_sample['mask2'])
        # ...
        if i_batch >= 0:
            break
            
    # 清理dummy数据 (可选)
    # import shutil
    # shutil.rmtree(cfg.project_root)
    # print(f"\nDummy data deleted from {cfg.project_root}")