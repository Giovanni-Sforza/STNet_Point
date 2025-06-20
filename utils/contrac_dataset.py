import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re

class PairedEventContrastiveDatasetV3(Dataset):
    def __init__(self,
                 root_dir_source1: str, # event_source_1 文件夹的完整路径
                 root_dir_source2: str, # event_source_2 文件夹的完整路径
                 mask_dir_source1: str,   # event_source_1 下 mask 文件夹的完整路径
                 mask_dir_source2: str,   # event_source_2 下 mask 文件夹的完整路径 (如果mask是共享的，可能只需要一个mask_dir)
                 split_file_source1: str, # 记录source1数据文件名的txt文件完整路径
                 split_file_source2: str, # 记录source2数据文件名的txt文件完整路径
                 shared_mask_from_source1: bool = True, # 新增：如果mask共享，是否从source1的文件名推断mask名
                 transform=None):
        """
        Args:
            root_dir_source1 (string): source1 事件数据文件夹的完整路径。
            root_dir_source2 (string): source2 事件数据文件夹的完整路径。
            mask_dir_source1 (string): source1 事件对应的mask文件夹的完整路径。
            mask_dir_source2 (string): source2 事件对应的mask文件夹的完整路径。
                                       如果 shared_mask_from_source1 为 True，则此参数可以忽略或设为与 mask_dir_source1 相同。
            split_file_source1 (string): 记录source1数据文件名列表的txt文件完整路径。
            split_file_source2 (string): 记录source2数据文件名列表的txt文件完整路径。
            shared_mask_from_source1 (bool): 指示成对的事件是否共享同一个mask文件，
                                             并且该mask文件名从source1的事件文件名推断。
                                             如果为False，则会为source2也单独推断和加载mask。
            transform (callable, optional): 可选的转换操作。
        """
        self.root_dir_s1 = root_dir_source1
        self.root_dir_s2 = root_dir_source2
        self.mask_dir_s1 = mask_dir_source1
        self.mask_dir_s2 = mask_dir_source2 # 如果共享，这个可能和 mask_dir_s1 一样
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
            mask_path_for_s1 = os.path.join(self.mask_dir_s1, fname1)
            if not os.path.exists(mask_path_for_s1):
                raise FileNotFoundError(f"Mask file for source1 not found: {mask_path_for_s1}")

            if not self.shared_mask_from_source1: # 如果不共享，也检查s2的mask
                core_name2, _, _ = self._parse_event_filename(fname2)
                mask_path_for_s2 = os.path.join(self.mask_dir_s2, fname2)
                if not os.path.exists(mask_path_for_s2):
                    raise FileNotFoundError(f"Mask file for source2 not found: {mask_path_for_s2}")
            # 如果共享，我们已经检查了基于core_name1的mask，这里不再重复检查同一个文件


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
        if self.shared_mask_from_source1:
            # 假设成对的事件共享同一个 mask 文件，其名称由 source1 的 core_name 决定
            # 并且这个 mask 文件位于 self.mask_dir_s1 (或者一个统一的、单独指定的共享 mask 目录)
            # 如果 core_name_s1 和 core_name_s2 应该相同，这里可以加一个断言
            # assert core_name_s1 == core_name_s2, f"Shared mask assumed, but core names differ: {core_name_s1} vs {core_name_s2}"
            mask_filename_to_load = event_filename_s1
            mask_data_path = os.path.join(self.mask_dir_s1, mask_filename_to_load) # 使用 mask_dir_s1
            mask_data = np.load(mask_data_path)
            mask_data1_tensor = torch.from_numpy(mask_data).bool()
            mask_data2_tensor = mask_data1_tensor # 直接引用同一个mask张量
        else:
            # 为 event1 加载 mask
            mask_data1_path = os.path.join(self.mask_dir_s1, event_filename_s1)
            mask_data1 = np.load(mask_data1_path)
            mask_data1_tensor = torch.from_numpy(mask_data1).bool()

            # 为 event2 加载 mask
            mask_data2_path = os.path.join(self.mask_dir_s2, event_filename_s2) # 使用 mask_dir_s2
            mask_data2 = np.load(mask_data2_path)
            mask_data2_tensor = torch.from_numpy(mask_data2).bool()


        # 转换为张量 (event data)
        event_data1_tensor = torch.from_numpy(event_data1).float()
        event_data2_tensor = torch.from_numpy(event_data2).float()

        sample = {
            'event1': event_data1_tensor,
            'mask1': mask_data1_tensor, # 如果共享，mask1和mask2是同一个对象
            'class_label1': torch.tensor(class_label_s1, dtype=torch.long),
            'point_count1': torch.tensor(point_count_s1, dtype=torch.long),

            'event2': event_data2_tensor,
            'mask2': mask_data2_tensor, # 如果共享，mask1和mask2是同一个对象
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
        mask_dir_source1=cfg.s1_mask_dir,
        mask_dir_source2=cfg.s2_mask_dir, # 即使共享，也传递，Dataset内部逻辑处理
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