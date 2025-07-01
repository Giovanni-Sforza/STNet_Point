# finetune_spt.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset,Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import random
import logging
from omegaconf import OmegaConf
import shutil
import time
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import io
from PIL import Image
from collections import defaultdict
import re
from scipy.stats import mode as sp_mode
from tqdm import tqdm

try:
    from utils.STNet_dataset import MemoryEfficientRandomPairedDataset , VotingPairedDataset
    # Import create_model and the model classes if needed for type hinting or direct instantiation
    #from utils.STNet_dataset2 import TrulyMemoryEfficientDataset
    from model.STNet_point2 import create_model
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure 'utils' and 'model' directories are in PYTHONPATH or accessible,")
    print("and model.SPT contains create_model, SPT, and P_by_P_feature.")
    exit(1)
def batcher(iterable, batch_size):
    """Yields successive n-sized chunks from an iterable."""
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]

class FineTuningTrainer:
    def __init__(self, config, pretrained_checkpoint_path=None):
        self.config = config
        self.cur_device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")
        logging.info(f"FineTuningTrainer: Using device: {self.cur_device}")
        self.current_epoch = 23
        self.current_iter = 0
        self.best_spt_accuracy = -1.0
        self.best_spt_loss = float('inf')
        self.accumulation_steps = 1 # 在这里定义累积步数
        self.current_iter = 0
        self.output_dir = config.training.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # --- Configuration for classification task type ---
        self.classification_mode = config.spt_downstream.get("classification_mode", "13_class") # "13_class" or "3_class"
        if self.classification_mode == "3_class":
            self.effective_num_classes_for_task = 3
            # Define the mapping from 0-12 (original 13 classes) to 0-2 (3 super-classes)
            # Original 1-13 classes:
            # Super-class 0 (was 1): classes 1,2,3,7,8 (0-indexed: 0,1,2,6,7)
            # Super-class 1 (was 2): classes 4,5,6     (0-indexed: 3,4,5)
            # Super-class 2 (was 3): classes 9,10,11,12,13 (0-indexed: 8,9,10,11,12)
            self.map_13_to_3_class = {
                0: 0, 1: 0, 2: 0, 6: 0, 7: 0,  # Super-class 0
                3: 1, 4: 1, 5: 1,              # Super-class 1
                8: 2, 9: 2, 10: 2, 11: 2, 12: 2 # Super-class 2
            }
            # For confusion matrix labels (0-indexed for 3 classes)
            self.cm_display_labels_effective = [0, 1, 2]
            self.cm_display_names_effective = ["SuperClass1(1,2,3,7,8)", "SuperClass2(4,5,6)", "SuperClass3(9-13)"]
            logging.info("Fine-tuning for 3 super-classes.")
        elif self.classification_mode == "13_class":
            self.effective_num_classes_for_task = 4
            # Custom order for 13-class confusion matrix (0-indexed)
            # Original labels: 1,  2,  3,  7,  8,  4,  5,  6,  9, 10, 11, 12, 13
            # 0-indexed:       0,  1,  2,  6,  7,  3,  4,  5,  8,  9, 10, 11, 12
            #self.cm_display_labels_effective = [0, 1, 2, 6, 7, 3, 4, 5, 8, 9, 10, 11, 12]
            # Corresponding names for display (using 1-based for clarity in plot)
            self.cm_display_labels_effective = [0, 1,2,3]
            self.cm_display_names_effective = [str(l+1) for l in self.cm_display_labels_effective]
            logging.info("Fine-tuning for 13 original classes.")
        else:
            raise ValueError(f"Unsupported classification_mode: {self.classification_mode}")

        # --- Initialize SPT Model using create_model ---
        if not hasattr(config, 'spt_model_params'):
            raise ValueError("Configuration missing 'spt_model_params' section for SPT model creation.")
        
        # IMPORTANT: The 'num_class' in spt_model_params passed to create_model
        # should now be self.effective_num_classes_for_task so the classifier head is correct.
        spt_creation_config = OmegaConf.to_container(config.spt_model_params, resolve=True) # Make it a dict
        spt_creation_config['num_class'] = self.effective_num_classes_for_task # Override num_class for SPT creation
        
        self.spt_model = create_model(OmegaConf.create(spt_creation_config)).to(self.cur_device) # Pass modified config
        logging.info(f"SPT model created for fine-tuning with {self.effective_num_classes_for_task} output classes.")

        if pretrained_checkpoint_path:
            self._load_pretrained_backbone(pretrained_checkpoint_path)
        else:
            logging.warning("No pretrained_checkpoint_path provided. Backbone starts with random weights.")

        self.optimizer = getattr(optim, config.optimizer.name)(
            self.spt_model.parameters(), lr=config.optimizer.lr,
            weight_decay=config.optimizer.get('weight_decay', 0)
        )
        ft_scheduler_config = config.get('finetune_scheduler', config.finetune_scheduler)
        if not hasattr(ft_scheduler_config, 'T_max'):
             raise ValueError("Fine-tune scheduler config (CosineAnnealingLR) missing T_max.")
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=ft_scheduler_config.T_max,
            eta_min=ft_scheduler_config.get("eta_min", 0)
        )
        self.classification_criterion = nn.CrossEntropyLoss().to(self.cur_device)
        
        # This was for the original 13-class interpretation based on dataset label structure
        # Now, the actual number of classes for the task is self.effective_num_classes_for_task
        # self.actual_num_classes = config.spt_model_params.num_class # This would be 13 if based on old config
        # self.actual_num_classes = self.effective_num_classes_for_task # Already set

        self._load_spt_checkpoint()

        if not hasattr(self, 'writer') or self.writer is None:
            self.log_dir = os.path.join(self.output_dir, 'logs_finetune')
            if self.config.training.get("clean_log_dir_on_start", True) and self.current_iter == 0 and os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir)
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)
        logging.info(f"Fine-tuning TensorBoard logs: {self.log_dir}")

        data_cfg_section = config.get("finetune_data", config.finetune_data)
        self.train_dataset = MemoryEfficientRandomPairedDataset(
            root_dir_source1=data_cfg_section.s1_event_dir, root_dir_source2=data_cfg_section.s1_event_dir,
            split_file_source1=data_cfg_section.train1_split_file, split_file_source2=data_cfg_section.train2_split_file,max_pairs_per_class=50000
        )
        self.train_data_loader = DataLoader(
            self.train_dataset, batch_size=data_cfg_section.batch_size, shuffle=True,
            num_workers=data_cfg_section.num_workers, pin_memory=True,
            persistent_workers=True if data_cfg_section.num_workers > 0 else False,
            drop_last=True
        )
        self.train_data_loader_iter = None
        val_cfg = self.config.get("validation_voting",{})
        self.val_dataset = None
        if val_cfg.get("enabled"):
            # --- 使用新的投票数据集 ---
            logging.info("Voting validation enabled. Initializing VotingPairedDataset.")
            self.val_dataset = VotingPairedDataset(
                root_dir_source1=data_cfg_section.s1_event_dir,
                root_dir_source2=data_cfg_section.s1_event_dir,
                split_file_source1=data_cfg_section.val1_split_file,
                split_file_source2=data_cfg_section.val2_split_file,
                num_votes=val_cfg.get("num_votes", 50)
            )
            # 对于这个特殊的数据集，batch_size 必须是 1，因为每个样本返回的数据大小都不同
            self.val_data_loader = DataLoader(
                self.val_dataset, batch_size=1, shuffle=False, # BATCH SIZE MUST BE 1
                num_workers=data_cfg_section.num_workers,
                pin_memory=True
            )
        else:
            if data_cfg_section.get("val1_split_file") and data_cfg_section.get("val2_split_file"):
                self.val_dataset = MemoryEfficientRandomPairedDataset(
                    root_dir_source1=data_cfg_section.s1_event_dir, root_dir_source2=data_cfg_section.s1_event_dir,
                    split_file_source1=data_cfg_section.val1_split_file, split_file_source2=data_cfg_section.val2_split_file,max_pairs_per_class=500,
                    mode="test"
                )
                self.val_data_loader = DataLoader(
                    self.val_dataset, batch_size=data_cfg_section.batch_size, shuffle=False,
                    num_workers=data_cfg_section.num_workers, pin_memory=True,
                    persistent_workers=True if data_cfg_section.num_workers > 0 else False,
                    drop_last=True
                )
        
        self.max_iter = config.training.max_iter
        self.log_freq = config.training.log_freq
        self.val_freq = config.training.val_freq
        #print(self.spt_model.classifier[-1].out_features)

    @staticmethod
    def _transfer(cor):
        r = cor[:,: ,0]
        phi = cor[:,:, 1]
        z = cor[:,:, 2]#/1.7

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
        return a_cartesian_points,cor

    # 在 FineTuningTrainer 类中添加这个新方法

    def _preload_and_preprocess_validation_data(self, data_loader, device):
        """
        Iterates through the validation dataloader once to preload and preprocess
        all validation data into memory for efficient processing.
        
        Returns:
            - all_xyz (Tensor): Preprocessed coordinates for all samples.
            - all_points (Tensor): Preprocessed features for all samples.
            - all_labels (Tensor): Original class labels (1-26 or whatever).
        """
        logging.info("Pre-loading and pre-processing all validation data...")
        
        all_xyz_list, all_points_list, all_labels_list = [], [], []
        
        # 我们需要一个临时的、标准的 MemoryEfficientRandomPairedDataset 来加载数据
        # 因为它能高效地加载单个文件
        data_cfg = self.config.finetune_data
        # 合并 val1 和 val2 的文件列表
        with open(data_cfg.val1_split_file, 'r') as f1, open(data_cfg.val2_split_file, 'r') as f2:
            all_val_files_list = [line.strip() for line in f1 if line.strip()] + \
                                [line.strip() for line in f2 if line.strip()]
        
        # 去重，以防有文件同时在两个列表里
        all_val_files = sorted(list(set(all_val_files_list)))

        # 创建一个临时的单文件数据集
        from torch.utils.data import TensorDataset # 假设你的文件可以被这样处理
        
        # 更好的方法是创建一个简单的 SingleFileDataset
        class SingleFileDataset(Dataset):
            def __init__(self, root_dir, filenames):
                self.root_dir = root_dir
                self.filenames = filenames
            def __len__(self):
                return len(self.filenames)
            def __getitem__(self, idx):
                fname = self.filenames[idx]
                path = os.path.join(self.root_dir, fname)
                data = np.load(path)
                # 从文件名解析标签
                # (这里的解析逻辑需要和你已有的 _parse_event_filename 保持一致)
                name_without_ext = os.path.splitext(fname)[0]
                match = re.match(r'^(\d+)-(\d+)_(\d+)$', name_without_ext)
                label = int(match.group(2)) if match else -1
                return torch.from_numpy(data).float(), torch.tensor(label)

        temp_dataset = SingleFileDataset(data_cfg.s1_event_dir, all_val_files)
        temp_loader = DataLoader(temp_dataset, batch_size=data_loader.batch_size, 
                                num_workers=data_loader.num_workers, pin_memory=True)

        self.spt_model.eval()
        with torch.no_grad():
            for data_batch, labels_batch in tqdm(temp_loader, desc="Pre-processing Validation Data"):
                data_batch = data_batch.to(device)
                # 使用一个简化的 process 函数，只处理单边数据
                xyz, points, _, _, _ = self._process_spt_batch({
                    'event1': data_batch,
                    'event2': data_batch, # Dummy
                    'class_label1': labels_batch,
                    'class_label2': labels_batch # Dummy
                })

                all_xyz_list.append(xyz.cpu()) # 移回CPU以节省显存
                all_points_list.append(points.cpu())
                all_labels_list.append(labels_batch.cpu())

        all_xyz = torch.cat(all_xyz_list, dim=0)
        all_points = torch.cat(all_points_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0)
        
        # 创建一个从文件名到预处理数据索引的映射
        filename_to_idx = {fname: i for i, fname in enumerate(all_val_files)}

        logging.info(f"Pre-processing complete. Loaded {len(all_labels)} unique validation samples.")
        return all_xyz, all_points, all_labels, filename_to_idx
    def _load_pretrained_backbone(self, checkpoint_path):
        logging.info(f"Loading pretrained backbone weights from: {checkpoint_path}")
        try:
            pretrained_ckpt = torch.load(checkpoint_path, map_location=self.cur_device)

            # 依然先获取模型的 state_dict
            if 'spt_model_state_dict' in pretrained_ckpt:
                backbone_state_dict = pretrained_ckpt['spt_model_state_dict']
            elif 'model_state_dict' in pretrained_ckpt:
                # 兼容旧格式
                backbone_state_dict = pretrained_ckpt['model_state_dict']
            else:
                # 兼容直接保存 state_dict 的情况
                logging.info("Checkpoint does not contain a model key, assuming the file is the state_dict itself.")
                backbone_state_dict = pretrained_ckpt
                
            # --- 核心修改在这里 ---
            # 由于键名已经匹配，我们不再需要任何前缀处理
            # 直接将加载的 state_dict 加载到当前模型中
            # strict=False 仍然非常重要，因为它允许我们只加载部分权重（例如，不加载分类头）
            incompatible_keys = self.spt_model.load_state_dict(backbone_state_dict, strict=False)

            # 检查加载结果
            if incompatible_keys.missing_keys:
                # 缺失的键应该是我们不期望加载的部分，比如分类头，这是正常的
                logging.info(f"Partially loaded model. Missing keys (as expected for fine-tuning): {incompatible_keys.missing_keys[:5]}...")
            if incompatible_keys.unexpected_keys:
                # 如果预训练模型的分类头和当前模型不一样，这里可能会有“意外的键”
                # 这也是正常的，我们只关心主干网络
                logging.warning(f"Partially loaded model. Unexpected keys in checkpoint (likely old classifier head): {incompatible_keys.unexpected_keys[:5]}...")

            logging.info(f"Successfully loaded weights from checkpoint. Ready for fine-tuning.")

            # 冻结主干网络的代码
            if self.config.training.get("freeze_backbone_on_load", False):
                # 假设主干网络是除了 'classifier' 之外的所有模块
                # 这个列表你需要根据你的模型结构确认
                backbone_modules_to_freeze = ['sa1', 'sa2', 'sa3', 'tff1', 'tff2', 'tff3', 'sa_fusion1', 'sa_fusion2', 'gff1', 'gff2']
                logging.info(f"Freezing backbone parameters for modules: {backbone_modules_to_freeze}")
                
                frozen_count = 0
                for name, param in self.spt_model.named_parameters():
                    if any(name.startswith(mod) for mod in backbone_modules_to_freeze):
                        if param.requires_grad:
                            param.requires_grad = False
                            frozen_count += 1
                logging.info(f"Froze {frozen_count} parameters in the backbone.")
                
        except FileNotFoundError:
            logging.error(f"Pretrained backbone checkpoint DNE: {checkpoint_path}.")
        except Exception as e:
            logging.error(f"Error loading pretrained backbone: {e}", exc_info=True)



    def _process_spt_batch(self, batch_dict):
        # ... (代码分离 xyz, points 不变)
        coord_dim = self.config.spt_model_params.get('coord_dim',3)
        input_feature_dim = self.config.spt_model_params.get('input_feature_dim',0)

        event1_data = batch_dict['event1'].to(self.cur_device, non_blocking=True)
        xyz1 = event1_data[:, :, :coord_dim]
        points1 = None
        """if input_feature_dim > 0:
            if event1_data.shape[2] > coord_dim:
                 points1 = event1_data[:, :, coord_dim : coord_dim + input_feature_dim] # Fixed slicing
                 if points1.shape[2] != input_feature_dim: points1 = None; logging.warning("P1 dim mismatch")
            else: logging.warning(f"Not enough features in event1_data for points1")"""

        event2_data = batch_dict['event2'].to(self.cur_device, non_blocking=True)
        xyz2 = event2_data[:, :, :coord_dim]
        points2 = None
        """if input_feature_dim > 0:
            if event2_data.shape[2] > coord_dim:
                points2 = event2_data[:, :, coord_dim : coord_dim + input_feature_dim] # Fixed slicing
                if points2.shape[2] != input_feature_dim: points2 = None; logging.warning("P2 dim mismatch")
            else: logging.warning(f"Not enough features in event2_data for points2")"""

        original_labels_1_26 = batch_dict['class_label1'].to(self.cur_device, non_blocking=True)
        original_labels_2_26 = batch_dict['class_label2'].to(self.cur_device, non_blocking=True)
        # Step 1: Map original 1-26 labels to 0-12 (13 effective classes)
        labels_0_12 = (original_labels_1_26 - 1) % 13 # Assumes actual_num_classes for this mapping is 13
        labels_1_12 = (original_labels_2_26 - 1) % 13 
        assert torch.all(labels_0_12 == labels_1_12), "Mismatched base class labels in a pair!"
        #if torch.all(labels_0_12 == labels_1_12):
        #    print("eq")
        #else:
        #    print("errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
        # Step 2: Map 0-12 labels to final task labels (0-2 for 3-class, or identity for 13-class)
        if self.classification_mode == "3_class":
            # Apply the predefined mapping for 3 super-classes
            # Vectorized mapping:
            task_labels = torch.full_like(labels_0_12, -1) # Initialize with an invalid label
            for original_13_class_idx, super_class_idx in self.map_13_to_3_class.items():
                task_labels[labels_0_12 == original_13_class_idx] = super_class_idx
            
            if torch.any(task_labels == -1): # Check if any label was not mapped
                unmapped_originals = original_labels_1_26[task_labels == -1]
                logging.warning(f"Some labels could not be mapped to 3 super-classes. Original (1-26): {unmapped_originals.unique().cpu().tolist()}. Mapped 0-12: {labels_0_12[task_labels == -1].unique().cpu().tolist()}")
                # Handle unmapped labels, e.g., skip them or assign a default. For now, they remain -1.
                # The CrossEntropyLoss will ignore targets with negative values if ignore_index is set,
                # or you might need to filter these samples out before loss calculation.
                # For simplicity here, we assume all will be mapped.

        elif self.classification_mode == "13_class":
            task_labels = labels_0_12 # Use 0-12 labels directly
        else:
            raise ValueError(f"Invalid classification_mode: {self.classification_mode}")
            
        xyz1, cor1 = self._transfer(xyz1)
        xyz2,cor2 = self._transfer(xyz2)
        
        #print(points2.shape)
        #points1 = torch.cat((cor1,points1),dim=-1) if points1 is not None else cor1
        #points2 = torch.cat((cor2,points2),dim=-1) if points2 is not None else cor2
        points1 = cor1
        points2 = cor2
        if torch.any(torch.isnan(xyz1)) or torch.any(torch.isinf(xyz1)):
            print("!!!!!!!!!! WARNING: NaNs or Infs found in xyz1 !!!!!!!!!!")
        elif torch.any(torch.isnan(xyz2)) or torch.any(torch.isinf(xyz2)):
            print("!!!!!!!!!! WARNING: NaNs or Infs found in xyz2 !!!!!!!!!!")
        elif torch.any(torch.isnan(points1)) or torch.any(torch.isinf(points1)):
            print("!!!!!!!!!! WARNING: NaNs or Infs found in xyz2 !!!!!!!!!!")
        elif torch.any(torch.isnan(points2)) or torch.any(torch.isinf(points2)):
            print("!!!!!!!!!! WARNING: NaNs or Infs found in xyz2 !!!!!!!!!!")
        else:
            return xyz1.contiguous(), points1.permute(0,2,1).contiguous(),  xyz2.contiguous(), points2.permute(0,2,1).contiguous(),  task_labels
    
    def _get_train_data_iterator(self):
        # This method is now simpler as the reshuffle logic is in _train_iter_spt
        if self.train_data_loader_iter is None:
            self.train_data_loader_iter = iter(self.train_data_loader)
        return self.train_data_loader_iter
    def _train_iter_spt(self):
        # ... (代码基本不变, 确保使用 self.spt_model) ...
        self.spt_model.train()
        start_time = time.time()
        self.optimizer.zero_grad()
        data_iterator = self._get_train_data_iterator()
        try: 
            batch_dict = next(data_iterator)
        except StopIteration:
        #if self.current_iter%4000==0:
            # [MODIFIED] This is the core logic for reshuffling!
            # The iterator is exhausted, which means one "epoch" is complete.
            logging.info(f"--- End of conceptual epoch {self.current_epoch}. Reshuffling training pairs. ---")
            
            # 1. Increment the epoch counter
            self.current_epoch += 1
            
            # 2. Call the shuffle_pairs method on the dataset with the new epoch number
            # This will generate a new set of random pairs for the next "epoch".
            if self.train_dataset.sampling_strategy == 'random':
                self.train_dataset.shuffle_pairs(epoch=self.current_epoch)
            
            # 3. Create a new iterator from the dataloader.
            # The dataloader will now use the newly shuffled pairs from the dataset.
            self.train_data_loader_iter = iter(self.train_data_loader)
            batch_dict = next(self.train_data_loader_iter)
        #else: batch_dict = next(data_iterator)
        xyz1, points1, xyz2, points2, target_labels = self._process_spt_batch(batch_dict)
        
        # Filter out samples with unmapped labels if any occurred in 3-class mode
        if self.classification_mode == "3_class" and torch.any(target_labels < 0):
            valid_indices = target_labels >= 0
            if not torch.all(valid_indices): # If there are any invalid labels
                xyz1= xyz1[valid_indices]
                xyz2= xyz2[valid_indices]
                if points1 is not None: points1 = points1[valid_indices]
                if points2 is not None: points2 = points2[valid_indices]
                target_labels = target_labels[valid_indices]
                if target_labels.numel() == 0: # If all samples in batch became invalid
                    logging.warning(f"Iter {self.current_iter + 1}: All samples in batch had unmappable labels. Skipping batch.")
                    self.current_iter +=1 # Still count iteration
                    return 0.0, time.time() - start_time

            # --- 在这里开启异常检测 ---
        with torch.autograd.set_detect_anomaly(True):
            logits = self.spt_model((xyz1,points1),(xyz2,points2))
            loss = self.classification_criterion(logits, target_labels)
            if torch.isfinite(loss):
                loss.backward()
                loss = loss / self.accumulation_steps
                self.optimizer.step()
            else:
                logging.warning(f"Iter {self.current_iter + 1}: Non-finite loss: {loss.item()}. Skipping step.")
                loss = torch.tensor(0.0)
            """batch_time = time.time() - start_time
            self.current_iter += 1
            return loss.item(), batch_time"""
            if (self.current_iter + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.spt_model.parameters(), max_norm=1.0)
                self.optimizer.step()       # 更新模型权重
                self.optimizer.zero_grad()  # 清空梯度

        batch_time = time.time() - start_time
        self.current_iter += 1

        # 注意：返回的loss是缩放后的loss，如果你想看原始loss，需要乘以 accumulation_steps
        return loss.item() * self.accumulation_steps, batch_time


    def _validate_spt_iter_standard(self):
        if not self.val_data_loader: return 0.0, 0.0, None
        self.spt_model.eval()
        total_loss, num_batches = 0.0, 0
        all_preds_for_cm, all_labels_for_cm = [], []
        with torch.no_grad():
            for batch_dict in self.val_data_loader:
                xyz1, points1,  xyz2, points2, target_labels = self._process_spt_batch(batch_dict)
                
                # Filter out samples with unmapped labels if any occurred
                if self.classification_mode == "3_class" and torch.any(target_labels < 0):
                    valid_indices = target_labels >= 0
                    if not torch.all(valid_indices):
                        # No need to filter xyz etc. for validation if only preds/labels are used for metrics
                        target_labels = target_labels[valid_indices]
                        # Logits would need to be filtered too if used per sample
                        # For simplicity, we'll proceed but metrics will be on valid_indices only
                        if target_labels.numel() == 0: continue # Skip if no valid samples in batch

                logits = self.spt_model((xyz1,points1),(xyz2,points2))
                
                # If we filtered target_labels, we should filter logits too before loss/preds
                if self.classification_mode == "3_class" and 'valid_indices' in locals() and not torch.all(valid_indices):
                    logits = logits[valid_indices]


                loss = self.classification_criterion(logits, target_labels)
                if torch.isfinite(loss): total_loss += loss.item()
                
                _, predicted_classes = torch.max(logits, 1)
                all_preds_for_cm.append(predicted_classes.cpu())
                all_labels_for_cm.append(target_labels.cpu()) # target_labels are already processed
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        cm_image, spt_accuracy = None, 0.0

        if all_preds_for_cm and all_labels_for_cm:
            all_preds_for_cm_tensor = torch.cat(all_preds_for_cm)
            all_labels_for_cm_tensor = torch.cat(all_labels_for_cm)

            correct = (all_preds_for_cm_tensor == all_labels_for_cm_tensor).sum().item()
            total = all_labels_for_cm_tensor.size(0)
            spt_accuracy = (correct / total) * 100.0 if total > 0 else 0.0
            try:
                # Use self.cm_display_labels_effective for the desired order and labels
                cm = sk_confusion_matrix(all_labels_for_cm_tensor.numpy(), 
                                         all_preds_for_cm_tensor.numpy(), 
                                         labels=self.cm_display_labels_effective)
                # Use self.cm_display_names_effective for plotting
                cm_image = self._plot_confusion_matrix_to_image(cm, self.cm_display_names_effective)
            except Exception as e: logging.error(f"Error generating CM for SPT: {e}")

        if spt_accuracy == 0.0 and total > 0: # If accuracy is zero, print some examples
            logging.warning("Validation accuracy is 0%. Printing some target/prediction pairs:")
            for i in range(min(100, total)): # Print up to 10 examples
                logging.warning(f"  Sample {i}: Target={all_labels_for_cm_tensor[i].item()}, Predicted={all_preds_for_cm_tensor[i].item()}")
        return avg_loss, spt_accuracy, cm_image
    
    def _parse_event_filename(self, full_filename):
        """Helper to parse class label from filename. Copied from dataset for decoupling."""
        name_without_ext = os.path.splitext(full_filename)[0]
        # This regex should match your filename format, e.g., 'point_count-class_id_...'
        # Example: 1024-5_3.npy -> class_id is 5
        match = re.match(r'^(\d+)-(\d+)_(\d+)$', name_without_ext)
        if match:
            return int(match.group(2))
        # Fallback or different regex if needed
        try: # Let's assume a simpler format if the main one fails, e.g. class_id is part of the name
            return int(name_without_ext.split('_')[-1]) # Adjust if your format is different
        except (ValueError, IndexError):
            raise ValueError(f"Filename {full_filename} does not match expected format for class parsing.")

    def _load_and_group_validation_files(self):
        """Loads validation split files and groups source2 by class."""
        data_cfg = self.config.get("finetune_data", self.config.finetune_data)
        s1_event_dir = data_cfg.s1_event_dir

        # 1. Load all file paths from split files
        try:
            with open(data_cfg.val1_split_file, 'r') as f:
                val1_files = [line.strip() for line in f if line.strip()]
            with open(data_cfg.val2_split_file, 'r') as f:
                val2_files = [line.strip() for line in f if line.strip()]
        except FileNotFoundError as e:
            logging.error(f"Validation split file not found: {e}")
            return None, None

        # 2. Group source2 files by their class label
        s2_files_by_class = defaultdict(list)
        for fname in val2_files:
            try:
                class_label = self._parse_event_filename(fname)
                if os.path.exists(os.path.join(s1_event_dir, fname)):
                    s2_files_by_class[class_label].append(fname)
            except ValueError as e:
                logging.warning(f"Skipping file in val2 due to parsing error: {fname} ({e})")
                
        return val1_files, s2_files_by_class

    

    def _plot_confusion_matrix_to_image(self, cm, class_names_ordered): # class_names are now ordered
        figure = plt.figure(figsize=(max(8, len(class_names_ordered)*0.8), max(6, len(class_names_ordered)*0.6))) # Dynamic size
        if sns is not None:
            sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, 
                        xticklabels=class_names_ordered, yticklabels=class_names_ordered, # Use ordered names
                        annot_kws={"size": 8})
        else: # Fallback
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion matrix")
            plt.colorbar()
            tick_marks = np.arange(len(class_names_ordered))
            plt.xticks(tick_marks, class_names_ordered, rotation=45, ha="right")
            plt.yticks(tick_marks, class_names_ordered)
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], 'd'), 
                         horizontalalignment="center", 
                         color="white" if cm[i, j] > thresh else "black", fontsize=8)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=120); plt.close(figure); buf.seek(0)
        image = Image.open(buf)
        if image.mode == 'RGBA': image = image.convert('RGB')
        return image

    def _validate_spt_iter(self):
        if not self.val_data_loader: return 0.0, 0.0, None
        if not self.config.get('validation_voting', {}).get('enabled'):
            # 如果不是投票模式，调用旧的、非投票的验证函数
            return self._validate_spt_iter_standard()

        logging.info("--- Starting HIGH-PERFORMANCE Validation with Ensemble Voting ---")
        
        # 1. 预加载和预处理所有数据
        # 这一步是耗时操作，但只做一次！
        all_xyz, all_points, all_labels, filename_to_idx = self._preload_and_preprocess_validation_data(
            self.val_data_loader, self.cur_device
        )

        # 2. 准备投票任务
        # 获取 source1 的文件列表，并构建 s2 的按类别分组的文件列表
        data_cfg = self.config.finetune_data
        with open(data_cfg.val1_split_file, 'r') as f:
            s1_files = [line.strip() for line in f if line.strip()]
        with open(data_cfg.val2_split_file, 'r') as f:
            s2_files_list = [line.strip() for line in f if line.strip()]

        s2_files_by_class = defaultdict(list)
        for fname in s2_files_list:
            try:
                # (这里的解析逻辑需要和你已有的 _parse_event_filename 保持一致)
                name_without_ext = os.path.splitext(fname)[0]
                match = re.match(r'^(\d+)-(\d+)_(\d+)$', name_without_ext)
                label = int(match.group(2))
                s2_files_by_class[label].append(fname)
            except (ValueError, AttributeError):
                continue

        self.spt_model.eval()
        all_final_predictions = []
        all_true_labels = []
        
        voting_cfg = self.config.validation_voting
        num_votes = voting_cfg.get("num_votes", 50)
        internal_batch_size = voting_cfg.get("batch_size", 32)

        # 3. 外层循环：遍历所有待测的 s1 样本
        for s1_filename in tqdm(s1_files, desc="Executing Voting Logic"):
            if s1_filename not in filename_to_idx:
                logging.warning(f"File {s1_filename} from split file not found in pre-processed data. Skipping.")
                continue
            
            # 4. 获取 s1 的预处理数据和真实标签
            s1_idx = filename_to_idx[s1_filename]
            event1_xyz = all_xyz[s1_idx:s1_idx+1] # Shape [1, P, 3]
            event1_points = all_points[s1_idx:s1_idx+1] # Shape [1, F, P]
            s1_original_label = all_labels[s1_idx].item()
            
            # 将原始标签转换为任务标签 (0-12 or 0-2)
            base_label = (s1_original_label - 1) % 13
            if self.classification_mode == '13_class':
                true_task_label = base_label
            else: # 3-class
                true_task_label = self.map_13_to_3_class.get(base_label, -1)
            
            if true_task_label == -1: continue
            all_true_labels.append(true_task_label)

            # 5. 采样投票者
            target_s2_class = s1_original_label + 13 

            
            candidate_voters = s2_files_by_class.get(target_s2_class, [])
            if not candidate_voters:
                all_final_predictions.append(-1) # 标记为预测失败
                continue

            num_to_sample = min(num_votes, len(candidate_voters))
            voter_filenames = random.sample(candidate_voters, num_to_sample)
            voter_indices = [filename_to_idx[fname] for fname in voter_filenames if fname in filename_to_idx]

            if not voter_indices:
                all_final_predictions.append(-1)
                continue
                
            # 6. 获取所有投票者的预处理数据
            voters_xyz = all_xyz[voter_indices].to(self.cur_device)
            voters_points = all_points[voter_indices].to(self.cur_device)

            # 7. 高效批处理预测
            individual_predictions = []
            with torch.no_grad():
                event1_xyz_gpu = event1_xyz.to(self.cur_device)
                event1_points_gpu = event1_points.to(self.cur_device)

                for i in range(0, len(voter_indices), internal_batch_size):
                    v_xyz_b = voters_xyz[i:i+internal_batch_size]
                    v_pts_b = voters_points[i:i+internal_batch_size]
                    current_b_size = v_xyz_b.shape[0]

                    e1_xyz_b = event1_xyz_gpu.expand(current_b_size, -1, -1).contiguous()
                    e1_pts_b = event1_points_gpu.expand(current_b_size, -1, -1).contiguous()
                    
                    logits = self.spt_model((e1_xyz_b, e1_pts_b), (v_xyz_b, v_pts_b))
                    preds = torch.argmax(logits, dim=1).cpu().tolist()
                    individual_predictions.extend(preds)
            
            # 8. 投票
            final_prediction = sp_mode(individual_predictions, keepdims=False)[0]
            all_final_predictions.append(final_prediction)

        # 9. 计算最终指标 (与之前相同)
        valid_indices = [i for i, p in enumerate(all_final_predictions) if p != -1]
        if not valid_indices: return 0.0, 0.0, None

        final_preds_np = np.array(all_final_predictions)[valid_indices]
        true_labels_np = np.array(all_true_labels)[valid_indices]

        accuracy = (final_preds_np == true_labels_np).mean() * 100.0
        cm = sk_confusion_matrix(true_labels_np, final_preds_np, labels=self.cm_display_labels_effective)
        cm_image = self._plot_confusion_matrix_to_image(cm, self.cm_display_names_effective)
        
        logging.info(f"High-performance voting complete. Accuracy: {accuracy:.2f}%")
        return 0.0, accuracy, cm_image

    

    # ... (train_loop_spt, _save_spt_checkpoint, _load_spt_checkpoint, _reset_spt_state unchanged from previous full version)
    def train_loop_spt(self):
        logging.info(f"Starting SPT fine-tuning from iter {self.current_iter + 1}, target: {self.max_iter}")
        if self.val_data_loader and (self.current_iter == 0 or (self.val_freq > 0 and self.current_iter % self.val_freq != 0 and self.current_iter > 0) or self.val_freq == 0):
             if self.current_iter == 0 or self.val_freq == 0 or (self.val_freq > 0 and self.current_iter > 0):
                avg_loss, acc, cm_img = self._validate_spt_iter()
                logging.info(f"Initial/Resumed SPT Val (Iter {self.current_iter}): Loss: {avg_loss:.4f}, ACC: {acc:.2f}%")
                if self.writer:
                    self.writer.add_scalar('Loss/SPT_validation', avg_loss, self.current_iter)
                    self.writer.add_scalar('Accuracy/SPT_validation', acc, self.current_iter)
                    if cm_img: self.writer.add_image('ConfusionMatrix/SPT_validation', np.array(cm_img), self.current_iter, dataformats='HWC')
                if acc > self.best_spt_accuracy: self.best_spt_accuracy = acc
                if avg_loss < self.best_spt_loss: self.best_spt_loss = avg_loss
        while self.current_iter < self.max_iter:
            loss_item, batch_time = self._train_iter_spt()
            if self.current_iter % self.log_freq == 0:
                lr = self.scheduler.get_last_lr()[0]
                logging.info(f"Iter: {self.current_iter}/{self.max_iter}, SPT Train Loss: {loss_item:.4f}, LR: {lr:.6f}, Batch Time: {batch_time:.3f}s")
                if self.writer:
                    self.writer.add_scalar('Loss/SPT_train', loss_item, self.current_iter)
                    self.writer.add_scalar('LearningRate/SPT', lr, self.current_iter)
                    self.writer.add_scalar('Time/SPT_batch_time_seconds', batch_time, self.current_iter)
            if self.val_data_loader and self.current_iter > 0 and self.val_freq > 0 and self.current_iter % self.val_freq == 0:
                avg_val_loss, spt_accuracy, cm_image = self._validate_spt_iter()
                logging.info(f"Iter: {self.current_iter}/{self.max_iter}, SPT Val Loss: {avg_val_loss:.4f}, SPT Val ACC: {spt_accuracy:.2f}%")
                if self.writer:
                    self.writer.add_scalar('Loss/SPT_validation', avg_val_loss, self.current_iter)
                    self.writer.add_scalar('Accuracy/SPT_validation', spt_accuracy, self.current_iter)
                    if cm_image: self.writer.add_image('ConfusionMatrix/SPT_validation', np.array(cm_image), self.current_iter, dataformats='HWC')
                is_best = spt_accuracy > self.best_spt_accuracy
                if is_best: self.best_spt_accuracy = spt_accuracy; logging.info(f"New best SPT accuracy: {self.best_spt_accuracy:.2f}%")
                if avg_val_loss < self.best_spt_loss: self.best_spt_loss = avg_val_loss
                self._save_spt_checkpoint(is_best_model=is_best)
            self.scheduler.step()
        if self.val_data_loader:
            avg_val_loss_final, spt_accuracy_final, cm_image_final = self._validate_spt_iter()
            logging.info(f"Final SPT Val (Iter {self.current_iter}): Loss: {avg_val_loss_final:.4f}, ACC: {spt_accuracy_final:.2f}%")
            if self.writer:
                self.writer.add_scalar('Loss/SPT_validation', avg_val_loss_final, self.current_iter)
                self.writer.add_scalar('Accuracy/SPT_validation', spt_accuracy_final, self.current_iter)
                if cm_image_final: self.writer.add_image('ConfusionMatrix/SPT_validation', np.array(cm_image_final), self.current_iter, dataformats='HWC')
            is_best_final = spt_accuracy_final > self.best_spt_accuracy
            if is_best_final: self.best_spt_accuracy = spt_accuracy_final
            self._save_spt_checkpoint(is_best_model=is_best_final, filename_prefix="final_")
        else: self._save_spt_checkpoint(is_best_model=False, filename_prefix="final_")
        if self.writer: self.writer.close()
        logging.info("SPT Fine-tuning completed!")
    
    def train_loop_STNet(self):
        logging.info(f"Starting SPT fine-tuning from iter {self.current_iter + 1}, target: {self.max_iter}")
        
        def run_validation(iteration):
            if self.val_data_loader:
                logging.info(f"--- Running validation at iteration {iteration} ---")
                

                avg_loss, acc, cm_img = self._validate_spt_iter()

                logging.info(f"Validation (Iter {iteration}): Loss: {avg_loss:.4f}, ACC: {acc:.2f}%")
                if self.writer:
                    self.writer.add_scalar('Loss/SPT_validation', avg_loss, iteration)
                    self.writer.add_scalar('Accuracy/SPT_validation', acc, iteration)
                    if cm_img: self.writer.add_image('ConfusionMatrix/SPT_validation', np.array(cm_img), iteration, dataformats='HWC')
                
                is_best = acc > self.best_spt_accuracy
                if is_best:
                    self.best_spt_accuracy = acc
                    logging.info(f"New best SPT accuracy: {self.best_spt_accuracy:.2f}%")
                
                if avg_loss < self.best_spt_loss and not self.config.get('validation_voting', {}).get('enabled'):
                    self.best_spt_loss = avg_loss

                return is_best
            return False # No validation performed
         # Initial validation run
        if self.current_iter == 0:
            run_validation(self.current_iter)

        while self.current_iter < self.max_iter:
            loss_item, batch_time = self._train_iter_spt()
            
            if self.current_iter % self.log_freq == 0:
                lr = self.scheduler.get_last_lr()[0]
                logging.info(f"Iter: {self.current_iter}/{self.max_iter}, SPT Train Loss: {loss_item:.4f}, LR: {lr:.6f}, Batch Time: {batch_time:.3f}s")
                if self.writer:
                    # ... (logging to writer remains the same)
                    self.writer.add_scalar('Loss/SPT_train', loss_item, self.current_iter)
                    self.writer.add_scalar('LearningRate/SPT', lr, self.current_iter)
                    self.writer.add_scalar('Time/SPT_batch_time_seconds', batch_time, self.current_iter)

            # Periodic validation run
            if self.val_freq > 0 and self.current_iter > 0 and self.current_iter % self.val_freq == 0:
                is_best_model = run_validation(self.current_iter)
                self._save_spt_checkpoint(is_best_model=is_best_model)
            
            self.scheduler.step()

        # Final validation run
        is_best_final = run_validation(self.current_iter)
        self._save_spt_checkpoint(is_best_model=is_best_final, filename_prefix="final_")
        
        if self.writer: self.writer.close()
        logging.info("SPT Fine-tuning completed!")

    def _save_spt_checkpoint(self, is_best_model, filename_prefix=""):
        state = {'iter': self.current_iter, 'spt_model_state_dict': self.spt_model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.scheduler.state_dict(),
                 'best_spt_accuracy': self.best_spt_accuracy, 'best_spt_loss': self.best_spt_loss}
        latest_pth = os.path.join(self.checkpoint_dir, f'{filename_prefix}spt_latest.pth')
        latest_cfg = os.path.join(self.checkpoint_dir, f'{filename_prefix}spt_latest_config.yaml')
        logging.info(f"Saving SPT latest checkpoint to {latest_pth} (Iter: {self.current_iter})")
        torch.save(state, latest_pth)
        try: OmegaConf.save(self.config, latest_cfg)
        except Exception as e: logging.error(f"Could not save SPT cfg {latest_cfg}: {e}")
        if is_best_model:
            best_pth = os.path.join(self.checkpoint_dir, f'{filename_prefix}spt_best_acc.pth')
            best_cfg = os.path.join(self.checkpoint_dir, f'{filename_prefix}spt_best_acc_config.yaml')
            logging.info(f"Saving new best SPT checkpoint to {best_pth}")
            try: shutil.copyfile(latest_pth, best_pth)
            except Exception as e: logging.error(f"Could not copy {latest_pth} to {best_pth}: {e}")
            try: OmegaConf.save(self.config, best_cfg) # Or copy latest_cfg
            except Exception as e: logging.error(f"Could not save SPT best cfg {best_cfg}: {e}")

    def _load_spt_checkpoint(self):
        ckpt_to_load = None
        resume_path_cfg = self.config.training.get("resume_from_spt_checkpoint")
        if resume_path_cfg:
            paths_to_try = [os.path.join(self.checkpoint_dir, resume_path_cfg), resume_path_cfg]
            for p in paths_to_try:
                if os.path.isfile(p): ckpt_to_load = p; break
            if ckpt_to_load: logging.info(f"Attempting to resume SPT from specified: {ckpt_to_load}")
            else: logging.warning(f"Specified SPT checkpoint '{resume_path_cfg}' DNE. Will try spt_latest.pth.")
        if ckpt_to_load is None:
            default_latest = os.path.join(self.checkpoint_dir, 'spt_latest.pth')
            if os.path.exists(default_latest): ckpt_to_load = default_latest; logging.info(f"Attempting to resume SPT from default: {ckpt_to_load}")
        if ckpt_to_load and os.path.exists(ckpt_to_load):
            try:
                logging.info(f"Loading SPT checkpoint from {ckpt_to_load}")
                ckpt = torch.load(ckpt_to_load, map_location=self.cur_device)
                self.spt_model.load_state_dict(ckpt['spt_model_state_dict'])
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                self.current_iter = ckpt.get('iter', 0)
                self.best_spt_accuracy = ckpt.get('best_spt_accuracy', -1.0)
                self.best_spt_loss = ckpt.get('best_spt_loss', float('inf'))
                logging.info(f"Resumed SPT from iter {self.current_iter}. Best SPT ACC: {self.best_spt_accuracy:.2f}%.")
            except Exception as e: logging.error(f"Error loading SPT ckpt {ckpt_to_load}: {e}. Starting SPT fresh."); self._reset_spt_state()
        else: logging.info("No SPT ckpt found. Starting SPT fresh (backbone might be pretrained)."); self._reset_spt_state()

    def _reset_spt_state(self):
        self.current_iter = 0; self.best_spt_accuracy = -1.0; self.best_spt_loss = float('inf')


# --- Configuration for Fine-tuning ---
def get_finetune_config():
    # Using your provided pretrain config as a base, and adding/modifying for finetuning
    



    config_dict = {
        "gpu_id": 0,
        "finetune_data": { # Data specific to this fine-tuning task
            "s1_event_dir": "contra_data_clean_noipad/data_view1",
            "s2_event_dir": "contra_data_clean_noipad/data_view1",
            "train1_split_file": "contra_data_clean_noipad/fine_train_Ru copy.txt",
            "train2_split_file": "contra_data_clean_noipad/fine_train_Zr copy.txt",
            "val1_split_file": "contra_data_clean_noipad/fine_test_Ru copy 2.txt", # Added from your prev request
            "val2_split_file": "contra_data_clean_noipad/fine_test_Zr copy 2.txt", # Added from your prev request
            "use_shared_mask_from_s1": False,
            "batch_size": 32,
            "num_workers": 0, # Start with 0 for debugging
        },
        # Config for SPT model instance to be created by create_model
        "spt_model_params": {
            'model_name' : 'STNet_Point', # Tells create_model to make an SPT
            # Backbone (P_by_P_feature part) parameters - MUST MATCH PRETRAINED ARCHITECTURE
            'coord_dim': 3,
            'input_feature_dim': 3,     # For P_by_P_feature, if it takes separate `points` features from dataset.
                                        # If your PairedEventContrastiveDatasetV3's 'event1'/'event2'
                                        # already contains all features and you split xyz/points in _process_spt_batch,
                                        # this might be less relevant here but crucial for P_by_P_feature's own init.
            'output_dim': 512,           # Final projection dim if P_by_P_feature had one (otherwise transformer_dim)
            # SPT specific parts (fusion, classifier head) - These are passed to SPT's __init__ by create_model
            # The 'num_class' here will be dynamically set by FineTuningTrainer based on classification_mode
            'num_class':  4, # 13 OR This will be overridden by FineTuningTrainer
            'feature_out': False,       # SPT should output logits

            # The create_model for 'SPT' needs to handle how these top-level SPT params
            # relate to configuring its internal P_by_P_feature and its fusion/classifier.
            # E.g., it might pass a subset of these to an internal P_by_P_feature instantiation.
        },
        # SPT downstream specific parameters (used by FineTuningTrainer to configure SPT parts if not in spt_model_params directly)
        "spt_downstream": {
            "classification_mode": "13_class",  # "13_class" or "3_class"
        },
        "optimizer": { "name": "AdamW", "lr": 1.0e-05, "weight_decay": 0.01 }, # LR for fine-tuning
        "finetune_scheduler": { # Scheduler for fine-tuning
            "name": "CosineAnnealingLR", "T_max": 3000000, "eta_min": 1e-7
        },
        "training": {
            "max_iter": 3000000, "log_freq": 100, "val_freq": 10000,
            "output_dir": "./STNet_novote_v2", # CHANGE THIS FOR DIFFERENT RUNS
            "clean_log_dir_on_start": True,
            "pretrained_backbone_checkpoint": None,#"STNet_novote_v1/checkpoints/spt_latest.pth",#"STNet_3SA_v1_with_position_embedding_global_attention_4class/checkpoints/spt_best_acc.pth",#"STNet_all_afterdebug_v5_3abstraction_4class/checkpoints/spt_latest.pth",#"STNet_test_v8_3abstraction_4class/checkpoints/123latest.pth", # CRITICAL: Set actual path
            "resume_from_spt_checkpoint":None, # e.g., "spt_latest.pth" to resume finetuning
            "freeze_backbone_on_load": False, 
        },
        # --- 新增：投票验证配置 ---
        "validation_voting": {
            "enabled": False,  # 设置为 True 来启用投票验证
            "num_votes": 30,  # 每个 source1 样本与多少个 source2 样本配对进行投票
            "batch_size": 16, # 投票时内部处理的批次大小，以防显存不足
            "mega_batch_size": 1 
        }
    }
    return OmegaConf.create(config_dict)

# --- Main Execution ---
if __name__ == "__main__":
    # ... (logging setup and seed remains the same as previous main) ...
    try: ft_config = get_finetune_config()
    except Exception as e: print(f"FATAL: Error loading ft-config: {e}"); exit(1)
    os.makedirs(ft_config.training.output_dir, exist_ok=True)
    log_ft_path = os.path.join(ft_config.training.output_dir, "finetuning_spt_run.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_ft_path)])
    logging.info("Using Fine-tuning Configuration:\n%s", OmegaConf.to_yaml(ft_config))
    seed = 3407; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    try:
        ckpt_path = ft_config.training.pretrained_backbone_checkpoint
        if not ckpt_path or not os.path.exists(ckpt_path):
            logging.warning(f"Pretrained backbone ckpt path invalid: {ckpt_path}. Backbone random.")
            # Consider exiting if backbone is essential: exit(1)
        trainer = FineTuningTrainer(ft_config, pretrained_checkpoint_path=ckpt_path)
        trainer.train_loop_STNet()
    except Exception as e: logging.critical("Unhandled exception during SPT fine-tuning:", exc_info=True)