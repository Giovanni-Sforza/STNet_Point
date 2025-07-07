# pretrain_moco.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import random
import logging
from omegaconf import OmegaConf
import shutil
import time
from tqdm import tqdm
import copy

# 假设你的模型创建函数和骨干网络在这里
# IMPORTANT: create_model as used here should return a SINGLE-STREAM encoder.
from model.STNet_point2 import create_model

# --- 1. 数据增强和数据集 ---

# 这个类现在是可选的，因为数据已经预增强。
class PointCloudAugmentation:
    def __init__(self, rotate_axis='z', rotation_angle=180.0, scale_range=(0.85, 1.15), jitter_sigma=0.01, jitter_clip=0.05):
        self.rotate_axis = rotate_axis
        self.rotation_angle = rotation_angle
        self.scale_range = scale_range
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip
        logging.info("Initialized PointCloudAugmentation.")

    def __call__(self, points):
        """
        Apply augmentations to a point cloud tensor.
        Args:
            points (torch.Tensor): A tensor of shape (N, 3) representing the point cloud.
        Returns:
            torch.Tensor: The augmented point cloud.
        """
        # --- 随机旋转 ---
        angle = np.random.uniform(-self.rotation_angle, self.rotation_angle) * (np.pi / 180.0)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        if self.rotate_axis.lower() == 'z':
            rotation_matrix = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=points.dtype, device=points.device)
        elif self.rotate_axis.lower() == 'y':
            rotation_matrix = torch.tensor([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=points.dtype, device=points.device)
        else: # x-axis
             rotation_matrix = torch.tensor([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]], dtype=points.dtype, device=points.device)
        
        rotated_points = points @ rotation_matrix.T

        # --- 随机缩放 ---
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        scaled_points = rotated_points * scale

        # --- 随机抖动 (Jitter) ---
        jitter = torch.randn_like(scaled_points) * self.jitter_sigma
        jitter = torch.clamp(jitter, -self.jitter_clip, self.jitter_clip)
        jittered_points = scaled_points + jitter

        return jittered_points

### MODIFIED ###
# 新的数据集，用于加载预先生成的 query-key 对
class PairedContrastivePointCloudDataset(Dataset):
    """
    加载已预先生成并存储在两个不同文件列表中的对比学习对 (query, key)。
    """
    def __init__(self, root_dir_q, root_dir_k, split_file_q, split_file_k):
        self.root_dir_q = root_dir_q
        self.root_dir_k = root_dir_k

        try:
            with open(split_file_q, 'r') as f:
                self.file_list_q = [line.strip() for line in f if line.strip()]
            logging.info(f"Loaded {len(self.file_list_q)} query files from {split_file_q}")

            with open(split_file_k, 'r') as f:
                self.file_list_k = [line.strip() for line in f if line.strip()]
            logging.info(f"Loaded {len(self.file_list_k)} key files from {split_file_k}")
        except FileNotFoundError as e:
            logging.error(f"Split file not found: {e}")
            raise

        # 关键断言：确保两个列表的长度一致，它们是一一对应的正样本对
        assert len(self.file_list_q) == len(self.file_list_k), \
            f"Query file list ({len(self.file_list_q)}) and Key file list ({len(self.file_list_k)}) must have the same length."

    def __len__(self):
        return len(self.file_list_q)

    def __getitem__(self, idx):
        # 加载 query 视图
        file_path_q = os.path.join(self.root_dir_q, self.file_list_q[idx])
        point_cloud_q = np.load(file_path_q)
        view_q = torch.from_numpy(point_cloud_q).float()

        # 加载 key 视图
        file_path_k = os.path.join(self.root_dir_k, self.file_list_k[idx])
        point_cloud_k = np.load(file_path_k)
        view_k = torch.from_numpy(point_cloud_k).float()
        
        return view_q, view_k


# --- 2. MoCo 训练器 ---
class MoCoTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")
        logging.info(f"MoCoTrainer: Using device: {self.device}")

        # --- MoCo specific parameters --- (不变)
        self.moco_m = config.moco.m
        self.moco_k = config.moco.k
        self.moco_t = config.moco.t
        self.feature_dim = config.model_params.output_dim

        # --- 模型初始化 --- (不变)
        logging.info("Creating Query Encoder (encoder_q)...")
        self.encoder_q = create_model(config.model_params).to(self.device)
        
        logging.info("Creating Key Encoder (encoder_k)...")
        self.encoder_k = create_model(config.model_params).to(self.device)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # --- 创建队列 --- (不变)
        self.register_buffer('queue', F.normalize(torch.randn(self.feature_dim, self.moco_k), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # --- 优化器和调度器 --- (不变)
        self.optimizer = getattr(optim, config.optimizer.name)(
            self.encoder_q.parameters(), lr=config.optimizer.lr,
            weight_decay=config.optimizer.get('weight_decay', 0)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.training.max_iter,
            eta_min=config.scheduler.get("eta_min", 0)
        )
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # --- 目录和日志 --- (不变)
        self.output_dir = config.training.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_dir = os.path.join(self.output_dir, 'logs_moco')
        self.writer = SummaryWriter(log_dir=self.log_dir)

        ### MODIFIED ###
        # --- 数据加载 ---
        data_cfg = config.data
        # 不再需要现场增强
        # augmentation = PointCloudAugmentation() 
        train_dataset = PairedContrastivePointCloudDataset(
            root_dir_q=data_cfg.q_data_dir,
            root_dir_k=data_cfg.k_data_dir,
            split_file_q=data_cfg.q_split_file,
            split_file_k=data_cfg.k_split_file
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=data_cfg.batch_size, shuffle=True,
            num_workers=data_cfg.num_workers, pin_memory=True, drop_last=True
        )
        self.train_iter = iter(self.train_loader)

        # --- 训练状态 --- (不变)
        self.current_iter = 0
        self.best_loss = float('inf')
        self._load_checkpoint()

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
        return a_cartesian_points,cor[:,:, 0:3].permute(0,2,1).contiguous()

    # 所有 MoCo 核心方法 (_momentum_update_key_encoder, _dequeue_and_enqueue, _train_step, train_loop, _save_checkpoint, _load_checkpoint)
    # 的内部逻辑都保持不变，因为它们处理的是模型和特征，与数据源的加载方式解耦。
    # 这里为了简洁，我将省略这些方法的代码，它们与上一个回答中的完全相同。
    def register_buffer(self, name, tensor):
        setattr(self, name, tensor.to(self.device))
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]; ptr = int(self.queue_ptr)
        assert self.moco_k % batch_size == 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_ptr[0] = (ptr + batch_size) % self.moco_k

    # In MoCoTrainer class

    def _train_step(self):
        self.encoder_q.train()
        start_time = time.time()
        
        try:
            im_q, im_k = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            im_q, im_k = next(self.train_iter)

        im_q, im_k = im_q.to(self.device), im_k.to(self.device)
        xyz1, cor1 = self._transfer(im_q)
        xyz2,cor2 = self._transfer(im_k)
        xyz1,xyz2,cor1,cor2 = xyz1.to(self.device),xyz2.to(self.device),cor1.to(self.device),cor2.to(self.device)


        # 计算 query 特征
        q = self.encoder_q((xyz1, cor1)) # 假设输出是 (B, C, 1) 或 (B, 1, C)
        
        # --- 关键修改 1 ---
        # 展平/压缩特征，使其成为 (B, C)
        q = q.view(q.shape[0], -1) 
        q = F.normalize(q, dim=1)

        # 计算 key 特征 (无梯度)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k((xyz2,cor2)) # 假设输出是 (B, C, 1) 或 (B, 1, C)
            
            # --- 关键修改 2 ---
            # 同样展平/压缩 key 特征
            k = k.view(k.shape[0], -1)
            k = F.normalize(k, dim=1)

        # 现在 q 和 k 的形状都是 (B, C)，einsum 可以正常工作
        # --- 计算对比损失 ---
        # 正样本 logits: (B, 1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # 负样本 logits: (B, K)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # 组合 logits: (B, 1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.moco_t

        # 标签
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        loss = self.criterion(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新队列
        self._dequeue_and_enqueue(k)
        
        batch_time = time.time() - start_time
        self.current_iter += 1
        return loss.item(), batch_time
    def train_loop(self):
        logging.info(f"Starting MoCo pre-training from iter {self.current_iter + 1}, target: {self.config.training.max_iter}")
        while self.current_iter < self.config.training.max_iter:
            loss, batch_time = self._train_step()
            if self.current_iter % self.config.training.log_freq == 0:
                lr = self.scheduler.get_last_lr()[0]
                logging.info(f"Iter: {self.current_iter}/{self.config.training.max_iter}, Loss: {loss:.4f}, LR: {lr:.6f}, Time: {batch_time:.3f}s")
                self.writer.add_scalar('Loss/train', loss, self.current_iter)
                self.writer.add_scalar('LearningRate', lr, self.current_iter)
                if loss < self.best_loss: self.best_loss = loss; logging.info(f"New best loss: {self.best_loss:.4f}. Saving best model."); self._save_checkpoint(is_best=True)
            if self.current_iter % self.config.training.save_freq == 0: self._save_checkpoint(is_best=False)
            self.scheduler.step()
        self._save_checkpoint(is_best=False, filename_prefix="final_"); self.writer.close(); logging.info("MoCo pre-training completed!")
    def _save_checkpoint(self, is_best, filename_prefix=""):
        state = {'iter': self.current_iter, 'encoder_q_state_dict': self.encoder_q.state_dict(), 'encoder_k_state_dict': self.encoder_k.state_dict(),'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.scheduler.state_dict(),'queue': self.queue, 'queue_ptr': self.queue_ptr, 'best_loss': self.best_loss, 'config': self.config}
        latest_path = os.path.join(self.checkpoint_dir, f'{filename_prefix}moco_latest.pth'); torch.save(state, latest_path); logging.info(f"Saved checkpoint to {latest_path}")
        if is_best: best_path = os.path.join(self.checkpoint_dir, 'moco_best_loss.pth'); shutil.copyfile(latest_path, best_path); logging.info(f"Copied best model to {best_path}")
    def _load_checkpoint(self):
        resume_path = self.config.training.get("resume_from_checkpoint")
        if not resume_path: return
        path_to_load = os.path.join(self.checkpoint_dir, resume_path) if not os.path.isabs(resume_path) else resume_path
        if not os.path.isfile(path_to_load): logging.warning(f"Resume checkpoint not found: {path_to_load}. Starting fresh."); return
        logging.info(f"Loading checkpoint from {path_to_load}")
        ckpt = torch.load(path_to_load, map_location=self.device)
        self.encoder_q.load_state_dict(ckpt['encoder_q_state_dict']); self.encoder_k.load_state_dict(ckpt['encoder_k_state_dict']); self.optimizer.load_state_dict(ckpt['optimizer_state_dict']); self.scheduler.load_state_dict(ckpt['scheduler_state_dict']); self.queue = ckpt['queue'].to(self.device); self.queue_ptr = ckpt['queue_ptr'].to(self.device); self.current_iter = ckpt['iter']; self.best_loss = ckpt.get('best_loss', float('inf')); logging.info(f"Resumed from iteration {self.current_iter}. Best loss: {self.best_loss:.4f}")


# --- 3. 配置和主函数 ---
### MODIFIED ###
def get_moco_config():
    config_dict = {
        "gpu_id": 0,
        "data": {
            # 假设q和k视图文件在同一个根目录下
            "q_data_dir": "contra_data_clean_noipad/data_view1",
            "k_data_dir": "contra_data_clean_noipad/data_view2",
            # 指向两个不同的文件列表
            "q_split_file": "contra_data_clean_noipad/pretrain_all1.txt", 
            "k_split_file": "contra_data_clean_noipad/pretrain_all2.txt", 
            "batch_size": 128,
            "num_workers": 0,
        },
        # 模型参数: 保持不变
        "model_params": {
            'model_name' : 'STNet_Point',
            'coord_dim': 3,
            'input_feature_dim': 3,
            'output_dim': 512, 
            "pretrain": True
        },
        # MoCo 参数: 保持不变
        "moco": {
            "m": 0.999,
            "k": 65536,
            "t": 0.07,
        },
        "optimizer": { "name": "AdamW", "lr": 1.0e-4, "weight_decay": 0.01 },
        "scheduler": { "name": "CosineAnnealingLR", "eta_min": 1e-7 },
        "training": {
            "max_iter": 200000,
            "log_freq": 100,
            "save_freq": 5000,
            "output_dir": "./moco_pretrain_v2",
            "resume_from_checkpoint": None, 
        },
    }
    return OmegaConf.create(config_dict)

# 主函数 `if __name__ == "__main__":` 保持不变
if __name__ == "__main__":
    config = get_moco_config()
    os.makedirs(config.training.output_dir, exist_ok=True)
    log_path = os.path.join(config.training.output_dir, "moco_pretrain_run.log")
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)])
    
    logging.info("Using MoCo Pre-training Configuration:\n%s", OmegaConf.to_yaml(config))
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        trainer = MoCoTrainer(config)
        trainer.train_loop()
    except Exception as e:
        logging.critical("Unhandled exception during MoCo training:", exc_info=True)