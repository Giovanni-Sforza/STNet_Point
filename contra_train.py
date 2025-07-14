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
from collections import deque # ### NEW ### 用于高效地存储历史loss

# 假设你的模型创建函数和骨干网络在这里
from model.STNet_point2 import create_model

# --- 1. 数据增强和数据集 ---

### MODIFIED ###
# 增强类现在支持动态调整难度
class CylindricalPointCloudAugmentation:
    def __init__(self, phi_rotation_range_deg, r_scale_range, eta_shift_range, jitter_params, adaptive_steps=None):
        """
        在 (r, phi, eta) 坐标系中对点云进行增强, 支持自适应难度调整。
        """
        # 存储基础参数
        self.base_phi_range_deg = list(phi_rotation_range_deg)
        self.base_r_scale_range = list(r_scale_range)
        self.base_eta_shift_range = list(eta_shift_range)
        self.base_jitter_sigma = jitter_params.get('sigma', 0.01)
        self.base_jitter_clip = jitter_params.get('clip', 0.05)
        
        # ### NEW ### 存储自适应步长
        self.adaptive_steps = adaptive_steps if adaptive_steps else {}
        self.r_scale_step = self.adaptive_steps.get('r_scale_step', 0.02)
        self.eta_shift_step = self.adaptive_steps.get('eta_shift_step', 0.05)
        self.jitter_sigma_step = self.adaptive_steps.get('jitter_sigma_step', 0.002)

        # 初始化当前参数
        self.current_phi_range_deg = self.base_phi_range_deg
        self.current_r_scale_range = self.base_r_scale_range
        self.current_eta_shift_range = self.base_eta_shift_range
        self.current_jitter_sigma = self.base_jitter_sigma
        
        logging.info("Initialized CylindricalPointCloudAugmentation with ADAPTIVE capability.")
        self.update_difficulty(0) # 使用初始难度0进行初始化日志记录

    ### NEW ###
    def update_difficulty(self, level: int):
        """根据难度等级更新增强参数。"""
        # 更新 r 缩放范围
        self.current_r_scale_range = [
            self.base_r_scale_range[0] - level * self.r_scale_step,
            self.base_r_scale_range[1] + level * self.r_scale_step
        ]
        # 更新 eta 平移范围
        self.current_eta_shift_range = [
            self.base_eta_shift_range[0] - level * self.eta_shift_step,
            self.base_eta_shift_range[1] + level * self.eta_shift_step
        ]
        # 更新 jitter sigma
        self.current_jitter_sigma = self.base_jitter_sigma + level * self.jitter_sigma_step

        logging.info(f"[Adaptive Aug] Difficulty level set to {level}. New params: "
                     f"r_scale={self.current_r_scale_range}, "
                     f"eta_shift={self.current_eta_shift_range}, "
                     f"jitter_sigma={self.current_jitter_sigma:.4f}")

    def __call__(self, points):
        """使用当前难度参数对点云进行增强。"""
        # 使用 self.current_... 参数进行增强
        current_phi_range_rad = (np.deg2rad(self.current_phi_range_deg[0]), np.deg2rad(self.current_phi_range_deg[1]))
        delta_phi = np.random.uniform(current_phi_range_rad[0], current_phi_range_rad[1])
        scale_r = np.random.uniform(self.current_r_scale_range[0], self.current_r_scale_range[1])
        shift_eta = np.random.uniform(self.current_eta_shift_range[0], self.current_eta_shift_range[1])

        r, phi, eta = points[..., 0], points[..., 1], points[..., 2]
        phi_aug = phi + delta_phi
        r_aug = r * scale_r
        eta_aug = eta + shift_eta
        phi_aug = (phi_aug + np.pi) % (2 * np.pi) - np.pi
        
        jitter = torch.randn_like(points[..., 0:3]) * self.current_jitter_sigma
        jitter = torch.clamp(jitter, -self.base_jitter_clip, self.base_jitter_clip)
        
        augmented_coords = torch.stack([r_aug, phi_aug, eta_aug], dim=-1)
        jittered_points = augmented_coords + jitter
        return jittered_points

# 数据集类保持不变
class ContrastivePointCloudDataset(Dataset):
    def __init__(self, root_dir, split_file):
        self.root_dir = root_dir
        try:
            with open(split_file, 'r') as f:
                self.file_list = [line.strip() for line in f if line.strip()]
            logging.info(f"Loaded {len(self.file_list)} files from {split_file}")
        except FileNotFoundError as e:
            logging.error(f"Split file not found: {e}")
            raise
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        point_cloud = np.load(file_path)
        return torch.from_numpy(point_cloud).float()


# --- 2. MoCo 训练器 ---
class MoCoTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")
        # ... (模型、优化器等初始化不变) ...
        self.moco_m, self.moco_k, self.moco_t = config.moco.m, config.moco.k, config.moco.t
        self.feature_dim = config.model_params.output_dim
        self.encoder_q = create_model(config.model_params).to(self.device)
        self.encoder_k = create_model(config.model_params).to(self.device)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer('queue', F.normalize(torch.randn(self.feature_dim, self.moco_k), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.optimizer = getattr(optim, config.optimizer.name)(self.encoder_q.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.get('weight_decay', 0))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.training.max_iter, eta_min=config.scheduler.get("eta_min", 0))
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.output_dir = config.training.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_dir = os.path.join(self.output_dir, 'logs_moco')
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # --- 数据加载和增强 ---
        data_cfg = config.data
        aug_cfg = config.augmentation 
        
        # ### MODIFIED ###
        # 1. 初始化增强器时传入自适应步长配置
        self.augmentation = CylindricalPointCloudAugmentation(
            **aug_cfg.base_params, 
            adaptive_steps=aug_cfg.get('adaptive_steps')
        )
        
        # 2. 初始化数据集和加载器 (不变)
        train_dataset = ContrastivePointCloudDataset(root_dir=data_cfg.root_dir, split_file=data_cfg.split_file)
        self.train_loader = DataLoader(train_dataset, batch_size=data_cfg.batch_size, shuffle=True, num_workers=data_cfg.num_workers, pin_memory=True, drop_last=True)
        self.train_iter = iter(self.train_loader)

        # ### NEW ### 自适应增强状态初始化
        self.adaptive_cfg = config.get('adaptive_augmentation', {})
        if self.adaptive_cfg.get('enabled', False):
            logging.info("Adaptive augmentation is ENABLED.")
            self.loss_history = deque(maxlen=self.adaptive_cfg.history_len)
            self.difficulty_level = 0
            self.last_update_iter = 0
        else:
            logging.info("Adaptive augmentation is DISABLED.")

        # --- 训练状态 --- (不变)
        self.current_iter = 0
        self.best_loss = float('inf')
        self._load_checkpoint()

    # _transfer, MoCo核心方法等保持不变
    @staticmethod
    def _transfer(cor):
        r, phi, z = cor[..., 0], cor[..., 1], cor[..., 2]
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        z_cartesian = r*torch.sinh(z)
        x, y, z_cartesian = x/4.5, y/4.5, z_cartesian/16
        a_cartesian_points = torch.stack([x, y, z_cartesian], dim=-1)
        return a_cartesian_points, cor.permute(0,2,1).contiguous()
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

    def _train_step(self):
        # ... (_train_step 的内部逻辑完全不变) ...
        self.encoder_q.train()
        start_time = time.time()
        try:
            original_cylindrical_points = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            original_cylindrical_points = next(self.train_iter)
        original_cylindrical_points = original_cylindrical_points.to(self.device)
        im_q_cylindrical = self.augmentation(original_cylindrical_points)
        im_k_cylindrical = self.augmentation(original_cylindrical_points)
        xyz1, cor1 = self._transfer(im_q_cylindrical)
        xyz2, cor2 = self._transfer(im_k_cylindrical)
        q = self.encoder_q((xyz1, cor1)); q = q.view(q.shape[0], -1); q = F.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k((xyz2,cor2)); k = k.view(k.shape[0], -1); k = F.normalize(k, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.moco_t
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        loss = self.criterion(logits, labels)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        self._dequeue_and_enqueue(k)
        batch_time = time.time() - start_time
        self.current_iter += 1
        return loss.item(), batch_time
    
    ### NEW ###
    def _update_augmentation_difficulty(self):
        """检查并更新数据增强的难度。"""
        # 1. 检查功能是否启用
        if not self.adaptive_cfg.get('enabled', False):
            return

        # 2. 检查冷却期和最大难度
        if self.current_iter - self.last_update_iter < self.adaptive_cfg.cooldown:
            return
        if self.difficulty_level >= self.adaptive_cfg.max_level:
            logging.info("[Adaptive Aug] Reached max difficulty level. No further updates.")
            # 禁用进一步检查以节省计算
            self.adaptive_cfg['enabled'] = False 
            return
        
        # 3. 检查历史loss是否已满 (确保有足够数据做判断)
        if len(self.loss_history) < self.adaptive_cfg.history_len:
            return

        # 4. 计算统计数据
        avg_loss = np.mean(self.loss_history)
        std_loss = np.std(self.loss_history)
        
        # 5. 判断是否满足更新条件
        loss_is_low = avg_loss < self.adaptive_cfg.loss_threshold
        loss_is_stable = std_loss < self.adaptive_cfg.std_threshold

        if loss_is_low and loss_is_stable:
            logging.info(f"[Adaptive Aug] Conditions met: avg_loss={avg_loss:.3f} < {self.adaptive_cfg.loss_threshold}, "
                         f"std_loss={std_loss:.3f} < {self.adaptive_cfg.std_threshold}. Increasing difficulty.")
            # 更新难度
            self.difficulty_level += 1
            self.augmentation.update_difficulty(self.difficulty_level)
            self.writer.add_scalar('Augmentation/DifficultyLevel', self.difficulty_level, self.current_iter)
            
            # 重置状态
            self.last_update_iter = self.current_iter
            self.loss_history.clear() # 清空历史，用新难度的loss重新填充
        
    def train_loop(self):
        logging.info(f"Starting MoCo pre-training from iter {self.current_iter + 1}, target: {self.config.training.max_iter}")
        while self.current_iter < self.config.training.max_iter:
            loss, batch_time = self._train_step()
            
            # ### MODIFIED ### 将当前loss添加到历史记录
            if self.adaptive_cfg.get('enabled', False):
                self.loss_history.append(loss)
            
            if self.current_iter % self.config.training.log_freq == 0:
                lr = self.scheduler.get_last_lr()[0]
                logging.info(f"Iter: {self.current_iter}/{self.config.training.max_iter}, Loss: {loss:.4f}, LR: {lr:.6f}, Time: {batch_time:.3f}s")
                self.writer.add_scalar('Loss/train', loss, self.current_iter)
                self.writer.add_scalar('LearningRate', lr, self.current_iter)
                if loss < self.best_loss: self.best_loss = loss; logging.info(f"New best loss: {self.best_loss:.4f}. Saving best model."); self._save_checkpoint(is_best=True)

                # ### NEW ### 在log频率处检查是否需要更新难度
                self._update_augmentation_difficulty()

            if self.current_iter % self.config.training.save_freq == 0: self._save_checkpoint(is_best=False)
            self.scheduler.step()
        self._save_checkpoint(is_best=False, filename_prefix="final_"); self.writer.close(); logging.info("MoCo pre-training completed!")

    ### MODIFIED ###
    def _save_checkpoint(self, is_best, filename_prefix=""):
        state = {'iter': self.current_iter, 'encoder_q_state_dict': self.encoder_q.state_dict(), 'encoder_k_state_dict': self.encoder_k.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.scheduler.state_dict(), 'queue': self.queue, 'queue_ptr': self.queue_ptr, 'best_loss': self.best_loss, 'config': self.config}
        # 保存自适应增强的状态
        if self.adaptive_cfg.get('enabled', False):
            state['adaptive_state'] = {
                'loss_history': self.loss_history,
                'difficulty_level': self.difficulty_level,
                'last_update_iter': self.last_update_iter
            }
        
        latest_path = os.path.join(self.checkpoint_dir, f'{filename_prefix}moco_latest.pth'); torch.save(state, latest_path); logging.info(f"Saved checkpoint to {latest_path}")
        if is_best: 
            best_path = os.path.join(self.checkpoint_dir, 'moco_best_loss.pth')
            shutil.copyfile(latest_path, best_path)
            logging.info(f"Copied best model to {best_path}")
            # 保存纯净的骨干网络权重
            torch.save(self.encoder_q.state_dict(), os.path.join(self.checkpoint_dir, 'moco_best_backbone.pth'))
            logging.info(f"Saved clean backbone weights for fine-tuning.")

    ### MODIFIED ###
    def _load_checkpoint(self):
        resume_path = self.config.training.get("resume_from_checkpoint")
        if not resume_path: return
        path_to_load = os.path.join(self.checkpoint_dir, resume_path) if not os.path.isabs(resume_path) else resume_path
        if not os.path.isfile(path_to_load): logging.warning(f"Resume checkpoint not found: {path_to_load}. Starting fresh."); return
        logging.info(f"Loading checkpoint from {path_to_load}")
        ckpt = torch.load(path_to_load, map_location=self.device)
        self.encoder_q.load_state_dict(ckpt['encoder_q_state_dict']); self.encoder_k.load_state_dict(ckpt['encoder_k_state_dict']); self.optimizer.load_state_dict(ckpt['optimizer_state_dict']); self.scheduler.load_state_dict(ckpt['scheduler_state_dict']); self.queue = ckpt['queue'].to(self.device); self.queue_ptr = ckpt['queue_ptr'].to(self.device); self.current_iter = ckpt['iter']; self.best_loss = ckpt.get('best_loss', float('inf'))
        
        # 恢复自适应增强的状态
        adaptive_state = ckpt.get('adaptive_state')
        if adaptive_state and self.adaptive_cfg.get('enabled', False):
            self.loss_history = adaptive_state['loss_history']
            self.difficulty_level = adaptive_state['difficulty_level']
            self.last_update_iter = adaptive_state['last_update_iter']
            self.augmentation.update_difficulty(self.difficulty_level) # 恢复增强器到正确的难度
            logging.info(f"Resumed adaptive augmentation state. Difficulty level: {self.difficulty_level}")

        logging.info(f"Resumed from iteration {self.current_iter}. Best loss: {self.best_loss:.4f}")

# --- 3. 配置和主函数 ---
### MODIFIED ###
def get_moco_config():
    config_dict = {
        "gpu_id": 0,
        "data": { "root_dir": "contra_data_clean_noipad/data_view1", "split_file": "contra_data_clean_noipad/pretrain_all1.txt", "batch_size": 128, "num_workers": 0, },
        
        "augmentation": {
            # 基础参数
            "base_params": {
                "phi_rotation_range_deg": [-180.0, 180.0],
                "r_scale_range": [0.9, 1.1],
                "eta_shift_range": [-0.2, 0.2],
                "jitter_params": {"sigma": 0.01, "clip": 0.05}
            },
            # 难度增加的步长
            "adaptive_steps": {
                "r_scale_step": 0.02,
                "eta_shift_step": 0.05,
                "jitter_sigma_step": 0.002
            }
        },
        
        ### NEW ### 自适应增强策略的控制参数
        "adaptive_augmentation": {
            "enabled": True,                 # 总开关
            "history_len": 200,              # 用于计算统计量的loss历史长度
            "cooldown": 2000,                # 每次更新难度的最小间隔(iter)
            "max_level": 5,                  # 最大难度等级
            "loss_threshold": 1.0,           # 平均loss低于此值才考虑增加难度
            "std_threshold": 0.1,            # loss标准差低于此值才认为收敛
        },

        "model_params": { 'model_name' : 'STNet_Point', 'coord_dim': 3, 'input_feature_dim': 3, 'output_dim': 512, "pretrain": True },
        "moco": { "m": 0.999, "k": 65536, "t": 0.07 },
        "optimizer": { "name": "AdamW", "lr": 1.0e-4, "weight_decay": 0.01 },
        "scheduler": { "name": "CosineAnnealingLR", "eta_min": 1e-7 },
        "training": { "max_iter": 200000, "log_freq": 100, "save_freq": 5000, "output_dir": "./moco_pretrain_v5_adaptive_aug", "resume_from_checkpoint": None, },
    }
    return OmegaConf.create(config_dict)

# 主函数 `if __name__ == "__main__":` 保持不变
if __name__ == "__main__":
    config = get_moco_config()
    os.makedirs(config.training.output_dir, exist_ok=True)
    log_path = os.path.join(config.training.output_dir, "moco_pretrain_run.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(log_path)])
    logging.info("Using MoCo Pre-training Configuration:\n%s", OmegaConf.to_yaml(config))
    seed = 42; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    try:
        trainer = MoCoTrainer(config)
        trainer.train_loop()
    except Exception as e:
        logging.critical("Unhandled exception during MoCo training:", exc_info=True)