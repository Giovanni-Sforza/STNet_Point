import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf, DictConfig

import os
import logging
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# --- Import your custom modules ---
from utils.PPC_dataset import EventClusterDataset
from model.PPC import create_model

class Trainer:
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_epoch = 0
        self.best_acc = 0.0

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.exp_dir = os.path.join(config.output_dir, f"{config.exp_name}_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)

        self._setup_logging()
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, 'runs'))

        self.logger.info("================ Configuration ================")
        self.logger.info("\n" + OmegaConf.to_yaml(config))
        self.logger.info("===============================================")
        
        self._setup_data()
        self._setup_model()
        
        if self.config.model.get('pretrained_path'):
            self._load_weights(self.config.model.pretrained_path)
            
        self._setup_optimizer()
        
        if self.config.train.resume_path:
            self._load_checkpoint(self.config.train.resume_path)

    def _setup_logging(self):
        log_file = os.path.join(self.exp_dir, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger()

    def _setup_data(self):
        self.logger.info("Setting up datasets...")
        self.train_dataset = EventClusterDataset(
            data_dir=self.config.data.data_dir,
            file_list_path=self.config.data.train_list,
            cluster_size=self.config.data.num_events
        )
        self.val_dataset = EventClusterDataset(
            data_dir=self.config.data.data_dir,
            file_list_path=self.config.data.val_list,
            cluster_size=self.config.data.num_events
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )
        self.logger.info(f"Train dataset: {len(self.train_dataset)} clusters.")
        self.logger.info(f"Validation dataset: {len(self.val_dataset)} clusters.")

    def _setup_model(self):
        self.logger.info("Creating model...")
        self.model = create_model(self.config.model)
        self.model.to(self.device)
        self.logger.info(f"Model: {self.config.model.model_name}")
        self.logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def _setup_optimizer(self):
        self.logger.info("Setting up optimizer and loss function...")
        if self.config.train.optimizer.type == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.train.optimizer.lr,
                weight_decay=self.config.train.optimizer.weight_decay
            )
        else:
            raise NotImplementedError(f"Optimizer {self.config.train.optimizer.type} not supported.")

        self.criterion = nn.CrossEntropyLoss()
        
        if self.config.train.scheduler.type == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.train.scheduler.T_max
            )
        else:
            raise NotImplementedError(f"Scheduler {self.config.train.scheduler.type} not supported.")

    def _save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'config': self.config
        }
        
        filename = os.path.join(self.exp_dir, f'ckpt_epoch_{epoch}.pth')
        torch.save(state, filename)
        self.logger.info(f"Saved checkpoint to {filename}")

        latest_filename = os.path.join(self.exp_dir, 'ckpt_latest.pth')
        torch.save(state, latest_filename)

        if is_best:
            best_filename = os.path.join(self.exp_dir, 'spt_best_acc.pth')
            torch.save(state, best_filename)
            self.logger.info(f"Saved new best model to {best_filename}")

    def _load_weights(self, path: str):
        self.logger.info(f"Loading pretrained model weights from: {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.logger.info("Checkpoint file contains 'model_state_dict' key. Loading weights from it.")
                state_dict = checkpoint['model_state_dict']
            else:
                self.logger.info("Assuming the file is a model state_dict itself.")
                state_dict = checkpoint
            self.model.load_state_dict(state_dict)
            self.logger.info("Successfully loaded pretrained weights into the model.")
        except FileNotFoundError:
            self.logger.error(f"Weight file not found at {path}. The model will start with random weights.")
        except Exception as e:
            self.logger.error(f"An error occurred while loading weights from {path}: {e}. The model will start with random weights.")

    def _load_checkpoint(self, path: str):
        self.logger.info(f"Attempting to resume training from checkpoint: {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_acc = checkpoint.get('best_acc', 0.0)
            self.logger.info(f"Successfully resumed training. Starting from epoch {self.start_epoch}.")
        except FileNotFoundError:
            self.logger.error(f"Checkpoint file not found at {path}. Starting training from scratch.")
        except KeyError as e:
            self.logger.error(f"Checkpoint file at {path} is missing key: {e}. Cannot resume. Starting from scratch.")
        except Exception as e:
            self.logger.error(f"An error occurred while loading the checkpoint from {path}: {e}. Starting training from scratch.")

    def _train_one_epoch(self, epoch):
        self.model.train()
        self.train_dataset.shuffle_and_recluster()
        
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.train.max_epochs} [Training]")
        
        for i, ((xyz, feat), labels) in enumerate(pbar):
            xyz, feat, labels = xyz.to(self.device), feat.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(xyz, feat)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.writer.add_scalar('LearningRate', self.scheduler.get_last_lr()[0], epoch)
        self.logger.info(f"Epoch {epoch} Training | Average Loss: {avg_loss:.4f}")
        
        self.scheduler.step()

    # --- 修改代码: 在验证函数中加入损失计算 ---
    def _validate(self, epoch):
        """
        Validation loop with multiple runs for stable evaluation.
        It now calculates and logs both loss and accuracy.
        """
        self.model.eval()
        
        num_runs = self.config.validation.get('num_runs', 1)
        self.logger.info(f"Starting validation with {num_runs} run(s) for stable metrics...")

        master_preds = []
        master_labels = []
        run_accuracies = [] 
        run_losses = [] # --- 新增代码: 用于记录每次运行的损失 ---

        with torch.no_grad():
            for i in range(num_runs):
                self.logger.info(f"  > Validation Run {i + 1}/{num_runs}...")
                
                self.val_dataset.shuffle_and_recluster()
                val_loader_run = DataLoader(
                    self.val_dataset,
                    batch_size=self.config.train.batch_size,
                    shuffle=False,
                    num_workers=self.config.data.num_workers
                )
                
                pbar = tqdm(val_loader_run, desc=f"  Run {i+1}/{num_runs}", leave=False)
                
                current_run_preds = []
                current_run_labels = []
                total_val_loss = 0.0 # --- 新增代码: 初始化单次运行的总损失 ---

                for (xyz, feat), labels in pbar:
                    xyz, feat, labels = xyz.to(self.device), feat.to(self.device), labels.to(self.device)
                    logits = self.model(xyz, feat)
                    
                    # --- 新增代码: 计算并累加损失 ---
                    loss = self.criterion(logits, labels)
                    total_val_loss += loss.item()
                    # --- 结束新增代码 ---

                    preds = torch.argmax(logits, dim=1)
                    
                    current_run_preds.extend(preds.cpu().numpy())
                    current_run_labels.extend(labels.cpu().numpy())
                
                # --- 新增代码: 计算并记录本次运行的平均损失和准确率 ---
                avg_run_loss = total_val_loss / len(val_loader_run)
                run_losses.append(avg_run_loss)
                # --- 结束新增代码 ---

                run_acc = accuracy_score(current_run_labels, current_run_preds)
                run_accuracies.append(run_acc)
                self.logger.info(f"  > Run {i + 1} Accuracy: {run_acc:.4f}, Avg Loss: {avg_run_loss:.4f}")

                master_preds.extend(current_run_preds)
                master_labels.extend(current_run_labels)

        # --- 修改代码: 计算聚合后的最终指标 (包括损失) ---
        final_acc = accuracy_score(master_labels, master_preds)
        mean_acc = np.mean(run_accuracies)
        std_acc = np.std(run_accuracies)
        mean_loss = np.mean(run_losses)
        std_loss = np.std(run_losses)

        # --- 修改代码: 记录到TensorBoard (包括损失) ---
        self.writer.add_scalar('Loss/val_mean', mean_loss, epoch)
        self.writer.add_scalar('Loss/val_std', std_loss, epoch)
        self.writer.add_scalar('Accuracy/val_aggregated', final_acc, epoch)
        self.writer.add_scalar('Accuracy/val_mean', mean_acc, epoch)
        self.writer.add_scalar('Accuracy/val_std', std_acc, epoch)
        
        # --- 修改代码: 更新日志信息 (包括损失) ---
        log_message = (
            f"Epoch {epoch} Validation Summary | "
            f"Mean Loss: {mean_loss:.4f} (+/- {std_loss:.4f}) | "
            f"Aggregated Acc: {final_acc:.4f} | "
            f"Mean Acc: {mean_acc:.4f} (+/- {std_acc:.4f})"
        )
        self.logger.info(log_message)

        cm_fig = self._plot_confusion_matrix(master_preds, master_labels, "Validation")
        self.writer.add_figure('ConfusionMatrix/val', cm_fig, epoch)
        
        return final_acc
    # --- 结束修改代码 ---

    def _plot_confusion_matrix(self, preds, labels, title_prefix=''):
        cm = confusion_matrix(labels, preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=self.config.data.class_names, 
                    yticklabels=self.config.data.class_names)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'{title_prefix} Confusion Matrix')
        plt.tight_layout()
        return fig

    def train(self):
        self.logger.info("Starting training process...")
        for epoch in range(self.start_epoch, self.config.train.max_epochs):
            self._train_one_epoch(epoch)
            
            if (epoch + 1) % self.config.train.val_every_n_epochs == 0:
                current_acc = self._validate(epoch)
                
                if current_acc > self.best_acc:
                    self.best_acc = current_acc
                    self.logger.info(f"New best accuracy: {self.best_acc:.4f}")
                    self._save_checkpoint(epoch, is_best=True)

            if (epoch + 1) % self.config.train.save_every_n_epochs == 0:
                self._save_checkpoint(epoch)
        
        self.writer.close()
        self.logger.info("Training finished!")

def main():
    config_path = "./configs/base_config.yaml"
    config = OmegaConf.load(config_path)
    
    cli_conf = OmegaConf.from_cli()
    config = OmegaConf.merge(config, cli_conf)

    try:
        trainer = Trainer(config)
        trainer.train()
    except KeyboardInterrupt:
        logging.getLogger().info("Training interrupted by user. Exiting...")
    except Exception as e:
        logging.getLogger().critical(f"An unhandled exception occurred: {e}", exc_info=True)

if __name__ == '__main__':
    main()