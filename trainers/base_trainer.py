import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

class BaseTrainer:
    def __init__(self, config):
        """初始化训练器
        Args:
            config: 训练配置字典，包含所有必要的训练参数
        """
        self.config = config
        self.device = config.get('device', 'cpu')
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.train_dataloader = None
        self.val_dataloader = None
        
        # 训练状态
        self.start_epoch = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop = False
        
        # 记录训练过程
        self.train_losses = []
        self.val_losses = []
        
        # 结果保存路径
        self.setup_directories()
        
    def setup_directories(self):
        """设置保存结果的目录结构"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = self.config.get('model_name', 'model')
        results_base_dir = self.config.get('results_base_dir', 'results')
        
        self.run_dir = os.path.join(results_base_dir, f"{model_name}_{timestamp}")
        self.checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')
        self.images_dir = os.path.join(self.run_dir, 'images')
        
        for directory in [self.run_dir, self.checkpoint_dir, self.images_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def setup_model(self):
        """设置模型、损失函数和优化器"""
        raise NotImplementedError("子类必须实现setup_model方法")
    
    def setup_data(self):
        """设置数据加载器"""
        raise NotImplementedError("子类必须实现setup_data方法")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点
        Args:
            checkpoint_path: 检查点文件路径
        """
        if os.path.exists(checkpoint_path):
            print(f"从检查点恢复训练: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_loss = checkpoint.get('loss', float('inf'))
            print(f"已加载epoch {checkpoint['epoch']}的模型，loss为{checkpoint.get('loss', 'N/A')}")
        else:
            print(f"未找到检查点 {checkpoint_path}，从头开始训练。")
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """保存检查点
        Args:
            epoch: 当前训练轮次
            loss: 当前loss值
            is_best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        
        # 保存当前epoch的检查点
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            best_model_path = os.path.join(self.run_dir, f"best_model.pth")
            torch.save(self.model.state_dict(), best_model_path)
            print(f"保存最佳模型到 {best_model_path}，验证loss: {loss:.4f}")
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        
        for batch_data in tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch+1}", unit="batch"):
            loss = self.process_batch(batch_data, is_training=True)
            running_loss += loss
            
        epoch_loss = running_loss / len(self.train_dataloader)
        self.train_losses.append(epoch_loss)
        return epoch_loss
    
    def validate_epoch(self):
        """验证一个epoch"""
        if not self.val_dataloader:
            return float('inf')
            
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for batch_data in tqdm(self.val_dataloader, desc="Validation", unit="batch"):
                loss = self.process_batch(batch_data, is_training=False)
                running_loss += loss
                
        epoch_loss = running_loss / len(self.val_dataloader)
        self.val_losses.append(epoch_loss)
        return epoch_loss
    
    def process_batch(self, batch_data, is_training=True):
        """处理一个batch的数据
        Args:
            batch_data: 一个batch的数据
            is_training: 是否是训练模式
        Returns:
            loss: 当前batch的loss值
        """
        raise NotImplementedError("子类必须实现process_batch方法")
    
    def check_early_stopping(self, val_loss):
        """检查是否需要早停
        Args:
            val_loss: 当前验证loss
        Returns:
            bool: 是否需要早停
        """
        patience = self.config.get('early_stopping_patience')
        min_delta = self.config.get('early_stopping_min_delta', 0.0)
        
        if patience is None:
            return False
            
        if val_loss < self.best_loss - min_delta:
            self.best_loss = val_loss
            self.epochs_no_improve = 0
            return False
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= patience:
                print(f"触发早停机制，{patience}个epoch未改善。")
                return True
        return False
    
    def plot_losses(self):
        """绘制损失曲线"""
        plt.figure()
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        loss_curve_path = os.path.join(self.images_dir, 'loss_curve.png')
        plt.savefig(loss_curve_path)
        plt.close()
        print(f"损失曲线已保存到 {loss_curve_path}")
    
    def train(self):
        """执行完整的训练流程"""
        print(f"使用设备: {self.device}")
        self.setup_model()
        self.setup_data()
        
        # 如果指定了检查点，从检查点恢复
        if 'resume_checkpoint' in self.config:
            self.load_checkpoint(self.config['resume_checkpoint'])
        
        epochs = self.config.get('epochs', 10)
        print(f"开始训练，从epoch {self.start_epoch}开始...")
        
        for epoch in range(self.start_epoch, epochs):
            self.current_epoch = epoch
            
            # 训练阶段
            train_loss = self.train_epoch()
            print(f"Epoch {epoch+1}/{epochs}, 训练Loss: {train_loss:.4f}")
            
            # 验证阶段
            val_loss = self.validate_epoch()
            print(f"Epoch {epoch+1}/{epochs}, 验证Loss: {val_loss:.4f}")
            
            # 保存检查点
            is_best = val_loss < self.best_loss
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # 检查是否需要早停
            if self.check_early_stopping(val_loss):
                break
        
        print("训练完成")
        self.plot_losses()