import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
import os
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config_manager import ConfigManager
from utils.logger import TrainingLogger, setup_logging

class BaseTrainer:
    """训练器基类
    
    提供统一的训练接口和基础功能：
    - 配置管理和验证
    - 日志记录和监控
    - 检查点保存和恢复
    - 混合精度训练支持
    - 早停机制
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化训练器
        
        Args:
            config: 训练配置字典或ConfigManager实例
        """
        # 配置管理
        if isinstance(config, ConfigManager):
            self.config_manager = config
        else:
            self.config_manager = ConfigManager(config)
            
        # 为了向后兼容，保留config属性
        self.config = self.config_manager.to_dict()
        
        # 基础属性初始化
        self._initialize_attributes()
        
        # 设置日志系统
        self._setup_logging()
        
        # 设置目录结构
        self.setup_directories()
        
        # 记录配置信息
        self.logger.log_config(self.config)
        
        # 打印训练配置信息
        self._print_training_info()
        
    def _initialize_attributes(self):
        """初始化基础属性"""
        # 设备配置
        self.device = self.config_manager.get('device', 'cpu')
        
        # 模型相关
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None  # 学习率调度器
        
        # 数据相关
        self.train_dataloader = None
        self.val_dataloader = None
        
        # 训练配置
        self.accumulation_steps = self.config_manager.get('accumulation_steps', 1)
        self.use_mixed_precision = (
            self.config_manager.get('use_mixed_precision', True) and 
            torch.cuda.is_available()
        )
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # 训练状态
        self.start_epoch = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop = False
        self.training_start_time = None
        
        # 记录训练过程
        self.train_losses = []
        self.val_losses = []
        
        # 确保所有必需的属性都存在
        self._ensure_required_attributes()
        
    def _ensure_required_attributes(self):
        """确保所有必需的属性都存在，避免AttributeError"""
        required_attrs = [
            'results_dir', 'run_dir', 'checkpoint_dir', 
            'images_dir', 'logs_dir'
        ]
        
        for attr in required_attrs:
            if not hasattr(self, attr):
                setattr(self, attr, None)
                
    def _setup_logging(self):
        """设置日志系统"""
        log_dir = self.config_manager.get('log_dir')
        if not log_dir:
            # 如果没有指定日志目录，使用默认位置
            log_dir = 'logs'
            
        self.logger = setup_logging(
            log_dir=log_dir,
            console_output=self.config_manager.get('console_logging', True)
        )
        
    def _print_training_info(self):
        """打印训练配置信息"""
        info_items = [
            f"设备: {self.device}",
            f"梯度累积步数: {self.accumulation_steps}",
            f"混合精度训练: {'启用' if self.use_mixed_precision else '禁用'}",
            f"批次大小: {self.config_manager.get('batch_size', 'N/A')}",
            f"学习率: {self.config_manager.get('learning_rate', 'N/A')}",
        ]
        
        for info in info_items:
            self.logger.info(info)
        
    def setup_directories(self):
        """设置保存结果的目录结构"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = self.config.get('model_name', 'model')
        results_base_dir = self.config.get('results_base_dir', 'results')
        
        # 设置主要目录路径
        self.results_dir = results_base_dir  # 添加results_dir属性
        self.run_dir = os.path.join(results_base_dir, f"{model_name}_{timestamp}")
        self.checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')
        self.images_dir = os.path.join(self.run_dir, 'images')
        self.logs_dir = os.path.join(self.run_dir, 'logs')  # 添加日志目录
        
        # 创建所有必要的目录
        for directory in [self.results_dir, self.run_dir, self.checkpoint_dir, self.images_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
            
        print(f"结果保存目录: {self.run_dir}")
        print(f"检查点目录: {self.checkpoint_dir}")
        print(f"图像目录: {self.images_dir}")
        print(f"日志目录: {self.logs_dir}")
    
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
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
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
        
        # 如果使用混合精度训练，保存scaler状态
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
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
        
        # 重置优化器梯度
        self.optimizer.zero_grad()
        
        for batch_idx, batch_data in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch+1}", unit="batch")):
            loss = self.process_batch(batch_data, is_training=True, batch_idx=batch_idx)
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
    
    def process_batch(self, batch_data, is_training=True, batch_idx=0):
        """处理一个batch的数据
        Args:
            batch_data: 一个batch的数据
            is_training: 是否是训练模式
            batch_idx: 当前batch的索引（用于梯度累积）
        Returns:
            loss: 当前batch的loss值
        """
        raise NotImplementedError("子类必须实现process_batch方法")
    
    def backward_and_step(self, loss, batch_idx):
        """执行反向传播和优化器步骤（支持梯度累积和混合精度）
        Args:
            loss: 损失值
            batch_idx: 当前batch索引
        """
        # 将loss除以累积步数，以获得平均梯度
        loss = loss / self.accumulation_steps
        
        if self.use_mixed_precision:
            # 混合精度训练的反向传播
            self.scaler.scale(loss).backward()
            
            # 检查是否到达累积步数或是最后一个batch
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                # 梯度裁剪（可选）
                if self.config.get('gradient_clip_val'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip_val'])
                
                # 执行优化器步骤
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            # 标准精度训练的反向传播
            loss.backward()
            
            # 检查是否到达累积步数或是最后一个batch
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                # 梯度裁剪（可选）
                if self.config.get('gradient_clip_val'):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip_val'])
                
                # 执行优化器步骤
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        # 返回未缩放的loss用于记录
        return loss.item() * self.accumulation_steps
    
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
        try:
            self.training_start_time = time.time()
            
            self.logger.info(f"开始训练流程，使用设备: {self.device}")
            
            # 设置模型和数据
            self._safe_setup()
            
            # 记录模型信息
            if self.model:
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                self.logger.log_model_info(self.model, total_params, trainable_params)
            
            # 如果指定了检查点，从检查点恢复
            if 'resume_checkpoint' in self.config:
                self._safe_load_checkpoint(self.config['resume_checkpoint'])
            
            epochs = self.config_manager.get('epochs', 10)
            self.logger.info(f"开始训练，从epoch {self.start_epoch+1}开始，总共{epochs}个epochs")
            
            # 主训练循环
            for epoch in range(self.start_epoch, epochs):
                self.current_epoch = epoch
                
                try:
                    # 记录epoch开始
                    self.logger.log_epoch_start(epoch, epochs)
                    
                    # 训练阶段
                    train_loss = self.train_epoch()
                    
                    # 验证阶段
                    val_loss = self.validate_epoch()
                    
                    # 记录GPU内存使用
                    self.logger.log_gpu_memory()
                    
                    # 记录epoch结束
                    self.logger.log_epoch_end(epoch, train_loss, val_loss)
                    
                    # 保存检查点
                    is_best = val_loss < self.best_loss
                    self._safe_save_checkpoint(epoch, val_loss, is_best)
                    
                    # 检查是否需要早停
                    if self.check_early_stopping(val_loss):
                        patience = self.config_manager.get('early_stopping_patience')
                        self.logger.log_early_stopping(patience, self.epochs_no_improve)
                        break
                        
                except Exception as e:
                    self.logger.error(f"Epoch {epoch+1} 训练过程中发生错误", exception=e)
                    # 根据配置决定是否继续训练
                    if self.config_manager.get('stop_on_error', True):
                        raise
                    else:
                        self.logger.warning("继续下一个epoch的训练")
                        continue
            
            # 训练完成
            total_time = time.time() - self.training_start_time
            self.logger.log_training_complete(total_time, self.best_loss)
            
            # 生成训练报告
            report = self.logger.create_training_report()
            self.logger.info("训练报告", report=report)
            
            # 绘制损失曲线
            self._safe_plot_losses()
            
        except Exception as e:
            self.logger.error("训练过程中发生严重错误", exception=e)
            raise
            
    def _safe_setup(self):
        """安全地设置模型和数据"""
        try:
            self.setup_model()
            self.logger.info("模型设置完成")
        except Exception as e:
            self.logger.error("模型设置失败", exception=e)
            raise
            
        try:
            self.setup_data()
            self.logger.info("数据设置完成")
        except Exception as e:
            self.logger.error("数据设置失败", exception=e)
            raise
            
    def _safe_load_checkpoint(self, checkpoint_path: str):
        """安全地加载检查点"""
        try:
            self.load_checkpoint(checkpoint_path)
        except Exception as e:
            self.logger.error(f"加载检查点失败: {checkpoint_path}", exception=e)
            if self.config_manager.get('require_checkpoint', False):
                raise
            else:
                self.logger.warning("忽略检查点加载错误，从头开始训练")
                
    def _safe_save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """安全地保存检查点"""
        try:
            self.save_checkpoint(epoch, loss, is_best)
            # 记录检查点保存
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            self.logger.log_checkpoint_save(checkpoint_path, epoch, loss, is_best)
        except Exception as e:
            self.logger.error(f"保存检查点失败: epoch {epoch+1}", exception=e)
            # 检查点保存失败不应该中断训练
            
    def _safe_plot_losses(self):
        """安全地绘制损失曲线"""
        try:
            self.plot_losses()
        except Exception as e:
            self.logger.error("绘制损失曲线失败", exception=e)
            # 绘图失败不应该影响训练结果