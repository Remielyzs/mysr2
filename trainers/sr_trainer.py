import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
import os
from .base_trainer import BaseTrainer
from ..models.simple_srcnn import SimpleSRCNN
from ..losses import MSELoss, L1Loss, CombinedLoss
from ..data_utils import SRDataset

class SRTrainer(BaseTrainer):
    """超分辨率模型训练器（支持梯度累积和混合精度训练）"""
    
    def __init__(self, config):
        """初始化SR训练器
        Args:
            config: 训练配置字典
        """
        super().__init__(config)
        
        # 梯度累积和混合精度训练配置
        self.accumulation_steps = config.get('accumulation_steps', 1)
        self.use_mixed_precision = config.get('use_mixed_precision', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_mixed_precision else None
        self.gradient_clip_val = config.get('gradient_clip_val', None)
        
        # 打印训练配置信息
        print(f"梯度累积步数: {self.accumulation_steps}")
        print(f"混合精度训练: {'启用' if self.use_mixed_precision else '禁用'}")
        if self.gradient_clip_val:
            print(f"梯度裁剪阈值: {self.gradient_clip_val}")
        
        # 设置模型、数据和优化器
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
    
    def setup_model(self):
        """设置模型"""
        model_class = self.config.get('model_class', SimpleSRCNN)
        
        # 计算输入通道数（RGB + 可选的边缘检测通道）
        input_channels = 3
        edge_methods = self.config.get('edge_methods', [])
        if edge_methods:
            input_channels += len(edge_methods)
        
        # 创建模型
        model_params = {
            'upscale_factor': self.config.get('upscale_factor', 2),
            'input_channels': input_channels
        }
        
        self.model = model_class(**model_params)
        self.model = self.model.to(self.device)
        
        print(f"模型已创建: {model_class.__name__}")
        print(f"输入通道数: {input_channels}")
        print(f"放大倍数: {model_params['upscale_factor']}")
        
        # 打印模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"总参数数: {total_params:,}")
        print(f"可训练参数数: {trainable_params:,}")
    
    def setup_data(self):
        """设置数据加载器"""
        # 数据变换
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # 创建数据集
        train_dataset = SRDataset(
            lr_dir=self.config['train_lr_dir'],
            hr_dir=self.config['train_hr_dir'],
            patch_size=self.config.get('patch_size', 64),
            upscale_factor=self.config.get('upscale_factor', 2),
            edge_methods=self.config.get('edge_methods', []),
            transform=transform
        )
        
        val_dataset = SRDataset(
            lr_dir=self.config['val_lr_dir'],
            hr_dir=self.config['val_hr_dir'],
            patch_size=self.config.get('patch_size', 64),
            upscale_factor=self.config.get('upscale_factor', 2),
            edge_methods=self.config.get('edge_methods', []),
            transform=transform
        )
        
        # 创建数据加载器
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"训练数据集大小: {len(train_dataset)}")
        print(f"验证数据集大小: {len(val_dataset)}")
    
    def setup_optimizer(self):
        """设置损失函数和优化器"""
        # 设置损失函数
        loss_type = self.config.get('loss_type', 'mse')
        if loss_type == 'mse':
            self.criterion = MSELoss()
        elif loss_type == 'l1':
            self.criterion = L1Loss()
        elif loss_type == 'combined':
            self.criterion = CombinedLoss()
        else:
            self.criterion = MSELoss()
        
        # 设置优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        print(f"损失函数: {loss_type}")
        print(f"优化器: Adam")
        print(f"学习率: {self.config.get('learning_rate', 1e-4)}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点（支持混合精度scaler状态）"""
        if os.path.exists(checkpoint_path):
            print(f"从检查点恢复训练: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载混合精度scaler状态
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print("已加载混合精度scaler状态")
            
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_loss = checkpoint.get('loss', float('inf'))
            print(f"已加载epoch {checkpoint['epoch']}的模型，loss为{checkpoint.get('loss', 'N/A')}")
        else:
            print(f"未找到检查点 {checkpoint_path}，从头开始训练。")
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """保存检查点（包含混合精度scaler状态）"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
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
        """训练一个epoch（支持梯度累积和混合精度）"""
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
    
    def process_batch(self, batch_data, is_training=True, batch_idx=0):
        """处理一个batch的数据（支持混合精度和梯度累积）"""
        lr_images, hr_images = batch_data
        lr_images = lr_images.to(self.device)
        hr_images = hr_images.to(self.device)
        
        if is_training:
            if self.use_mixed_precision:
                # 使用混合精度训练
                with autocast():
                    outputs = self.model(lr_images)
                    loss = self.criterion(outputs, hr_images)
            else:
                # 标准精度训练
                outputs = self.model(lr_images)
                loss = self.criterion(outputs, hr_images)
            
            # 执行反向传播和优化器步骤
            self.backward_and_step(loss, batch_idx)
            
            return loss.item()
        else:
            # 验证模式
            with torch.no_grad():
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(lr_images)
                        loss = self.criterion(outputs, hr_images)
                else:
                    outputs = self.model(lr_images)
                    loss = self.criterion(outputs, hr_images)
                
                return loss.item()
    
    def backward_and_step(self, loss, batch_idx):
        """执行反向传播和优化器步骤（支持梯度累积和混合精度）"""
        # 将loss除以累积步数，以获得平均梯度
        loss = loss / self.accumulation_steps
        
        if self.use_mixed_precision:
            # 混合精度训练的反向传播
            self.scaler.scale(loss).backward()
            
            # 检查是否到达累积步数或是最后一个batch
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                # 梯度裁剪（可选）
                if self.gradient_clip_val:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
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
                if self.gradient_clip_val:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                # 执行优化器步骤
                self.optimizer.step()
                self.optimizer.zero_grad()