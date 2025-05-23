import torch
import torch.nn as nn
from trainers.base_trainer import BaseTrainer
from models.lora_diffusion_sr import LoRADiffusionSR
from torch.utils.data import DataLoader
from data_utils import SRDataset
import os

class LoRATrainer(BaseTrainer):
    """专门用于训练LoRA超分辨率模型的训练器"""
    
    def setup_model(self):
        """设置LoRA模型、损失函数和优化器"""
        # 获取模型配置
        base_model_params = {
            'input_channels': 3,
            'unet_channels': self.config.get('unet_channels', (64, 128, 256)),
            'num_blocks': self.config.get('num_blocks', 2)
        }
        
        # 创建LoRA模型
        edge_methods = self.config.get('edge_detection_methods', [])
        edge_channels = len(edge_methods) if edge_methods else 0
        
        self.model = LoRADiffusionSR(
            base_model_params=base_model_params,
            lora_rank=self.config.get('lora_rank', 4),
            lora_alpha=self.config.get('lora_alpha', 1.0),
            edge_channels=edge_channels
        ).to(self.device)
        
        # 冻结基础模型参数
        self.model.freeze_base_model()
        
        # 设置损失函数
        self.criterion = self.config.get('criterion', nn.MSELoss())
        
        # 设置优化器，只优化LoRA参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
    
    def setup_data(self):
        """设置训练和验证数据加载器"""
        # 数据转换
        transform = self.config.get('transform', None)
        
        # 创建训练数据集
        train_dataset = SRDataset(
            lr_data_dir=self.config['lr_data_dir'],
            hr_data_dir=self.config['hr_data_dir'],
            transform=transform,
            mode='train',
            edge_methods=self.config.get('edge_detection_methods', []),
            device=self.device,
            lr_patch_size=self.config.get('lr_patch_size'),
            upscale_factor=self.config.get('upscale_factor'),
            image_size=self.config.get('image_size')
        )
        
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        # 如果提供了验证数据，创建验证数据加载器
        if 'val_lr_data_dir' in self.config and 'val_hr_data_dir' in self.config:
            val_dataset = SRDataset(
                lr_dir=None,
                hr_dir=None,
                transform=transform,
                mode='eval',
                edge_methods=self.config.get('edge_detection_methods', []),
                device=self.device,
                lr_patch_size=self.config.get('lr_patch_size'),
                upscale_factor=self.config.get('upscale_factor'),
                val_lr_dir=self.config['val_lr_data_dir'],
                val_hr_dir=self.config['val_hr_data_dir'],
                image_size=self.config.get('image_size')
            )
            
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=1,  # 验证时使用较小的batch size
                shuffle=False,
                num_workers=self.config.get('num_workers', 4),
                pin_memory=True
            )
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_dataloader)
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            lr_imgs = batch['lr'].to(self.device)
            hr_imgs = batch['hr'].to(self.device)
            edge_features = [edge.to(self.device) for edge in batch['edge_features']] if 'edge_features' in batch else None
            
            self.optimizer.zero_grad()
            
            # 前向传播
            sr_output = self.model(lr_imgs, edge_features)
            
            # 计算损失
            loss = self.criterion(sr_output, hr_imgs)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Batch [{batch_idx}/{num_batches}] Loss: {loss.item():.4f}')
        
        return total_loss / num_batches
    
    def validate(self):
        """验证模型性能"""
        if not self.val_dataloader:
            return None
        
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_dataloader)
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                lr_imgs = batch['lr'].to(self.device)
                hr_imgs = batch['hr'].to(self.device)
                edge_features = [edge.to(self.device) for edge in batch['edge_features']] if 'edge_features' in batch else None
                
                sr_output = self.model(lr_imgs, edge_features)
                loss = self.criterion(sr_output, hr_imgs)
                
                total_loss += loss.item()
        
        return total_loss / num_batches