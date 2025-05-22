import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import glob
from PIL import Image
from .base_trainer import BaseTrainer
from data_utils import SRDataset

class SRTrainer(BaseTrainer):
    def __init__(self, config):
        """初始化超分辨率模型训练器
        Args:
            config: 训练配置字典
        """
        super().__init__(config)
        self.use_text_descriptions = config.get('use_text_descriptions', False)
        self.edge_detection_methods = config.get('edge_detection_methods', None)
        
    def setup_model(self):
        """设置模型、损失函数和优化器"""
        model_class = self.config['model_class']
        model_params = self.config.get('model_params', {})
        
        # 处理输入通道数
        num_edge_channels = len(self.edge_detection_methods) if self.edge_detection_methods else 0
        in_channels = 3 + num_edge_channels
        
        # 设置模型参数
        if 'in_channels' not in model_params:
            model_params['in_channels'] = in_channels
        if 'upscale_factor' not in model_params and self.config.get('upscale_factor'):
            model_params['upscale_factor'] = self.config['upscale_factor']
        
        self.model = model_class(**model_params)
        self.model.to(self.device)
        
        # 设置损失函数
        self.criterion = self.config.get('criterion', nn.MSELoss())
        
        # 设置优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001)
        )
    
    def setup_data(self):
        """设置数据加载器"""
        # 基本转换
        transform = transforms.ToTensor()
        
        # 准备训练数据的文本描述
        train_text_descriptions = None
        if self.use_text_descriptions:
            train_text_descriptions = self._prepare_text_descriptions(
                self.config['lr_data_dir']
            )
        
        # 创建训练数据集
        train_dataset = SRDataset(
            self.config['lr_data_dir'],
            self.config['hr_data_dir'],
            text_descriptions=train_text_descriptions,
            transform=transform,
            mode='train',
            edge_methods=self.edge_detection_methods,
            device=self.device,
            lr_patch_size=self.config.get('lr_patch_size'),
            upscale_factor=self.config.get('upscale_factor'),
            image_size=self.config.get('image_size')
        )
        
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True
        )
        
        # 如果提供了验证数据目录，创建验证数据加载器
        val_lr_dir = self.config.get('val_lr_data_dir')
        val_hr_dir = self.config.get('val_hr_data_dir')
        
        if val_lr_dir and val_hr_dir and os.path.exists(val_lr_dir) and os.path.exists(val_hr_dir):
            val_text_descriptions = None
            if self.use_text_descriptions:
                val_text_descriptions = self._prepare_text_descriptions(val_lr_dir)
            
            val_dataset = SRDataset(
                lr_dir=None,
                hr_dir=None,
                text_descriptions=val_text_descriptions,
                transform=transform,
                mode='eval',
                edge_methods=self.edge_detection_methods,
                device=self.device,
                lr_patch_size=self.config.get('lr_patch_size'),
                upscale_factor=self.config.get('upscale_factor'),
                val_lr_dir=val_lr_dir,
                val_hr_dir=val_hr_dir,
                image_size=self.config.get('image_size')
            )
            
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=1,  # 验证时使用batch_size=1
                shuffle=False
            )
    
    def _prepare_text_descriptions(self, data_dir):
        """准备图像的文本描述
        Args:
            data_dir: 数据目录
        Returns:
            list: 文本描述列表
        """
        supported_formats = ('.png', '.tiff', '.tif', '.npz')
        num_images = len([
            f for ext in supported_formats 
            for f in glob.glob(os.path.join(data_dir, f'*{ext}'))
        ])
        
        if num_images == 0:
            return None
            
        return [f"Sample text for image {i}" for i in range(num_images)]
    
    def process_batch(self, batch_data, is_training=True):
        """处理一个batch的数据
        Args:
            batch_data: 一个batch的数据
            is_training: 是否是训练模式
        Returns:
            float: 当前batch的loss值
        """
        # 解包数据
        if self.use_text_descriptions and is_training:
            lr_images, hr_images, text_descs = batch_data
        else:
            lr_images, hr_images = batch_data
            text_descs = None
        
        # 将数据移到指定设备
        lr_images = lr_images.to(self.device)
        hr_images = hr_images.to(self.device)
        
        # 在训练模式下，清零梯度
        if is_training:
            self.optimizer.zero_grad()
        
        # 前向传播
        if self.use_text_descriptions and hasattr(self.model, 'forward') and \
           'text_input' in self.model.forward.__code__.co_varnames:
            outputs = self.model(lr_images, text_input=text_descs)
        else:
            outputs = self.model(lr_images)
        
        # 计算损失
        loss = self.criterion(outputs, hr_images)
        
        # 在训练模式下，进行反向传播和优化
        if is_training:
            loss.backward()
            self.optimizer.step()
        
        return loss.item()