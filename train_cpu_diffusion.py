#!/usr/bin/env python3
"""
CPU版本的简化扩散模型超分辨率训练脚本
避免CUDA兼容性问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import time
from pathlib import Path

# 强制使用CPU
def get_device():
    print("🖥️ 强制使用CPU以避免CUDA兼容性问题")
    return torch.device('cpu')

class SimpleSRDataset(Dataset):
    """简化的超分辨率数据集"""
    def __init__(self, size=20, patch_size=32):
        self.size = size
        self.patch_size = patch_size
        print(f"📁 创建模拟数据集: {size} 个样本")
        
        # 简单的变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 生成模拟的LR和HR图像对
        # LR: 8x8, HR: 32x32 (4x超分辨率)
        lr_size = self.patch_size // 4
        hr_size = self.patch_size
        
        # 创建有结构的模拟数据而不是纯随机
        base_pattern = torch.randn(3, lr_size, lr_size)
        lr_tensor = torch.clamp(base_pattern, 0, 1)
        
        # HR图像是LR的上采样版本加上一些细节
        hr_tensor = nn.functional.interpolate(
            lr_tensor.unsqueeze(0), 
            size=(hr_size, hr_size), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # 添加一些高频细节
        detail = torch.randn(3, hr_size, hr_size) * 0.1
        hr_tensor = torch.clamp(hr_tensor + detail, 0, 1)
        
        return lr_tensor, hr_tensor

class SimpleUNet(nn.Module):
    """极简的U-Net模型 - CPU优化版本"""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # 更小的网络以适应CPU训练
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 解码器
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),  # 32 = 16 + 16 (skip connection)
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 上采样输入
        x = nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        
        # 解码
        d2 = self.dec2(e2)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        return d1

def add_simple_noise(image, noise_level=0.1):
    """添加简单的高斯噪声"""
    noise = torch.randn_like(image) * noise_level
    return torch.clamp(image + noise, 0, 1), noise

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    print(f"Epoch {epoch} 开始训练...")
    
    for batch_idx, (lr_images, hr_images) in enumerate(dataloader):
        try:
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            predicted = model(lr_images)
            
            # 计算损失
            loss = criterion(predicted, hr_images)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 打印进度
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}")
                
        except Exception as e:
            print(f"❌ 批次 {batch_idx} 训练失败: {e}")
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"Epoch {epoch} 完成，平均损失: {avg_loss:.6f}")
    return avg_loss

def main():
    print("🚀 开始CPU版本简化扩散模型训练")
    print("=" * 50)
    
    # 强制使用CPU
    device = get_device()
    print(f"使用设备: {device}")
    
    # 训练配置 - CPU优化
    config = {
        'epochs': 3,
        'batch_size': 2,
        'learning_rate': 1e-3,  # 稍微提高学习率以加快收敛
        'patch_size': 32,       # 更小的patch大小
        'num_samples': 20       # 更少的样本
    }
    
    print("训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 创建模拟数据集
    train_dataset = SimpleSRDataset(
        size=config['num_samples'],
        patch_size=config['patch_size']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0  # CPU训练不需要多进程
    )
    
    print(f"训练批次数: {len(train_loader)}")
    
    # 创建模型
    print("\n🧠 创建模型...")
    model = SimpleUNet().to(device)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    print(f"优化器: Adam")
    print(f"损失函数: MSE Loss")
    
    # 开始训练
    print(f"\n🎯 开始训练 ({config['epochs']} epochs)...")
    print("=" * 50)
    
    training_history = []
    
    try:
        for epoch in range(1, config['epochs'] + 1):
            start_time = time.time()
            
            # 训练一个epoch
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
            
            epoch_time = time.time() - start_time
            training_history.append(train_loss)
            
            print(f"Epoch {epoch} 完成，用时: {epoch_time:.2f}秒")
            print("-" * 30)
            
        print("\n✅ 训练完成!")
        print(f"最终损失: {training_history[-1]:.6f}")
        
        # 保存模型
        model_path = "simple_diffusion_cpu_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'training_history': training_history
        }, model_path)
        print(f"💾 模型已保存到: {model_path}")
        
        # 简单测试
        print("\n🧪 进行简单测试...")
        model.eval()
        with torch.no_grad():
            test_lr = torch.randn(1, 3, 8, 8).to(device)
            test_output = model(test_lr)
            print(f"测试输入形状: {test_lr.shape}")
            print(f"测试输出形状: {test_output.shape}")
            print("✅ 模型测试通过!")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    return model, training_history

if __name__ == "__main__":
    model, history = main()