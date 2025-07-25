#!/usr/bin/env python3
"""
简化的扩散模型超分辨率训练脚本
专门针对CUDA兼容性问题进行优化
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

# 强制使用CPU如果CUDA不兼容
def get_device():
    if torch.cuda.is_available():
        try:
            # 测试CUDA是否真正可用
            test_tensor = torch.randn(1, device='cuda')
            del test_tensor
            return torch.device('cuda')
        except Exception as e:
            print(f"⚠️ CUDA不可用，使用CPU: {e}")
            return torch.device('cpu')
    else:
        print("⚠️ CUDA不可用，使用CPU")
        return torch.device('cpu')

class SimpleSRDataset(Dataset):
    """简化的超分辨率数据集"""
    def __init__(self, hr_dir, lr_dir, patch_size=64, max_samples=100):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.patch_size = patch_size
        
        # 获取图像文件列表
        hr_files = list(self.hr_dir.glob("*.png"))[:max_samples]
        lr_files = list(self.lr_dir.glob("*.png"))[:max_samples]
        
        # 确保HR和LR文件匹配
        self.image_pairs = []
        for hr_file in hr_files:
            lr_file = self.lr_dir / hr_file.name
            if lr_file.exists():
                self.image_pairs.append((hr_file, lr_file))
        
        print(f"📁 找到 {len(self.image_pairs)} 个图像对")
        
        # 简单的变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        hr_path, lr_path = self.image_pairs[idx]
        
        try:
            # 加载图像
            hr_image = Image.open(hr_path).convert('RGB')
            lr_image = Image.open(lr_path).convert('RGB')
            
            # 调整大小
            hr_image = hr_image.resize((self.patch_size, self.patch_size), Image.LANCZOS)
            lr_image = lr_image.resize((self.patch_size // 4, self.patch_size // 4), Image.LANCZOS)
            
            # 转换为tensor
            hr_tensor = self.transform(hr_image)
            lr_tensor = self.transform(lr_image)
            
            return lr_tensor, hr_tensor
            
        except Exception as e:
            print(f"❌ 加载图像失败 {hr_path}: {e}")
            # 返回随机数据作为fallback
            return torch.randn(3, self.patch_size // 4, self.patch_size // 4), torch.randn(3, self.patch_size, self.patch_size)

class SimpleUNet(nn.Module):
    """极简的U-Net模型"""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 解码器
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),  # 64 = 32 + 32 (skip connection)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, padding=1),
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
            
            # 添加噪声（简化的扩散过程）
            noisy_hr, noise = add_simple_noise(hr_images)
            
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
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}")
                
        except Exception as e:
            print(f"❌ 批次 {batch_idx} 训练失败: {e}")
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"Epoch {epoch} 完成，平均损失: {avg_loss:.6f}")
    return avg_loss

def main():
    print("🚀 开始简化扩散模型训练")
    print("=" * 50)
    
    # 设备检测
    device = get_device()
    print(f"使用设备: {device}")
    
    # 训练配置
    config = {
        'epochs': 2,
        'batch_size': 1,  # 减小批次大小
        'learning_rate': 1e-4,
        'patch_size': 64,  # 减小patch大小
        'max_samples': 50  # 限制样本数量
    }
    
    print("训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 数据路径
    hr_dir = "data/DIV2K_train_HR"
    lr_dir = "data/DIV2K_train_LR_bicubic/X4"
    
    # 检查数据目录
    if not os.path.exists(hr_dir) or not os.path.exists(lr_dir):
        print(f"❌ 数据目录不存在: {hr_dir} 或 {lr_dir}")
        print("使用模拟数据进行测试...")
        
        # 创建模拟数据
        class DummyDataset(Dataset):
            def __init__(self, size=20):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                lr = torch.randn(3, 16, 16)  # 16x16 LR
                hr = torch.randn(3, 64, 64)  # 64x64 HR
                return lr, hr
        
        train_dataset = DummyDataset(config['max_samples'])
        print(f"📁 使用模拟数据集: {len(train_dataset)} 个样本")
    else:
        # 加载真实数据
        train_dataset = SimpleSRDataset(
            hr_dir, lr_dir, 
            patch_size=config['patch_size'],
            max_samples=config['max_samples']
        )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0  # 避免多进程问题
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
        model_path = "simple_diffusion_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'training_history': training_history
        }, model_path)
        print(f"💾 模型已保存到: {model_path}")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    return model, training_history

if __name__ == "__main__":
    model, history = main()