#!/usr/bin/env python3
"""
优化的扩散模型训练脚本
基于环境测试结果进行优化
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from PIL import Image

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.diffusion_sr import DiffusionSRModel

class SimpleSRDataset:
    """简化的超分辨率数据集"""
    
    def __init__(self, lr_dir, hr_dir, patch_size=64):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.patch_size = patch_size
        
        # 获取文件列表
        self.lr_files = sorted(list(self.lr_dir.glob('*.png')))
        self.hr_files = sorted(list(self.hr_dir.glob('*.png')))
        
        # 确保文件数量匹配
        assert len(self.lr_files) == len(self.hr_files), f"LR和HR文件数量不匹配: {len(self.lr_files)} vs {len(self.hr_files)}"
        
        print(f"数据集加载完成: {len(self.lr_files)} 个图像对")
    
    def __len__(self):
        return len(self.lr_files)
    
    def __getitem__(self, idx):
        # 加载图像
        lr_path = self.lr_files[idx]
        hr_path = self.hr_files[idx]
        
        lr_image = Image.open(lr_path).convert('RGB')
        hr_image = Image.open(hr_path).convert('RGB')
        
        # 调整大小
        lr_image = lr_image.resize((self.patch_size, self.patch_size), Image.BICUBIC)
        hr_image = hr_image.resize((self.patch_size * 4, self.patch_size * 4), Image.BICUBIC)
        
        # 转换为tensor
        lr_tensor = torch.from_numpy(np.array(lr_image)).permute(2, 0, 1).float() / 255.0
        hr_tensor = torch.from_numpy(np.array(hr_image)).permute(2, 0, 1).float() / 255.0
        
        return lr_tensor, hr_tensor

def add_noise(hr_image, timestep, num_timesteps=1000):
    """添加噪声到高分辨率图像"""
    # 定义噪声调度参数
    beta_start = 0.0001
    beta_end = 0.02
    
    # 计算beta值
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    
    # 确保timestep在有效范围内
    timestep = torch.clamp(torch.tensor(timestep), 0, num_timesteps - 1).long()
    
    # 获取当前时间步的alpha_cumprod值
    alpha_t = alpha_cumprod[timestep].to(hr_image.device)
    
    # 生成噪声
    noise = torch.randn_like(hr_image)
    
    # 添加噪声 - 确保所有操作都在tensor上进行
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
    
    # 扩展维度以匹配图像张量
    while len(sqrt_alpha_t.shape) < len(hr_image.shape):
        sqrt_alpha_t = sqrt_alpha_t.unsqueeze(-1)
        sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.unsqueeze(-1)
    
    noisy_hr = sqrt_alpha_t * hr_image + sqrt_one_minus_alpha_t * noise
    
    return noisy_hr, noise

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    print(f"\nEpoch {epoch + 1} 开始训练...")
    
    for batch_idx, (lr_images, hr_images) in enumerate(dataloader):
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)
        
        batch_size = lr_images.size(0)
        
        # 随机采样时间步长
        timesteps = torch.randint(0, 1000, (batch_size,)).float().to(device)
        
        # 为每个样本添加噪声
        noisy_hr_list = []
        noise_list = []
        
        for i in range(batch_size):
            noisy_hr, noise = add_noise(hr_images[i:i+1], timesteps[i].item())
            noisy_hr_list.append(noisy_hr)
            noise_list.append(noise)
        
        noisy_hr_batch = torch.cat(noisy_hr_list, dim=0)
        noise_batch = torch.cat(noise_list, dim=0)
        
        # 前向传播
        optimizer.zero_grad()
        predicted_noise = model.forward_with_noisy_hr(lr_images, noisy_hr_batch, timesteps)
        
        # 计算损失
        loss = criterion(predicted_noise, noise_batch)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # 打印进度
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.6f}, Avg Loss: {avg_loss:.6f}")
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1} 完成, 平均损失: {avg_loss:.6f}")
    
    return avg_loss

def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    print("开始验证...")
    
    with torch.no_grad():
        for batch_idx, (lr_images, hr_images) in enumerate(dataloader):
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            
            batch_size = lr_images.size(0)
            
            # 使用固定的时间步长进行验证
            timesteps = torch.full((batch_size,), 500.0).to(device)
            
            # 添加噪声
            noisy_hr_list = []
            noise_list = []
            
            for i in range(batch_size):
                noisy_hr, noise = add_noise(hr_images[i:i+1], timesteps[i].item())
                noisy_hr_list.append(noisy_hr)
                noise_list.append(noise)
            
            noisy_hr_batch = torch.cat(noisy_hr_list, dim=0)
            noise_batch = torch.cat(noise_list, dim=0)
            
            # 前向传播
            predicted_noise = model.forward_with_noisy_hr(lr_images, noisy_hr_batch, timesteps)
            
            # 计算损失
            loss = criterion(predicted_noise, noise_batch)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    print(f"验证完成, 平均损失: {avg_loss:.6f}")
    
    return avg_loss

def main():
    """主训练函数"""
    print("🚀 开始扩散模型超分辨率训练")
    print("=" * 60)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 训练配置
    config = {
        'scale_factor': 4,
        'num_timesteps': 1000,
        'unet_channels': [32, 64],  # 适中的通道数
        'attention_resolutions': [],  # 禁用注意力以节省内存
        'num_res_blocks': 1,
        'dropout': 0.0
    }
    
    # 训练参数
    epochs = 3  # 减少epoch数以快速测试
    batch_size = 2  # 小批次大小以适应GPU内存
    learning_rate = 1e-4
    patch_size = 32  # 小patch以节省内存
    
    print(f"训练配置:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Patch Size: {patch_size}")
    
    # 创建数据集
    print("\n📁 加载数据集...")
    train_dataset = SimpleSRDataset(
        'data/split_sample/train/lr',
        'data/split_sample/train/hr',
        patch_size=patch_size
    )
    
    val_dataset = SimpleSRDataset(
        'data/split_sample/val/lr',
        'data/split_sample/val/hr',
        patch_size=patch_size
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows上设为0避免多进程问题
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    
    # 创建模型
    print("\n🧠 创建模型...")
    model = DiffusionSRModel(config).to(device)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 创建优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    criterion = nn.MSELoss()
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"优化器: AdamW")
    print(f"损失函数: MSE Loss")
    print(f"学习率调度: Cosine Annealing")
    
    # 创建结果目录
    results_dir = Path('results/diffusion_training')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    print(f"\n🎯 开始训练 ({epochs} epochs)...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"\nEpoch {epoch + 1}/{epochs} 总结:")
        print(f"  训练损失: {train_loss:.6f}")
        print(f"  验证损失: {val_loss:.6f}")
        print(f"  学习率: {current_lr:.2e}")
        print(f"  耗时: {epoch_time:.1f}秒")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }
            torch.save(checkpoint, results_dir / 'best_model.pth')
            print(f"  ✅ 保存最佳模型 (验证损失: {val_loss:.6f})")
        
        print("-" * 60)
    
    total_time = time.time() - start_time
    
    # 训练完成
    print(f"\n🎉 训练完成!")
    print(f"总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"模型保存在: {results_dir / 'best_model.pth'}")
    
    # 保存训练历史
    import json
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'total_time': total_time,
        'config': config
    }
    
    with open(results_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"训练历史保存在: {results_dir / 'training_history.json'}")
    
    return model, history

if __name__ == '__main__':
    try:
        model, history = main()
        print("\n✅ 训练成功完成!")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)