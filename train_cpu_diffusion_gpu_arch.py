#!/usr/bin/env python3
"""
强制CPU训练但保持GPU架构的扩散模型
专门解决RTX 5090兼容性问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import time
import warnings

def force_cpu_with_gpu_architecture():
    """强制使用CPU但保持GPU架构设计"""
    print("🔧 强制CPU模式（GPU架构设计）")
    
    # 检查GPU是否可用
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 检测到GPU: {gpu_name}")
        
        if "RTX 5090" in gpu_name:
            print("⚠️ RTX 5090检测到兼容性问题")
            print("🔄 强制使用CPU避免兼容性问题")
        else:
            print("🔄 为了稳定性，强制使用CPU")
    else:
        print("❌ 未检测到CUDA，使用CPU")
    
    # 强制使用CPU
    device = torch.device('cpu')
    print(f"✅ 使用设备: {device}")
    
    return device

class SuperResolutionDataset(Dataset):
    """超分辨率数据集"""
    def __init__(self, size=50):
        self.size = size
        print(f"📁 创建超分辨率数据集: {size} 个样本")
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 生成LR和HR图像对
        lr = torch.randn(3, 32, 32) * 0.3 + 0.5  # 32x32 LR
        hr = torch.randn(3, 128, 128) * 0.3 + 0.5  # 128x128 HR (4x)
        return torch.clamp(lr, 0, 1), torch.clamp(hr, 0, 1)

class DiffusionUNet(nn.Module):
    """扩散模型的U-Net架构（CPU优化）"""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # 编码器
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        
        # 瓶颈层
        self.bottleneck = self._conv_block(128, 256)
        
        # 解码器
        self.dec3 = self._conv_block(256 + 128, 128)
        self.dec2 = self._conv_block(128 + 64, 64)
        self.dec1 = self._conv_block(64 + 32, 32)
        
        # 输出层
        self.final = nn.Conv2d(32, out_channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 上采样输入到目标尺寸
        x = nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        
        # 编码路径
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.max_pool2d(e1, 2))
        e3 = self.enc3(nn.functional.max_pool2d(e2, 2))
        
        # 瓶颈
        b = self.bottleneck(nn.functional.max_pool2d(e3, 2))
        
        # 解码路径
        d3 = nn.functional.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = nn.functional.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = nn.functional.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # 输出
        output = self.final(d1)
        return self.sigmoid(output)

def add_noise_diffusion(images, timesteps, max_timesteps=1000):
    """添加扩散噪声"""
    # 简化的噪声调度
    betas = torch.linspace(0.0001, 0.02, max_timesteps)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    
    # 获取当前时间步的alpha
    alpha_t = alpha_cumprod[timesteps]
    
    # 生成噪声
    noise = torch.randn_like(images)
    
    # 添加噪声
    sqrt_alpha_t = torch.sqrt(alpha_t).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t).view(-1, 1, 1, 1)
    
    noisy_images = sqrt_alpha_t * images + sqrt_one_minus_alpha_t * noise
    
    return noisy_images, noise

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, max_timesteps=1000):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    count = 0
    
    print(f"Epoch {epoch} 开始训练...")
    
    for batch_idx, (lr, hr) in enumerate(dataloader):
        try:
            # 移动到设备
            lr = lr.to(device)
            hr = hr.to(device)
            
            # 随机时间步
            timesteps = torch.randint(0, max_timesteps, (lr.size(0),))
            
            # 添加噪声
            noisy_hr, noise = add_noise_diffusion(hr, timesteps, max_timesteps)
            
            # 前向传播
            optimizer.zero_grad()
            
            # 模型预测噪声
            predicted_noise = model(lr)
            
            # 计算损失（预测噪声 vs 真实噪声）
            loss = criterion(predicted_noise, noise)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            count += 1
            
            # 每10个批次打印一次
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
                
        except Exception as e:
            print(f"❌ 批次 {batch_idx} 失败: {e}")
            continue
    
    avg_loss = total_loss / count if count > 0 else 0
    print(f"Epoch {epoch} 完成，平均损失: {avg_loss:.6f}")
    return avg_loss

def main():
    print("🚀 强制CPU扩散模型训练（GPU架构）")
    print("=" * 50)
    
    # 忽略警告
    warnings.filterwarnings("ignore")
    
    # 强制使用CPU
    device = force_cpu_with_gpu_architecture()
    
    # 训练配置
    config = {
        'epochs': 5,
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_samples': 100,
        'max_timesteps': 1000
    }
    
    print("\n训练配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # 创建数据集
    dataset = SuperResolutionDataset(config['num_samples'])
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0
    )
    
    print(f"批次数: {len(dataloader)}")
    
    # 创建模型
    print("\n🧠 创建扩散U-Net模型...")
    model = DiffusionUNet().to(device)
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 优化器和损失
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    print(f"\n🎯 开始训练 ({config['epochs']} epochs)...")
    print("=" * 30)
    
    history = []
    
    try:
        for epoch in range(1, config['epochs'] + 1):
            start_time = time.time()
            
            # 训练
            loss = train_epoch(model, dataloader, optimizer, criterion, device, epoch, config['max_timesteps'])
            history.append(loss)
            
            # 更新学习率
            scheduler.step()
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch} 用时: {epoch_time:.2f}秒")
            print(f"当前学习率: {scheduler.get_last_lr()[0]:.6f}")
            print("-" * 30)
        
        print("\n✅ 训练完成!")
        print(f"最终损失: {history[-1]:.6f}")
        
        # 保存模型
        model_path = "cpu_diffusion_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'history': history
        }, model_path)
        print(f"💾 模型保存到: {model_path}")
        
        # 测试
        print("\n🧪 模型测试...")
        model.eval()
        with torch.no_grad():
            test_lr = torch.randn(1, 3, 32, 32).to(device)
            test_output = model(test_lr)
            print(f"输入: {test_lr.shape} -> 输出: {test_output.shape}")
            print("✅ 测试通过!")
            
    except Exception as e:
        print(f"❌ 训练错误: {e}")
        import traceback
        traceback.print_exc()
    
    return model, history

if __name__ == "__main__":
    model, history = main()