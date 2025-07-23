#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的扩散模型训练测试
用于诊断内存和训练问题
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.simple_unet import SimpleUNet
from models.noise_scheduler import NoiseScheduler

def create_dummy_data(num_samples=10, lr_size=32, hr_size=64):
    """创建虚拟数据用于测试"""
    lr_images = torch.randn(num_samples, 3, lr_size, lr_size)
    hr_images = torch.randn(num_samples, 3, hr_size, hr_size)
    return lr_images, hr_images

def test_simple_training():
    """测试简化的训练流程"""
    print("开始简化训练测试...")
    
    # 检查CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"当前已用内存: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
    
    # 创建虚拟数据
    print("创建测试数据...")
    lr_images, hr_images = create_dummy_data(num_samples=8, lr_size=32, hr_size=64)
    dataset = TensorDataset(lr_images, hr_images)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 创建简化模型
    print("创建简化模型...")
    
    # LR特征提取器
    lr_encoder = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    ).to(device)
    
    # 简化的U-Net
    unet = SimpleUNet(
        in_channels=35,  # LR特征(32) + HR图像(3)
        out_channels=3,
        channels=[8, 16],  # 非常小的通道数
        time_emb_dim=32
    ).to(device)
    
    # 噪声调度器
    noise_scheduler = NoiseScheduler(
        num_timesteps=20,  # 很少的时间步
        beta_start=0.001,
        beta_end=0.02,
        schedule='linear'
    )
    
    # 优化器
    optimizer = torch.optim.Adam(list(lr_encoder.parameters()) + list(unet.parameters()), lr=1e-4)
    
    # 损失函数
    criterion = nn.MSELoss()
    
    print(f"模型参数数量:")
    print(f"  LR编码器: {sum(p.numel() for p in lr_encoder.parameters()):,}")
    print(f"  U-Net: {sum(p.numel() for p in unet.parameters()):,}")
    print(f"  总计: {sum(p.numel() for p in lr_encoder.parameters()) + sum(p.numel() for p in unet.parameters()):,}")
    
    # 训练循环
    print("\n开始训练...")
    
    for epoch in range(2):
        print(f"\nEpoch {epoch + 1}/2")
        epoch_loss = 0.0
        
        for batch_idx, (lr_batch, hr_batch) in enumerate(dataloader):
            try:
                # 移动数据到设备
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)
                
                print(f"  Batch {batch_idx + 1}: LR shape {lr_batch.shape}, HR shape {hr_batch.shape}")
                
                if torch.cuda.is_available():
                    print(f"    GPU内存使用: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
                
                # 提取LR特征
                lr_features = lr_encoder(lr_batch)
                print(f"    LR特征形状: {lr_features.shape}")
                
                # 随机时间步
                timesteps = torch.randint(0, noise_scheduler.num_timesteps, (lr_batch.size(0),), device=device)
                
                # 添加噪声
                noise = torch.randn_like(hr_batch)
                noisy_hr, _ = noise_scheduler.add_noise(hr_batch, timesteps, noise)
                
                # 确保空间维度匹配
                print(f"    噪声HR形状: {noisy_hr.shape}")
                if lr_features.shape[2:] != noisy_hr.shape[2:]:
                    # 如果空间维度不匹配，调整noisy_hr的大小
                    noisy_hr = torch.nn.functional.interpolate(
                        noisy_hr, size=lr_features.shape[2:], mode='bilinear', align_corners=False
                    )
                    print(f"    调整后噪声HR形状: {noisy_hr.shape}")
                
                # 拼接输入
                unet_input = torch.cat([lr_features, noisy_hr], dim=1)
                print(f"    U-Net输入形状: {unet_input.shape}")
                
                # 前向传播
                optimizer.zero_grad()
                predicted_noise = unet(unet_input, timesteps)
                
                print(f"    预测噪声形状: {predicted_noise.shape}")
                
                # 计算损失
                loss = criterion(predicted_noise, noise)
                print(f"    损失: {loss.item():.6f}")
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if torch.cuda.is_available():
                    print(f"    训练后GPU内存: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
                
            except Exception as e:
                print(f"    错误: {e}")
                if torch.cuda.is_available():
                    print(f"    错误时GPU内存: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
                raise e
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"  平均损失: {avg_loss:.6f}")
    
    print("\n✓ 简化训练测试完成！")
    
    if torch.cuda.is_available():
        print(f"最终GPU内存使用: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
        torch.cuda.empty_cache()
        print(f"清理后GPU内存: {torch.cuda.memory_allocated() / 1e9:.3f} GB")

if __name__ == "__main__":
    test_simple_training()