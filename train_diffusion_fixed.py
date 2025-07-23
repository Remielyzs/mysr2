#!/usr/bin/env python3
"""
修复版扩散模型训练脚本
基于成功的简化测试脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.simple_unet import SimpleUNet
from models.noise_scheduler import NoiseScheduler
from data_utils import SRDataset

def create_lr_encoder(in_channels=3, out_channels=32):
    """创建LR编码器"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels//2, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels//2, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    )

def main():
    """主训练函数"""
    print("=" * 60)
    print("修复版扩散模型超分辨率训练")
    print("=" * 60)
    
    # 检查CUDA可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name()}")
    
    # 数据目录
    data_dir = Path("data/split_sample")
    print(f"✓ 数据目录: {data_dir}")
    
    # 训练配置
    config = {
        'num_timesteps': 50,
        'batch_size': 1,
        'learning_rate': 1e-3,
        'num_epochs': 3,
        'lr_channels': 32,
        'unet_channels': [16, 32],
        'time_emb_dim': 64
    }
    
    print(f"\n训练配置:")
    for key, value in config.items():
        print(f"- {key}: {value}")
    
    # 创建数据集
    print("\n创建数据集...")
    train_dataset = SRDataset(
        lr_dir=str(data_dir / "train" / "lr"),
        hr_dir=str(data_dir / "train" / "hr"),
        upscale_factor=2,
        lr_patch_size=64,
        mode='train'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"✓ 训练数据集大小: {len(train_dataset)}")
    
    # 创建模型
    print("\n创建模型...")
    lr_encoder = create_lr_encoder(3, config['lr_channels']).to(device)
    
    unet = SimpleUNet(
        in_channels=config['lr_channels'] + 3,  # LR特征 + 噪声HR
        out_channels=3,
        channels=config['unet_channels'],
        time_emb_dim=config['time_emb_dim']
    ).to(device)
    
    # 创建噪声调度器
    noise_scheduler = NoiseScheduler(
        num_timesteps=config['num_timesteps'],
        beta_start=0.001,
        beta_end=0.02,
        schedule='linear'
    )
    
    # 计算参数数量
    lr_params = sum(p.numel() for p in lr_encoder.parameters())
    unet_params = sum(p.numel() for p in unet.parameters())
    total_params = lr_params + unet_params
    
    print(f"✓ LR编码器参数: {lr_params:,}")
    print(f"✓ U-Net参数: {unet_params:,}")
    print(f"✓ 总参数: {total_params:,}")
    
    # 创建优化器
    all_params = list(lr_encoder.parameters()) + list(unet.parameters())
    optimizer = optim.AdamW(all_params, lr=config['learning_rate'], weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    print(f"✓ 优化器: AdamW")
    print(f"✓ 损失函数: MSELoss")
    
    # 开始训练
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    
    lr_encoder.train()
    unet.train()
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (lr_batch, hr_batch) in enumerate(train_loader):
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx + 1}: LR shape {lr_batch.shape}, HR shape {hr_batch.shape}")
                if torch.cuda.is_available():
                    print(f"    GPU内存: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
            
            try:
                # 提取LR特征
                lr_features = lr_encoder(lr_batch)
                
                # 随机时间步
                timesteps = torch.randint(0, noise_scheduler.num_timesteps, (lr_batch.size(0),), device=device)
                
                # 添加噪声
                noise = torch.randn_like(hr_batch)
                noisy_hr, _ = noise_scheduler.add_noise(hr_batch, timesteps, noise)
                
                # 确保空间维度匹配
                if lr_features.shape[2:] != noisy_hr.shape[2:]:
                    noisy_hr = torch.nn.functional.interpolate(
                        noisy_hr, size=lr_features.shape[2:], mode='bilinear', align_corners=False
                    )
                
                # 拼接输入
                unet_input = torch.cat([lr_features, noisy_hr], dim=1)
                
                # 前向传播
                optimizer.zero_grad()
                predicted_noise = unet(unet_input, timesteps)
                
                # 计算损失
                loss = criterion(predicted_noise, noise)
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 100 == 0:
                    print(f"    损失: {loss.item():.6f}")
                    if torch.cuda.is_available():
                        print(f"    训练后GPU内存: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
                
            except Exception as e:
                print(f"    错误: {e}")
                if torch.cuda.is_available():
                    print(f"    错误时GPU内存: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
                continue
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"  平均损失: {avg_loss:.6f}")
        
        # 保存检查点
        if (epoch + 1) % 1 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'lr_encoder_state_dict': lr_encoder.state_dict(),
                'unet_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch + 1}.pth')
            print(f"  ✓ 保存检查点: checkpoint_epoch_{epoch + 1}.pth")
    
    print("\n" + "=" * 60)
    print("✓ 训练完成！")
    print("=" * 60)
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"最终GPU内存: {torch.cuda.memory_allocated() / 1e9:.3f} GB")

if __name__ == "__main__":
    main()