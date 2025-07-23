#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练好的扩散模型
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.simple_unet import SimpleUNet
from models.noise_scheduler import NoiseScheduler

# 备用简化的U-Net模型（如果导入失败）
class BackupSimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channels=[16, 32], time_emb_dim=64):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 编码器
        self.encoder = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoder.append(nn.Sequential(
                nn.Conv2d(prev_ch, ch, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.ReLU()
            ))
            prev_ch = ch
        
        # 解码器
        self.decoder = nn.ModuleList()
        for i in range(len(channels) - 1, -1, -1):
            if i == 0:
                out_ch = out_channels
            else:
                out_ch = channels[i-1]
            
            self.decoder.append(nn.Sequential(
                nn.Conv2d(channels[i], out_ch, 3, padding=1),
                nn.ReLU() if i > 0 else nn.Identity()
            ))
    
    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.time_mlp(t)
        
        # 编码
        features = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            features.append(x)
        
        # 解码
        for i, decoder_block in enumerate(self.decoder):
            x = decoder_block(x)
        
        return x

def create_lr_encoder(in_channels=3, out_channels=32):
    """创建LR编码器"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels//2, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels//2, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    )

def load_model(checkpoint_path, device='cuda'):
    """加载训练好的模型"""
    print(f"加载模型: {checkpoint_path}")
    
    # 创建模型（需要匹配训练时的配置）
    model = SimpleUNet(
        in_channels=35,  # lr_channels(32) + 3
        out_channels=3,
        channels=[16, 32],
        time_emb_dim=64
    ).to(device)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['unet_state_dict'])
    model.eval()
    
    print(f"✓ 模型加载成功，训练轮次: {checkpoint['epoch']}")
    print(f"✓ 训练损失: {checkpoint['loss']:.6f}")
    
    return model

def test_model_inference(model, device='cuda'):
    """测试模型推理"""
    print("\n测试模型推理...")
    
    # 创建噪声调度器
    scheduler = NoiseScheduler(num_timesteps=50, beta_start=0.001, beta_end=0.02, schedule='linear')
    
    # 创建LR编码器
    lr_encoder = create_lr_encoder(3, 32).to(device)
    lr_encoder.eval()
    
    # 创建随机输入
    batch_size = 1
    height, width = 64, 64
    
    # 随机LR图像
    lr_image = torch.randn(batch_size, 3, height//2, width//2).to(device)
    
    # 随机噪声HR图像
    noisy_hr = torch.randn(batch_size, 3, height, width).to(device)
    
    # 随机时间步
    timesteps = torch.randint(0, 50, (batch_size,)).to(device)
    
    print(f"LR图像形状: {lr_image.shape}")
    print(f"噪声HR图像形状: {noisy_hr.shape}")
    print(f"时间步: {timesteps}")
    
    with torch.no_grad():
        # LR编码
        lr_features = lr_encoder(lr_image)
        print(f"LR特征形状: {lr_features.shape}")
        
        # 拼接LR特征和噪声HR
        model_input = torch.cat([lr_features, noisy_hr], dim=1)
        print(f"模型输入形状: {model_input.shape}")
        
        # 模型推理
        output = model(model_input, timesteps)
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # 检查输出是否合理
        if torch.isnan(output).any():
            print("❌ 输出包含NaN值")
        elif torch.isinf(output).any():
            print("❌ 输出包含无穷值")
        else:
            print("✓ 输出正常")
    
    return output

def main():
    print("="*60)
    print("测试训练好的扩散模型")
    print("="*60)
    
    # 设备检查
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ 使用设备: {device}")
    
    # 检查点路径
    checkpoint_path = "checkpoint_epoch_3.pth"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ 检查点文件不存在: {checkpoint_path}")
        return
    
    try:
        # 加载模型
        model = load_model(checkpoint_path, device)
        
        # 测试推理
        output = test_model_inference(model, device)
        
        print("\n" + "="*60)
        print("✓ 模型测试完成！")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()