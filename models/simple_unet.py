#!/usr/bin/env python3
"""
简化的U-Net实现，专门用于扩散模型
避免复杂的skip connection问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeEmbedding(nn.Module):
    """时间步长嵌入"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        
        # 添加时间嵌入
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.block2(h)
        
        return h + self.shortcut(x)

class SimpleUNet(nn.Module):
    """简化的U-Net，避免复杂的skip connection"""
    def __init__(self, in_channels=67, out_channels=3, channels=[32, 64], time_emb_dim=128):
        super().__init__()
        
        self.time_emb_dim = time_emb_dim
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # 输入层
        self.input_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        
        # 编码器
        self.down_blocks = nn.ModuleList()
        ch = channels[0]
        for next_ch in channels[1:]:
            self.down_blocks.append(nn.ModuleList([
                ResBlock(ch, ch, time_emb_dim),
                ResBlock(ch, next_ch, time_emb_dim),
                nn.Conv2d(next_ch, next_ch, 3, stride=2, padding=1)  # 下采样
            ]))
            ch = next_ch
        
        # 中间层
        self.mid_block = ResBlock(ch, ch, time_emb_dim)
        
        # 解码器
        self.up_blocks = nn.ModuleList()
        for prev_ch in reversed(channels[:-1]):
            self.up_blocks.append(nn.ModuleList([
                nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1),  # 上采样
                ResBlock(ch, ch, time_emb_dim),
                ResBlock(ch, prev_ch, time_emb_dim)
            ]))
            ch = prev_ch
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )
    
    def forward(self, x, time):
        # 时间嵌入
        time_emb = self.time_embedding(time)
        time_emb = self.time_mlp(time_emb)
        
        # 输入
        x = self.input_conv(x)
        
        # 编码器
        for res1, res2, down in self.down_blocks:
            x = res1(x, time_emb)
            x = res2(x, time_emb)
            x = down(x)
        
        # 中间层
        x = self.mid_block(x, time_emb)
        
        # 解码器
        for up, res1, res2 in self.up_blocks:
            x = up(x)
            x = res1(x, time_emb)
            x = res2(x, time_emb)
        
        # 输出
        x = self.output_conv(x)
        
        return x

if __name__ == '__main__':
    # 测试
    model = SimpleUNet()
    x = torch.randn(2, 67, 32, 32)
    t = torch.randint(0, 1000, (2,))
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {out.shape}")
    print("简化U-Net测试通过！")