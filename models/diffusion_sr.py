import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .simple_unet import SimpleUNet

class TimeEmbedding(nn.Module):
    """时间步长嵌入层"""
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
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0):
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
            nn.Dropout(dropout),
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

class AttentionBlock(nn.Module):
    """自注意力块"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(8, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.to_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.group_norm(x)
        
        qkv = self.to_qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.view(b, c, h * w).transpose(1, 2)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w).transpose(1, 2)
        
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(attn, v)
        out = out.transpose(1, 2).view(b, c, h, w)
        out = self.to_out(out)
        
        return x + out

class UNet(nn.Module):
    """U-Net架构用于扩散模型"""
    def __init__(self, in_channels=3, out_channels=3, channels=[64, 128, 256, 512], 
                 attention_resolutions=[16, 8], num_res_blocks=2, dropout=0.0):
        super().__init__()
        
        # 时间嵌入
        time_emb_dim = channels[0] * 4
        self.time_embedding = TimeEmbedding(channels[0])
        self.time_mlp = nn.Sequential(
            nn.Linear(channels[0], time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 初始卷积
        self.init_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        self.down_attentions = nn.ModuleList()
        
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(in_ch, out_ch, time_emb_dim, dropout))
                in_ch = out_ch
            
            self.down_blocks.append(blocks)
            
            # 添加注意力层
            if 2**(len(channels)-i-1) in attention_resolutions:
                self.down_attentions.append(AttentionBlock(out_ch))
            else:
                self.down_attentions.append(nn.Identity())
            
            # 下采样（除了最后一层）
            if i < len(channels) - 1:
                self.down_blocks.append(nn.ModuleList([nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)]))
                self.down_attentions.append(nn.Identity())
        
        # 中间块
        mid_ch = channels[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_emb_dim, dropout)
        self.mid_attention = AttentionBlock(mid_ch)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_emb_dim, dropout)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        self.up_attentions = nn.ModuleList()
        
        channels_reversed = list(reversed(channels))
        for i, out_ch in enumerate(channels_reversed):
            in_ch = channels_reversed[i-1] if i > 0 else channels_reversed[0]
            
            # 上采样
            if i > 0:
                self.up_blocks.append(nn.ModuleList([nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)]))
                self.up_attentions.append(nn.Identity())
            
            # 残差块
            blocks = nn.ModuleList()
            # 跳跃连接使输入通道数翻倍
            skip_ch = out_ch if i == 0 else out_ch
            for j in range(num_res_blocks + 1):
                if j == 0 and i > 0:
                    blocks.append(ResBlock(out_ch + skip_ch, out_ch, time_emb_dim, dropout))
                else:
                    blocks.append(ResBlock(out_ch, out_ch, time_emb_dim, dropout))
            
            self.up_blocks.append(blocks)
            
            # 添加注意力层
            if 2**(i+1) in attention_resolutions:
                self.up_attentions.append(AttentionBlock(out_ch))
            else:
                self.up_attentions.append(nn.Identity())
        
        # 输出层
        self.out_norm = nn.GroupNorm(8, channels[0])
        self.out_conv = nn.Conv2d(channels[0], out_channels, 3, padding=1)
    
    def forward(self, x, time):
        # 时间嵌入
        time_emb = self.time_embedding(time)
        time_emb = self.time_mlp(time_emb)
        
        # 初始卷积
        x = self.init_conv(x)
        
        # 保存跳跃连接
        skip_connections = [x]
        
        # 下采样
        for i, (blocks, attention) in enumerate(zip(self.down_blocks, self.down_attentions)):
            if len(blocks) == 1 and isinstance(blocks[0], nn.Conv2d):
                # 下采样层
                x = blocks[0](x)
            else:
                # 残差块
                for block in blocks:
                    x = block(x, time_emb)
                x = attention(x)
                skip_connections.append(x)
        
        # 中间块
        x = self.mid_block1(x, time_emb)
        x = self.mid_attention(x)
        x = self.mid_block2(x, time_emb)
        
        # 上采样
        skip_idx = len(skip_connections) - 1
        for i, (blocks, attention) in enumerate(zip(self.up_blocks, self.up_attentions)):
            if len(blocks) == 1 and isinstance(blocks[0], nn.ConvTranspose2d):
                # 上采样层
                x = blocks[0](x)
            else:
                # 残差块
                if i > 0 and skip_idx >= 0:
                    # 添加跳跃连接
                    skip = skip_connections[skip_idx]
                    skip_idx -= 1
                    x = torch.cat([x, skip], dim=1)
                
                for j, block in enumerate(blocks):
                    x = block(x, time_emb)
                x = attention(x)
        
        # 输出
        x = self.out_norm(x)
        x = F.silu(x)
        x = self.out_conv(x)
        
        return x

class DiffusionSRModel(nn.Module):
    def __init__(self, config):
        super(DiffusionSRModel, self).__init__()
        self.config = config
        self.scale_factor = config.get('scale_factor', 4)
        
        # 低分辨率图像编码器
        self.lr_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1)
        )
        
        # 简化的U-Net用于噪声预测 - 使用最小配置以适应小尺寸输入
        self.unet = SimpleUNet(
            in_channels=67,  # LR特征(64) + 噪声HR图像(3)
            out_channels=3,  # 预测的噪声
            channels=config.get('unet_channels', [16, 32]),  # 进一步减少通道数
            time_emb_dim=64
        )
        
        print("Initializing Diffusion-based Super-Resolution Model")
        print(f"Scale factor: {self.scale_factor}")
        print(f"U-Net channels: {config.get('unet_channels', [32, 64])}")
        print(f"Attention resolutions: {config.get('attention_resolutions', [])} (disabled for memory efficiency)")
        print(f"Num res blocks: {config.get('num_res_blocks', 1)}")

    def forward(self, lr_image, text_description=None, noise_level=None):
        """
        扩散模型的前向传播逻辑

        参数:
            lr_image (torch.Tensor): 低分辨率图像张量 (B, C_in, H_in, W_in)
            text_description (torch.Tensor, optional): 文本描述的编码表示 (B, SeqLen, EmbDim)。默认为None。
            noise_level (torch.Tensor, optional): 噪声水平或时间步长 (B,)。默认为None。

        返回:
            torch.Tensor: 预测的噪声 (B, C_out, H_out, W_out)
        """
        if noise_level is None:
            raise ValueError("扩散模型需要提供noise_level（时间步长）")
        
        b, c, h, w = lr_image.shape
        h_out = h * self.scale_factor
        w_out = w * self.scale_factor
        
        # 1. 编码低分辨率图像
        lr_features = self.lr_encoder(lr_image)
        
        # 2. 上采样LR特征到HR尺寸
        lr_features_upsampled = F.interpolate(
            lr_features, 
            size=(h_out, w_out), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 3. 使用U-Net预测噪声
        # 注意：在训练时，这里应该接收带噪声的HR图像作为输入
        # 但为了简化，我们直接使用上采样的LR特征
        predicted_noise = self.unet(lr_features_upsampled, noise_level)
        
        return predicted_noise
    
    def forward_with_noisy_hr(self, lr_image, noisy_hr_image, noise_level):
        """
        训练时的前向传播，接收带噪声的HR图像
        
        参数:
            lr_image (torch.Tensor): 低分辨率图像 (B, C, H, W)
            noisy_hr_image (torch.Tensor): 带噪声的高分辨率图像 (B, C, H*scale, W*scale)
            noise_level (torch.Tensor): 时间步长 (B,)
            
        返回:
            torch.Tensor: 预测的噪声
        """
        # 1. 编码低分辨率图像
        lr_features = self.lr_encoder(lr_image)
        
        # 2. 上采样LR特征到HR尺寸
        _, _, h_hr, w_hr = noisy_hr_image.shape
        lr_features_upsampled = F.interpolate(
            lr_features, 
            size=(h_hr, w_hr), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 3. 将LR特征与带噪声的HR图像结合
        # 由于通道数不匹配(LR特征64通道，HR图像3通道)，使用连接而不是相加
        combined_input = torch.cat([lr_features_upsampled, noisy_hr_image], dim=1)
        
        # 4. 使用U-Net预测噪声
        predicted_noise = self.unet(combined_input, noise_level)
        
        return predicted_noise

if __name__ == '__main__':
    dummy_config = {
        'scale_factor': 4,
        # 其他扩散模型相关配置
        'num_timesteps': 1000,
        'unet_channels': (64, 128, 256)
    }
    model = DiffusionSRModel(config=dummy_config)
    
    dummy_lr_image = torch.randn(1, 3, 32, 32)
    print(f"Input LR image shape: {dummy_lr_image.shape}")
    
    # 模拟训练过程中的一次前向传播
    # 实际应用中，还需要提供噪声水平 t
    dummy_t = torch.randint(0, dummy_config['num_timesteps'], (1,)).float()
    predicted_output = model(dummy_lr_image, noise_level=dummy_t)
    print(f"Output (predicted noise/image) shape: {predicted_output.shape}")

    expected_h = 32 * dummy_config['scale_factor']
    expected_w = 32 * dummy_config['scale_factor']
    assert predicted_output.shape == (1, 3, expected_h, expected_w), \
        f"Output shape mismatch! Expected (1, 3, {expected_h}, {expected_w}), got {predicted_output.shape}"
    print("DiffusionSRModel basic test passed.")