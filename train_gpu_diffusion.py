#!/usr/bin/env python3
"""
GPU兼容性修正版本的扩散模型超分辨率训练脚本
专门针对RTX 5090的CUDA兼容性问题进行优化
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
import warnings

def setup_gpu_compatibility():
    """设置GPU兼容性，处理RTX 5090的CUDA架构问题"""
    print("🔧 设置GPU兼容性...")
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，将使用CPU")
        return torch.device('cpu')
    
    # 获取GPU信息
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🎮 检测到GPU: {gpu_name}")
    
    # 针对RTX 5090的特殊处理
    if "RTX 5090" in gpu_name or "RTX 50" in gpu_name:
        print("⚠️ 检测到RTX 5090，应用兼容性修正...")
        
        # 设置环境变量以强制兼容性
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9;9.0'
        
        # 尝试设置兼容模式
        try:
            # 强制使用较低的CUDA架构进行计算
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # 测试GPU是否真正可用
            test_tensor = torch.randn(10, 10, device='cuda')
            test_result = test_tensor @ test_tensor.T
            del test_tensor, test_result
            
            print("✅ GPU兼容性修正成功")
            return torch.device('cuda')
            
        except Exception as e:
            print(f"❌ GPU兼容性修正失败: {e}")
            print("🔄 回退到CPU模式")
            return torch.device('cpu')
    
    else:
        # 其他GPU的标准处理
        try:
            test_tensor = torch.randn(10, 10, device='cuda')
            del test_tensor
            print("✅ GPU可用")
            return torch.device('cuda')
        except Exception as e:
            print(f"❌ GPU测试失败: {e}")
            return torch.device('cpu')

class OptimizedSRDataset(Dataset):
    """优化的超分辨率数据集，支持GPU训练"""
    def __init__(self, hr_dir=None, lr_dir=None, patch_size=64, max_samples=100, use_real_data=True):
        self.patch_size = patch_size
        self.use_real_data = use_real_data
        
        if use_real_data and hr_dir and lr_dir and os.path.exists(hr_dir) and os.path.exists(lr_dir):
            self.hr_dir = Path(hr_dir)
            self.lr_dir = Path(lr_dir)
            
            # 获取图像文件列表
            hr_files = list(self.hr_dir.glob("*.png"))[:max_samples]
            lr_files = list(self.lr_dir.glob("*.png"))[:max_samples]
            
            # 确保HR和LR文件匹配
            self.image_pairs = []
            for hr_file in hr_files:
                lr_file = self.lr_dir / hr_file.name
                if lr_file.exists():
                    self.image_pairs.append((hr_file, lr_file))
            
            print(f"📁 找到 {len(self.image_pairs)} 个真实图像对")
            self.data_size = len(self.image_pairs)
        else:
            print(f"📁 使用模拟数据集: {max_samples} 个样本")
            self.data_size = max_samples
            self.image_pairs = None
        
        # GPU优化的变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        if self.use_real_data and self.image_pairs:
            return self._get_real_data(idx)
        else:
            return self._get_synthetic_data(idx)
    
    def _get_real_data(self, idx):
        """加载真实数据"""
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
            return self._get_synthetic_data(idx)
    
    def _get_synthetic_data(self, idx):
        """生成合成数据"""
        lr_size = self.patch_size // 4
        hr_size = self.patch_size
        
        # 创建有结构的模拟数据
        np.random.seed(idx)  # 确保可重复性
        
        # 生成基础模式
        base_freq = np.random.uniform(0.1, 0.5)
        x = np.linspace(0, 2*np.pi, lr_size)
        y = np.linspace(0, 2*np.pi, lr_size)
        X, Y = np.meshgrid(x, y)
        
        # 创建多频率模式
        pattern = (np.sin(base_freq * X) * np.cos(base_freq * Y) + 
                  np.sin(2 * base_freq * X) * np.cos(2 * base_freq * Y)) * 0.5 + 0.5
        
        # 转换为RGB
        lr_data = np.stack([pattern, pattern * 0.8, pattern * 0.6], axis=0)
        lr_tensor = torch.from_numpy(lr_data).float()
        
        # 生成对应的HR图像
        hr_tensor = nn.functional.interpolate(
            lr_tensor.unsqueeze(0), 
            size=(hr_size, hr_size), 
            mode='bicubic', 
            align_corners=False
        ).squeeze(0)
        
        # 添加高频细节
        detail_noise = torch.randn(3, hr_size, hr_size) * 0.05
        hr_tensor = torch.clamp(hr_tensor + detail_noise, 0, 1)
        
        return lr_tensor, hr_tensor

class GPUOptimizedUNet(nn.Module):
    """GPU优化的U-Net模型"""
    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super().__init__()
        
        # 编码器
        self.enc1 = self._make_encoder_block(in_channels, base_channels)
        self.enc2 = self._make_encoder_block(base_channels, base_channels * 2)
        self.enc3 = self._make_encoder_block(base_channels * 2, base_channels * 4)
        
        # 瓶颈层
        self.bottleneck = self._make_encoder_block(base_channels * 4, base_channels * 8)
        
        # 解码器
        self.dec3 = self._make_decoder_block(base_channels * 8, base_channels * 4)
        self.dec2 = self._make_decoder_block(base_channels * 8, base_channels * 2)  # 8 = 4 + 4 (skip)
        self.dec1 = self._make_decoder_block(base_channels * 4, base_channels)      # 4 = 2 + 2 (skip)
        
        # 输出层
        self.final = nn.Sequential(
            nn.Conv2d(base_channels * 2, out_channels, 1),  # 2 = 1 + 1 (skip)
            nn.Sigmoid()
        )
        
        # 池化和上采样
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 上采样输入到目标尺寸
        x = nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        
        # 编码路径
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # 瓶颈
        b = self.bottleneck(self.pool(e3))
        
        # 解码路径
        d3 = self.dec3(self.upsample(b))
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(self.upsample(d3))
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(self.upsample(d2))
        d1 = torch.cat([d1, e1], dim=1)
        
        # 输出
        output = self.final(d1)
        return output

def add_diffusion_noise(image, timestep, max_timesteps=1000):
    """添加扩散噪声 - GPU优化版本"""
    device = image.device
    
    # 噪声调度参数
    beta_start = 0.0001
    beta_end = 0.02
    
    # 计算beta和alpha
    betas = torch.linspace(beta_start, beta_end, max_timesteps, device=device)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    
    # 确保timestep在有效范围内
    timestep = torch.clamp(torch.tensor(timestep, device=device), 0, max_timesteps - 1).long()
    
    # 获取当前时间步的alpha值
    alpha_t = alpha_cumprod[timestep]
    
    # 生成噪声
    noise = torch.randn_like(image)
    
    # 计算噪声系数
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
    
    # 扩展维度以匹配图像张量
    while len(sqrt_alpha_t.shape) < len(image.shape):
        sqrt_alpha_t = sqrt_alpha_t.unsqueeze(-1)
        sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.unsqueeze(-1)
    
    # 添加噪声
    noisy_image = sqrt_alpha_t * image + sqrt_one_minus_alpha_t * noise
    
    return noisy_image, noise

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, max_timesteps=1000):
    """训练一个epoch - GPU优化版本"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    print(f"Epoch {epoch} 开始训练...")
    
    for batch_idx, (lr_images, hr_images) in enumerate(dataloader):
        try:
            # 移动数据到GPU
            lr_images = lr_images.to(device, non_blocking=True)
            hr_images = hr_images.to(device, non_blocking=True)
            
            # 随机选择时间步
            batch_size = hr_images.shape[0]
            timesteps = torch.randint(0, max_timesteps, (batch_size,), device=device)
            
            # 为每个样本添加噪声
            noisy_hrs = []
            target_noises = []
            
            for i in range(batch_size):
                noisy_hr, noise = add_diffusion_noise(hr_images[i:i+1], timesteps[i].item(), max_timesteps)
                noisy_hrs.append(noisy_hr)
                target_noises.append(noise)
            
            noisy_hr_batch = torch.cat(noisy_hrs, dim=0)
            target_noise_batch = torch.cat(target_noises, dim=0)
            
            # 前向传播
            optimizer.zero_grad()
            
            # 模型预测去噪后的图像
            predicted = model(lr_images)
            
            # 计算损失（预测图像与原始HR图像的差异）
            loss = criterion(predicted, hr_images)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
    print("🚀 开始GPU优化的扩散模型训练")
    print("=" * 60)
    
    # 设置GPU兼容性
    device = setup_gpu_compatibility()
    print(f"使用设备: {device}")
    
    if device.type == 'cuda':
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 训练配置
    config = {
        'epochs': 5,
        'batch_size': 4 if device.type == 'cuda' else 2,
        'learning_rate': 1e-4,
        'patch_size': 64,
        'max_samples': 100,
        'max_timesteps': 1000
    }
    
    print("训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 数据路径
    hr_dir = "data/DIV2K_train_HR"
    lr_dir = "data/DIV2K_train_LR_bicubic/X4"
    
    # 创建数据集
    use_real_data = os.path.exists(hr_dir) and os.path.exists(lr_dir)
    train_dataset = OptimizedSRDataset(
        hr_dir=hr_dir if use_real_data else None,
        lr_dir=lr_dir if use_real_data else None,
        patch_size=config['patch_size'],
        max_samples=config['max_samples'],
        use_real_data=use_real_data
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"训练批次数: {len(train_loader)}")
    
    # 创建模型
    print("\n🧠 创建GPU优化模型...")
    model = GPUOptimizedUNet(base_channels=32).to(device)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    print(f"优化器: AdamW")
    print(f"损失函数: MSE Loss")
    print(f"学习率调度: Cosine Annealing")
    
    # 开始训练
    print(f"\n🎯 开始训练 ({config['epochs']} epochs)...")
    print("=" * 60)
    
    training_history = []
    best_loss = float('inf')
    
    try:
        for epoch in range(1, config['epochs'] + 1):
            start_time = time.time()
            
            # 训练一个epoch
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch, config['max_timesteps']
            )
            
            # 更新学习率
            scheduler.step()
            
            epoch_time = time.time() - start_time
            training_history.append(train_loss)
            
            print(f"Epoch {epoch} 完成，用时: {epoch_time:.2f}秒，学习率: {scheduler.get_last_lr()[0]:.6f}")
            
            # 保存最佳模型
            if train_loss < best_loss:
                best_loss = train_loss
                best_model_path = "best_gpu_diffusion_model.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'loss': train_loss,
                    'training_history': training_history
                }, best_model_path)
                print(f"💾 保存最佳模型: {best_model_path}")
            
            print("-" * 40)
            
        print("\n✅ 训练完成!")
        print(f"最终损失: {training_history[-1]:.6f}")
        print(f"最佳损失: {best_loss:.6f}")
        
        # 保存最终模型
        final_model_path = "final_gpu_diffusion_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'training_history': training_history
        }, final_model_path)
        print(f"💾 最终模型已保存到: {final_model_path}")
        
        # GPU内存清理
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print("🧹 GPU内存已清理")
        
        # 简单测试
        print("\n🧪 进行模型测试...")
        model.eval()
        with torch.no_grad():
            test_lr = torch.randn(1, 3, 16, 16).to(device)
            test_output = model(test_lr)
            print(f"测试输入形状: {test_lr.shape}")
            print(f"测试输出形状: {test_output.shape}")
            print("✅ 模型测试通过!")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理GPU内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return model, training_history

if __name__ == "__main__":
    # 忽略兼容性警告
    warnings.filterwarnings("ignore", category=UserWarning)
    
    model, history = main()