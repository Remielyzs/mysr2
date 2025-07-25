#!/usr/bin/env python3
"""
超保守的RTX 5090兼容训练脚本
使用最小的模型和最安全的操作
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import time
import warnings

def safe_gpu_setup():
    """最安全的GPU设置"""
    print("🔧 安全GPU设置...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，使用CPU")
        return torch.device('cpu')
    
    try:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 检测到GPU: {gpu_name}")
        
        # 设置最保守的环境
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9;9.0'
        
        # 禁用所有优化
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        
        # 简单测试
        print("🧪 GPU测试...")
        test_tensor = torch.tensor([1.0, 2.0], device='cuda')
        result = test_tensor * 2
        print(f"测试结果: {result}")
        
        # 清理
        del test_tensor, result
        torch.cuda.empty_cache()
        
        print("✅ GPU测试成功")
        return torch.device('cuda')
        
    except Exception as e:
        print(f"❌ GPU设置失败: {e}")
        print("🔄 回退到CPU")
        return torch.device('cpu')

class TinyDataset(Dataset):
    """超小数据集"""
    def __init__(self, size=10):
        self.size = size
        print(f"📁 创建微型数据集: {size} 个样本")
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 生成非常小的张量
        lr = torch.randn(1, 8, 8) * 0.1 + 0.5  # 单通道 8x8
        hr = torch.randn(1, 16, 16) * 0.1 + 0.5  # 单通道 16x16
        return torch.clamp(lr, 0, 1), torch.clamp(hr, 0, 1)

class TinyModel(nn.Module):
    """超小模型"""
    def __init__(self):
        super().__init__()
        
        # 最简单的网络
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),  # 1->8通道
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),  # 8->1通道
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 简单上采样
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.net(x)

def safe_train_step(model, lr_batch, hr_batch, optimizer, criterion, device):
    """安全的训练步骤"""
    try:
        # 移动数据
        lr_batch = lr_batch.to(device, non_blocking=False)
        hr_batch = hr_batch.to(device, non_blocking=False)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(lr_batch)
        loss = criterion(output, hr_batch)
        
        # 检查损失
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ 检测到NaN/Inf损失，跳过此批次")
            return None
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return loss.item()
        
    except Exception as e:
        print(f"❌ 训练步骤失败: {e}")
        return None

def main():
    print("🚀 超保守RTX 5090兼容训练")
    print("=" * 40)
    
    # 忽略警告
    warnings.filterwarnings("ignore")
    
    # 设置设备
    device = safe_gpu_setup()
    print(f"使用设备: {device}")
    
    if device.type == 'cuda':
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU内存: {memory_gb:.1f} GB")
    
    # 超保守配置
    config = {
        'epochs': 2,
        'batch_size': 1,  # 单个样本
        'learning_rate': 1e-4,
        'num_samples': 5  # 只有5个样本
    }
    
    print("\n训练配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # 创建数据
    dataset = TinyDataset(config['num_samples'])
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,  # 不打乱
        num_workers=0,
        pin_memory=False
    )
    
    print(f"批次数: {len(dataloader)}")
    
    # 创建模型
    print("\n🧠 创建超小模型...")
    model = TinyModel().to(device)
    
    # 参数统计
    params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {params:,}")
    
    # 优化器和损失
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])  # 使用SGD
    criterion = nn.MSELoss()
    
    print(f"\n🎯 开始超保守训练...")
    print("=" * 25)
    
    history = []
    
    try:
        for epoch in range(1, config['epochs'] + 1):
            print(f"\nEpoch {epoch}:")
            model.train()
            
            epoch_losses = []
            
            for batch_idx, (lr_batch, hr_batch) in enumerate(dataloader):
                print(f"  处理批次 {batch_idx + 1}/{len(dataloader)}...")
                
                # 安全训练步骤
                loss = safe_train_step(model, lr_batch, hr_batch, optimizer, criterion, device)
                
                if loss is not None:
                    epoch_losses.append(loss)
                    print(f"    损失: {loss:.6f}")
                else:
                    print("    批次跳过")
                
                # 频繁清理GPU内存
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # 小延迟
                time.sleep(0.1)
            
            # 计算平均损失
            if epoch_losses:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                history.append(avg_loss)
                print(f"  Epoch {epoch} 平均损失: {avg_loss:.6f}")
            else:
                print(f"  Epoch {epoch} 没有有效损失")
        
        print("\n✅ 训练完成!")
        
        if history:
            print(f"最终损失: {history[-1]:.6f}")
            
            # 保存模型
            model_path = "tiny_rtx5090_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'history': history
            }, model_path)
            print(f"💾 模型保存到: {model_path}")
        
        # 简单测试
        print("\n🧪 模型测试...")
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 1, 8, 8).to(device)
            test_output = model(test_input)
            print(f"输入: {test_input.shape} -> 输出: {test_output.shape}")
            print("✅ 测试通过!")
        
        # 最终清理
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print("🧹 GPU内存已清理")
            
    except Exception as e:
        print(f"❌ 训练错误: {e}")
        import traceback
        traceback.print_exc()
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return model, history

if __name__ == "__main__":
    model, history = main()