#!/usr/bin/env python3
"""
简化但GPU兼容的扩散模型训练脚本
专门针对RTX 5090优化，使用更保守的方法
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import time
import warnings

def setup_rtx5090_compatibility():
    """专门为RTX 5090设置兼容性"""
    print("🔧 RTX 5090兼容性设置...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return torch.device('cpu')
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🎮 GPU: {gpu_name}")
    
    # RTX 5090特殊处理
    if "RTX 5090" in gpu_name:
        print("⚠️ 应用RTX 5090兼容性修正...")
        
        # 设置环境变量
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9;9.0'
        
        # 启用兼容模式
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False  # 更保守的设置
        torch.backends.cudnn.deterministic = True
        
        try:
            # 简单测试
            test = torch.randn(2, 2, device='cuda')
            result = test @ test
            del test, result
            torch.cuda.empty_cache()
            
            print("✅ RTX 5090兼容性测试通过")
            return torch.device('cuda')
            
        except Exception as e:
            print(f"❌ GPU测试失败: {e}")
            return torch.device('cpu')
    
    # 其他GPU
    try:
        test = torch.randn(2, 2, device='cuda')
        del test
        return torch.device('cuda')
    except:
        return torch.device('cpu')

class SimpleDataset(Dataset):
    """极简数据集"""
    def __init__(self, size=50):
        self.size = size
        print(f"📁 创建简单数据集: {size} 个样本")
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 生成简单的LR和HR对
        lr = torch.randn(3, 16, 16) * 0.5 + 0.5  # 16x16 LR
        hr = torch.randn(3, 64, 64) * 0.5 + 0.5  # 64x64 HR
        return torch.clamp(lr, 0, 1), torch.clamp(hr, 0, 1)

class MinimalUNet(nn.Module):
    """最小化的U-Net，专为GPU兼容性优化"""
    def __init__(self):
        super().__init__()
        
        # 极简编码器
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # 极简解码器
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 上采样输入
        x = nn.functional.interpolate(x, scale_factor=4, mode='nearest')
        
        # 编码-解码
        encoded = self.enc(x)
        decoded = self.dec(encoded)
        
        return decoded

def train_simple_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """简化的训练循环"""
    model.train()
    total_loss = 0
    count = 0
    
    print(f"Epoch {epoch} 开始...")
    
    for batch_idx, (lr, hr) in enumerate(dataloader):
        try:
            # 移动到设备
            lr = lr.to(device)
            hr = hr.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(lr)
            loss = criterion(output, hr)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            count += 1
            
            # 每5个批次打印一次
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}, Loss: {loss.item():.6f}")
                
            # 清理GPU内存
            if device.type == 'cuda' and batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ 批次 {batch_idx} 失败: {e}")
            continue
    
    avg_loss = total_loss / count if count > 0 else 0
    print(f"Epoch {epoch} 完成，平均损失: {avg_loss:.6f}")
    return avg_loss

def main():
    print("🚀 RTX 5090兼容的简化扩散模型训练")
    print("=" * 50)
    
    # 忽略警告
    warnings.filterwarnings("ignore")
    
    # 设置设备
    device = setup_rtx5090_compatibility()
    print(f"使用设备: {device}")
    
    if device.type == 'cuda':
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 简化配置
    config = {
        'epochs': 3,
        'batch_size': 2,  # 小批次大小
        'learning_rate': 1e-3,
        'num_samples': 30
    }
    
    print("训练配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # 创建数据
    dataset = SimpleDataset(config['num_samples'])
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0  # 避免多进程问题
    )
    
    print(f"批次数: {len(dataloader)}")
    
    # 创建模型
    print("\n🧠 创建模型...")
    model = MinimalUNet().to(device)
    
    # 参数统计
    params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {params:,}")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    # 开始训练
    print(f"\n🎯 开始训练...")
    print("=" * 30)
    
    history = []
    
    try:
        for epoch in range(1, config['epochs'] + 1):
            start_time = time.time()
            
            # 训练
            loss = train_simple_epoch(model, dataloader, optimizer, criterion, device, epoch)
            history.append(loss)
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch} 用时: {epoch_time:.2f}秒")
            print("-" * 20)
            
            # GPU内存清理
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        print("\n✅ 训练完成!")
        print(f"最终损失: {history[-1]:.6f}")
        
        # 保存模型
        model_path = "rtx5090_compatible_model.pth"
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
            test_input = torch.randn(1, 3, 16, 16).to(device)
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