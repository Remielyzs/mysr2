#!/usr/bin/env python3
"""
简化的扩散模型训练测试脚本
用于验证环境和基本功能
"""

import sys
import os
import torch
import torch.nn as nn
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_environment():
    """测试训练环境"""
    print("=== 环境测试 ===")
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查PyTorch
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        device = torch.device('cuda')
    else:
        print("CUDA不可用，使用CPU")
        device = torch.device('cpu')
    
    return device

def test_data():
    """测试数据加载"""
    print("\n=== 数据测试 ===")
    
    data_dirs = [
        'data/split_sample/train/lr',
        'data/split_sample/train/hr',
        'data/split_sample/val/lr',
        'data/split_sample/val/hr'
    ]
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            count = len(os.listdir(data_dir))
            print(f"{data_dir}: {count} 个文件")
        else:
            print(f"{data_dir}: 目录不存在")
            return False
    
    return True

def test_model():
    """测试模型创建"""
    print("\n=== 模型测试 ===")
    
    try:
        from models.diffusion_sr import DiffusionSRModel
        
        config = {
            'scale_factor': 4,
            'num_timesteps': 1000,
            'unet_channels': [32, 64],  # 简化配置
            'attention_resolutions': [],
            'num_res_blocks': 1,
            'dropout': 0.0
        }
        
        model = DiffusionSRModel(config)
        print("✓ 扩散模型创建成功")
        
        # 测试前向传播
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # 创建测试输入
        lr_image = torch.randn(1, 3, 32, 32).to(device)
        noise_level = torch.randint(0, 1000, (1,)).float().to(device)
        
        with torch.no_grad():
            output = model(lr_image, noise_level=noise_level)
        
        print(f"✓ 前向传播成功: 输入 {lr_image.shape} -> 输出 {output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_training():
    """测试简单训练循环"""
    print("\n=== 简单训练测试 ===")
    
    try:
        from models.diffusion_sr import DiffusionSRModel
        import torch.optim as optim
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        config = {
            'scale_factor': 4,
            'num_timesteps': 1000,
            'unet_channels': [16, 32],  # 最小配置
            'attention_resolutions': [],
            'num_res_blocks': 1,
            'dropout': 0.0
        }
        
        model = DiffusionSRModel(config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        print("✓ 模型和优化器创建成功")
        
        # 模拟一个训练步骤
        model.train()
        
        # 创建模拟数据
        lr_image = torch.randn(2, 3, 32, 32).to(device)
        hr_image = torch.randn(2, 3, 128, 128).to(device)
        noise_level = torch.randint(0, 1000, (2,)).float().to(device)
        
        # 添加噪声到HR图像
        noise = torch.randn_like(hr_image)
        noisy_hr = hr_image + 0.1 * noise
        
        # 前向传播
        optimizer.zero_grad()
        predicted_noise = model.forward_with_noisy_hr(lr_image, noisy_hr, noise_level)
        
        # 计算损失
        loss = criterion(predicted_noise, noise)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        print(f"✓ 训练步骤成功: 损失 = {loss.item():.6f}")
        return True
        
    except Exception as e:
        print(f"✗ 训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("扩散模型训练环境测试")
    print("=" * 50)
    
    # 测试环境
    device = test_environment()
    
    # 测试数据
    data_ok = test_data()
    
    # 测试模型
    model_ok = test_model()
    
    # 测试训练
    training_ok = test_simple_training()
    
    print("\n=== 测试总结 ===")
    print(f"数据准备: {'✓' if data_ok else '✗'}")
    print(f"模型创建: {'✓' if model_ok else '✗'}")
    print(f"训练测试: {'✓' if training_ok else '✗'}")
    
    if data_ok and model_ok and training_ok:
        print("\n🎉 所有测试通过！可以开始正式训练。")
        return True
    else:
        print("\n❌ 部分测试失败，请检查环境配置。")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)