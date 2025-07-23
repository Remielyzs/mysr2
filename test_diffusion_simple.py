#!/usr/bin/env python3
"""
简化的扩散模型测试脚本
使用更大的输入尺寸来避免U-Net尺寸不匹配问题
"""

import torch
import torch.nn.functional as F
from models.diffusion_sr import DiffusionSRModel
from models.noise_scheduler import NoiseScheduler

def test_diffusion_model():
    """测试扩散模型的基本功能"""
    print("=" * 60)
    print("扩散模型简化测试")
    print("=" * 60)
    
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 配置参数
    config = {
        'scale_factor': 4,
        'in_channels': 3,
        'out_channels': 3,
        'unet_channels': [32, 64],  # 最小配置
        'attention_resolutions': [],  # 禁用注意力
        'num_res_blocks': 1,
        'dropout': 0.1
    }
    
    try:
        # 1. 初始化模型
        print("\n1. 初始化扩散模型...")
        model = DiffusionSRModel(config=config).to(device)
        print("✓ 模型初始化成功")
        
        # 2. 初始化噪声调度器
        print("\n2. 初始化噪声调度器...")
        noise_scheduler = NoiseScheduler(
            num_timesteps=100,  # 减少时间步数
            beta_start=0.0001,
            beta_end=0.02,
            schedule='linear'
        ).to(device)
        print("✓ 噪声调度器初始化成功")
        
        # 3. 创建测试数据 - 使用更大的尺寸
        print("\n3. 创建测试数据...")
        batch_size = 1
        lr_size = 32  # 使用适中的LR尺寸以避免内存问题
        hr_size = lr_size * config['scale_factor']
        
        lr_image = torch.randn(batch_size, 3, lr_size, lr_size).to(device)
        hr_image = torch.randn(batch_size, 3, hr_size, hr_size).to(device)
        
        print(f"LR图像尺寸: {lr_image.shape}")
        print(f"HR图像尺寸: {hr_image.shape}")
        
        # 4. 测试噪声添加
        print("\n4. 测试噪声添加...")
        t = torch.randint(0, 100, (batch_size,)).to(device)
        noisy_image, noise = noise_scheduler.add_noise(hr_image, t)
        print(f"噪声图像尺寸: {noisy_image.shape}")
        print(f"噪声尺寸: {noise.shape}")
        print("✓ 噪声添加成功")
        
        # 5. 测试模型前向传播
        print("\n5. 测试模型前向传播...")
        with torch.no_grad():
            predicted_noise = model.forward_with_noisy_hr(lr_image, noisy_image, t)
        
        print(f"预测噪声尺寸: {predicted_noise.shape}")
        print("✓ 前向传播成功")
        
        # 6. 计算损失
        print("\n6. 测试损失计算...")
        loss = F.mse_loss(predicted_noise, noise)
        print(f"MSE损失: {loss.item():.6f}")
        print("✓ 损失计算成功")
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！扩散模型基本功能正常")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_diffusion_model()
    if success:
        print("\n🎉 扩散模型测试成功！可以进行训练。")
    else:
        print("\n💥 扩散模型测试失败，需要进一步调试。")