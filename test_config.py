#!/usr/bin/env python3
"""
测试配置传递
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.experiment_config import ExperimentConfig

def test_config():
    """测试配置传递"""
    
    # 快速测试配置
    selected_config = {
        "num_timesteps": 50,
        "beta_start": 0.001,
        "beta_end": 0.02,
        "noise_schedule": "linear",
        "batch_size": 1,
        "learning_rate": 1e-3,
        "num_epochs": 3,
        "accumulation_steps": 4,
        "use_mixed_precision": False,
        "gradient_clip_val": 1.0,
        "weight_decay": 1e-4,
        "num_workers": 0,
        "unet_channels": [16, 32]  # 使用最小的通道配置
    }
    
    # 创建训练配置
    config = ExperimentConfig()
    
    # 更新配置参数
    config.model_type = "diffusion"
    config.scale_factor = 4
    config.in_channels = 3
    config.out_channels = 3
    
    # 扩散模型特有参数
    config.num_timesteps = selected_config['num_timesteps']
    config.beta_start = selected_config['beta_start']
    config.beta_end = selected_config['beta_end']
    config.noise_schedule = selected_config['noise_schedule']
    
    # 训练参数
    config.batch_size = selected_config['batch_size']
    config.learning_rate = selected_config['learning_rate']
    config.epochs = selected_config['num_epochs']
    config.accumulation_steps = selected_config['accumulation_steps']
    config.use_mixed_precision = selected_config['use_mixed_precision']
    config.gradient_clip_val = selected_config['gradient_clip_val']
    config.weight_decay = selected_config['weight_decay']
    config.num_workers = selected_config['num_workers']
    
    # 模型架构参数
    if 'unet_channels' in selected_config:
        config.unet_channels = selected_config['unet_channels']
    
    print("配置测试:")
    print(f"- unet_channels: {getattr(config, 'unet_channels', '未设置')}")
    print(f"- config.get('unet_channels'): {config.get('unet_channels', '默认值')}")
    
    # 测试训练器配置传递
    from trainers.diffusion_trainer import DiffusionTrainer
    
    print("\n创建训练器...")
    try:
        trainer = DiffusionTrainer(config)
        print("✓ 训练器创建成功")
    except Exception as e:
        print(f"✗ 训练器创建失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_config()