#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩散模型超分辨率训练脚本

这个脚本演示了如何使用DiffusionTrainer训练扩散模型进行图像超分辨率任务。
扩散模型通过学习逐步去噪的过程来生成高质量的超分辨率图像。
"""

import os
import sys
import torch
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from trainers.diffusion_trainer import DiffusionTrainer
from config.experiment_config import ExperimentConfig

def main():
    """
    主训练函数
    """
    print("=" * 60)
    print("扩散模型超分辨率训练")
    print("=" * 60)
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"✓ CUDA可用，设备数量: {torch.cuda.device_count()}")
        print(f"✓ 当前设备: {torch.cuda.get_device_name()}")
        device = 'cuda'
    else:
        print("⚠ CUDA不可用，使用CPU训练")
        device = 'cpu'
    
    # 检查数据目录
    data_dir = project_root / "data"
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        print("请确保数据目录存在并包含训练数据")
        return
    
    print(f"✓ 数据目录: {data_dir}")
    
    # 配置选项
    configs = {
        "基础配置": {
            "num_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "noise_schedule": "linear",
            "batch_size": 4,
            "learning_rate": 1e-4,
            "num_epochs": 100,
            "accumulation_steps": 1,
            "use_mixed_precision": False,
            "gradient_clip_val": 1.0,
             "weight_decay": 1e-4,
              "num_workers": 0  # 避免CUDA多进程问题
        },
        "优化配置": {
            "num_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "noise_schedule": "cosine",
            "batch_size": 2,  # 扩散模型内存需求较大
            "learning_rate": 1e-4,
            "num_epochs": 200,
            "accumulation_steps": 4,  # 通过梯度累积模拟更大的batch size
            "use_mixed_precision": True,
            "gradient_clip_val": 1.0,
            "weight_decay": 1e-4,
            "num_workers": 0  # 避免CUDA多进程问题
        },
        "快速测试": {
            "num_timesteps": 50,  # 减少时间步长用于快速测试
            "beta_start": 0.001,
            "beta_end": 0.02,
            "noise_schedule": "linear",
            "batch_size": 1,
            "learning_rate": 1e-3,
            "num_epochs": 3,
            "accumulation_steps": 4,
            "use_mixed_precision": False,  # 禁用混合精度避免问题
            "gradient_clip_val": 1.0,
            "weight_decay": 1e-4,
            "num_workers": 0,
            "unet_channels": [16, 32]  # 使用最小的通道配置
        }
    }
    
    # 自动选择快速测试配置
    config_name = "快速测试"
    selected_config = configs[config_name]
    print("\n🚀 自动选择快速测试配置进行训练")
    
    # 创建训练配置
    config = ExperimentConfig()
    
    # 更新配置参数
    config.model_type = "diffusion"
    config.scale_factor = 4
    config.in_channels = 3
    config.out_channels = 3
    config.data_dir = str(data_dir)
    config.device = device
    
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
    
    # 设置patch_size以减少内存使用
    config.patch_size = 64  # 使用较小的patch size以减少内存使用
    
    print(f"\n训练配置详情:")
    print(f"- 模型类型: 扩散模型")
    print(f"- 放大倍数: {config.scale_factor}x")
    print(f"- 输入通道: {config.in_channels}")
    print(f"- 输出通道: {config.out_channels}")
    print(f"- 扩散时间步长: {config.num_timesteps}")
    print(f"- 噪声调度: {config.noise_schedule}")
    print(f"- Beta范围: [{config.beta_start}, {config.beta_end}]")
    print(f"- 批次大小: {config.batch_size}")
    print(f"- 学习率: {config.learning_rate}")
    print(f"- 训练轮数: {config.epochs}")
    print(f"- 梯度累积步数: {config.accumulation_steps}")
    print(f"- 混合精度训练: {config.use_mixed_precision}")
    print(f"- 梯度裁剪: {config.gradient_clip_val}")
    print(f"- 权重衰减: {config.weight_decay}")
    print(f"- 数据加载器工作进程: {config.num_workers}")
    print(f"- 设备: {config.device}")
    
    try:
        # 初始化扩散训练器
        print("\n初始化扩散训练器...")
        trainer = DiffusionTrainer(config)
        
        print("\n模型架构信息:")
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"- 总参数量: {total_params:,}")
        print(f"- 可训练参数量: {trainable_params:,}")
        
        # 显示扩散模型特有信息
        print(f"\n扩散模型信息:")
        print(f"- 噪声调度类型: {trainer.noise_schedule}")
        print(f"- 时间步长范围: [0, {trainer.num_timesteps-1}]")
        print(f"- Beta参数范围: [{trainer.beta_start}, {trainer.beta_end}]")
        print(f"- 当前Beta值范围: [{trainer.betas.min():.6f}, {trainer.betas.max():.6f}]")
        
        # 开始训练
        print("\n" + "="*60)
        print("开始扩散模型训练")
        print("="*60)
        
        trainer.train()
        
        print("\n" + "="*60)
        print("训练完成！")
        print("="*60)
        
        # 显示训练结果
        results_dir = Path(trainer.results_dir)
        print(f"\n训练结果保存在: {results_dir}")
        print(f"- 模型检查点: {results_dir / 'checkpoints'}")
        print(f"- 训练日志: {results_dir / 'logs'}")
        
        # 扩散模型的特殊说明
        print("\n扩散模型训练说明:")
        print("- 扩散模型通过学习逐步去噪过程来生成高质量图像")
        print("- 训练过程中模型学习预测每个时间步的噪声")
        print("- 推理时通过多步去噪生成最终的超分辨率图像")
        print("- 相比传统方法，扩散模型能生成更多样化和高质量的结果")
        
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n\n训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)