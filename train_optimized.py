#!/usr/bin/env python3
"""
优化训练脚本 - 支持梯度累积和混合精度训练

这个脚本演示了如何使用新的训练优化功能：
1. 梯度累积 - 在显存受限时模拟大batch训练
2. 混合精度训练 - 减少显存占用并加速训练
3. 梯度裁剪 - 提高训练稳定性

使用方法:
    python train_optimized.py
"""

import os
import sys
import torch
from config.experiment_config import ExperimentConfig
from trainers.sr_trainer import SRTrainer

def main():
    """主训练函数"""
    print("=" * 60)
    print("超分辨率模型优化训练")
    print("支持梯度累积和混合精度训练")
    print("=" * 60)
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name()}")
        print(f"CUDA内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("警告: CUDA不可用，将使用CPU训练")
    
    # 创建实验配置
    config_manager = ExperimentConfig()
    
    # 创建优化配置
    print("\n创建优化训练配置...")
    
    # 配置选项
    configs = [
        # 基础配置（无优化）
        config_manager.create_optimized_config(
            accumulation_steps=1,
            use_mixed_precision=False,
            gradient_clip_val=None
        ),
        # 混合精度训练
        config_manager.create_optimized_config(
            accumulation_steps=1,
            use_mixed_precision=True,
            gradient_clip_val=1.0
        ),
        # 梯度累积 + 混合精度
        config_manager.create_optimized_config(
            accumulation_steps=4,
            use_mixed_precision=True,
            gradient_clip_val=1.0
        ),
        # 大梯度累积 + 混合精度（适用于显存极度受限的情况）
        config_manager.create_optimized_config(
            accumulation_steps=8,
            use_mixed_precision=True,
            gradient_clip_val=0.5
        )
    ]
    
    # 让用户选择配置
    print("\n可用的训练配置:")
    for i, config in enumerate(configs):
        print(f"{i+1}. {config['experiment_name']}")
        print(f"   - 梯度累积步数: {config['accumulation_steps']}")
        print(f"   - 混合精度训练: {'启用' if config['use_mixed_precision'] else '禁用'}")
        print(f"   - 梯度裁剪: {config.get('gradient_clip_val', '禁用')}")
        print()
    
    while True:
        try:
            choice = int(input("请选择配置 (1-4): ")) - 1
            if 0 <= choice < len(configs):
                selected_config = configs[choice]
                break
            else:
                print("无效选择，请重新输入")
        except ValueError:
            print("请输入有效数字")
    
    print(f"\n已选择配置: {selected_config['experiment_name']}")
    
    # 检查数据目录
    data_dirs = [
        selected_config['train_lr_dir'],
        selected_config['train_hr_dir'],
        selected_config['val_lr_dir'],
        selected_config['val_hr_dir']
    ]
    
    missing_dirs = [d for d in data_dirs if not os.path.exists(d)]
    if missing_dirs:
        print("\n错误: 以下数据目录不存在:")
        for d in missing_dirs:
            print(f"  - {d}")
        print("\n请确保数据已正确准备。可以运行以下命令准备数据:")
        print("  python download_div2k.py")
        print("  python split_data.py")
        return
    
    # 创建训练器
    print("\n初始化训练器...")
    trainer = SRTrainer(selected_config)
    
    # 显示训练信息
    print("\n训练配置摘要:")
    print(f"  模型: {selected_config['model_class'].__name__}")
    print(f"  训练轮次: {selected_config['epochs']}")
    print(f"  批次大小: {selected_config['batch_size']}")
    print(f"  学习率: {selected_config['learning_rate']}")
    print(f"  设备: {selected_config['device']}")
    print(f"  放大倍数: {selected_config['upscale_factor']}")
    
    # 计算有效批次大小
    effective_batch_size = selected_config['batch_size'] * selected_config['accumulation_steps']
    print(f"  有效批次大小: {effective_batch_size}")
    
    if selected_config['use_mixed_precision']:
        print("  ✓ 混合精度训练已启用")
    if selected_config['accumulation_steps'] > 1:
        print(f"  ✓ 梯度累积已启用 (步数: {selected_config['accumulation_steps']})")
    if selected_config.get('gradient_clip_val'):
        print(f"  ✓ 梯度裁剪已启用 (阈值: {selected_config['gradient_clip_val']})")
    
    # 开始训练
    print("\n开始训练...")
    try:
        trainer.train()
        print("\n训练完成！")
        
        # 显示训练结果
        if trainer.train_losses:
            print(f"最终训练损失: {trainer.train_losses[-1]:.6f}")
        if trainer.val_losses:
            print(f"最终验证损失: {trainer.val_losses[-1]:.6f}")
        
        print(f"最佳模型保存在: {trainer.run_dir}")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def check_requirements():
    """检查训练要求"""
    # 检查PyTorch版本
    torch_version = torch.__version__
    print(f"PyTorch版本: {torch_version}")
    
    # 检查是否支持混合精度训练
    if hasattr(torch.cuda.amp, 'GradScaler'):
        print("✓ 支持混合精度训练")
    else:
        print("✗ 不支持混合精度训练，需要PyTorch 1.6+")
    
    # 检查CUDA版本
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
    
    print()

if __name__ == "__main__":
    print("检查系统要求...")
    check_requirements()
    main()