#!/usr/bin/env python3
"""
改进的扩散模型超分辨率训练脚本

主要改进：
- 统一的配置管理和验证
- 结构化日志记录和监控
- 健壮的错误处理机制
- 更好的代码组织和可维护性
"""

import os
import sys
import argparse
from pathlib import Path
import traceback

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.config_manager import ConfigManager
from utils.logger import setup_logging
from trainers.diffusion_trainer import DiffusionTrainer

def create_default_config():
    """创建默认配置"""
    return {
        # 基础配置
        'model_name': 'diffusion_sr',
        'device': 'cuda',
        'results_base_dir': 'results',
        
        # 数据配置
        'data_dir': 'data/split_sample',
        'train_lr_dir': 'data/split_sample/train/lr',
        'train_hr_dir': 'data/split_sample/train/hr', 
        'val_lr_dir': 'data/split_sample/val/lr',
        'val_hr_dir': 'data/split_sample/val/hr',
        'patch_size': 64,
        'upscale_factor': 4,
        
        # 训练配置
        'epochs': 5,
        'batch_size': 4,
        'learning_rate': 1e-4,
        'weight_decay': 1e-2,
        'num_workers': 2,
        
        # 优化配置
        'use_mixed_precision': True,
        'accumulation_steps': 1,
        'gradient_clip_val': 1.0,
        
        # 扩散模型配置
        'num_timesteps': 1000,
        'beta_start': 0.001,
        'beta_end': 0.02,
        'noise_schedule': 'linear',
        'unet_channels': [32, 64, 128],  # 简化的通道配置
        'attention_resolutions': [16, 8],
        'num_res_blocks': 2,
        'dropout': 0.0,
        
        # 早停配置
        'early_stopping_patience': None,  # 禁用早停
        'early_stopping_min_delta': 0.001,
        
        # 错误处理配置
        'stop_on_error': True,
        'require_checkpoint': False,
        'console_logging': True,
    }

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='改进的扩散模型超分辨率训练',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='配置文件路径（JSON格式）'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/split_sample',
        help='数据目录路径'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='训练轮次'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='批次大小'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='学习率'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='训练设备'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='恢复训练的检查点路径'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='日志保存目录'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='仅验证配置，不执行训练'
    )
    
    return parser.parse_args()

def update_config_from_args(config, args):
    """根据命令行参数更新配置"""
    # 更新数据路径
    if args.data_dir:
        data_dir = Path(args.data_dir)
        config.update({
            'data_dir': str(data_dir),
            'train_lr_dir': str(data_dir / 'train' / 'lr'),
            'train_hr_dir': str(data_dir / 'train' / 'hr'),
            'val_lr_dir': str(data_dir / 'val' / 'lr'),
            'val_hr_dir': str(data_dir / 'val' / 'hr'),
        })
    
    # 更新训练参数
    config.update({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'device': args.device,
        'log_dir': args.log_dir,
    })
    
    # 恢复训练
    if args.resume:
        config['resume_checkpoint'] = args.resume
    
    # 调试模式
    if args.debug:
        config.update({
            'console_logging': True,
            'stop_on_error': True,
        })
    
    return config

def validate_environment():
    """验证训练环境"""
    issues = []
    
    # 检查CUDA可用性
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("CUDA不可用，将使用CPU训练（速度较慢）")
        else:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"检测到 {gpu_count} 个GPU: {gpu_name}")
    except ImportError:
        issues.append("PyTorch未安装")
        return issues
    
    # 检查内存
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.available < 4 * 1024**3:  # 4GB
            issues.append(f"可用内存较少: {memory.available / 1024**3:.1f}GB")
    except ImportError:
        pass
    
    return issues

def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 设置基础日志
        logger = setup_logging(
            log_dir=args.log_dir,
            console_output=True
        )
        
        logger.info("=== 扩散模型超分辨率训练开始 ===")
        
        # 验证环境
        env_issues = validate_environment()
        for issue in env_issues:
            logger.warning(f"环境问题: {issue}")
        
        # 创建配置
        if args.config and Path(args.config).exists():
            logger.info(f"从文件加载配置: {args.config}")
            config_manager = ConfigManager.load_from_file(args.config)
        else:
            logger.info("使用默认配置")
            default_config = create_default_config()
            config_manager = ConfigManager(default_config)
        
        # 根据命令行参数更新配置
        updated_config = update_config_from_args(config_manager.to_dict(), args)
        config_manager.update_config(updated_config)
        
        # 打印配置信息
        if args.debug:
            config_manager.print_config()
        
        # 保存配置到日志目录
        config_save_path = Path(args.log_dir) / 'training_config.json'
        config_manager.save_to_file(config_save_path)
        logger.info(f"配置已保存到: {config_save_path}")
        
        # 仅验证模式
        if args.validate_only:
            logger.info("配置验证完成，退出")
            return
        
        # 创建训练器
        logger.info("创建扩散模型训练器")
        trainer_config = config_manager.create_trainer_config('diffusion')
        trainer = DiffusionTrainer(trainer_config)
        
        # 开始训练
        logger.info("开始训练流程")
        trainer.train()
        
        # 训练完成
        logger.info("=== 训练流程完成 ===")
        
        # 生成最终报告
        report = trainer.logger.create_training_report()
        print("\n" + report)
        
    except KeyboardInterrupt:
        logger.warning("训练被用户中断")
        sys.exit(1)
        
    except Exception as e:
        logger.error("训练过程中发生严重错误", exception=e)
        
        # 在调试模式下打印完整的堆栈跟踪
        if args.debug if 'args' in locals() else False:
            traceback.print_exc()
        
        sys.exit(1)

if __name__ == '__main__':
    main()