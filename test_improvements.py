#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试代码质量和可维护性改进

这个脚本用于验证所有改进是否正常工作，包括：
1. ConfigManager 配置管理
2. TrainingLogger 日志系统
3. BaseTrainer 和 DiffusionTrainer 的改进
4. 错误处理和目录管理
"""

import os
import sys
import torch
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config_manager import ConfigManager
from utils.logger import TrainingLogger
from trainers.diffusion_trainer import DiffusionTrainer

def test_config_manager():
    """测试配置管理器"""
    print("\n=== 测试 ConfigManager ===")
    
    # 测试基本功能
    # 提供必需的配置项
    required_config = {
        'train_lr_dir': '/tmp/train/lr',
        'train_hr_dir': '/tmp/train/hr',
        'val_lr_dir': '/tmp/val/lr',
        'val_hr_dir': '/tmp/val/hr'
    }
    config = ConfigManager(required_config)
    
    # 测试默认值
    assert config.get('batch_size') == 8, "默认batch_size应该是8"
    assert config.get('learning_rate') == 1e-4, "默认learning_rate应该是1e-4"
    
    # 测试设置和获取
    config['custom_param'] = 'test_value'  # 使用字典式访问
    assert config.get('custom_param') == 'test_value', "自定义参数设置失败"
    
    # 测试验证（通过update_config方法）
    try:
        test_config = ConfigManager(required_config.copy())
        test_config.update_config({'batch_size': -1})  # 应该失败
        assert False, "负数batch_size应该被拒绝"
    except ValueError:
        pass  # 预期的错误
    
    # 测试序列化（通过直接访问_config）
    config_dict = config._config.copy()
    assert isinstance(config_dict, dict), "_config应该是字典"
    
    # 测试从字典创建（确保使用有效的配置）
    valid_config_dict = required_config.copy()
    valid_config_dict.update({'batch_size': 8, 'learning_rate': 1e-4})
    new_config = ConfigManager(valid_config_dict)
    assert new_config.get('batch_size') == 8, "从字典创建的配置应该相同"
    
    print("✓ ConfigManager 测试通过")

def test_training_logger():
    """测试训练日志器"""
    print("\n=== 测试 TrainingLogger ===")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = TrainingLogger(
            name="test_logger",
            log_dir=temp_dir,
            level="INFO"
        )
        
        # 测试基本日志
        logger.info("测试信息日志")
        logger.warning("测试警告日志")
        logger.error("测试错误日志")
        
        # 测试epoch日志
        logger.log_epoch_start(0, 10)
        logger.log_epoch_end(0, 0.5, 0.4)
        
        # 测试模型日志
        model = torch.nn.Linear(10, 1)
        logger.log_model_info(model, 11, 11)
        
        # 测试配置日志
        config = {'test': 'value'}
        logger.log_config(config)
        
        # 测试检查点日志
        logger.log_checkpoint_save("/tmp/test.pth", 0, 0.5, True)
        
        # 测试训练完成日志
        logger.log_training_complete(3600, 0.3)
        
        # 测试报告生成
        report = logger.create_training_report()
        assert isinstance(report, dict), "训练报告应该是字典"
        
        # 检查日志文件是否创建
        log_files = list(Path(temp_dir).glob("*.log"))
        assert len(log_files) > 0, "应该创建日志文件"
    
    print("✓ TrainingLogger 测试通过")

def test_diffusion_trainer_initialization():
    """测试DiffusionTrainer初始化"""
    print("\n=== 测试 DiffusionTrainer 初始化 ===")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建配置
        required_config = {
            'train_lr_dir': '/tmp/train/lr',
            'train_hr_dir': '/tmp/train/hr',
            'val_lr_dir': '/tmp/val/lr',
            'val_hr_dir': '/tmp/val/hr'
        }
        config = ConfigManager(required_config)
        config.update_config({
            'run_dir': temp_dir,
            'device': 'cpu',
            'unet_channels': [32, 64],
            'num_timesteps': 100
        })
        
        # 测试DiffusionTrainer初始化
        try:
            trainer = DiffusionTrainer(config)
            
            # 检查必要的属性是否存在
            assert hasattr(trainer, 'results_dir'), "应该有results_dir属性"
            assert hasattr(trainer, 'logs_dir'), "应该有logs_dir属性"
            assert hasattr(trainer, 'checkpoint_dir'), "应该有checkpoint_dir属性"
            assert hasattr(trainer, 'config_manager'), "应该有config_manager属性"
            assert hasattr(trainer, 'logger'), "应该有logger属性"
            
            # 检查目录是否创建
            assert os.path.exists(trainer.results_dir), "results_dir应该存在"
            assert os.path.exists(trainer.logs_dir), "logs_dir应该存在"
            assert os.path.exists(trainer.checkpoint_dir), "checkpoint_dir应该存在"
            
            print("✓ DiffusionTrainer 初始化成功")
            
        except Exception as e:
            print(f"✗ DiffusionTrainer 初始化失败: {e}")
            raise

def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    # 测试ConfigManager错误处理
    required_config = {
        'train_lr_dir': '/tmp/train/lr',
        'train_hr_dir': '/tmp/train/hr',
        'val_lr_dir': '/tmp/val/lr',
        'val_hr_dir': '/tmp/val/hr'
    }
    config = ConfigManager(required_config)
    
    # 测试类型错误
    try:
        config.update_config({'batch_size': 'invalid_type'})
        assert False, "应该抛出类型错误"
    except (ValueError, TypeError):
        pass  # 预期的错误
    
    # 测试负值错误
    try:
        config.update_config({'batch_size': -5})
        assert False, "应该拒绝负数batch_size"
    except ValueError:
        pass  # 预期的错误
    
    print("✓ 错误处理测试通过")

def test_directory_management():
    """测试目录管理"""
    print("\n=== 测试目录管理 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        required_config = {
            'train_lr_dir': '/tmp/train/lr',
            'train_hr_dir': '/tmp/train/hr',
            'val_lr_dir': '/tmp/val/lr',
            'val_hr_dir': '/tmp/val/hr'
        }
        config = ConfigManager(required_config)
        config.update_config({
            'run_dir': temp_dir,
            'device': 'cpu'
        })
        
        # 创建trainer
        trainer = DiffusionTrainer(config)
        
        # 检查所有必要的目录是否创建
        expected_dirs = [
            trainer.run_dir,
            trainer.checkpoint_dir,
            trainer.results_dir,
            trainer.logs_dir,
            trainer.images_dir
        ]
        
        for dir_path in expected_dirs:
            assert os.path.exists(dir_path), f"目录 {dir_path} 应该存在"
            assert os.path.isdir(dir_path), f"{dir_path} 应该是目录"
        
        print("✓ 目录管理测试通过")

def test_configuration_inheritance():
    """测试配置继承"""
    print("\n=== 测试配置继承 ===")
    
    # 创建基础配置（包含必需的配置项）
    base_config = {
        'batch_size': 8,
        'learning_rate': 2e-4,
        'custom_param': 'base_value',
        'train_lr_dir': '/tmp/train/lr',
        'train_hr_dir': '/tmp/train/hr',
        'val_lr_dir': '/tmp/val/lr',
        'val_hr_dir': '/tmp/val/hr'
    }
    
    # 创建继承配置
    override_config = {
        'batch_size': 16,  # 覆盖
        'new_param': 'new_value'  # 新增
    }
    
    config = ConfigManager(base_config)
    config.update_config(override_config)
    
    # 检查继承结果
    assert config.get('batch_size') == 16, "batch_size应该被覆盖"
    assert config.get('learning_rate') == 2e-4, "learning_rate应该保持不变"
    assert config.get('custom_param') == 'base_value', "custom_param应该保持不变"
    assert config.get('new_param') == 'new_value', "new_param应该被添加"
    
    print("✓ 配置继承测试通过")

def main():
    """运行所有测试"""
    print("开始测试代码质量和可维护性改进...")
    
    try:
        test_config_manager()
        test_training_logger()
        test_diffusion_trainer_initialization()
        test_error_handling()
        test_directory_management()
        test_configuration_inheritance()
        
        print("\n🎉 所有测试通过！改进实施成功。")
        print("\n改进总结:")
        print("1. ✓ ConfigManager: 统一配置管理，支持验证和继承")
        print("2. ✓ TrainingLogger: 结构化日志系统，支持多级别日志")
        print("3. ✓ BaseTrainer: 增强错误处理和目录管理")
        print("4. ✓ DiffusionTrainer: 修复AttributeError，集成新系统")
        print("5. ✓ 错误处理: 全面的异常捕获和日志记录")
        print("6. ✓ 目录管理: 自动创建和验证必要目录")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()