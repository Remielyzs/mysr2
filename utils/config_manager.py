#!/usr/bin/env python3
"""
配置管理模块

提供统一的配置管理、验证和默认值处理功能
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

class ConfigManager:
    """配置管理器
    
    功能：
    - 配置验证
    - 默认值管理
    - 配置继承和覆盖
    - 配置序列化和反序列化
    """
    
    # 默认配置模板
    DEFAULT_CONFIG = {
        # 基础配置
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_name': 'sr_model',
        'results_base_dir': 'results',
        'checkpoint_dir': 'checkpoints',
        
        # 训练配置
        'epochs': 100,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'weight_decay': 1e-2,
        'num_workers': 4,
        
        # 数据配置
        'patch_size': 64,
        'upscale_factor': 4,
        
        # 优化配置
        'use_mixed_precision': True,
        'accumulation_steps': 1,
        'gradient_clip_val': 1.0,
        
        # 早停配置
        'early_stopping_patience': None,
        'early_stopping_min_delta': 0.0,
        
        # 扩散模型特有配置
        'num_timesteps': 1000,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'noise_schedule': 'linear',
        'unet_channels': [64, 128, 256, 512],
        'attention_resolutions': [16, 8],
        'num_res_blocks': 2,
        'dropout': 0.0
    }
    
    # 必需的配置项
    REQUIRED_KEYS = {
        'train_lr_dir',
        'train_hr_dir',
        'val_lr_dir', 
        'val_hr_dir'
    }
    
    # 配置项类型验证
    TYPE_VALIDATORS = {
        'epochs': int,
        'batch_size': int,
        'learning_rate': float,
        'weight_decay': float,
        'num_workers': int,
        'patch_size': int,
        'upscale_factor': int,
        'use_mixed_precision': bool,
        'accumulation_steps': int,
        'gradient_clip_val': float,
        'num_timesteps': int,
        'beta_start': float,
        'beta_end': float,
        'noise_schedule': str,
        'unet_channels': list,
        'attention_resolutions': list,
        'num_res_blocks': int,
        'dropout': float
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化配置管理器
        
        Args:
            config: 用户提供的配置字典
        """
        self.logger = logging.getLogger(__name__)
        self._config = self.DEFAULT_CONFIG.copy()
        
        if config:
            self.update_config(config)
            
    def update_config(self, config: Dict[str, Any]) -> None:
        """更新配置
        
        Args:
            config: 新的配置字典
        """
        # 处理ExperimentConfig对象
        if hasattr(config, '__dict__'):
            config = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
            
        self._config.update(config)
        self.validate_config()
        
    def validate_config(self) -> None:
        """验证配置的完整性和正确性"""
        errors = []
        warnings = []
        
        # 检查必需的配置项
        for key in self.REQUIRED_KEYS:
            if key not in self._config or self._config[key] is None:
                errors.append(f"缺少必需的配置项: {key}")
                
        # 检查数据路径是否存在
        for path_key in ['train_lr_dir', 'train_hr_dir', 'val_lr_dir', 'val_hr_dir']:
            if path_key in self._config and self._config[path_key]:
                path = Path(self._config[path_key])
                if not path.exists():
                    warnings.append(f"路径不存在: {path_key} = {path}")
                    
        # 类型验证
        for key, expected_type in self.TYPE_VALIDATORS.items():
            if key in self._config:
                value = self._config[key]
                if value is not None and not isinstance(value, expected_type):
                    try:
                        # 尝试类型转换
                        self._config[key] = expected_type(value)
                        warnings.append(f"配置项 {key} 已自动转换为 {expected_type.__name__} 类型")
                    except (ValueError, TypeError):
                        errors.append(f"配置项 {key} 类型错误，期望 {expected_type.__name__}，实际 {type(value).__name__}")
                        
        # 值范围验证
        self._validate_ranges(errors, warnings)
        
        # 输出验证结果
        if warnings:
            for warning in warnings:
                self.logger.warning(warning)
                
        if errors:
            error_msg = "配置验证失败:\n" + "\n".join(errors)
            raise ValueError(error_msg)
            
        self.logger.info("配置验证通过")
        
    def _validate_ranges(self, errors: list, warnings: list) -> None:
        """验证配置值的范围"""
        # 学习率范围
        if 'learning_rate' in self._config:
            lr = self._config['learning_rate']
            if lr <= 0 or lr > 1:
                errors.append(f"学习率应在(0, 1]范围内，当前值: {lr}")
                
        # batch size范围
        if 'batch_size' in self._config:
            bs = self._config['batch_size']
            if bs <= 0:
                errors.append(f"batch_size应大于0，当前值: {bs}")
            elif bs > 64:
                warnings.append(f"batch_size较大({bs})，可能导致内存不足")
                
        # 扩散模型参数范围
        if 'beta_start' in self._config and 'beta_end' in self._config:
            beta_start = self._config['beta_start']
            beta_end = self._config['beta_end']
            if beta_start >= beta_end:
                errors.append(f"beta_start({beta_start})应小于beta_end({beta_end})")
                
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值
        """
        return self._config.get(key, default)
        
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self._config[key]
        
    def __setitem__(self, key: str, value: Any) -> None:
        """支持字典式设置"""
        self._config[key] = value
        
    def __contains__(self, key: str) -> bool:
        """支持in操作符"""
        return key in self._config
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._config.copy()
        
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """保存配置到文件
        
        Args:
            file_path: 文件路径
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 过滤不可序列化的对象
        serializable_config = {}
        for key, value in self._config.items():
            try:
                json.dumps(value)  # 测试是否可序列化
                serializable_config[key] = value
            except (TypeError, ValueError):
                serializable_config[key] = str(value)
                
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"配置已保存到: {file_path}")
        
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'ConfigManager':
        """从文件加载配置
        
        Args:
            file_path: 文件路径
            
        Returns:
            ConfigManager实例
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        return cls(config)
        
    def create_trainer_config(self, trainer_type: str = 'diffusion') -> Dict[str, Any]:
        """为特定训练器类型创建配置
        
        Args:
            trainer_type: 训练器类型 ('diffusion', 'sr', 'lora')
            
        Returns:
            训练器配置字典
        """
        config = self.to_dict()
        
        # 根据训练器类型添加特定配置
        if trainer_type == 'diffusion':
            config.update({
                'model_name': f"diffusion_{config.get('model_name', 'sr')}",
                'use_mixed_precision': True,  # 扩散模型推荐使用混合精度
                'gradient_clip_val': 1.0,     # 扩散模型需要梯度裁剪
            })
        elif trainer_type == 'sr':
            config.update({
                'model_name': f"sr_{config.get('model_name', 'basic')}",
            })
        elif trainer_type == 'lora':
            config.update({
                'model_name': f"lora_{config.get('model_name', 'diffusion')}",
                'lora_rank': config.get('lora_rank', 16),
                'lora_alpha': config.get('lora_alpha', 32),
            })
            
        return config
        
    def print_config(self) -> None:
        """打印当前配置"""
        print("\n=== 当前配置 ===")
        for key, value in sorted(self._config.items()):
            print(f"{key}: {value}")
        print("===============\n")


# 导入torch用于设备检测
try:
    import torch
except ImportError:
    # 如果torch未安装，使用CPU作为默认设备
    class MockTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False
    torch = MockTorch()