import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union

class TrainConfig:
    """训练配置管理类，用于管理和验证训练参数"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """初始化训练配置
        Args:
            config_dict: 配置字典
        """
        self.config = self._validate_and_process_config(config_dict)
    
    def _validate_and_process_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证和处理配置参数
        Args:
            config: 原始配置字典
        Returns:
            Dict[str, Any]: 处理后的配置字典
        Raises:
            ValueError: 当必需参数缺失或参数值无效时
        """
        # 复制配置字典以避免修改原始数据
        processed_config = config.copy()
        
        # 验证必需参数
        required_params = [
            'model_class',
            'lr_data_dir',
            'hr_data_dir'
        ]
        
        for param in required_params:
            if param not in config:
                raise ValueError(f"缺少必需参数: {param}")
        
        # 验证数据目录
        if not os.path.exists(config['lr_data_dir']):
            raise ValueError(f"LR数据目录不存在: {config['lr_data_dir']}")
        if not os.path.exists(config['hr_data_dir']):
            raise ValueError(f"HR数据目录不存在: {config['hr_data_dir']}")
        
        # 设置默认值
        defaults = {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'use_text_descriptions': False,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'criterion': nn.MSELoss(),
            'results_base_dir': 'results',
            'model_name': 'model',
            'early_stopping_min_delta': 0.0
        }
        
        for key, value in defaults.items():
            if key not in processed_config:
                processed_config[key] = value
        
        # 验证和处理特定参数
        self._validate_image_size_params(processed_config)
        self._validate_edge_detection_params(processed_config)
        self._validate_validation_data_params(processed_config)
        
        return processed_config
    
    def _validate_image_size_params(self, config: Dict[str, Any]) -> None:
        """验证图像尺寸相关参数
        Args:
            config: 配置字典
        Raises:
            ValueError: 当参数无效时
        """
        # 如果既没有指定lr_patch_size也没有指定image_size，设置默认值
        if 'lr_patch_size' not in config and 'image_size' not in config:
            config['lr_patch_size'] = 48
            config['upscale_factor'] = 2
        
        # 如果指定了image_size，确保也指定了upscale_factor
        if 'image_size' in config and 'upscale_factor' not in config:
            raise ValueError("当指定image_size时，必须同时指定upscale_factor")
    
    def _validate_edge_detection_params(self, config: Dict[str, Any]) -> None:
        """验证边缘检测相关参数
        Args:
            config: 配置字典
        """
        if 'edge_detection_methods' in config:
            valid_methods = {'sobel', 'canny', 'laplacian'}
            methods = config['edge_detection_methods']
            if not isinstance(methods, (list, tuple)):
                raise ValueError("edge_detection_methods必须是列表或元组")
            if not all(method in valid_methods for method in methods):
                raise ValueError(f"无效的边缘检测方法。有效方法为: {valid_methods}")
    
    def _validate_validation_data_params(self, config: Dict[str, Any]) -> None:
        """验证验证数据相关参数
        Args:
            config: 配置字典
        """
        # 如果没有指定验证数据目录，尝试从训练数据目录推断
        if 'val_lr_data_dir' not in config:
            config['val_lr_data_dir'] = config['lr_data_dir'].replace('/lr', '/val/lr')
        if 'val_hr_data_dir' not in config:
            config['val_hr_data_dir'] = config['hr_data_dir'].replace('/hr', '/val/hr')
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值
        Args:
            key: 配置键
            default: 默认值
        Returns:
            Any: 配置值
        """
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """通过字典方式访问配置值
        Args:
            key: 配置键
        Returns:
            Any: 配置值
        """
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """检查配置是否包含指定键
        Args:
            key: 配置键
        Returns:
            bool: 是否包含该键
        """
        return key in self.config