import torch
import torch.nn as nn
from models.simple_srcnn import SimpleSRCNN
from losses import L1Loss, CombinedLoss, EdgeLoss

class ExperimentConfig:
    """
    实验配置管理类，用于管理所有实验相关的配置
    """
    def __init__(self):
        # 数据目录配置
        self.TRAIN_LR_DIR = './data/DIV2K/DIV2K_train_LR_bicubic/X2'
        self.TRAIN_HR_DIR = './data/DIV2K/DIV2K_train_HR'
        self.VAL_LR_DIR = './data/DIV2K/DIV2K_valid_LR_bicubic/X2'
        self.VAL_HR_DIR = './data/DIV2K/DIV2K_valid_HR'
        
        # 基础训练参数
        self.model_class = SimpleSRCNN
        self.epochs = 50
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.train_lr_dir = self.TRAIN_LR_DIR
        self.train_hr_dir = self.TRAIN_HR_DIR
        self.val_lr_dir = self.VAL_LR_DIR
        self.val_hr_dir = self.VAL_HR_DIR
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.upscale_factor = 2
        self.patch_size = 64
        
        # 梯度累积和混合精度训练配置
        self.accumulation_steps = 1  # 梯度累积步数
        self.use_mixed_precision = True  # 启用混合精度训练
        self.gradient_clip_val = 1.0  # 梯度裁剪阈值
        self.weight_decay = 1e-5  # 权重衰减
        self.num_workers = 4  # 数据加载器工作进程数
        
        self.BASE_TRAIN_PARAMS = {
            'model_class': SimpleSRCNN,
            'epochs': 100,  # 用于快速测试的较小轮数
            'batch_size': 256,
            'learning_rate': 0.001,
            'lr_data_dir': self.TRAIN_LR_DIR,
            'hr_data_dir': self.TRAIN_HR_DIR,
            'use_text_descriptions': False,
            'criterion': nn.MSELoss(),  # 默认损失函数
            'results_base_dir': 'results_edge_experiments',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'upscale_factor': 2,  # 放大倍数
            'lr_patch_size': 48  # 训练用的LR图像块大小
        }
        
        # 边缘检测方法组合
        self.EDGE_COMBINATIONS = [
            [],  # 无边缘检测
            ['sobel'],  # Sobel边缘检测
            ['canny'],  # Canny边缘检测
            ['laplacian'],  # Laplacian边缘检测
            ['sobel', 'canny'],  # Sobel和Canny组合
            ['sobel', 'laplacian'],  # Sobel和Laplacian组合
            ['canny', 'laplacian'],  # Canny和Laplacian组合
            ['sobel', 'canny', 'laplacian']  # 三种方法组合
        ]
        
        # 损失函数配置
        self.LOSS_FUNCTIONS = {
            'mse': nn.MSELoss(),
            'l1': L1Loss(),
            'sobel_edge_loss': EdgeLoss(edge_detector_type='sobel'),
            'canny_edge_loss': EdgeLoss(edge_detector_type='canny'),
            'laplacian_edge_loss': EdgeLoss(edge_detector_type='laplacian'),
        }
        
        # 边缘检测方法和损失函数组合
        self.edge_combinations = self.EDGE_COMBINATIONS
        self.loss_functions = ['mse', 'l1']
        
        # 实验配置列表
        self.TRAINING_RUNS = self._create_training_runs()
    
    def create_training_configs(self):
        """创建不同的训练配置组合
        Returns:
            list: 训练配置列表
        """
        configs = []
        
        for edge_combo in self.edge_combinations:
            for loss_func in self.loss_functions:
                config = {
                    'model_class': self.model_class,
                    'epochs': self.epochs,
                    'batch_size': self.batch_size,
                    'learning_rate': self.learning_rate,
                    'train_lr_dir': self.train_lr_dir,
                    'train_hr_dir': self.train_hr_dir,
                    'val_lr_dir': self.val_lr_dir,
                    'val_hr_dir': self.val_hr_dir,
                    'device': self.device,
                    'upscale_factor': self.upscale_factor,
                    'patch_size': self.patch_size,
                    'edge_methods': edge_combo,
                    'loss_type': loss_func,
                    'experiment_name': f"edge_{'-'.join(edge_combo) if edge_combo else 'none'}_loss_{loss_func}",
                    # 梯度累积和混合精度训练配置
                    'accumulation_steps': self.accumulation_steps,
                    'use_mixed_precision': self.use_mixed_precision,
                    'gradient_clip_val': self.gradient_clip_val,
                    'weight_decay': self.weight_decay,
                    'num_workers': self.num_workers
                }
                configs.append(config)
        
        return configs
    
    def create_optimized_config(self, accumulation_steps=4, use_mixed_precision=True, gradient_clip_val=1.0):
        """创建优化的训练配置（支持梯度累积和混合精度）
        Args:
            accumulation_steps: 梯度累积步数
            use_mixed_precision: 是否使用混合精度训练
            gradient_clip_val: 梯度裁剪阈值
        Returns:
            dict: 优化的训练配置
        """
        config = {
            'model_class': self.model_class,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'train_lr_dir': self.train_lr_dir,
            'train_hr_dir': self.train_hr_dir,
            'val_lr_dir': self.val_lr_dir,
            'val_hr_dir': self.val_hr_dir,
            'device': self.device,
            'upscale_factor': self.upscale_factor,
            'patch_size': self.patch_size,
            'edge_methods': [],  # 默认不使用边缘检测
            'loss_type': 'mse',  # 默认使用MSE损失
            'experiment_name': f"optimized_acc{accumulation_steps}_mp{use_mixed_precision}",
            # 优化配置
            'accumulation_steps': accumulation_steps,
            'use_mixed_precision': use_mixed_precision,
            'gradient_clip_val': gradient_clip_val,
            'weight_decay': self.weight_decay,
            'num_workers': self.num_workers
        }
        return config
    
    def _create_training_runs(self):
        """
        创建实验配置列表
        Returns:
            list: 实验配置列表
        """
        return [
            {
                'name': 'no_edge_mse',
                'edge_methods': [],
                'criterion': self.LOSS_FUNCTIONS['mse'],
            },
            {
                'name': 'sobel_edge_mse',
                'edge_methods': ['sobel'],
                'criterion': self.LOSS_FUNCTIONS['mse'],
            },
            {
                'name': 'canny_edge_mse',
                'edge_methods': ['canny'],
                'criterion': self.LOSS_FUNCTIONS['mse'],
            },
            {
                'name': 'sobel_canny_edge_mse',
                'edge_methods': ['sobel', 'canny'],
                'criterion': self.LOSS_FUNCTIONS['mse'],
            },
            {
                'name': 'sobel_edge_combined_loss',
                'edge_methods': ['sobel'],
                'criterion': CombinedLoss([
                    (self.LOSS_FUNCTIONS['mse'], 1.0),
                    (self.LOSS_FUNCTIONS['sobel_edge_loss'], 0.1)
                ]),
            },
            {
                'name': 'canny_edge_combined_loss',
                'edge_methods': ['canny'],
                'criterion': CombinedLoss([
                    (self.LOSS_FUNCTIONS['mse'], 1.0),
                    (self.LOSS_FUNCTIONS['canny_edge_loss'], 0.1)
                ]),
            },
            {
                'name': 'sobel_canny_edge_combined_loss',
                'edge_methods': ['sobel', 'canny'],
                'criterion': CombinedLoss([
                    (self.LOSS_FUNCTIONS['mse'], 1.0),
                    (self.LOSS_FUNCTIONS['sobel_edge_loss'], 0.1),
                    (self.LOSS_FUNCTIONS['canny_edge_loss'], 0.1)
                ]),
            },
            {
                'name': 'laplacian_edge_mse',
                'edge_methods': ['laplacian'],
                'criterion': self.LOSS_FUNCTIONS['mse'],
            },
            {
                'name': 'laplacian_edge_combined_loss',
                'edge_methods': ['laplacian'],
                'criterion': CombinedLoss([
                    (self.LOSS_FUNCTIONS['mse'], 1.0),
                    (self.LOSS_FUNCTIONS['laplacian_edge_loss'], 0.1)
                ]),
            },
            {
                'name': 'sobel_laplacian_edge_combined_loss',
                'edge_methods': ['sobel', 'laplacian'],
                'criterion': CombinedLoss([
                    (self.LOSS_FUNCTIONS['mse'], 1.0),
                    (self.LOSS_FUNCTIONS['sobel_edge_loss'], 0.1),
                    (self.LOSS_FUNCTIONS['laplacian_edge_loss'], 0.1)
                ]),
            },
            {
                'name': 'canny_laplacian_edge_combined_loss',
                'edge_methods': ['canny', 'laplacian'],
                'criterion': CombinedLoss([
                    (self.LOSS_FUNCTIONS['mse'], 1.0),
                    (self.LOSS_FUNCTIONS['canny_edge_loss'], 0.1),
                    (self.LOSS_FUNCTIONS['laplacian_edge_loss'], 0.1)
                ]),
            },
            {
                'name': 'sobel_canny_laplacian_edge_combined_loss',
                'edge_methods': ['sobel', 'canny', 'laplacian'],
                'criterion': CombinedLoss([
                    (self.LOSS_FUNCTIONS['mse'], 1.0),
                    (self.LOSS_FUNCTIONS['sobel_edge_loss'], 0.1),
                    (self.LOSS_FUNCTIONS['canny_edge_loss'], 0.1),
                    (self.LOSS_FUNCTIONS['laplacian_edge_loss'], 0.1)
                ]),
            },
        ]