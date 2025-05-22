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
        
        # 实验配置列表
        self.TRAINING_RUNS = self._create_training_runs()
    
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