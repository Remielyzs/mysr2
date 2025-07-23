import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
import os
import numpy as np
from tqdm import tqdm
import math
from pathlib import Path
import json
import sys
import traceback

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config_manager import ConfigManager
from utils.logger import TrainingLogger

from trainers.base_trainer import BaseTrainer
from models.diffusion_sr import DiffusionSRModel
from losses import MSELoss, L1Loss
from data_utils import SRDataset

class DiffusionTrainer(BaseTrainer):
    """扩散模型超分辨率训练器
    
    实现了扩散模型的特殊训练逻辑，包括：
    - 噪声调度器
    - 时间步长采样
    - 噪声预测损失
    - 支持梯度累积和混合精度训练
    """
    
    def __init__(self, config):
        """初始化扩散模型训练器
        
        Args:
            config: 训练配置字典或ConfigManager实例
        """
        # 确保配置是ConfigManager实例
        if not isinstance(config, ConfigManager):
            if hasattr(config, '__dict__') and not isinstance(config, dict):
                # ExperimentConfig对象转换为字典
                config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
                config = ConfigManager(config_dict)
            elif isinstance(config, dict):
                config = ConfigManager(config)
            else:
                raise TypeError("配置必须是字典、ConfigManager或具有__dict__属性的对象")
        
        # 调用父类初始化
        super().__init__(config)
        
        # 确保必要的目录属性存在
        self._ensure_required_directories()
        
        # 扩散模型特有配置
        self.num_timesteps = self.config_manager.get('num_timesteps', 1000)
        self.beta_start = self.config_manager.get('beta_start', 0.0001)
        self.beta_end = self.config_manager.get('beta_end', 0.02)
        self.gradient_clip_val = self.config_manager.get('gradient_clip_val', 1.0)
        self.noise_schedule = self.config_manager.get('noise_schedule', 'linear')
        
        try:
            # 初始化噪声调度器
            self.setup_noise_scheduler()
            
            # 记录扩散模型特有信息
            diffusion_info = {
                'num_timesteps': self.num_timesteps,
                'noise_schedule': self.noise_schedule,
                'beta_range': f'[{self.beta_start}, {self.beta_end}]',
                'gradient_clip_val': self.gradient_clip_val
            }
            
            self.logger.info("扩散模型训练器初始化完成", extra=diffusion_info)
        except Exception as e:
            self.logger.error(f"扩散模型训练器初始化失败: {str(e)}")
            traceback.print_exc()
            raise
    
    def _ensure_required_directories(self):
        """确保必要的目录属性存在"""
        required_dirs = ['checkpoint_dir', 'run_dir', 'results_dir', 'logs_dir']
        for dir_attr in required_dirs:
            if not hasattr(self, dir_attr) or getattr(self, dir_attr) is None:
                # 如果目录属性不存在，创建默认目录
                base_dir = self.config_manager.get('results_base_dir', 'results')
                model_name = self.config_manager.get('model_name', 'diffusion_model')
                dir_path = Path(base_dir) / model_name / dir_attr.replace('_dir', '')
                dir_path.mkdir(parents=True, exist_ok=True)
                setattr(self, dir_attr, str(dir_path))
                self.logger.debug(f"创建默认{dir_attr}: {dir_path}")
    
    def setup_noise_scheduler(self):
        """设置噪声调度器
        
        实现不同的噪声调度策略：
        - linear: 线性调度
        - cosine: 余弦调度
        """
        try:
            if self.noise_schedule == 'linear':
                # 线性噪声调度
                self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
            elif self.noise_schedule == 'cosine':
                # 余弦噪声调度
                steps = self.num_timesteps + 1
                x = torch.linspace(0, self.num_timesteps, steps)
                alphas_cumprod = torch.cos(((x / self.num_timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
                alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
                betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
                self.betas = torch.clip(betas, 0.0001, 0.9999)
            else:
                raise ValueError(f"不支持的噪声调度类型: {self.noise_schedule}")
            
            # 计算扩散过程中的关键参数
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
            
            # 用于采样的参数
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
            
            # 将所有噪声调度器参数移动到正确的设备
            self.betas = self.betas.to(self.device)
            self.alphas = self.alphas.to(self.device)
            self.alphas_cumprod = self.alphas_cumprod.to(self.device)
            self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)
            self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
            self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(self.device)
            
            self.logger.info(f"噪声调度器设置完成，Beta值范围: [{self.betas.min():.6f}, {self.betas.max():.6f}]")
        except Exception as e:
            self.logger.error(f"噪声调度器设置失败: {str(e)}")
            raise
    
    def setup_model(self):
        """设置扩散模型"""
        try:
            model_config = {
                'scale_factor': self.config_manager.get('upscale_factor', 4),
                'num_timesteps': self.num_timesteps,
                'input_channels': 3,  # RGB图像
                'output_channels': 3,  # RGB图像
                'unet_channels': self.config_manager.get('unet_channels', [64, 128, 256, 512]),
                'attention_resolutions': self.config_manager.get('attention_resolutions', [16, 8]),
                'num_res_blocks': self.config_manager.get('num_res_blocks', 2),
                'dropout': self.config_manager.get('dropout', 0.0)
            }
            
            self.model = DiffusionSRModel(config=model_config)
            self.model = self.model.to(self.device)
            
            # 打印模型参数数量
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            model_info = {
                'scale_factor': model_config['scale_factor'],
                'total_params': f"{total_params:,}",
                'trainable_params': f"{trainable_params:,}",
                'unet_channels': model_config['unet_channels'],
                'attention_resolutions': model_config['attention_resolutions']
            }
            
            self.logger.info("扩散模型已创建", extra=model_info)
            return True
        except Exception as e:
            self.logger.error(f"模型设置失败: {str(e)}")
            traceback.print_exc()
            return False
    
    def setup_data(self):
        """设置数据加载器"""
        try:
            # 数据变换
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
            # 获取数据路径
            train_lr_dir = self.config_manager.get('train_lr_dir')
            train_hr_dir = self.config_manager.get('train_hr_dir')
            val_lr_dir = self.config_manager.get('val_lr_dir')
            val_hr_dir = self.config_manager.get('val_hr_dir')
            
            # 验证数据路径
            for path_name, path in [('train_lr_dir', train_lr_dir), ('train_hr_dir', train_hr_dir), 
                                  ('val_lr_dir', val_lr_dir), ('val_hr_dir', val_hr_dir)]:
                if not path or not Path(path).exists():
                    self.logger.warning(f"{path_name} 路径不存在或无效: {path}")
            
            # 创建数据集
            train_dataset = SRDataset(
                lr_dir=train_lr_dir,
                hr_dir=train_hr_dir,
                lr_patch_size=self.config_manager.get('patch_size', 64),
                upscale_factor=self.config_manager.get('upscale_factor', 4),
                edge_methods=None,  # 扩散模型通常不使用边缘检测
                transform=transform
            )
            
            val_dataset = SRDataset(
                lr_dir=None,  # 主要lr_dir参数
                hr_dir=None,  # 主要hr_dir参数
                mode='eval',
                val_lr_dir=val_lr_dir,
                val_hr_dir=val_hr_dir,
                lr_patch_size=self.config_manager.get('patch_size', 64),
                upscale_factor=self.config_manager.get('upscale_factor', 4),
                edge_methods=None,
                device=self.device
            )
            
            # 创建数据加载器
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.config_manager.get('batch_size', 8),
                shuffle=True,
                num_workers=self.config_manager.get('num_workers', 4),
                pin_memory=True if 'cuda' in str(self.device) else False
            )
            
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.config_manager.get('batch_size', 8),
                shuffle=False,
                num_workers=self.config_manager.get('num_workers', 4),
                pin_memory=True if 'cuda' in str(self.device) else False
            )
            
            data_info = {
                'train_dataset_size': len(train_dataset),
                'val_dataset_size': len(val_dataset),
                'train_lr_dir': train_lr_dir,
                'train_hr_dir': train_hr_dir,
                'val_lr_dir': val_lr_dir,
                'val_hr_dir': val_hr_dir,
                'patch_size': self.config_manager.get('patch_size', 64),
                'upscale_factor': self.config_manager.get('upscale_factor', 4)
            }
            
            self.logger.info("数据加载器设置完成", extra=data_info)
            return True
        except Exception as e:
            self.logger.error(f"数据加载器设置失败: {str(e)}")
            traceback.print_exc()
            return False
    
    def setup_optimizer(self):
        """设置损失函数和优化器"""
        try:
            # 扩散模型通常使用MSE损失来预测噪声
            self.criterion = MSELoss()
            
            # 获取优化器参数
            learning_rate = self.config_manager.get('learning_rate', 1e-4)
            weight_decay = self.config_manager.get('weight_decay', 1e-2)
            
            # 设置优化器（扩散模型通常使用较小的学习率）
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
            
            optimizer_info = {
                'optimizer': 'AdamW',
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'loss_function': 'MSE (噪声预测)'
            }
            
            self.logger.info("优化器设置完成", extra=optimizer_info)
            return True
        except Exception as e:
            self.logger.error(f"优化器设置失败: {str(e)}")
            traceback.print_exc()
            return False
    
    def add_noise(self, hr_images, timesteps):
        """向高分辨率图像添加噪声
        
        Args:
            hr_images: 高分辨率图像 (B, C, H, W)
            timesteps: 时间步长 (B,)
            
        Returns:
            noisy_images: 添加噪声后的图像
            noise: 添加的噪声
        """
        # 生成随机噪声
        noise = torch.randn_like(hr_images)
        
        # 获取对应时间步的参数（已经在正确的设备上）
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        # 添加噪声: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        noisy_images = sqrt_alphas_cumprod_t * hr_images + sqrt_one_minus_alphas_cumprod_t * noise
        
        return noisy_images, noise
    
    def process_batch(self, batch_data, is_training=True, batch_idx=0):
        """处理一个batch的数据（扩散模型特殊逻辑）"""
        lr_images, hr_images = batch_data
        lr_images = lr_images.to(self.device)
        hr_images = hr_images.to(self.device)
        
        # 随机采样时间步长
        batch_size = hr_images.shape[0]
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        
        if is_training:
            # 训练模式
            if self.use_mixed_precision:
                # 混合精度训练
                with autocast():
                    noisy_hr_images, noise = self.add_noise(hr_images, timesteps)
                    predicted_noise = self.model.forward_with_noisy_hr(
                        lr_images, noisy_hr_images, timesteps
                    )
                    
                    # 计算噪声预测损失
                    loss = self.criterion(predicted_noise, noise)
                
                # 混合精度反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度累积处理
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.gradient_clip_val > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # 标准精度训练
                noisy_hr_images, noise = self.add_noise(hr_images, timesteps)
                predicted_noise = self.model.forward_with_noisy_hr(
                    lr_images, noisy_hr_images, timesteps
                )
                loss = self.criterion(predicted_noise, noise)
                
                # 标准反向传播
                loss.backward()
                
                # 梯度累积处理
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            return loss.item()
        else:
            # 验证模式
            with torch.no_grad():
                if self.use_mixed_precision:
                    with autocast():
                        noisy_hr_images, noise = self.add_noise(hr_images, timesteps)
                        predicted_noise = self.model.forward_with_noisy_hr(
                            lr_images, noisy_hr_images, timesteps
                        )
                        loss = self.criterion(predicted_noise, noise)
                else:
                    noisy_hr_images, noise = self.add_noise(hr_images, timesteps)
                    predicted_noise = self.model.forward_with_noisy_hr(
                        lr_images, noisy_hr_images, timesteps
                    )
                    loss = self.criterion(predicted_noise, noise)
                
                return loss.item()
    
    def sample(self, lr_image, num_inference_steps=50):
        """使用DDPM采样生成超分辨率图像
        
        Args:
            lr_image: 低分辨率输入图像 (1, C, H, W)
            num_inference_steps: 推理步数
            
        Returns:
            sr_image: 生成的超分辨率图像
        """
        self.model.eval()
        
        with torch.no_grad():
            # 初始化为纯噪声
            scale_factor = self.config.get('upscale_factor', 4)
            b, c, h, w = lr_image.shape
            hr_shape = (b, c, h * scale_factor, w * scale_factor)
            
            # 从纯噪声开始
            x = torch.randn(hr_shape, device=lr_image.device)
            
            # 计算采样步长
            timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long)
            
            for i, t in enumerate(tqdm(timesteps, desc="DDPM采样")):
                t_batch = torch.full((b,), t, device=lr_image.device, dtype=torch.long)
                
                # 预测噪声
                predicted_noise = self.model(lr_image, noise_level=t_batch)
                
                # DDPM去噪步骤
                alpha_t = self.alphas[t]
                alpha_cumprod_t = self.alphas_cumprod[t]
                beta_t = self.betas[t]
                
                # 计算去噪后的图像
                if i < len(timesteps) - 1:
                    # 不是最后一步，添加噪声
                    noise = torch.randn_like(x)
                    x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
                else:
                    # 最后一步，不添加噪声
                    x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise)
                
                # 裁剪到有效范围
                x = torch.clamp(x, -1, 1)
            
            return x
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点（包含扩散模型特有状态）"""
        if os.path.exists(checkpoint_path):
            print(f"从检查点恢复训练: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载混合精度scaler状态
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print("已加载混合精度scaler状态")
            
            # 加载扩散模型特有状态
            if 'noise_scheduler_state' in checkpoint:
                scheduler_state = checkpoint['noise_scheduler_state']
                self.betas = scheduler_state['betas']
                self.alphas = scheduler_state['alphas']
                self.alphas_cumprod = scheduler_state['alphas_cumprod']
                self.sqrt_alphas_cumprod = scheduler_state['sqrt_alphas_cumprod']
                self.sqrt_one_minus_alphas_cumprod = scheduler_state['sqrt_one_minus_alphas_cumprod']
                print("已加载噪声调度器状态")
            
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_loss = checkpoint.get('loss', float('inf'))
            print(f"已加载epoch {checkpoint['epoch']}的模型，loss为{checkpoint.get('loss', 'N/A')}")
        else:
            print(f"未找到检查点 {checkpoint_path}，从头开始训练。")
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """保存检查点（包含扩散模型特有状态）"""
        try:
            # 确保目录存在
            self._ensure_required_directories()
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'config': self.config_manager.to_dict(),
                'noise_scheduler_state': {
                    'betas': self.betas,
                    'alphas': self.alphas,
                    'alphas_cumprod': self.alphas_cumprod,
                    'sqrt_alphas_cumprod': self.sqrt_alphas_cumprod,
                    'sqrt_one_minus_alphas_cumprod': self.sqrt_one_minus_alphas_cumprod
                }
            }
            
            # 如果使用混合精度训练，保存scaler状态
            if self.scaler:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            # 保存当前epoch的检查点
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint, checkpoint_path)
            
            # 保存最新模型（方便恢复训练）
            latest_model_path = os.path.join(self.checkpoint_dir, "latest_model.pth")
            torch.save(checkpoint, latest_model_path)
            
            # 如果是最佳模型，额外保存一份
            if is_best:
                best_model_path = os.path.join(self.results_dir, f"best_model.pth")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'loss': loss,
                    'config': self.config_manager.to_dict()
                }, best_model_path)
                self.logger.info(f"保存最佳模型，验证loss: {loss:.4f}", extra={'path': best_model_path})
            
            self.logger.info(f"保存检查点，epoch: {epoch+1}, loss: {loss:.4f}", 
                           extra={'path': checkpoint_path})
            return True
        except Exception as e:
            self.logger.error(f"保存检查点失败: {str(e)}")
            traceback.print_exc()
            return False